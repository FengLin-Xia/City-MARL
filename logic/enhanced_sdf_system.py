#!/usr/bin/env python3
"""
高斯核地价场系统 - 实现连续的城市地价分布
"""

import numpy as np
import math
from typing import List, Dict, Tuple
import json
import os

class GaussianLandPriceSystem:
    """高斯核地价场系统"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sdf_config = config.get('gaussian_land_price_system', {})
        
        # 地图尺寸
        self.map_size = [256, 256]
        
        # 交通枢纽位置
        self.transport_hubs = []
        
        # 地价场
        self.land_price_field = None
        
        # 演化历史
        self.evolution_history = []
        
        # 当前月份
        self.current_month = 0
        
        # 获取配置参数
        self.meters_per_pixel = self.sdf_config.get('meters_per_pixel', 2.0)
        
        # 高斯核参数（像素单位）- 从配置读取米值并换算
        hub_sigma_base_m = float(self.sdf_config.get('hub_sigma_base_m', 40))
        road_sigma_base_m = float(self.sdf_config.get('road_sigma_base_m', 20))
        self.hub_sigma_base = int(hub_sigma_base_m / self.meters_per_pixel)
        self.road_sigma_base = int(road_sigma_base_m / self.meters_per_pixel)
        
        # 演化参数
        self.hub_growth_rate = float(self.sdf_config.get('hub_growth_rate', 0.03))
        self.road_growth_rate = float(self.sdf_config.get('road_growth_rate', 0.02))
        self.max_hub_multiplier = float(self.sdf_config.get('max_hub_multiplier', 2.0))
        self.max_road_multiplier = float(self.sdf_config.get('max_road_multiplier', 2.5))
        
        # 地价值参数
        self.hub_peak_value = float(self.sdf_config.get('hub_peak_value', 1.0))
        self.road_peak_value = float(self.sdf_config.get('road_peak_value', 0.7))
        self.min_threshold = float(self.sdf_config.get('min_threshold', 0.1))

        # 额外的“静态 Hub 点核”（始终叠加，不受演化启停影响）
        self.extra_hub_point_peak = float(self.sdf_config.get('extra_hub_point_peak', 0.0))
        # 以像素为单位的 sigma；若 <=0 则回退到 hub_sigma_base
        self.extra_hub_point_sigma_px = float(self.sdf_config.get('extra_hub_point_sigma_px', 0.0))

        # 河流配置（可选）：从 terrain_features.rivers 读取；若无则回退 river.txt
        self.rivers: List[Dict] = []
        tf = self.config.get('terrain_features', {}) if isinstance(self.config, dict) else {}
        rivers_cfg = tf.get('rivers', []) if isinstance(tf, dict) else []
        for r in rivers_cfg:
            coords = r.get('coordinates', []) or []
            # 若未给坐标，尝试用 river.txt
            if (not coords) and os.path.exists('river.txt'):
                try:
                    tmp: List[Tuple[float, float]] = []
                    import re
                    with open('river.txt', 'r', encoding='utf-8') as f:
                        for line in f:
                            s = line.strip()
                            if not s or s.startswith('#'):
                                continue
                            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
                            if len(nums) >= 2:
                                tmp.append((float(nums[0]), float(nums[1])))
                    coords = tmp
                except Exception:
                    coords = []
            if not coords:
                continue
            lp = r.get('land_price', {}) or {}
            self.rivers.append({
                'coords': [(float(x), float(y)) for x, y in coords if isinstance(x, (int, float)) or isinstance(y, (int, float))],
                'enabled': bool(lp.get('enabled', True)),
                'peak_value': float(lp.get('peak_value', 0.8)),
                'decay_rate': float(lp.get('decay_rate', 0.05)),  # for exponential mode
                'max_influence_distance': float(lp.get('max_influence_distance', 20.0)),
                'decay_mode': str(lp.get('decay_mode', 'exponential')),  # 'exponential' | 'gaussian' | 'lorentzian'
                'sigma_px': float(lp.get('sigma_px', 6.0)),       # for gaussian mode
                'gamma_px': float(lp.get('gamma_px', 6.0)),       # for lorentzian mode
            })

        # 若未配置，尝试从 river.txt 加载一条河流（默认开启）
        if not self.rivers and os.path.exists('river.txt'):
            try:
                coords: List[Tuple[float, float]] = []
                import re
                with open('river.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith('#'):
                            continue
                        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
                        if len(nums) >= 2:
                            coords.append((float(nums[0]), float(nums[1])))
                if len(coords) >= 2:
                    self.rivers.append({
                        'coords': coords,
                        'enabled': True,
                        'peak_value': 0.8,
                        'decay_rate': 0.05,
                        'max_influence_distance': 20.0,
                        'decay_mode': 'exponential',
                        'sigma_px': 6.0,
                        'gamma_px': 6.0,
                    })
            except Exception:
                pass
        
        # --- 融合模式配置 ---
        # fusion_mode: 'max' | 'weighted_sum'
        self.fusion_mode = str(self.sdf_config.get('fusion_mode', 'max')).lower()
        # fusion_weights: {hub, road, river, build}
        fw = self.sdf_config.get('fusion_weights', {}) or {}
        self.fusion_weights = {
            'hub': float(fw.get('hub', 1.0)),
            'road': float(fw.get('road', 1.0)),
            'river': float(fw.get('river', 1.0)),
            'build': float(fw.get('build', 1.0)),
        }
        # 组件先归一化到[0,1]再加权
        self.component_normalize = bool(self.sdf_config.get('component_normalize', True))
        # 融合结果归一化：'none' | 'minmax' | 'clip01'
        self.fusion_normalize = str(self.sdf_config.get('fusion_normalize', 'minmax')).lower()

        # --- 建筑点核（可选） ---
        bk = self.sdf_config.get('building_kernel', {}) or {}
        self.building_kernel_enabled = bool(bk.get('enabled', False))
        self.building_kernel_peak = float(bk.get('peak_value', 0.1))
        self.building_kernel_sigma_px = float(bk.get('sigma_px', 3.0))
        # 对于 max 融合，作为乘权后再与其他场取 max；对于 weighted_sum，作为一个独立分量参与加权
        self.building_kernel_weight = float(bk.get('weight', 1.0))
        # 生效建筑类型列表：如 ['public','industrial']
        types = bk.get('types', ['public', 'industrial'])
        self.building_kernel_types = [str(t) for t in types] if isinstance(types, list) else ['public', 'industrial']

        print(f"[LandPrice] Gaussian system initialized")
        
    def initialize_system(self, transport_hubs: List[List[int]], map_size: List[int]):
        """初始化系统"""
        self.transport_hubs = transport_hubs
        self.map_size = map_size
        self.land_price_field = self._create_initial_land_price()
        print(f"[LandPrice] Initialized with {len(transport_hubs)} hubs")
        
    def _create_initial_land_price(self) -> np.ndarray:
        """创建初始地价场"""
        return self._create_land_price_field(month=0, city_state=None)
        
    def _gaussian_2d(self, x: np.ndarray, y: np.ndarray, center_x: float, center_y: float, sigma: float, peak_value: float) -> np.ndarray:
        """创建2D高斯核"""
        distance_squared = (x - center_x)**2 + (y - center_y)**2
        gaussian = peak_value * np.exp(-distance_squared / (2 * sigma**2))
        return gaussian
    
    def _line_gaussian(self, x: np.ndarray, y: np.ndarray, hub1: List[int], hub2: List[int], sigma: float, peak_value: float) -> np.ndarray:
        """创建线状高斯核（道路影响）"""
        if len(self.transport_hubs) < 2:
            return np.zeros_like(x)
        
        dx = hub2[0] - hub1[0]
        dy = hub2[1] - hub1[1]
        length = math.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return np.zeros_like(x)
        
        ux, uy = dx / length, dy / length
        X, Y = np.meshgrid(np.arange(self.map_size[0]), np.arange(self.map_size[1]))
        
        px = X - hub1[0]
        py = Y - hub1[1]
        proj_length = px * ux + py * uy
        perp_distance = np.sqrt((X - (hub1[0] + proj_length * ux))**2 + (Y - (hub1[1] + proj_length * uy))**2)
        
        line_gaussian = np.zeros_like(X, dtype=float)
        road_mask = (proj_length >= 0) & (proj_length <= length)
        line_gaussian[road_mask] = peak_value * np.exp(-perp_distance[road_mask]**2 / (2 * sigma**2))
        
        return line_gaussian
    
    def _create_land_price_field(self, month: int = 0, city_state: Dict = None) -> np.ndarray:
        """创建地价场 - 支持渐进式演化"""
        hub_sigma = self._calculate_hub_sigma(month)
        road_sigma = self._calculate_road_sigma(month)
        
        X, Y = np.meshgrid(np.arange(self.map_size[0]), np.arange(self.map_size[1]))
        # 分量初始化
        land_price = np.zeros(self.map_size, dtype=float)
        hub_land_price = np.zeros(self.map_size, dtype=float)
        road_land_price = np.zeros(self.map_size, dtype=float)
        river_land_price = np.zeros(self.map_size, dtype=float)
        build_land_price = np.zeros(self.map_size, dtype=float)
        
        # 获取组件强度
        road_strength = self._get_component_strength('road', month)
        hub1_strength = self._get_component_strength('hub1', month)
        hub2_strength = self._get_component_strength('hub2', month)
        hub3_strength = self._get_component_strength('hub3', month)
        
        # 添加道路高斯核（如果激活）
        if road_strength > 0 and len(self.transport_hubs) >= 2:
            road_land_price = self._line_gaussian(X, Y, self.transport_hubs[0], self.transport_hubs[1], road_sigma, road_strength)
        
        # 添加Hub高斯核（根据强度）
        for i, hub in enumerate(self.transport_hubs):
            if i == 0 and hub1_strength > 0:  # Hub1
                hub_gaussian = self._gaussian_2d(X, Y, hub[0], hub[1], hub_sigma, hub1_strength)
                hub_land_price = np.maximum(hub_land_price, hub_gaussian)
            elif i == 1 and hub2_strength > 0:  # Hub2
                hub_gaussian = self._gaussian_2d(X, Y, hub[0], hub[1], hub_sigma, hub2_strength)
                hub_land_price = np.maximum(hub_land_price, hub_gaussian)
            elif i == 2 and hub3_strength > 0:  # Hub3
                hub_gaussian = self._gaussian_2d(X, Y, hub[0], hub[1], hub_sigma, hub3_strength)
                hub_land_price = np.maximum(hub_land_price, hub_gaussian)

            # 叠加静态 Hub 点核（可选）
            if self.extra_hub_point_peak > 0.0:
                stat_sigma = self.extra_hub_point_sigma_px if self.extra_hub_point_sigma_px > 0 else self.hub_sigma_base
                stat_gauss = self._gaussian_2d(X, Y, hub[0], hub[1], stat_sigma, self.extra_hub_point_peak)
                hub_land_price = np.maximum(hub_land_price, stat_gauss)

        # 添加河流边界核（若配置了）
        if self.rivers:
            river_lp = self._river_land_price_field()
            if river_lp is not None:
                river_land_price = river_lp

        # 建筑点核（可选）
        if self.building_kernel_enabled:
            build_land_price = self._building_kernel_field(city_state, X, Y)

        # 融合
        land_price = self._fuse_components(
            hub_land_price=hub_land_price,
            road_land_price=road_land_price,
            river_land_price=river_land_price,
            build_land_price=build_land_price,
        )
        
        land_price[land_price < self.min_threshold] = 0
        return land_price
    
    def _calculate_hub_sigma(self, month: int) -> float:
        """计算Hub高斯核的当前σ值"""
        growth_factor = 1 + (self.max_hub_multiplier - 1) * (1 - math.exp(-self.hub_growth_rate * month))
        return self.hub_sigma_base * min(growth_factor, self.max_hub_multiplier)
    
    def _calculate_road_sigma(self, month: int) -> float:
        """计算道路高斯核的当前σ值"""
        growth_factor = 1 + (self.max_road_multiplier - 1) * (1 - math.exp(-self.road_growth_rate * month))
        return self.road_sigma_base * min(growth_factor, self.max_road_multiplier)
    
    def _get_component_strength(self, component_type: str, current_month: int) -> float:
        """获取组件强度 - 支持渐进式演化"""
        # 获取演化配置
        evolution_config = self.config.get('land_price_evolution', {})
        
        if not evolution_config.get('enabled', False):
            # 如果未启用演化，返回默认强度
            if component_type == 'road':
                return self.road_peak_value
            else:
                return self.hub_peak_value
        
        # 道路组件强度
        if component_type == 'road':
            road_activation_month = evolution_config.get('road_activation_month', 0)
            road_peak_value = evolution_config.get('road_peak_value', 0.7)
            return road_peak_value if current_month >= road_activation_month else 0.0
        
        # Hub1和Hub2组件强度
        elif component_type in ['hub1', 'hub2']:
            hub_activation_month = evolution_config.get('hub_activation_month', 7)
            hub_growth_duration = evolution_config.get('hub_growth_duration_months', 6)
            hub_initial_peak = evolution_config.get('hub_initial_peak', 0.7)
            hub_final_peak = evolution_config.get('hub_final_peak', 1.0)
            growth_curve_type = evolution_config.get('growth_curve_type', 'smooth')
            
            if current_month < hub_activation_month:
                return 0.0
            elif current_month < hub_activation_month + hub_growth_duration:
                # 计算增长进度
                progress = (current_month - hub_activation_month) / hub_growth_duration
                progress = max(0.0, min(1.0, progress))  # 限制在[0,1]范围内
                
                # 应用增长曲线
                if growth_curve_type == 'linear':
                    curve_progress = progress
                elif growth_curve_type == 'smooth':
                    # S型增长曲线
                    steepness = evolution_config.get('smooth_curve_steepness', 10.0)
                    curve_progress = 1 / (1 + math.exp(-steepness * (progress - 0.5)))
                elif growth_curve_type == 'exponential':
                    curve_progress = progress ** 2
                else:
                    curve_progress = progress
                
                return hub_initial_peak + (hub_final_peak - hub_initial_peak) * curve_progress
            else:
                return hub_final_peak
        
        # Hub3组件强度（保持现有状态）
        elif component_type == 'hub3':
            hub3_keep_existing = evolution_config.get('hub3_keep_existing', True)
            if hub3_keep_existing:
                return self.hub_peak_value if current_month >= 0 else 0.0
            else:
                # 如果Hub3也参与演化，使用与Hub1/Hub2相同的逻辑
                return self._get_component_strength('hub1', current_month)
        
        return 0.0

    # ----------------------
    # 河流边界地价核
    # ----------------------
    def _dist_point_to_segment(self, px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        abx, aby = (bx - ax), (by - ay)
        apx, apy = (px - ax), (py - ay)
        ab2 = abx * abx + aby * aby
        if ab2 <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = (apx * abx + apy * aby) / ab2
        if t < 0.0:
            qx, qy = ax, ay
        elif t > 1.0:
            qx, qy = bx, by
        else:
            qx, qy = ax + t * abx, ay + t * aby
        return math.hypot(px - qx, py - qy)

    def _min_dist_to_polylines(self, px: float, py: float) -> float:
        best = 1e9
        for r in self.rivers:
            coords = r.get('coords', [])
            for i in range(len(coords) - 1):
                ax, ay = coords[i]
                bx, by = coords[i + 1]
                d = self._dist_point_to_segment(px, py, ax, ay, bx, by)
                if d < best:
                    best = d
        return best

    def _river_land_price_field(self) -> np.ndarray:
        if not self.rivers:
            return None
        W, H = int(self.map_size[0]), int(self.map_size[1])
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        field = np.zeros((H, W), dtype=float)
        # 对每个像素计算到所有河折线的最小距离，再取每条河的核最大值
        # 为效率，这里采用逐像素循环（图幅较小可接受）；必要时可向量化近似
        for yy in range(H):
            py = float(yy)
            for xx in range(W):
                px = float(xx)
                v_max = 0.0
                for r in self.rivers:
                    if not r.get('enabled', True):
                        continue
                    peak = float(r.get('peak_value', 0.8))
                    decay = float(r.get('decay_rate', 0.05))
                    rmax = float(r.get('max_influence_distance', 20.0))
                    mode = str(r.get('decay_mode', 'exponential'))
                    sigma = float(r.get('sigma_px', 6.0))
                    gamma = float(r.get('gamma_px', 6.0))
                    d = self._min_dist_to_polylines(px, py)
                    if d <= rmax:
                        if mode == 'gaussian':
                            # v = peak * exp( - d^2 / (2*sigma^2) )
                            denom = 2.0 * max(1e-6, sigma) * max(1e-6, sigma)
                            v = peak * math.exp(-(d * d) / denom)
                        elif mode == 'lorentzian':
                            # v = peak * gamma^2 / (d^2 + gamma^2)
                            g2 = max(1e-6, gamma) * max(1e-6, gamma)
                            v = peak * (g2 / (d * d + g2))
                        else:
                            # exponential: v = peak * exp(-decay * d)
                            v = peak * math.exp(-decay * d)
                        if v > v_max:
                            v_max = v
                field[yy, xx] = v_max
        return field

    # ----------------------
    # 建筑点核
    # ----------------------
    def _building_kernel_field(self, city_state: Dict, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if not self.building_kernel_enabled or city_state is None:
            return np.zeros(self.map_size, dtype=float)
        field = np.zeros(self.map_size, dtype=float)
        sigma = float(self.building_kernel_sigma_px if self.building_kernel_sigma_px > 0 else 3.0)
        peak = float(self.building_kernel_peak)
        for t in self.building_kernel_types:
            arr = city_state.get(t, []) if isinstance(city_state, dict) else []
            for b in arr:
                pos = b.get('xy') if isinstance(b, dict) else None
                if not pos or len(pos) < 2:
                    continue
                try:
                    cx = float(pos[0]); cy = float(pos[1])
                except Exception:
                    continue
                g = self._gaussian_2d(X, Y, cx, cy, sigma, peak)
                field = np.maximum(field, g)
        return field

    # ----------------------
    # 融合策略
    # ----------------------
    def _normalize_component(self, comp: np.ndarray) -> np.ndarray:
        if not self.component_normalize:
            return comp
        vmax = float(np.max(comp)) if comp.size > 0 else 0.0
        if vmax <= 1e-9:
            return np.zeros_like(comp)
        return comp / vmax

    def _post_normalize(self, fused: np.ndarray) -> np.ndarray:
        mode = self.fusion_normalize
        if mode == 'minmax':
            vmin = float(np.min(fused))
            vmax = float(np.max(fused))
            if vmax - vmin <= 1e-9:
                return np.zeros_like(fused)
            return (fused - vmin) / (vmax - vmin)
        if mode == 'clip01':
            return np.clip(fused, 0.0, 1.0)
        return fused

    def _fuse_components(
        self,
        hub_land_price: np.ndarray,
        road_land_price: np.ndarray,
        river_land_price: np.ndarray,
        build_land_price: np.ndarray,
    ) -> np.ndarray:
        if self.fusion_mode == 'weighted_sum':
            h = self._normalize_component(hub_land_price)
            r = self._normalize_component(road_land_price)
            v = self._normalize_component(river_land_price)
            b = self._normalize_component(build_land_price)
            fused = (
                self.fusion_weights.get('hub', 1.0) * h +
                self.fusion_weights.get('road', 1.0) * r +
                self.fusion_weights.get('river', 1.0) * v +
                self.fusion_weights.get('build', 1.0) * b
            )
            return self._post_normalize(fused)
        # max 模式
        fused = np.maximum(np.maximum(hub_land_price, road_land_price), river_land_price)
        if self.building_kernel_enabled:
            fused = np.maximum(fused, build_land_price * max(0.0, self.building_kernel_weight))
        return fused
    
    def _get_evolution_stage(self, month: int) -> Dict:
        """获取当前演化阶段配置"""
        hub_sigma = self._calculate_hub_sigma(month)
        road_sigma = self._calculate_road_sigma(month)
        
        # 获取组件强度
        road_strength = self._get_component_strength('road', month)
        hub1_strength = self._get_component_strength('hub1', month)
        hub2_strength = self._get_component_strength('hub2', month)
        hub3_strength = self._get_component_strength('hub3', month)
        
        # 根据渐进式演化定义阶段
        evolution_config = self.config.get('land_price_evolution', {})
        if evolution_config.get('enabled', False):
            road_activation_month = evolution_config.get('road_activation_month', 0)
            hub_activation_month = evolution_config.get('hub_activation_month', 7)
            hub_growth_duration = evolution_config.get('hub_growth_duration_months', 6)
            
            if month < road_activation_month:
                stage_name = "pre_road"
                description = "道路发展前"
            elif month < hub_activation_month:
                stage_name = "road_development"
                description = "道路优先发展"
            elif month < hub_activation_month + hub_growth_duration:
                stage_name = "hub_development"
                description = "Hub渐进增长"
            else:
                stage_name = "full_development"
                description = "完整地价场"
        else:
            # 原有的阶段定义
            if month < 6:
                stage_name = "initial"
                description = "初始阶段"
            elif month < 12:
                stage_name = "early_growth"
                description = "早期增长"
            elif month < 18:
                stage_name = "mid_growth"
                description = "中期增长"
            else:
                stage_name = "mature"
                description = "成熟阶段"
        
        return {
            'name': stage_name,
            'hub_sigma': hub_sigma,
            'road_sigma': road_sigma,
            'description': description,
            'month': month,
            'component_strengths': {
                'road': road_strength,
                'hub1': hub1_strength,
                'hub2': hub2_strength,
                'hub3': hub3_strength
            }
        }
    
    def update_land_price_field(self, month: int, city_state: Dict = None):
        """更新地价场"""
        self.current_month = month
        evolution_stage = self._get_evolution_stage(month)
        
        new_land_price = self._create_land_price_field(month, city_state)
        
        if self.land_price_field is not None:
            alpha = self.sdf_config.get('alpha_inertia', 0.25)
            self.land_price_field = (1 - alpha) * new_land_price + alpha * self.land_price_field
        else:
            self.land_price_field = new_land_price
        
        self.evolution_history.append({
            'month': month,
            'evolution_stage': evolution_stage,
            'land_price_stats': {
                'min': float(np.min(self.land_price_field)),
                'max': float(np.max(self.land_price_field)),
                'mean': float(np.mean(self.land_price_field)),
                'std': float(np.std(self.land_price_field))
            }
        })
        
        print(f"[LandPrice] field updated - month: {month}")
    
    def get_land_price_field(self) -> np.ndarray:
        """获取当前地价场"""
        return self.land_price_field
    
    def get_land_price_stats(self) -> Dict:
        """获取地价场统计信息"""
        if self.land_price_field is None:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
        
        return {
            'min': float(np.min(self.land_price_field)),
            'max': float(np.max(self.land_price_field)),
            'mean': float(np.mean(self.land_price_field)),
            'std': float(np.std(self.land_price_field))
        }
    
    def get_land_price(self, position: List[float]) -> float:
        """获取指定位置的地价值（支持浮点坐标，采用双线性插值）。"""
        if self.land_price_field is None:
            return 0.0

        try:
            xf = float(position[0]); yf = float(position[1])
        except Exception:
            return 0.0

        # 边界裁剪到 [0, W-1] / [0, H-1]
        W = int(self.map_size[0]); H = int(self.map_size[1])
        if W <= 0 or H <= 0:
            return 0.0
        if xf < 0.0: xf = 0.0
        if yf < 0.0: yf = 0.0
        if xf > (W - 1): xf = float(W - 1)
        if yf > (H - 1): yf = float(H - 1)

        # 双线性插值
        x0 = int(math.floor(xf)); x1 = min(x0 + 1, W - 1)
        y0 = int(math.floor(yf)); y1 = min(y0 + 1, H - 1)
        dx = xf - x0; dy = yf - y0

        f00 = float(self.land_price_field[y0, x0])
        f10 = float(self.land_price_field[y0, x1])
        f01 = float(self.land_price_field[y1, x0])
        f11 = float(self.land_price_field[y1, x1])

        val = (1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f10 + (1 - dx) * dy * f01 + dx * dy * f11
        return float(val)
    
    def get_evolution_history(self) -> List[Dict]:
        """获取演化历史"""
        return self.evolution_history
    
    def save_land_price_frame(self, month: int, output_dir: str = "land_price_frames"):
        """保存地价场帧"""
        if self.land_price_field is None:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        frame_data = {
            'month': month,
            'land_price_field': self.land_price_field.tolist(),
            'evolution_stage': self._get_evolution_stage(month),
            'land_price_stats': self.get_land_price_stats()
        }
        
        frame_file = os.path.join(output_dir, f"land_price_frame_month_{month:02d}.json")
        with open(frame_file, 'w', encoding='utf-8') as f:
            json.dump(frame_data, f, indent=2, ensure_ascii=False)
        
        print(f"[LandPrice] frame saved: {frame_file}")
    
    def get_land_price_components(self, month: int) -> Dict[str, np.ndarray]:
        """获取地价场的各个组成部分（与演化强度一致）"""
        X, Y = np.meshgrid(np.arange(self.map_size[0]), np.arange(self.map_size[1]))
        hub_sigma = self._calculate_hub_sigma(month)
        road_sigma = self._calculate_road_sigma(month)

        # 组件强度与 _create_land_price_field 保持一致
        road_strength = self._get_component_strength('road', month)
        hub1_strength = self._get_component_strength('hub1', month)
        hub2_strength = self._get_component_strength('hub2', month)
        hub3_strength = self._get_component_strength('hub3', month)

        hub_land_price = np.zeros(self.map_size, dtype=float)
        for i, hub in enumerate(self.transport_hubs):
            if i == 0 and hub1_strength > 0:
                s = hub1_strength
            elif i == 1 and hub2_strength > 0:
                s = hub2_strength
            elif i == 2 and hub3_strength > 0:
                s = hub3_strength
            else:
                s = 0.0
            if s > 0:
                hub_gaussian = self._gaussian_2d(X, Y, hub[0], hub[1], hub_sigma, s)
                hub_land_price = np.maximum(hub_land_price, hub_gaussian)

            # 同样在组件图中叠加“静态 Hub 点核”（若启用）
            if self.extra_hub_point_peak > 0.0:
                stat_sigma = self.extra_hub_point_sigma_px if self.extra_hub_point_sigma_px > 0 else self.hub_sigma_base
                stat_gauss = self._gaussian_2d(X, Y, hub[0], hub[1], stat_sigma, self.extra_hub_point_peak)
                hub_land_price = np.maximum(hub_land_price, stat_gauss)

        road_land_price = np.zeros(self.map_size, dtype=float)
        if len(self.transport_hubs) >= 2 and road_strength > 0:
            road_land_price = self._line_gaussian(X, Y, self.transport_hubs[0], self.transport_hubs[1], road_sigma, road_strength)

        river_land_price = np.zeros(self.map_size, dtype=float)
        if self.rivers:
            rlp = self._river_land_price_field()
            if rlp is not None:
                river_land_price = rlp

        # 建筑点核（基于当前 self.land_price_field 形成时对应城市状态；此处不额外注入）
        build_land_price = np.zeros(self.map_size, dtype=float)
        combined_land_price = self._fuse_components(
            hub_land_price=hub_land_price,
            road_land_price=road_land_price,
            river_land_price=river_land_price,
            build_land_price=build_land_price,
        )
        combined_land_price[combined_land_price < self.min_threshold] = 0

        return {
            'hub_land_price': hub_land_price,
            'road_land_price': road_land_price,
            'river_land_price': river_land_price,
            'combined_land_price': combined_land_price,
            'building_land_price': build_land_price
        }

# 为了保持兼容性，保留原来的类名作为别名
EnhancedSDFSystem = GaussianLandPriceSystem


