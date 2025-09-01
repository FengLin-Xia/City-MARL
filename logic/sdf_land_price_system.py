#!/usr/bin/env python3
"""
SDF地价系统 v2.3
实现基于SDF (Signed Distance Field) 的地价潜力场
"""

import numpy as np
import math
from typing import List, Dict, Tuple
import json

class SDFLandPriceSystem:
    """SDF地价系统：城市地价潜力场"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sdf_config = config.get('sdf_system', {})
        
        # SDF演化参数
        self.transition_start_month = self.sdf_config.get('transition', {}).get('point_to_line_start_month', 6)
        self.blend_duration = self.sdf_config.get('transition', {}).get('blend_duration_months', 12)
        self.lambda_perp_m = self.sdf_config.get('lambda_perp_m', 120)
        self.front_speed = self.sdf_config.get('front_speed_px_per_year', 200)
        
        # 经济权重
        self.economic_weights = self.sdf_config.get('economic_weights', {
            'accessibility': 0.6,
            'heatmap': 0.7,
            'public_facilities': 0.4,
            'unsuitability': -0.5
        })
        self.max_economic_influence = self.sdf_config.get('max_economic_influence', 0.3)
        
        # 系统状态
        self.current_month = 0
        self.sdf_field = None
        self.land_price_matrix = None
        self.transport_hubs = []
        self.map_size = [256, 256]
        
        # 演化历史
        self.sdf_evolution_history = []
        
    def initialize_sdf_field(self, map_size: List[int], transport_hubs: List[List[int]]):
        """初始化SDF地价场"""
        self.map_size = map_size
        self.transport_hubs = transport_hubs
        self.current_month = 0
        
        # 初始化点SDF（基于交通枢纽）
        self.sdf_field = self._create_point_sdf()
        self.land_price_matrix = self._sdf_to_land_price(self.sdf_field)
        
        print(f"🏗️ SDF地价系统初始化完成：{len(transport_hubs)} 个交通枢纽")
        
    def _create_point_sdf(self) -> np.ndarray:
        """创建基于交通枢纽的点SDF - 使用明确的几何函数"""
        sdf = np.zeros(self.map_size)
        
        # 获取点衰减长度（转换为像素）
        lambda_point_m = self.sdf_config.get('lambda_point_m', 100)
        lambda_point_px = lambda_point_m / self.sdf_config.get('meters_per_pixel', 2.0)
        
        for hub in self.transport_hubs:
            hub_x, hub_y = hub[0], hub[1]
            
            # 使用明确的点核函数：P_S(x) = max_{s∈S} exp(-||x-s||/λ_S)
            for y in range(self.map_size[1]):
                for x in range(self.map_size[0]):
                    distance = math.sqrt((x - hub_x)**2 + (y - hub_y)**2)
                    point_value = math.exp(-distance / lambda_point_px)
                    sdf[y, x] = max(sdf[y, x], point_value)
        
        return sdf
    
    def _create_line_sdf(self) -> np.ndarray:
        """创建基于主干道的线SDF - 使用明确的几何函数"""
        sdf = np.zeros(self.map_size)
        
        # 获取线衰减参数（转换为像素）
        lambda_perp_m = self.sdf_config.get('lambda_perp_m', 120)
        lambda_perp_px = lambda_perp_m / self.sdf_config.get('meters_per_pixel', 2.0)
        lambda_tangential_m = self.sdf_config.get('lambda_tangential_m', 200)
        lambda_tangential_px = lambda_tangential_m / self.sdf_config.get('meters_per_pixel', 2.0)
        use_tangential_decay = self.sdf_config.get('use_tangential_decay', False)
        
        # 假设主干道是连接交通枢纽的直线
        if len(self.transport_hubs) >= 2:
            hub1, hub2 = self.transport_hubs[0], self.transport_hubs[1]
            
            # 计算主干道参数
            dx = hub2[0] - hub1[0]
            dy = hub2[1] - hub1[1]
            length = math.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # 单位向量
                ux, uy = dx / length, dy / length
                
                # 为每个点计算到主干道的距离
                for y in range(self.map_size[1]):
                    for x in range(self.map_size[0]):
                        # 计算点到直线的投影
                        px, py = x - hub1[0], y - hub1[1]
                        proj_length = px * ux + py * uy
                        
                        # 投影点
                        proj_x = hub1[0] + proj_length * ux
                        proj_y = hub1[1] + proj_length * uy
                        
                        # 计算垂直距离（法向距离）
                        perp_distance = math.sqrt((x - proj_x)**2 + (y - proj_y)**2)
                        
                        # 使用明确的线核函数：P_Γ(x) = exp(-d⊥(x,Γ)/λ⊥) · exp(-|d∥(x,Γ)|/λ∥)
                        if 0 <= proj_length <= length:
                            # 在主干道范围内
                            line_value = math.exp(-perp_distance / lambda_perp_px)
                            if use_tangential_decay:
                                # 添加切向衰减
                                tangential_factor = math.exp(-abs(proj_length - length/2) / lambda_tangential_px)
                                line_value *= tangential_factor
                            sdf[y, x] = line_value
                        else:
                            # 超出主干道范围，使用到端点的距离
                            if proj_length < 0:
                                end_distance = math.sqrt((x - hub1[0])**2 + (y - hub1[1])**2)
                            else:
                                end_distance = math.sqrt((x - hub2[0])**2 + (y - hub2[1])**2)
                            sdf[y, x] = 0.5 * math.exp(-end_distance / lambda_perp_px)
        
        return sdf
    
    def update_sdf_field(self, month: int, city_state: Dict):
        """更新SDF地价场（年度更新）"""
        self.current_month = month
        
        # 计算演化阶段
        evolution_stage = self._calculate_evolution_stage(month)
        
        # 生成新的SDF场
        if evolution_stage < 0.5:
            # 点SDF阶段
            new_sdf = self._create_point_sdf()
        elif evolution_stage < 1.0:
            # 混合阶段
            point_sdf = self._create_point_sdf()
            line_sdf = self._create_line_sdf()
            blend_factor = (evolution_stage - 0.5) * 2  # 0到1
            new_sdf = (1 - blend_factor) * point_sdf + blend_factor * line_sdf
        else:
            # 线SDF阶段
            new_sdf = self._create_line_sdf()
        
        # 应用经济修正因子
        corrected_sdf = self._apply_economic_corrections(new_sdf, city_state)
        
        # 使用PRD中定义的时间演化公式：P_t(x) = clip((1-α) · P_base(x) + α · P_{t-1}(x) + β · E(x), 0, 1)
        alpha = self.sdf_config.get('alpha_inertia', 0.25)
        beta = self.sdf_config.get('max_economic_influence', 0.3)
        
        if self.sdf_field is not None:
            # 计算经济修正项
            economic_correction = corrected_sdf - new_sdf
            economic_correction = np.clip(economic_correction, -beta, beta)
            
            # 应用时间演化公式
            self.sdf_field = np.clip(
                (1 - alpha) * new_sdf + alpha * self.sdf_field + beta * economic_correction, 
                0, 1
            )
        else:
            self.sdf_field = corrected_sdf
        
        # 转换为地价矩阵
        self.land_price_matrix = self._sdf_to_land_price(self.sdf_field)
        
        # 记录演化历史
        self.sdf_evolution_history.append({
            'month': month,
            'evolution_stage': evolution_stage,
            'sdf_stats': {
                'min': float(np.min(self.sdf_field)),
                'max': float(np.max(self.sdf_field)),
                'mean': float(np.mean(self.sdf_field)),
                'std': float(np.std(self.sdf_field))
            }
        })
        
    def _calculate_evolution_stage(self, month: int) -> float:
        """计算SDF演化阶段（0=点SDF, 1=线SDF）"""
        if month < self.transition_start_month:
            return 0.0
        elif month < self.transition_start_month + self.blend_duration:
            return (month - self.transition_start_month) / self.blend_duration
        else:
            return 1.0
    
    def _apply_economic_corrections(self, sdf: np.ndarray, city_state: Dict) -> np.ndarray:
        """应用经济修正因子"""
        corrected_sdf = sdf.copy()
        
        # 1. 可达性修正
        if self.economic_weights['accessibility'] > 0:
            accessibility_correction = self._calculate_accessibility_correction(city_state)
            corrected_sdf += self.economic_weights['accessibility'] * accessibility_correction
        
        # 2. 热力图修正
        if self.economic_weights['heatmap'] > 0:
            heatmap_correction = self._calculate_heatmap_correction(city_state)
            corrected_sdf += self.economic_weights['heatmap'] * heatmap_correction
        
        # 3. 公共设施修正
        if self.economic_weights['public_facilities'] > 0:
            facility_correction = self._calculate_facility_correction(city_state)
            corrected_sdf += self.economic_weights['public_facilities'] * facility_correction
        
        # 4. 不适宜性修正
        if self.economic_weights['unsuitability'] < 0:
            unsuitability_correction = self._calculate_unsuitability_correction(city_state)
            corrected_sdf += self.economic_weights['unsuitability'] * unsuitability_correction
        
        # 限制修正幅度
        max_correction = self.max_economic_influence
        correction_factor = np.clip(corrected_sdf - sdf, -max_correction, max_correction)
        corrected_sdf = sdf + correction_factor
        
        # 确保SDF值在合理范围内
        corrected_sdf = np.clip(corrected_sdf, 0.0, 1.0)
        
        return corrected_sdf
    
    def _calculate_accessibility_correction(self, city_state: Dict) -> np.ndarray:
        """计算可达性修正"""
        correction = np.zeros(self.map_size)
        
        # 基于到交通枢纽的距离
        for y in range(self.map_size[1]):
            for x in range(self.map_size[0]):
                min_distance = float('inf')
                for hub in self.transport_hubs:
                    distance = math.sqrt((x - hub[0])**2 + (y - hub[1])**2)
                    min_distance = min(min_distance, distance)
                
                # 距离越近，可达性越高
                accessibility = 1.0 / (1.0 + min_distance / 100.0)
                correction[y, x] = accessibility
        
        return correction
    
    def _calculate_heatmap_correction(self, city_state: Dict) -> np.ndarray:
        """计算热力图修正"""
        correction = np.zeros(self.map_size)
        
        # 从轨迹系统获取热力图数据
        if 'trajectory_system' in city_state:
            trajectory_system = city_state['trajectory_system']
            if hasattr(trajectory_system, 'get_heatmap_data'):
                heatmap_data = trajectory_system.get_heatmap_data()
                if 'combined_heatmap' in heatmap_data:
                    combined_heatmap = heatmap_data['combined_heatmap']
                    if combined_heatmap.shape == self.map_size:
                        # 归一化热力图
                        max_heat = np.max(combined_heatmap)
                        if max_heat > 0:
                            correction = combined_heatmap / max_heat
        
        return correction
    
    def _calculate_facility_correction(self, city_state: Dict) -> np.ndarray:
        """计算公共设施修正"""
        correction = np.zeros(self.map_size)
        
        public_buildings = city_state.get('public', [])
        
        for building in public_buildings:
            facility_pos = building['xy']
            service_radius = building.get('service_radius', 50)
            
            for y in range(self.map_size[1]):
                for x in range(self.map_size[0]):
                    distance = math.sqrt((x - facility_pos[0])**2 + (y - facility_pos[1])**2)
                    if distance <= service_radius:
                        # 在服务半径内，地价提升
                        facility_effect = 1.0 - (distance / service_radius)
                        correction[y, x] = max(correction[y, x], facility_effect)
        
        return correction
    
    def _calculate_unsuitability_correction(self, city_state: Dict) -> np.ndarray:
        """计算不适宜性修正"""
        correction = np.zeros(self.map_size)
        
        # 建筑密度过高的区域
        building_density = np.zeros(self.map_size)
        
        all_buildings = []
        all_buildings.extend(city_state.get('public', []))
        all_buildings.extend(city_state.get('residential', []))
        all_buildings.extend(city_state.get('commercial', []))
        
        for building in all_buildings:
            building_pos = building['xy']
            x, y = int(building_pos[0]), int(building_pos[1])
            
            # 在建筑周围增加密度
            for dy in range(-20, 21):
                for dx in range(-20, 21):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                        distance = math.sqrt(dx**2 + dy**2)
                        if distance <= 20:
                            building_density[ny, nx] += 1.0 / (1.0 + distance)
        
        # 密度过高的区域地价降低
        max_density = np.max(building_density)
        if max_density > 0:
            normalized_density = building_density / max_density
            # 密度超过0.8的区域被认为不适宜
            correction = np.where(normalized_density > 0.8, normalized_density - 0.8, 0)
        
        return correction
    
    def _sdf_to_land_price(self, sdf: np.ndarray) -> np.ndarray:
        """将SDF转换为地价矩阵"""
        # 基础地价范围：50-300
        base_min_price = 50
        base_max_price = 300
        
        # SDF值转换为地价
        land_price = base_min_price + (base_max_price - base_min_price) * sdf
        
        return land_price
    
    def get_land_price_matrix(self) -> np.ndarray:
        """获取地价矩阵"""
        return self.land_price_matrix
    
    def get_land_price(self, position: List[int]) -> float:
        """获取指定位置的地价"""
        if self.land_price_matrix is None:
            return 100.0  # 默认地价
        
        x, y = int(position[0]), int(position[1])
        if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
            return float(self.land_price_matrix[y, x])
        else:
            return 100.0
    
    def get_land_price_stats(self) -> Dict:
        """获取地价统计信息"""
        if self.land_price_matrix is None:
            return {
                'min_price': 50.0,
                'max_price': 300.0,
                'avg_price': 100.0,
                'price_distribution': {}
            }
        
        return {
            'min_price': float(np.min(self.land_price_matrix)),
            'max_price': float(np.max(self.land_price_matrix)),
            'avg_price': float(np.mean(self.land_price_matrix)),
            'price_distribution': {
                'low': float(np.percentile(self.land_price_matrix, 25)),
                'medium': float(np.percentile(self.land_price_matrix, 50)),
                'high': float(np.percentile(self.land_price_matrix, 75))
            }
        }
    
    def get_sdf_field(self) -> np.ndarray:
        """获取SDF场"""
        return self.sdf_field
    
    def get_evolution_history(self) -> List[Dict]:
        """获取演化历史"""
        return self.sdf_evolution_history
    
    def save_sdf_data(self, output_dir: str, month: int):
        """保存SDF数据"""
        if self.sdf_field is not None:
            sdf_data = {
                'month': month,
                'sdf_field': self.sdf_field.tolist(),
                'land_price_matrix': self.land_price_matrix.tolist(),
                'evolution_stage': self._calculate_evolution_stage(month),
                'stats': self.get_land_price_stats()
            }
            
            filepath = f"{output_dir}/sdf_field_month_{month:02d}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sdf_data, f, indent=2, ensure_ascii=False)
