#!/usr/bin/env python3
"""
增强城市模拟系统 v3.3
基于高斯核地价场的城市发展模拟
实现地价场驱动的槽位生成和建筑选址
"""

import numpy as np
import math
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist

# 导入现有模块
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem
from logic.hysteresis_system import HysteresisSystem
from logic.public_facility_system import PublicFacilitySystem

@dataclass
class Slot:
    """槽位数据类"""
    pos: List[int]
    allowed_types: List[str]
    features: Dict[str, float]
    scores: Dict[str, float]
    used: bool = False
    dead: bool = False
    building_id: Optional[str] = None

@dataclass
class Layer:
    """层数据类"""
    layer_id: str
    status: str  # "locked", "active", "complete"
    activated_quarter: int
    slots: List[Slot]
    capacity: int
    dead_slots: int
    capacity_effective: int
    placed: int
    density: float
    layer_type: str  # "road", "radial"
    frozen_contour: Optional[List[List[int]]] = None  # 冻结的等值线
    
    def update_stats(self):
        """更新层统计信息"""
        self.placed = sum(1 for slot in self.slots if slot.used)
        self.dead_slots = sum(1 for slot in self.slots if slot.dead)
        self.capacity_effective = self.capacity - self.dead_slots
        
        if self.capacity_effective > 0:
            self.density = self.placed / self.capacity_effective
        else:
            self.density = 1.0

class GaussianLandPriceSystemV3_3:
    """高斯核地价场系统 v3.3 - 加权和+归一化融合"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sdf_config = config.get('gaussian_land_price_system', {})
        
        # 地图尺寸
        self.map_size = [110, 110]  # 默认尺寸
        
        # 地价场
        self.land_price_field = None
        
        # 获取配置参数
        self.meters_per_pixel = self.sdf_config.get('meters_per_pixel', 2.0)
        
        # 融合权重
        self.w_r = self.sdf_config.get('w_r', 0.6)      # 道路权重
        self.w_c = self.sdf_config.get('w_c', 0.5)      # 商业枢纽权重
        self.w_i = self.sdf_config.get('w_i', 0.5)      # 工业枢纽权重
        self.w_cor = self.sdf_config.get('w_cor', 0.2)  # 走廊权重
        self.bias = self.sdf_config.get('bias', 0.0)    # 偏置
        
        # 高斯核参数（米单位）
        self.hub_sigma_base_m = self.sdf_config.get('hub_sigma_base_m', 40)
        self.road_sigma_base_m = self.sdf_config.get('road_sigma_base_m', 20)
        
        # 演化参数
        self.hub_growth_rate = self.sdf_config.get('hub_growth_rate', 0.03)
        self.road_growth_rate = self.sdf_config.get('road_growth_rate', 0.02)
        self.max_hub_multiplier = self.sdf_config.get('max_hub_multiplier', 2.0)
        self.max_road_multiplier = self.sdf_config.get('max_road_multiplier', 2.5)
        
        # 归一化参数
        self.normalize = self.sdf_config.get('normalize', True)
        self.smoothstep_tau = self.sdf_config.get('smoothstep_tau', 0.0)
        
        # 当前月份
        self.current_month = 0
        
        # 演化历史
        self.evolution_history = []
    
    def initialize_system(self, transport_hubs: List[List[int]], map_size: List[int]):
        """初始化地价场系统"""
        self.map_size = map_size
        self.transport_hubs = transport_hubs
        
        # 生成初始地价场
        self._generate_land_price_field()
        
        print(f"✅ 地价场系统初始化完成，地图尺寸: {map_size}")
        print(f"   融合权重: w_r={self.w_r}, w_c={self.w_c}, w_i={self.w_i}, w_cor={self.w_cor}")
    
    def _generate_land_price_field(self):
        """生成地价场 - v3.3加权和+归一化融合"""
        height, width = self.map_size[1], self.map_size[0]
        
        # 初始化基础场
        P_base = np.zeros((height, width), dtype=np.float32)
        
        # 1. 道路核 (R)
        road_kernel = self._compute_road_kernel()
        P_base += self.w_r * road_kernel
        
        # 2. 商业枢纽核 (H_c)
        if len(self.transport_hubs) > 0:
            hub_com_kernel = self._compute_hub_kernel(self.transport_hubs[0], 'commercial')
            P_base += self.w_c * hub_com_kernel
        
        # 3. 工业枢纽核 (H_i)
        if len(self.transport_hubs) > 1:
            hub_ind_kernel = self._compute_hub_kernel(self.transport_hubs[1], 'industrial')
            P_base += self.w_i * hub_ind_kernel
        
        # 4. 走廊核 (C) - 可选
        corridor_kernel = self._compute_corridor_kernel()
        P_base += self.w_cor * corridor_kernel
        
        # 5. 偏置
        P_base += self.bias
        
        # 6. 归一化到[0,1]
        if self.normalize:
            p_min, p_max = P_base.min(), P_base.max()
            if p_max > p_min:
                self.land_price_field = np.clip((P_base - p_min) / (p_max - p_min), 0, 1)
            else:
                self.land_price_field = np.zeros_like(P_base)
        else:
            self.land_price_field = np.clip(P_base, 0, 1)
        
        # 7. 可选软阈值
        if self.smoothstep_tau > 0:
            self.land_price_field = self._apply_smoothstep(self.land_price_field, self.smoothstep_tau)
    
    def _compute_road_kernel(self) -> np.ndarray:
        """计算道路核"""
        height, width = self.map_size[1], self.map_size[0]
        kernel = np.zeros((height, width), dtype=np.float32)
        
        # 主干道中心线 (y = 55)
        center_y = height // 2
        
        # 当前道路σ（随时间演化）
        current_road_sigma = self.road_sigma_base_m * (1 + (self.max_road_multiplier - 1) * 
                                                      (1 - math.exp(-self.road_growth_rate * self.current_month)))
        current_road_sigma_px = current_road_sigma / self.meters_per_pixel
        
        # 计算到道路线的距离
        for y in range(height):
            d_perp = abs(y - center_y) * self.meters_per_pixel
            kernel[y, :] = math.exp(-(d_perp**2) / (2 * current_road_sigma**2))
        
        return kernel
    
    def _compute_hub_kernel(self, hub_pos: List[int], hub_type: str) -> np.ndarray:
        """计算枢纽核"""
        height, width = self.map_size[1], self.map_size[0]
        kernel = np.zeros((height, width), dtype=np.float32)
        
        hub_x, hub_y = hub_pos[0], hub_pos[1]
        
        # 当前枢纽σ（随时间演化）
        current_hub_sigma = self.hub_sigma_base_m * (1 + (self.max_hub_multiplier - 1) * 
                                                    (1 - math.exp(-self.hub_growth_rate * self.current_month)))
        
        # 各向异性参数
        if hub_type == 'commercial':
            sigma_perp = current_hub_sigma
            sigma_para = current_hub_sigma * 3  # 沿走廊方向更长
        else:  # industrial
            sigma_perp = current_hub_sigma * 1.2
            sigma_para = current_hub_sigma * 2.8
        
        # 计算各向异性高斯核
        for y in range(height):
            for x in range(width):
                dx = (x - hub_x) * self.meters_per_pixel
                dy = (y - hub_y) * self.meters_per_pixel
                
                # 各向异性距离
                d_eff = math.sqrt((dx**2 / sigma_para**2) + (dy**2 / sigma_perp**2))
                kernel[y, x] = math.exp(-d_eff**2 / 2)
        
        return kernel
    
    def _compute_corridor_kernel(self) -> np.ndarray:
        """计算走廊核"""
        height, width = self.map_size[1], self.map_size[0]
        kernel = np.zeros((height, width), dtype=np.float32)
        
        # 走廊中心线
        center_y = height // 2
        corridor_width = 20 / self.meters_per_pixel  # 20米走廊宽度
        
        # 走廊核（tanh型）
        for y in range(height):
            d_perp = abs(y - center_y) * self.meters_per_pixel
            if d_perp <= corridor_width:
                kernel[y, :] = 0.5 * (1 + math.tanh(2 * (1 - d_perp / corridor_width)))
        
        return kernel
    
    def _apply_smoothstep(self, field: np.ndarray, tau: float) -> np.ndarray:
        """应用软阈值"""
        # smoothstep函数: 3t² - 2t³
        t = np.clip(field / tau, 0, 1)
        return 3 * t**2 - 2 * t**3
    
    def update_land_price_field(self, month: int, city_state: Dict):
        """更新地价场"""
        self.current_month = month
        
        # 重新生成地价场
        self._generate_land_price_field()
        
        # 记录演化历史
        self.evolution_history.append({
            'month': month,
            'min': float(self.land_price_field.min()),
            'max': float(self.land_price_field.max()),
            'mean': float(self.land_price_field.mean())
        })
        
        print(f"📈 地价场更新 (月{month}): min={self.land_price_field.min():.3f}, "
              f"max={self.land_price_field.max():.3f}, mean={self.land_price_field.mean():.3f}")
    
    def get_land_price_field(self) -> np.ndarray:
        """获取地价场"""
        return self.land_price_field
    
    def get_land_price_stats(self) -> Dict:
        """获取地价统计"""
        return {
            'min': float(self.land_price_field.min()),
            'max': float(self.land_price_field.max()),
            'mean': float(self.land_price_field.mean()),
            'std': float(self.land_price_field.std())
        }
    
    def get_evolution_history(self) -> List[Dict]:
        """获取演化历史"""
        return self.evolution_history
    
    def save_land_price_frame(self, month: int, output_dir: str):
        """保存地价场帧"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存地价场数据
        np.save(os.path.join(output_dir, f'land_price_field_month_{month:02d}.npy'), 
                self.land_price_field)
        
        # 保存统计信息
        stats = self.get_land_price_stats()
        with open(os.path.join(output_dir, f'land_price_stats_month_{month:02d}.json'), 'w') as f:
            json.dump(stats, f, indent=2)

class FeatureScoringSystemV3_3:
    """特征评分系统 v3.3 - 包含f_price特征"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.meters_per_pixel = config.get('city', {}).get('meters_per_pixel', 2.0)
        
        # 评分权重（使用长名键）
        self.weights = {
            'commercial': {
                'f_price': 0.35, 'f_hub_com': 0.25, 'f_road': 0.20,
                'f_heat': 0.15, 'f_access': 0.05,
                'crowding': -0.03, 'junction_penalty': -0.02
            },
            'industrial': {
                'f_price': -0.20, 'f_hub_ind': 0.45, 'f_road': 0.25,
                'f_access': 0.05, 'crowding': -0.10, 'junction_penalty': -0.05
            },
            'residential': {
                'f_price': 0.10, 'f_road': 0.45, 'f_access': 0.15,
                'f_hub_com': -0.15, 'f_hub_ind': -0.10, 'crowding': -0.05
            }
        }
        
        print("✅ 特征评分系统v3.3初始化完成")
        print(f"   商业权重: f_price={self.weights['commercial']['f_price']}")
        print(f"   工业权重: f_price={self.weights['industrial']['f_price']}")
        print(f"   住宅权重: f_price={self.weights['residential']['f_price']}")
    
    def compute_features(self, pos: List[int], backbone_system, city_state: Dict, 
                        land_price_system: GaussianLandPriceSystemV3_3) -> Dict[str, float]:
        """计算槽位特征 - v3.3包含f_price"""
        features = {}
        
        # f_price: 地价（从land_price_field直接读取）
        land_price_field = land_price_system.get_land_price_field()
        y, x = pos[1], pos[0]
        if 0 <= y < land_price_field.shape[0] and 0 <= x < land_price_field.shape[1]:
            features['f_price'] = float(land_price_field[y, x])
        else:
            features['f_price'] = 0.0
        
        # f_road: 到走廊线的法向核
        features['f_road'] = self._compute_road_feature(pos, backbone_system)
        
        # f_hub_com: 商业顶点核
        features['f_hub_com'] = self._compute_hub_feature(pos, backbone_system.hub_commercial_pos, 
                                                         backbone_system.hub_commercial)
        
        # f_hub_ind: 工业顶点核
        features['f_hub_ind'] = self._compute_hub_feature(pos, backbone_system.hub_industrial_pos, 
                                                         backbone_system.hub_industrial)
        
        # f_access: 公共设施可达性
        features['f_access'] = self._compute_access_feature(pos, city_state)
        
        # f_heat: 居民轨迹热力
        features['f_heat'] = self._compute_heat_feature(pos, city_state)
        
        # crowding: 拥挤惩罚
        features['crowding'] = self._compute_crowding_penalty(pos, city_state)
        
        # junction_penalty: 路口惩罚
        features['junction_penalty'] = self._compute_junction_penalty(pos, backbone_system)
        
        return features
    
    def _compute_road_feature(self, pos: List[int], backbone_system) -> float:
        """计算道路特征：到走廊线的法向高斯核"""
        center_y = backbone_system.road_corridor['center_y']
        d_perp = abs(pos[1] - center_y) * self.meters_per_pixel
        sigma_perp = backbone_system.sigma_perp_m
        
        return math.exp(-(d_perp**2) / (2 * sigma_perp**2))
    
    def _compute_hub_feature(self, pos: List[int], hub_pos: List[int], hub_config: Dict) -> float:
        """计算枢纽特征：各向异性高斯核"""
        dx = (pos[0] - hub_pos[0]) * self.meters_per_pixel
        dy = (pos[1] - hub_pos[1]) * self.meters_per_pixel
        
        sigma_perp = hub_config.get('sigma_perp_m', 30)
        sigma_para = hub_config.get('sigma_parallel_m', 90)
        
        # 各向异性高斯（沿走廊方向更长）
        d_perp = abs(dy)
        d_para = abs(dx)
        
        return math.exp(-(d_perp**2 / (2 * sigma_perp**2) + d_para**2 / (2 * sigma_para**2)))
    
    def _compute_access_feature(self, pos: List[int], city_state: Dict) -> float:
        """计算公共设施可达性"""
        public_facilities = city_state.get('public', [])
        if not public_facilities:
            return 0.0
        
        max_access = 0.0
        for facility in public_facilities:
            facility_pos = facility.get('xy', [0, 0])
            distance = math.sqrt((pos[0] - facility_pos[0])**2 + (pos[1] - facility_pos[1])**2)
            distance_m = distance * self.meters_per_pixel
            
            # 服务半径内的可达性
            service_radius = facility.get('service_radius', 300)
            if distance_m <= service_radius:
                access = 1.0 - (distance_m / service_radius)
                max_access = max(max_access, access)
        
        return max_access
    
    def _compute_heat_feature(self, pos: List[int], city_state: Dict) -> float:
        """计算居民轨迹热力"""
        # 简化实现：基于附近居民密度
        residents = city_state.get('residents', [])
        if not residents:
            return 0.0
        
        heat = 0.0
        for resident in residents:
            resident_pos = resident.get('position', [0, 0])
            distance = math.sqrt((pos[0] - resident_pos[0])**2 + (pos[1] - resident_pos[1])**2)
            
            # 距离衰减
            if distance <= 50:  # 50像素范围内
                heat += math.exp(-distance / 20.0)
        
        return min(heat, 1.0)  # 归一化到[0,1]
    
    def _compute_crowding_penalty(self, pos: List[int], city_state: Dict) -> float:
        """计算拥挤惩罚"""
        # 计算附近建筑密度
        buildings = city_state.get('residential', []) + city_state.get('commercial', []) + city_state.get('industrial', [])
        
        nearby_buildings = 0
        for building in buildings:
            building_pos = building.get('xy', [0, 0])
            distance = math.sqrt((pos[0] - building_pos[0])**2 + (pos[1] - building_pos[1])**2)
            if distance <= 30:  # 30像素范围内
                nearby_buildings += 1
        
        # 归一化拥挤惩罚
        return min(nearby_buildings / 10.0, 1.0)
    
    def _compute_junction_penalty(self, pos: List[int], backbone_system) -> float:
        """计算路口惩罚"""
        # 计算到枢纽的距离惩罚
        hub_com_pos = backbone_system.hub_commercial_pos
        hub_ind_pos = backbone_system.hub_industrial_pos
        
        dist_to_com = math.sqrt((pos[0] - hub_com_pos[0])**2 + (pos[1] - hub_com_pos[1])**2)
        dist_to_ind = math.sqrt((pos[0] - hub_ind_pos[0])**2 + (pos[1] - hub_ind_pos[1])**2)
        
        # 距离枢纽太近有惩罚
        min_dist = min(dist_to_com, dist_to_ind)
        if min_dist <= 5:  # 5像素内
            return 1.0 - (min_dist / 5.0)
        else:
            return 0.0
    
    def compute_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """计算评分"""
        scores = {}
        
        for building_type in ['commercial', 'industrial', 'residential']:
            score = 0.0
            for feature_name, weight in self.weights[building_type].items():
                if feature_name in features:
                    score += weight * features[feature_name]
            scores[building_type] = score
        
        return scores

class ContourExtractionSystemV3_3:
    """等值线提取系统 v3.3 - 从地价场提取并冻结槽位"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.meters_per_pixel = config.get('city', {}).get('meters_per_pixel', 2.0)
        
        # 等值线配置
        self.contour_config = config.get('isocontour_layout', {})
        
        # 商业等值线级别
        self.commercial_levels = self.contour_config.get('commercial', {}).get('levels', [0.85, 0.78, 0.71])
        
        # 工业等值线级别（外侧优先）
        self.industrial_levels = self.contour_config.get('industrial', {}).get('levels', [0.60, 0.70, 0.80])
        
        # 住宅等值线带区
        self.residential_band = self.contour_config.get('residential', {}).get('band', [0.45, 0.65])
        
        # 采样参数
        self.arc_spacing = {
            'commercial': self.contour_config.get('commercial', {}).get('arc_spacing_m', [25, 35]),
            'industrial': self.contour_config.get('industrial', {}).get('arc_spacing_m', [35, 55]),
            'residential': self.contour_config.get('residential', {}).get('arc_spacing_m', [35, 55])
        }
        
        # 微偏移参数
        self.normal_offset_m = self.contour_config.get('normal_offset_m', 1.0)
        self.jitter_m = self.contour_config.get('jitter_m', 0.5)
        
        print("✅ 等值线提取系统v3.3初始化完成")
        print(f"   商业等值线: {self.commercial_levels}")
        print(f"   工业等值线: {self.industrial_levels}")
        print(f"   住宅带区: {self.residential_band}")
    
    def extract_contours_from_land_price(self, land_price_field: np.ndarray, building_type: str, 
                                       map_size: List[int]) -> List[List[List[int]]]:
        """从地价场提取等值线"""
        height, width = land_price_field.shape
        
        if building_type == 'commercial':
            levels = self.commercial_levels
        elif building_type == 'industrial':
            levels = self.industrial_levels
        elif building_type == 'residential':
            # 住宅使用带区中心等值线
            band_min, band_max = self.residential_band
            center_level = (band_min + band_max) / 2
            levels = [center_level]
        else:
            return []
        
        contours = []
        
        for level in levels:
            # 使用OpenCV提取等值线
            binary_image = (land_price_field >= level).astype(np.uint8) * 255
            
            # 查找轮廓
            contours_found, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours_found:
                # 简化轮廓
                epsilon = 0.02 * cv2.arcLength(contour, True)
                simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # 转换为坐标列表
                contour_points = []
                for point in simplified_contour:
                    x, y = point[0]
                    if 0 <= x < width and 0 <= y < height:
                        contour_points.append([int(x), int(y)])
                
                if len(contour_points) > 3:  # 至少需要3个点形成轮廓
                    contours.append(contour_points)
        
        return contours
    
    def sample_slots_on_contours(self, contours: List[List[List[int]]], building_type: str, 
                               map_size: List[int]) -> List[Slot]:
        """在等值线上采样槽位"""
        slots = []
        
        if not contours:
            return slots
        
        # 获取采样间距
        spacing_range = self.arc_spacing.get(building_type, [35, 55])
        min_spacing, max_spacing = spacing_range
        avg_spacing = (min_spacing + max_spacing) / 2
        spacing_px = avg_spacing / self.meters_per_pixel
        
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # 沿轮廓等弧长采样
            contour_slots = self._sample_along_contour(contour, spacing_px, building_type, map_size)
            slots.extend(contour_slots)
        
        return slots
    
    def _sample_along_contour(self, contour: List[List[int]], spacing_px: float, 
                            building_type: str, map_size: List[int]) -> List[Slot]:
        """沿轮廓等弧长采样"""
        slots = []
        
        if len(contour) < 2:
            return slots
        
        # 计算轮廓总长度
        total_length = 0
        segment_lengths = []
        
        for i in range(len(contour)):
            next_i = (i + 1) % len(contour)
            dx = contour[next_i][0] - contour[i][0]
            dy = contour[next_i][1] - contour[i][1]
            segment_length = math.sqrt(dx**2 + dy**2)
            segment_lengths.append(segment_length)
            total_length += segment_length
        
        if total_length < spacing_px:
            return slots
        
        # 计算采样点数量
        num_samples = max(1, int(total_length / spacing_px))
        actual_spacing = total_length / num_samples
        
        # 沿轮廓采样
        current_length = 0
        sample_index = 0
        
        for i in range(len(contour)):
            if sample_index >= num_samples:
                break
            
            next_i = (i + 1) % len(contour)
            segment_length = segment_lengths[i]
            
            # 检查是否需要在当前段内采样
            while (sample_index < num_samples and 
                   current_length + segment_length >= sample_index * actual_spacing):
                
                # 计算采样点位置
                t = (sample_index * actual_spacing - current_length) / segment_length
                t = max(0, min(1, t))  # 限制在[0,1]范围内
                
                # 线性插值
                x = int(contour[i][0] + t * (contour[next_i][0] - contour[i][0]))
                y = int(contour[i][1] + t * (contour[next_i][1] - contour[i][1]))
                
                # 检查位置有效性
                if 0 <= x < map_size[0] and 0 <= y < map_size[1]:
                    # 应用微偏移
                    x, y = self._apply_micro_offset(x, y, map_size)
                    
                    # 创建槽位
                    slot = Slot(
                        pos=[x, y],
                        allowed_types=[building_type],
                        features={},
                        scores={}
                    )
                    slots.append(slot)
                
                sample_index += 1
            
            current_length += segment_length
        
        return slots
    
    def _apply_micro_offset(self, x: int, y: int, map_size: List[int]) -> Tuple[int, int]:
        """应用微偏移"""
        # 法向偏移
        offset_px = self.normal_offset_m / self.meters_per_pixel
        jitter_px = self.jitter_m / self.meters_per_pixel
        
        # 随机偏移
        import random
        dx = random.uniform(-jitter_px, jitter_px)
        dy = random.uniform(-jitter_px, jitter_px)
        
        # 应用偏移
        new_x = int(x + dx)
        new_y = int(y + dy)
        
        # 边界检查
        new_x = max(0, min(map_size[0] - 1, new_x))
        new_y = max(0, min(map_size[1] - 1, new_y))
        
        return new_x, new_y

class RoadSamplingSystemV3_3:
    """道路采样系统 v3.3 - Road-L0沿线等弧长采样"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.meters_per_pixel = config.get('city', {}).get('meters_per_pixel', 2.0)
        
        # 政府骨架配置
        self.backbone_config = config.get('government_backbone', {})
        self.road_config = self.backbone_config.get('road_corridor', {})
        
        # 退线距离
        self.setback_m = self.road_config.get('setback_m', {})
        self.setback_com = self.setback_m.get('commercial', 8)
        self.setback_res = self.setback_m.get('residential', 10)
        self.setback_ind = self.setback_m.get('industrial', 14)
        
        print("✅ 道路采样系统v3.3初始化完成")
        print(f"   商业退线: {self.setback_com}m, 住宅退线: {self.setback_res}m, 工业退线: {self.setback_ind}m")
    
    def create_road_layers(self, map_size: List[int], center_y: int) -> List[Layer]:
        """创建道路层 - 沿线等弧长采样"""
        layers = []
        
        # 主干道中心线
        trunk_road = self.config.get('city', {}).get('trunk_road', [[20, center_y], [90, center_y]])
        start_x, end_x = trunk_road[0][0], trunk_road[1][0]
        
        # 创建商业、住宅、工业道路层
        for building_type in ['commercial', 'residential', 'industrial']:
            layer = self._create_road_layer(building_type, start_x, end_x, center_y, map_size)
            if layer:
                layers.append(layer)
        
        return layers
    
    def _create_road_layer(self, building_type: str, start_x: int, end_x: int, 
                          center_y: int, map_size: List[int]) -> Optional[Layer]:
        """创建单个道路层"""
        # 获取退线距离
        if building_type == 'commercial':
            setback = self.setback_com
        elif building_type == 'residential':
            setback = self.setback_res
        else:  # industrial
            setback = self.setback_ind
        
        setback_px = setback / self.meters_per_pixel
        
        # 计算退线位置
        y_upper = int(center_y - setback_px)
        y_lower = int(center_y + setback_px)
        
        # 沿线等弧长采样
        slots = []
        
        # 计算道路长度
        road_length = end_x - start_x
        road_length_m = road_length * self.meters_per_pixel
        
        # 采样间距（根据建筑类型调整）
        if building_type == 'commercial':
            spacing_m = 30  # 商业密度较高
        elif building_type == 'residential':
            spacing_m = 40  # 住宅中等密度
        else:  # industrial
            spacing_m = 50  # 工业密度较低
        
        spacing_px = spacing_m / self.meters_per_pixel
        num_samples = max(1, int(road_length / spacing_px))
        
        # 在两条退线上采样
        for side_y in [y_upper, y_lower]:
            if 0 <= side_y < map_size[1]:
                for i in range(num_samples):
                    x = int(start_x + i * (road_length / num_samples))
                    if 0 <= x < map_size[0]:
                        slot = Slot(
                            pos=[x, side_y],
                            allowed_types=[building_type],
                            features={},
                            scores={}
                        )
                        slots.append(slot)
        
        if slots:
            layer = Layer(
                layer_id=f"road_L0_{building_type}",
                status="locked",
                activated_quarter=-1,
                slots=slots,
                capacity=len(slots),
                dead_slots=0,
                capacity_effective=len(slots),
                placed=0,
                density=0.0,
                layer_type='road'
            )
            return layer
        
        return None

class UnifiedDecisionMakerV3_3:
    """统一决策器 v3.3 - 支持f_price特征"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_scoring_system = FeatureScoringSystemV3_3(config)
        
        print("✅ 统一决策器v3.3初始化完成")
    
    def place_on_active_layer(self, active_layer: Layer, quotas: Dict[str, int], 
                            city_state: Dict, backbone_system, 
                            land_price_system: GaussianLandPriceSystemV3_3) -> List[Dict]:
        """在激活层上放置建筑 - v3.3支持f_price特征"""
        placed_buildings = []
        
        # 按建筑类型分桶
        buckets = {"commercial": [], "residential": [], "industrial": []}
        
        # 计算每个槽位的特征和评分
        for slot in active_layer.slots:
            if slot.used or slot.dead:
                continue
            
            # 计算特征（包含f_price）
            features = self.feature_scoring_system.compute_features(
                slot.pos, backbone_system, city_state, land_price_system
            )
            slot.features = features
            
            # 计算评分
            scores = self.feature_scoring_system.compute_scores(features)
            slot.scores = scores
            
            # 检查分区约束
            allowed_types = self._check_zoning_constraints(slot.pos, backbone_system)
            
            # 为每个允许的建筑类型分桶
            for building_type in ['commercial', 'residential', 'industrial']:
                if building_type in allowed_types and building_type in scores:
                    buckets[building_type].append((slot, scores[building_type]))
        
        # 按评分排序并选择
        for building_type in ['commercial', 'residential', 'industrial']:
            if building_type not in quotas or quotas[building_type] <= 0:
                continue
            
            # 按评分降序排序
            buckets[building_type].sort(key=lambda x: x[1], reverse=True)
            
            # 选择前N个
            selected = buckets[building_type][:quotas[building_type]]
            
            for slot, score in selected:
                if not slot.used:
                    # 创建建筑
                    building = self._create_building(slot, building_type, score)
                    placed_buildings.append(building)
                    
                    # 标记槽位为已使用
                    slot.used = True
                    slot.building_id = building['building_id']
        
        # 更新层统计
        active_layer.update_stats()
        
        # 检查层是否完成
        if active_layer.density >= 0.95:
            active_layer.status = "complete"
            print(f"✅ 层 {active_layer.layer_id} 已完成 (密度: {active_layer.density:.1%})")
        
        return placed_buildings
    
    def _check_zoning_constraints(self, pos: List[int], backbone_system) -> List[str]:
        """检查分区约束"""
        allowed_types = ['commercial', 'residential', 'industrial']
        
        # 获取分区配置
        zoning_config = backbone_system.backbone_config.get('zoning', {})
        hub_com_radius_m = zoning_config.get('hub_com_radius_m', 350)
        hub_ind_radius_m = zoning_config.get('hub_ind_radius_m', 450)
        
        # 计算到枢纽的距离
        hub_com_pos = backbone_system.hub_commercial_pos
        hub_ind_pos = backbone_system.hub_industrial_pos
        
        dist_to_com = math.sqrt((pos[0] - hub_com_pos[0])**2 + (pos[1] - hub_com_pos[1])**2) * backbone_system.meters_per_pixel
        dist_to_ind = math.sqrt((pos[0] - hub_ind_pos[0])**2 + (pos[1] - hub_ind_pos[1])**2) * backbone_system.meters_per_pixel
        
        # 商业枢纽半径内优先商业
        if dist_to_com <= hub_com_radius_m:
            allowed_types = ['commercial', 'residential']  # 移除工业
        
        # 工业枢纽半径内优先工业
        if dist_to_ind <= hub_ind_radius_m:
            allowed_types = ['industrial', 'residential']  # 移除商业
        
        # 走廊中段优先住宅
        center_y = backbone_system.road_corridor['center_y']
        if abs(pos[1] - center_y) <= 20:  # 20像素内
            if 'residential' in allowed_types:
                allowed_types = ['residential', 'commercial']  # 住宅优先
        
        return allowed_types
    
    def _create_building(self, slot: Slot, building_type: str, score: float) -> Dict:
        """创建建筑"""
        building_id = f"{building_type}_{len(slot.pos)}"  # 简化ID生成
        
        building = {
            'building_id': building_id,
            'xy': slot.pos,
            'building_type': building_type,
            'score': score,
            'features': slot.features.copy(),
            'land_price_value': slot.features.get('f_price', 0.0),
            'month_placed': 0  # 将在主循环中设置
        }
        
        return building

class ProgressiveGrowthSystemV3_3:
    """渐进生长系统 v3.3 - 基于地价等值线的严格逐层生长"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.contour_extraction_system = ContourExtractionSystemV3_3(config)
        self.road_sampling_system = RoadSamplingSystemV3_3(config)
        
        # 层管理
        self.layers = []
        self.active_layers = []
        self.completed_layers = []
        
        # 生长状态
        self.road_layers_completed = False
        self.growth_phase = 'road_corridor'  # 'road_corridor' -> 'radial_expansion'
        
        print("✅ 渐进生长系统v3.3初始化完成")
    
    def initialize_layers(self, backbone_system, map_size: List[int]):
        """初始化层系统"""
        center_y = backbone_system.road_corridor['center_y']
        
        # 创建道路层
        road_layers = self.road_sampling_system.create_road_layers(map_size, center_y)
        self.layers.extend(road_layers)
        
        print(f"✅ 初始化了 {len(road_layers)} 个道路层")
    
    def activate_road_layers(self, quarter: int):
        """激活道路层"""
        road_layers = [layer for layer in self.layers if layer.layer_type == 'road' and layer.status == 'locked']
        
        for layer in road_layers:
            layer.status = 'active'
            layer.activated_quarter = quarter
            self.active_layers.append(layer)
            print(f"🛣️ 激活道路层: {layer.layer_id}")
    
    def check_road_layers_completion(self, quarter: int, map_size: List[int], 
                                   land_price_system: GaussianLandPriceSystemV3_3):
        """检查道路层完成情况"""
        # 检查所有道路层是否完成
        road_layers = [layer for layer in self.layers if layer.layer_type == 'road']
        completed_road_layers = [layer for layer in road_layers if layer.status == 'complete']
        
        if not self.road_layers_completed and len(completed_road_layers) == len(road_layers):
            self.road_layers_completed = True
            self.growth_phase = 'radial_expansion'
            
            # 创建放射扩张层
            self._create_radial_layers(quarter, map_size, land_price_system)
            
            print("✅ 道路层全部完成，进入放射扩张阶段")
    
    def _create_radial_layers(self, quarter: int, map_size: List[int], 
                            land_price_system: GaussianLandPriceSystemV3_3, phase: int = 0):
        """创建放射扩张层 - 从当前地价场提取等值线"""
        land_price_field = land_price_system.get_land_price_field()
        
        # 为每种建筑类型创建放射层
        for building_type in ['commercial', 'industrial', 'residential']:
            # 从地价场提取等值线
            contours = self.contour_extraction_system.extract_contours_from_land_price(
                land_price_field, building_type, map_size
            )
            
            if contours:
                # 在等值线上采样槽位
                slots = self.contour_extraction_system.sample_slots_on_contours(
                    contours, building_type, map_size
                )
                
                if slots:
                    # 创建层
                    layer = Layer(
                        layer_id=f"{building_type}_radial_P{phase}",
                        status="active",
                        activated_quarter=quarter,
                        slots=slots,
                        capacity=len(slots),
                        dead_slots=0,
                        capacity_effective=len(slots),
                        placed=0,
                        density=0.0,
                        layer_type='radial',
                        frozen_contour=contours[0] if contours else None
                    )
                    
                    self.layers.append(layer)
                    self.active_layers.append(layer)
                    
                    print(f"🎯 创建放射层: {layer.layer_id} (槽位数: {len(slots)})")
    
    def check_radial_layers_completion(self, quarter: int, map_size: List[int], 
                                     land_price_system: GaussianLandPriceSystemV3_3):
        """检查放射层完成情况并创建新层"""
        if not self.road_layers_completed:
            return
        
        # 获取当前激活的放射层
        active_radial_layers = [layer for layer in self.active_layers if layer.layer_type == 'radial']
        
        if not active_radial_layers:
            return
        
        # 检查是否有层完成
        completed_radial_layers = []
        for layer in active_radial_layers:
            layer.update_stats()
            if layer.density >= 0.95:
                layer.status = "complete"
                completed_radial_layers.append(layer)
                if layer in self.active_layers:
                    self.active_layers.remove(layer)
                self.completed_layers.append(layer)
                print(f"✅ 放射层 {layer.layer_id} 已完成 (密度: {layer.density:.1%})")
        
        # 如果有层完成，创建新的放射层
        if completed_radial_layers:
            # 计算下一个阶段号
            max_phase = 0
            for layer in self.layers:
                if 'radial_P' in layer.layer_id:
                    try:
                        phase_num = int(layer.layer_id.split('P')[1])
                        max_phase = max(max_phase, phase_num)
                    except:
                        pass
            
            next_phase = max_phase + 1
            print(f"🎯 创建下一阶段放射层 P{next_phase}...")
            self._create_radial_layers(quarter, map_size, land_price_system, next_phase)
    
    def get_active_layers(self) -> List[Layer]:
        """获取激活的层"""
        return self.active_layers
    
    def get_layer_status(self) -> Dict:
        """获取层状态"""
        return {
            'growth_phase': self.growth_phase,
            'road_layers_completed': self.road_layers_completed,
            'total_layers': len(self.layers),
            'active_layers': len(self.active_layers),
            'completed_layers': len(self.completed_layers),
            'layers': [
                {
                    'layer_id': layer.layer_id,
                    'status': layer.status,
                    'layer_type': layer.layer_type,
                    'activated_quarter': layer.activated_quarter,
                    'capacity': layer.capacity,
                    'dead_slots': layer.dead_slots,
                    'capacity_effective': layer.capacity_effective,
                    'placed': layer.placed,
                    'density': layer.density
                }
                for layer in self.layers
            ]
        }

class EnhancedCitySimulationV3_3:
    """增强城市模拟系统 v3.3 - 主模拟类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.city_config = config.get('city', {})
        
        # 地图尺寸
        self.map_size = self.city_config.get('map_size', [110, 110])
        self.meters_per_pixel = self.city_config.get('meters_per_pixel', 2.0)
        
        # 时间状态
        self.current_month = 0
        self.current_quarter = 0
        self.current_year = 0
        
        # 初始化子系统
        self.land_price_system = GaussianLandPriceSystemV3_3(config)
        self.progressive_growth_system = ProgressiveGrowthSystemV3_3(config)
        self.decision_maker = UnifiedDecisionMakerV3_3(config)
        
        # 政府骨架系统（简化版）
        self.backbone_system = self._create_backbone_system()
        
        # 城市状态
        self.city_state = {}
        
        print("✅ 增强城市模拟系统v3.3初始化完成")
    
    def _create_backbone_system(self):
        """创建政府骨架系统"""
        class BackboneSystem:
            def __init__(self, config, map_size):
                self.config = config
                self.map_size = map_size
                self.meters_per_pixel = config.get('city', {}).get('meters_per_pixel', 2.0)
                
                # 道路走廊
                self.road_corridor = {'center_y': map_size[1] // 2}
                self.sigma_perp_m = config.get('government_backbone', {}).get('road_corridor', {}).get('sigma_perp_m', 40)
                
                # 枢纽位置
                self.hub_commercial_pos = [map_size[0] // 3, map_size[1] // 2]
                self.hub_industrial_pos = [2 * map_size[0] // 3, map_size[1] // 2]
                
                # 枢纽配置
                self.hub_commercial = config.get('government_backbone', {}).get('hubs', {}).get('commercial', {})
                self.hub_industrial = config.get('government_backbone', {}).get('hubs', {}).get('industrial', {})
                
                # 分区配置
                self.backbone_config = config.get('government_backbone', {})
        
        backbone = BackboneSystem(self.config, self.map_size)
        return backbone
    
    def initialize_simulation(self):
        """初始化模拟"""
        # 交通枢纽
        transport_hubs = [
            self.backbone_system.hub_commercial_pos,
            self.backbone_system.hub_industrial_pos
        ]
        
        # 初始化地价场系统
        self.land_price_system.initialize_system(transport_hubs, self.map_size)
        
        # 初始化渐进生长系统
        self.progressive_growth_system.initialize_layers(self.backbone_system, self.map_size)
        
        # 初始化城市状态
        self.city_state = {
            'simulation_info': {
                'current_month': self.current_month,
                'current_quarter': self.current_quarter,
                'current_year': self.current_year
            },
            'land_price_field': self.land_price_system.get_land_price_field(),
            'land_price_stats': self.land_price_system.get_land_price_stats(),
            'buildings': {
                'public': [],
                'residential': [],
                'commercial': [],
                'industrial': []
            },
            'residents': [],
            'layers': self.progressive_growth_system.get_layer_status(),
            'backbone_info': {
                'road_corridor': self.backbone_system.road_corridor,
                'hub_commercial': self.backbone_system.hub_commercial_pos,
                'hub_industrial': self.backbone_system.hub_industrial_pos
            }
        }
        
        print("✅ 模拟初始化完成")
    
    def run_simulation(self, total_months: int = 24):
        """运行模拟"""
        print(f"🚀 开始运行模拟，总时长: {total_months} 个月")
        
        for month in range(total_months):
            self.current_month = month
            self.current_quarter = month // 3
            self.current_year = month // 12
            
            print(f"\n📅 第 {month} 个月 (第 {self.current_quarter} 季度, 第 {self.current_year} 年)")
            
            # 月度更新
            self._monthly_update()
            
            # 季度更新
            if month % 3 == 0:
                self._quarterly_update()
            
            # 年度更新
            if month % 12 == 0 and month > 0:
                self._yearly_update()
            
            # 保存输出
            self._save_monthly_outputs(month)
        
        print("✅ 模拟完成")
    
    def _monthly_update(self):
        """月度更新"""
        # 更新地价场
        self.land_price_system.update_land_price_field(self.current_month, self.city_state)
        
        # 更新城市状态中的地价场
        self.city_state['land_price_field'] = self.land_price_system.get_land_price_field()
        self.city_state['land_price_stats'] = self.land_price_system.get_land_price_stats()
        
        # 生成建筑
        self._generate_buildings()
        
        # 更新层状态
        self.city_state['layers'] = self.progressive_growth_system.get_layer_status()
    
    def _quarterly_update(self):
        """季度更新"""
        print(f"📊 第 {self.current_quarter} 季度更新...")
        
        # 激活道路层（第0季度）
        if self.current_quarter == 0:
            self.progressive_growth_system.activate_road_layers(self.current_quarter)
        
        # 检查道路层完成情况（每个季度都检查）
        self.progressive_growth_system.check_road_layers_completion(
            self.current_quarter, self.map_size, self.land_price_system
        )
        
        # 检查放射层完成情况（每个季度都检查）
        self.progressive_growth_system.check_radial_layers_completion(
            self.current_quarter, self.map_size, self.land_price_system
        )
    
    def _yearly_update(self):
        """年度更新"""
        print(f"📅 第 {self.current_year} 年更新...")
        
        # 地价场演化已在月度更新中处理
        pass
    
    def _generate_buildings(self):
        """生成建筑"""
        # 获取激活的层
        active_layers = self.progressive_growth_system.get_active_layers()
        
        if not active_layers:
            return
        
        # 获取季度配额
        quotas = self._get_quarterly_quotas()
        
        # 在激活层上放置建筑
        for layer in active_layers:
            placed_buildings = self.decision_maker.place_on_active_layer(
                layer, quotas, self.city_state, self.backbone_system, self.land_price_system
            )
            
            # 添加到城市状态
            for building in placed_buildings:
                building['month_placed'] = self.current_month
                building_type = building['building_type']
                self.city_state['buildings'][building_type].append(building)
            
            if placed_buildings:
                print(f"  🏗️ 在层 {layer.layer_id} 放置了 {len(placed_buildings)} 个建筑")
    
    def _get_quarterly_quotas(self) -> Dict[str, int]:
        """获取季度配额"""
        quotas_config = self.config.get('government_backbone', {}).get('quotas_per_quarter', {})
        
        # 根据季度调整配额
        quarter_index = min(self.current_quarter, len(quotas_config.get('residential', [10, 20])) - 1)
        
        return {
            'residential': quotas_config.get('residential', [10, 20])[quarter_index],
            'commercial': quotas_config.get('commercial', [5, 12])[quarter_index],
            'industrial': quotas_config.get('industrial', [4, 10])[quarter_index]
        }
    
    def _save_monthly_outputs(self, month: int):
        """保存月度输出"""
        output_dir = 'enhanced_simulation_v3_3_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存地价场
        self.land_price_system.save_land_price_frame(month, output_dir)
        
        # 保存建筑位置
        self._save_building_positions(month, output_dir)
        
        # 保存层状态
        self._save_layer_state(month, output_dir)
    
    def _save_building_positions(self, month: int, output_dir: str):
        """保存建筑位置"""
        if month == 0:
            # 第0个月保存完整状态
            self._save_full_building_state(month, output_dir)
        else:
            # 后续月份：只保存增量文件
            self._save_new_buildings_only(month, output_dir)
    
    def _save_full_building_state(self, month: int, output_dir: str):
        """保存完整建筑状态"""
        buildings_data = {
            'month': month,
            'buildings': self.city_state['buildings']
        }
        
        with open(os.path.join(output_dir, f'building_positions_month_{month:02d}.json'), 'w') as f:
            json.dump(buildings_data, f, indent=2)
    
    def _save_new_buildings_only(self, month: int, output_dir: str):
        """保存新增建筑"""
        # 简化实现：保存本月新增的建筑
        new_buildings = {
            'month': month,
            'new_buildings': []
        }
        
        for building_type in ['residential', 'commercial', 'industrial']:
            for building in self.city_state['buildings'][building_type]:
                if building.get('month_placed', -1) == month:
                    new_buildings['new_buildings'].append(building)
        
        with open(os.path.join(output_dir, f'building_delta_month_{month:02d}.json'), 'w') as f:
            json.dump(new_buildings, f, indent=2)
    
    def _save_layer_state(self, month: int, output_dir: str):
        """保存层状态"""
        layer_data = {
            'month': month,
            'quarter': self.current_quarter,
            'layers': self.progressive_growth_system.get_layer_status()
        }
        
        with open(os.path.join(output_dir, f'layer_state_month_{month:02d}.json'), 'w') as f:
            json.dump(layer_data, f, indent=2)

def main():
    """主函数"""
    print("🏗️ 增强城市模拟系统 v3.3")
    print("   基于高斯核地价场的城市发展模拟")
    print("   实现地价场驱动的槽位生成和建筑选址")
    
    # 加载配置
    config = {
        'city': {
            'map_size': [110, 110],
            'meters_per_pixel': 2.0,
            'trunk_road': [[20, 55], [90, 55]],
            'transport_hubs': [[37, 55], [73, 55]]
        },
        'government_backbone': {
            'road_corridor': {
                'sigma_perp_m': 40,
                'setback_m': {'commercial': 8, 'residential': 10, 'industrial': 14}
            },
            'hubs': {
                'commercial': {'sigma_perp_m': 30, 'sigma_parallel_m': 90},
                'industrial': {'sigma_perp_m': 35, 'sigma_parallel_m': 110}
            },
            'zoning': {
                'hub_com_radius_m': 350,
                'hub_ind_radius_m': 450,
                'mid_corridor_residential': True
            },
            'quotas_per_quarter': {
                'residential': [10, 20, 15, 25],
                'commercial': [5, 12, 8, 15],
                'industrial': [4, 10, 6, 12]
            },
            'strict_layering': True,
            'dead_slots_ratio_max': 0.05
        },
        'gaussian_land_price_system': {
            'w_r': 0.6, 'w_c': 0.5, 'w_i': 0.5, 'w_cor': 0.2, 'bias': 0.0,
            'hub_sigma_base_m': 40, 'road_sigma_base_m': 20,
            'hub_growth_rate': 0.03, 'road_growth_rate': 0.02,
            'max_hub_multiplier': 2.0, 'max_road_multiplier': 2.5,
            'normalize': True, 'smoothstep_tau': 0.0
        },
        'scoring_weights': {
            'commercial': {
                'f_price': 0.35, 'f_hub_com': 0.25, 'f_road': 0.20,
                'f_heat': 0.15, 'f_access': 0.05,
                'crowding': -0.03, 'junction_penalty': -0.02
            },
            'industrial': {
                'f_price': -0.20, 'f_hub_ind': 0.45, 'f_road': 0.25,
                'f_access': 0.05, 'crowding': -0.10, 'junction_penalty': -0.05
            },
            'residential': {
                'f_price': 0.10, 'f_road': 0.45, 'f_access': 0.15,
                'f_hub_com': -0.15, 'f_hub_ind': -0.10, 'crowding': -0.05
            }
        },
        'isocontour_layout': {
            'commercial': {'levels': [0.85, 0.78, 0.71], 'arc_spacing_m': [25, 35]},
            'industrial': {'levels': [0.60, 0.70, 0.80], 'arc_spacing_m': [35, 55]},
            'residential': {'band': [0.45, 0.65], 'arc_spacing_m': [35, 55]},
            'normal_offset_m': 1.0, 'jitter_m': 0.5
        }
    }
    
    # 创建并运行模拟
    simulation = EnhancedCitySimulationV3_3(config)
    simulation.initialize_simulation()
    simulation.run_simulation(total_months=24)
    
    print("🎉 模拟完成！输出文件保存在 enhanced_simulation_v3_3_output/ 目录")

if __name__ == "__main__":
    main()
