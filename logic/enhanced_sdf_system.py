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
        
        # 高斯核参数（像素单位）
        self.hub_sigma_base = int(40 / self.meters_per_pixel)
        self.road_sigma_base = int(20 / self.meters_per_pixel)
        
        # 演化参数
        self.hub_growth_rate = 0.03
        self.road_growth_rate = 0.02
        self.max_hub_multiplier = 2.0
        self.max_road_multiplier = 2.5
        
        # 地价值参数
        self.hub_peak_value = 1.0
        self.road_peak_value = 0.7
        self.min_threshold = 0.1
        
        print(f"🏗️ 高斯核地价场系统初始化完成")
        
    def initialize_system(self, transport_hubs: List[List[int]], map_size: List[int]):
        """初始化系统"""
        self.transport_hubs = transport_hubs
        self.map_size = map_size
        self.land_price_field = self._create_initial_land_price()
        print(f"✅ 高斯核地价场系统初始化完成：{len(transport_hubs)} 个交通枢纽")
        
    def _create_initial_land_price(self) -> np.ndarray:
        """创建初始地价场"""
        return self._create_land_price_field(month=0)
        
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
    
    def _create_land_price_field(self, month: int = 0) -> np.ndarray:
        """创建地价场"""
        hub_sigma = self._calculate_hub_sigma(month)
        road_sigma = self._calculate_road_sigma(month)
        
        X, Y = np.meshgrid(np.arange(self.map_size[0]), np.arange(self.map_size[1]))
        land_price = np.zeros(self.map_size, dtype=float)
        
        for hub in self.transport_hubs:
            hub_gaussian = self._gaussian_2d(X, Y, hub[0], hub[1], hub_sigma, self.hub_peak_value)
            land_price = np.maximum(land_price, hub_gaussian)
        
        if len(self.transport_hubs) >= 2:
            road_gaussian = self._line_gaussian(X, Y, self.transport_hubs[0], self.transport_hubs[1], road_sigma, self.road_peak_value)
            land_price = np.maximum(land_price, road_gaussian)
        
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
    
    def _get_evolution_stage(self, month: int) -> Dict:
        """获取当前演化阶段配置"""
        hub_sigma = self._calculate_hub_sigma(month)
        road_sigma = self._calculate_road_sigma(month)
        
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
            'month': month
        }
    
    def update_land_price_field(self, month: int, city_state: Dict = None):
        """更新地价场"""
        self.current_month = month
        evolution_stage = self._get_evolution_stage(month)
        
        new_land_price = self._create_land_price_field(month)
        
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
        
        print(f"✅ 地价场更新完成 - 月份: {month}")
    
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
    
    def get_land_price(self, position: List[int]) -> float:
        """获取指定位置的地价值"""
        if self.land_price_field is None:
            return 0.0
        
        x, y = position[0], position[1]
        if (x < 0 or x >= self.map_size[0] or y < 0 or y >= self.map_size[1]):
            return 0.0
        
        return float(self.land_price_field[y, x])
    
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
        
        print(f"💾 地价场帧保存: {frame_file}")
    
    def get_land_price_components(self, month: int) -> Dict[str, np.ndarray]:
        """获取地价场的各个组成部分"""
        X, Y = np.meshgrid(np.arange(self.map_size[0]), np.arange(self.map_size[1]))
        hub_sigma = self._calculate_hub_sigma(month)
        road_sigma = self._calculate_road_sigma(month)
        
        hub_land_price = np.zeros(self.map_size, dtype=float)
        for hub in self.transport_hubs:
            hub_gaussian = self._gaussian_2d(X, Y, hub[0], hub[1], hub_sigma, self.hub_peak_value)
            hub_land_price = np.maximum(hub_land_price, hub_gaussian)
        
        road_land_price = np.zeros(self.map_size, dtype=float)
        if len(self.transport_hubs) >= 2:
            road_land_price = self._line_gaussian(X, Y, self.transport_hubs[0], self.transport_hubs[1], road_sigma, self.road_peak_value)
        
        combined_land_price = np.maximum(hub_land_price, road_land_price)
        combined_land_price[combined_land_price < self.min_threshold] = 0
        
        return {
            'hub_land_price': hub_land_price,
            'road_land_price': road_land_price,
            'combined_land_price': combined_land_price
        }

# 为了保持兼容性，保留原来的类名作为别名
EnhancedSDFSystem = GaussianLandPriceSystem
