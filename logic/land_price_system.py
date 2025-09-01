#!/usr/bin/env python3
"""
地价系统模块
计算和更新城市地价分布
"""

import numpy as np
from typing import List, Dict, Tuple
import math

class LandPriceSystem:
    def __init__(self, config: Dict):
        """初始化地价系统"""
        self.config = config
        self.land_price_factors = config.get('land_price_factors', {})
        
        # 地价计算参数
        self.core_distance_weight = self.land_price_factors.get('core_distance_weight', 0.5)
        self.transport_weight = self.land_price_factors.get('transport_weight', 0.3)
        self.facility_weight = self.land_price_factors.get('facility_weight', 0.2)
        self.decay_rate = self.land_price_factors.get('decay_rate', 0.01)
        self.transport_decay_rate = self.land_price_factors.get('transport_decay_rate', 0.02)
        
        # 地价矩阵
        self.land_price_matrix = None
        self.base_price = 100
        
        # 交通枢纽点（主干道起始点）- 这是地价的核心点
        self.transport_hubs = [[40, 128], [216, 128]]
        
    def initialize_land_prices(self, map_size: List[int], transport_hubs: List[List[int]] = None):
        """初始化地价矩阵，以交通枢纽为核心点"""
        width, height = map_size
        self.land_price_matrix = np.zeros((height, width))
        
        # 使用传入的交通枢纽或默认值
        if transport_hubs:
            self.transport_hubs = transport_hubs
        
        # 计算初始地价分布
        for y in range(height):
            for x in range(width):
                self.land_price_matrix[y, x] = self._calculate_land_price([x, y])
    
    def _calculate_land_price(self, position: List[int]) -> float:
        """计算单个位置的地价，以最近的交通枢纽为核心"""
        x, y = position
        
        # 1. 核心距离影响（距离最近的交通枢纽）
        min_hub_distance = float('inf')
        for hub in self.transport_hubs:
            distance = self._calculate_distance(position, hub)
            min_hub_distance = min(min_hub_distance, distance)
        
        core_factor = math.exp(-min_hub_distance * self.decay_rate)
        
        # 2. 主干道便利性影响（距离主干道中心线）
        trunk_center_y = 128  # 主干道在y=128
        trunk_distance = abs(y - trunk_center_y)
        trunk_factor = math.exp(-trunk_distance * self.transport_decay_rate * 0.5)  # 权重较小
        
        # 3. 基础地价
        base_price = self.base_price
        
        # 综合计算
        land_price = base_price * (
            self.core_distance_weight * core_factor +
            self.transport_weight * trunk_factor +
            self.facility_weight * 1.0  # 初始时设施密度为1.0
        )
        
        return max(land_price, 50)  # 最低地价50
    
    def update_land_prices(self, city_state: Dict):
        """更新地价分布"""
        if self.land_price_matrix is None:
            return
        
        height, width = self.land_price_matrix.shape
        
        for y in range(height):
            for x in range(width):
                position = [x, y]
                
                # 计算设施密度影响
                facility_factor = self._calculate_facility_density(position, city_state)
                
                # 计算商业聚集度影响
                business_factor = self._calculate_business_cluster(position, city_state)
                
                # 计算居住需求影响
                residential_factor = self._calculate_residential_demand(position, city_state)
                
                # 更新地价
                base_price = self._calculate_land_price(position)
                updated_price = base_price * facility_factor * business_factor * residential_factor
                
                self.land_price_matrix[y, x] = max(updated_price, 50)
    
    def _calculate_facility_density(self, position: List[int], city_state: Dict) -> float:
        """计算公共设施密度影响"""
        public_buildings = city_state.get('public', [])
        if not public_buildings:
            return 1.0
        
        total_influence = 0
        for building in public_buildings:
            distance = self._calculate_distance(position, building['xy'])
            service_radius = building.get('service_radius', 50)
            
            if distance <= service_radius:
                influence = 1.0 - (distance / service_radius)
                total_influence += influence
        
        # 设施密度因子：1.0 + 总影响 * 0.1
        return 1.0 + total_influence * 0.1
    
    def _calculate_business_cluster(self, position: List[int], city_state: Dict) -> float:
        """计算商业聚集度影响"""
        commercial_buildings = city_state.get('commercial', [])
        if not commercial_buildings:
            return 1.0
        
        total_influence = 0
        for building in commercial_buildings:
            distance = self._calculate_distance(position, building['xy'])
            cluster_radius = 100  # 商业聚集影响半径
            
            if distance <= cluster_radius:
                influence = 1.0 - (distance / cluster_radius)
                total_influence += influence
        
        # 商业聚集因子：1.0 + 总影响 * 0.05
        return 1.0 + total_influence * 0.05
    
    def _calculate_residential_demand(self, position: List[int], city_state: Dict) -> float:
        """计算居住需求影响"""
        residents = city_state.get('residents', [])
        if not residents:
            return 1.0
        
        total_influence = 0
        for resident in residents:
            distance = self._calculate_distance(position, resident['pos'])
            demand_radius = 80  # 居住需求影响半径
            
            if distance <= demand_radius:
                influence = 1.0 - (distance / demand_radius)
                total_influence += influence
        
        # 居住需求因子：1.0 + 总影响 * 0.02
        return 1.0 + total_influence * 0.02
    
    def _calculate_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_land_price(self, position: List[int]) -> float:
        """获取指定位置的地价"""
        if self.land_price_matrix is None:
            return self.base_price
        
        x, y = position
        if 0 <= x < self.land_price_matrix.shape[1] and 0 <= y < self.land_price_matrix.shape[0]:
            return self.land_price_matrix[y, x]
        return self.base_price
    
    def get_land_price_stats(self) -> Dict:
        """获取地价统计信息"""
        if self.land_price_matrix is None:
            return {
                "min_price": self.base_price,
                "max_price": self.base_price,
                "avg_price": self.base_price,
                "price_distribution": {}
            }
        
        min_price = np.min(self.land_price_matrix)
        max_price = np.max(self.land_price_matrix)
        avg_price = np.mean(self.land_price_matrix)
        
        # 价格分布统计
        price_ranges = [(50, 100), (100, 150), (150, 200), (200, 250), (250, 300)]
        price_distribution = {}
        
        for low, high in price_ranges:
            count = np.sum((self.land_price_matrix >= low) & (self.land_price_matrix < high))
            price_distribution[f"{low}-{high}"] = int(count)
        
        return {
            "min_price": float(min_price),
            "max_price": float(max_price),
            "avg_price": float(avg_price),
            "price_distribution": price_distribution
        }
    
    def get_land_price_matrix(self) -> np.ndarray:
        """获取地价矩阵"""
        return self.land_price_matrix.copy() if self.land_price_matrix is not None else None
