#!/usr/bin/env python3
"""
增强的智能体系统
政府、企业、居民智能体的决策逻辑
"""

import random
import numpy as np
from typing import List, Dict, Tuple
import math

class GovernmentAgent:
    """政府智能体：负责公共设施建设"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.decision_frequency = config.get('decision_frequency', 30)
        self.budget_limit = config.get('budget_limit', 10000)
        self.public_facility_threshold = config.get('public_facility_threshold', 0.8)
        self.coverage_threshold = config.get('coverage_threshold', 0.6)
        self.avg_time_threshold = config.get('avg_time_threshold', 8)
        
    def make_decisions(self, city_state: Dict, land_price_system) -> List[Dict]:
        """政府决策：建设公共设施"""
        new_public_buildings = []
        
        # 检查是否需要建设公共设施
        if self._needs_public_facility(city_state):
            # 找到最佳建设位置
            best_position = self._find_best_public_facility_position(city_state, land_price_system)
            
            if best_position:
                new_building = {
                    "id": f"pub_{len(city_state.get('public', [])) + 1}",
                    "type": "public",
                    "xy": best_position,
                    "capacity": 500,
                    "current_usage": 0,
                    "service_radius": 50,
                    "construction_cost": 1000,
                    "revenue": 0
                }
                new_public_buildings.append(new_building)
        
        return new_public_buildings
    
    def _needs_public_facility(self, city_state: Dict) -> bool:
        """检查是否需要建设公共设施"""
        residents = city_state.get('residents', [])
        public_buildings = city_state.get('public', [])
        
        if not residents:
            return False
        
        # 计算公共设施覆盖率
        total_residents = len(residents)
        covered_residents = 0
        
        for resident in residents:
            resident_pos = resident['pos']
            for building in public_buildings:
                distance = self._calculate_distance(resident_pos, building['xy'])
                if distance <= building.get('service_radius', 50):
                    covered_residents += 1
                    break
        
        coverage_ratio = covered_residents / total_residents if total_residents > 0 else 0
        
        return coverage_ratio < self.coverage_threshold
    
    def _find_best_public_facility_position(self, city_state: Dict, land_price_system) -> List[int]:
        """找到最佳公共设施建设位置"""
        residents = city_state.get('residents', [])
        public_buildings = city_state.get('public', [])
        
        if not residents:
            return None
        
        # 找到需求最强烈的区域
        demand_map = np.zeros((256, 256))
        
        for resident in residents:
            resident_pos = resident['pos']
            x, y = int(resident_pos[0]), int(resident_pos[1])
            
            # 检查是否已有公共设施覆盖
            is_covered = False
            for building in public_buildings:
                distance = self._calculate_distance(resident_pos, building['xy'])
                if distance <= building.get('service_radius', 50):
                    is_covered = True
                    break
            
            if not is_covered:
                # 在需求地图上增加权重
                for dy in range(-50, 51):
                    for dx in range(-50, 51):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 256 and 0 <= ny < 256:
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance <= 50:
                                demand_map[ny, nx] += 1.0 / (1.0 + distance)
        
        # 结合地价因素
        land_prices = land_price_system.get_land_price_matrix()
        if land_prices is not None:
            # 偏好地价适中的区域（不是最高也不是最低）
            price_factor = 1.0 - abs(land_prices - np.mean(land_prices)) / np.std(land_prices)
            price_factor = np.clip(price_factor, 0.1, 1.0)
            demand_map *= price_factor
        
        # 找到需求最高的位置
        if np.max(demand_map) > 0:
            best_positions = np.where(demand_map == np.max(demand_map))
            y, x = best_positions[0][0], best_positions[1][0]
            return [int(x), int(y)]
        
        return None
    
    def _calculate_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

class BusinessAgent:
    """企业智能体：负责住宅和商业建筑建设（地价驱动）"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.profit_threshold = config.get('profit_threshold', 0.6)
        self.consecutive_days = config.get('consecutive_days', 2)
        self.expansion_probability = config.get('expansion_probability', 0.4)
        self.usage_ratio_threshold = config.get('usage_ratio_threshold', 0.7)
        
        # 地价驱动参数
        self.land_price_weight = config.get('land_price_weight', 0.6)
        self.heatmap_weight = config.get('heatmap_weight', 0.3)
        self.facility_weight = config.get('facility_weight', 0.1)
        self.candidate_top_percent = config.get('candidate_top_percent', 20)
        
    def make_decisions(self, city_state: Dict, land_price_system, trajectory_system=None) -> Tuple[List[Dict], List[Dict]]:
        """企业决策：建设住宅和商业建筑（地价驱动）"""
        new_residential_buildings = []
        new_commercial_buildings = []
        
        # 获取地价矩阵
        land_price_matrix = land_price_system.get_land_price_matrix()
        if land_price_matrix is None:
            return new_residential_buildings, new_commercial_buildings
        
        # 获取热力图数据
        heatmap_data = None
        if trajectory_system:
            heatmap_data = trajectory_system.get_heatmap_data()
        
        # 住宅建设决策
        if self._needs_residential_expansion(city_state):
            new_residential = self._decide_residential_development_enhanced(
                city_state, land_price_system, land_price_matrix, heatmap_data
            )
            new_residential_buildings.extend(new_residential)
        
        # 商业建设决策
        if self._needs_commercial_expansion(city_state):
            new_commercial = self._decide_commercial_development_enhanced(
                city_state, land_price_system, land_price_matrix, heatmap_data
            )
            new_commercial_buildings.extend(new_commercial)
        
        return new_residential_buildings, new_commercial_buildings
    
    def _needs_residential_expansion(self, city_state: Dict) -> bool:
        """检查是否需要住宅扩张"""
        residents = city_state.get('residents', [])
        residential_buildings = city_state.get('residential', [])
        
        if not residential_buildings:
            return len(residents) > 0
        
        # 计算住宅使用率
        total_capacity = sum(building.get('capacity', 200) for building in residential_buildings)
        total_population = len(residents)
        
        usage_ratio = total_population / total_capacity if total_capacity > 0 else 0
        
        # 如果使用率超过阈值，或者人口接近容量上限，则需要扩张
        capacity_threshold = 0.8  # 80%容量时开始建设
        return usage_ratio > self.usage_ratio_threshold or usage_ratio > capacity_threshold
    
    def _needs_commercial_expansion(self, city_state: Dict) -> bool:
        """检查是否需要商业扩张（基于人口基础和建筑密度）"""
        residents = city_state.get('residents', [])
        commercial_buildings = city_state.get('commercial', [])
        residential_buildings = city_state.get('residential', [])
        
        # 基础条件：有足够的人口和住宅
        if len(residents) < 30 or len(residential_buildings) < 3:
            return False
        
        # 商业建筑密度：每35个居民需要1个商业建筑（更积极）
        target_commercial = len(residents) // 35
        current_commercial = len(commercial_buildings)
        
        # 如果当前商业建筑数量少于目标，则需要建设
        return current_commercial < target_commercial
    
    def _decide_residential_development(self, city_state: Dict, land_price_system) -> List[Dict]:
        """住宅开发决策"""
        new_buildings = []
        
        # 找到最佳住宅建设位置
        best_position = self._find_best_residential_position(city_state, land_price_system)
        
        if best_position:
            new_building = {
                "id": f"res_{len(city_state.get('residential', [])) + 1}",
                "type": "residential",
                "xy": best_position,
                "capacity": 200,
                "current_usage": 0,
                "construction_cost": 500,
                "revenue_per_person": 10,
                "revenue": 0
            }
            new_buildings.append(new_building)
        
        return new_buildings
    
    def _decide_commercial_development(self, city_state: Dict, land_price_system) -> List[Dict]:
        """商业开发决策"""
        new_buildings = []
        
        # 找到最佳商业建设位置
        best_position = self._find_best_commercial_position(city_state, land_price_system)
        
        if best_position:
            new_building = {
                "id": f"com_{len(city_state.get('commercial', [])) + 1}",
                "type": "commercial",
                "xy": best_position,
                "capacity": 800,
                "current_usage": 0,
                "construction_cost": 1000,
                "revenue_per_person": 20,
                "revenue": 0
            }
            new_buildings.append(new_building)
        
        return new_buildings
    
    def _find_best_residential_position(self, city_state: Dict, land_price_system) -> List[int]:
        """找到最佳住宅建设位置"""
        # 住宅偏好：靠近主干道，地价适中
        trunk_center_y = 128
        
        # 在主干道附近寻找位置
        for _ in range(20):
            x = random.randint(20, 236)
            y = trunk_center_y + random.randint(-60, 60)
            
            if 20 <= y <= 236:
                position = [x, y]
                land_price = land_price_system.get_land_price(position)
                
                # 检查地价是否合理（不是最高也不是最低）
                if 80 <= land_price <= 200:
                    return position
        
        return None
    
    def _find_best_commercial_position(self, city_state: Dict, land_price_system) -> List[int]:
        """找到最佳商业建设位置"""
        # 商业偏好：靠近核心点，人流密集
        core_point = [128, 128]
        
        # 在核心点附近寻找位置
        for _ in range(20):
            x = core_point[0] + random.randint(-100, 100)
            y = core_point[1] + random.randint(-100, 100)
            
            if 20 <= x <= 236 and 20 <= y <= 236:
                position = [x, y]
                land_price = land_price_system.get_land_price(position)
                
                # 商业偏好较高地价区域
                if land_price >= 120:
                    return position
        
        return None
    
    def _decide_residential_development_enhanced(self, city_state: Dict, land_price_system, land_price_matrix, heatmap_data) -> List[Dict]:
        """增强版住宅开发决策（地价驱动）"""
        new_buildings = []
        
        # 首先检查是否需要住宅扩张
        if not self._needs_residential_expansion(city_state):
            return new_buildings
        
        # 获取最小距离配置
        min_distance = self.config.get('min_building_distance', {}).get('residential', 30)
        
        # 生成候选地块（排除已有建筑）
        candidates = self._get_available_land_price_zones(
            land_price_matrix, city_state, top_percent=self.candidate_top_percent, min_distance=min_distance
        )
        
        if not candidates:
            return new_buildings
        
        # 计算每个候选地块的评分
        scores = {}
        for candidate in candidates:
            score = (
                land_price_matrix[candidate[1], candidate[0]] * self.land_price_weight +
                self._calculate_facility_proximity(candidate, city_state) * self.facility_weight
            )
            scores[candidate] = score
        
        # 选择最佳位置
        if scores:
            best_loc = max(scores, key=scores.get)
            new_building = {
                "id": f"res_{len(city_state.get('residential', [])) + 1}",
                "type": "residential",
                "xy": [best_loc[0], best_loc[1]],
                "capacity": 200,
                "current_usage": 0,
                "construction_cost": 500,
                "revenue_per_person": 10,
                "revenue": 0
            }
            new_buildings.append(new_building)
        
        return new_buildings
    
    def _decide_commercial_development_enhanced(self, city_state: Dict, land_price_system, land_price_matrix, heatmap_data) -> List[Dict]:
        """增强版商业开发决策（地价驱动）"""
        new_buildings = []
        
        # 首先检查是否需要商业扩张
        if not self._needs_commercial_expansion(city_state):
            return new_buildings
        
        # 获取最小距离配置
        min_distance = self.config.get('min_building_distance', {}).get('commercial', 40)
        
        # 生成候选地块（排除已有建筑）
        candidates = self._get_available_land_price_zones(
            land_price_matrix, city_state, top_percent=self.candidate_top_percent, min_distance=min_distance
        )
        
        if not candidates:
            return new_buildings
        
        # 计算每个候选地块的评分
        scores = {}
        for candidate in candidates:
            land_price_score = land_price_matrix[candidate[1], candidate[0]] * self.land_price_weight
            
            # 热力图评分
            heatmap_score = 0
            if heatmap_data and 'combined_heatmap' in heatmap_data:
                heatmap = heatmap_data['combined_heatmap']
                if candidate[1] < heatmap.shape[0] and candidate[0] < heatmap.shape[1]:
                    heatmap_score = heatmap[candidate[1], candidate[0]] * self.heatmap_weight
            
            score = land_price_score + heatmap_score
            scores[candidate] = score
        
        # 选择最佳位置
        if scores:
            best_loc = max(scores, key=scores.get)
            new_building = {
                "id": f"com_{len(city_state.get('commercial', [])) + 1}",
                "type": "commercial",
                "xy": [best_loc[0], best_loc[1]],
                "capacity": 800,
                "current_usage": 0,
                "construction_cost": 1000,
                "revenue_per_person": 20,
                "revenue": 0
            }
            new_buildings.append(new_building)
        
        return new_buildings
    
    def _get_top_land_price_zones(self, land_price_matrix, city_state: Dict, top_percent=20):
        """获取地价最高的区域作为候选地块"""
        if land_price_matrix is None:
            return []
        
        # 计算地价阈值（top_percent%的最高地价）
        flat_prices = land_price_matrix.flatten()
        threshold = np.percentile(flat_prices, 100 - top_percent)
        
        # 找到高于阈值的位置
        candidates = []
        for y in range(land_price_matrix.shape[0]):
            for x in range(land_price_matrix.shape[1]):
                if land_price_matrix[y, x] >= threshold:
                    # 检查是否已有建筑
                    if self._is_position_available([x, y], city_state):
                        candidates.append((x, y))
        
        return candidates
    
    def _get_available_land_price_zones(self, land_price_matrix, city_state: Dict, top_percent=20, min_distance=30):
        """获取可用的高地价区域，确保不与现有建筑重叠"""
        if land_price_matrix is None:
            return []
        
        # 计算地价阈值（top_percent%的最高地价）
        flat_prices = land_price_matrix.flatten()
        threshold = np.percentile(flat_prices, 100 - top_percent)
        
        # 找到高于阈值的位置
        candidates = []
        for y in range(land_price_matrix.shape[0]):
            for x in range(land_price_matrix.shape[1]):
                if land_price_matrix[y, x] >= threshold:
                    position = [x, y]
                    # 检查是否与现有建筑保持最小距离
                    if self._is_position_available_with_distance(position, city_state, min_distance):
                        candidates.append((x, y))
        
        return candidates
    
    def _is_position_available_with_distance(self, position, city_state: Dict, min_distance=30):
        """检查位置是否可用（与现有建筑保持最小距离）"""
        if city_state is None:
            return True
        
        # 检查所有类型的建筑
        all_buildings = []
        all_buildings.extend(city_state.get('public', []))
        all_buildings.extend(city_state.get('residential', []))
        all_buildings.extend(city_state.get('commercial', []))
        
        # 检查是否有建筑在最小距离内
        for building in all_buildings:
            distance = self._calculate_distance(position, building['xy'])
            if distance < min_distance:
                return False
        
        return True
    
    def _calculate_facility_proximity(self, position, city_state: Dict) -> float:
        """计算与公共设施的接近度"""
        public_buildings = city_state.get('public', [])
        
        if not public_buildings:
            return 0.0
        
        total_proximity = 0.0
        for building in public_buildings:
            distance = self._calculate_distance(position, building['xy'])
            if distance <= 100:  # 100像素内的设施
                total_proximity += 1.0 / (1.0 + distance / 50)
        
        return min(total_proximity / len(public_buildings), 1.0)
    
    def _is_position_available(self, position, city_state: Dict = None):
        """检查位置是否可用（没有其他建筑）"""
        if city_state is None:
            return True
        
        # 检查所有类型的建筑
        all_buildings = []
        all_buildings.extend(city_state.get('public', []))
        all_buildings.extend(city_state.get('residential', []))
        all_buildings.extend(city_state.get('commercial', []))
        
        # 检查是否有建筑在相同位置
        for building in all_buildings:
            if building['xy'] == position:
                return False
        
        return True
    
    def _calculate_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

class ResidentAgent:
    """居民智能体：需求提供和日常行为"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.movement_speed = config.get('movement_speed', 4)
        self.preference_weight = config.get('preference_weight', {
            'cost': 0.4,
            'convenience': 0.3,
            'quality': 0.3
        })
        self.income_range = config.get('income_range', [3000, 8000])
        self.housing_cost_ratio = config.get('housing_cost_ratio', 0.4)
        
    def choose_residence(self, available_housing: List[Dict], city_state: Dict, land_price_system) -> Dict:
        """选择居住地"""
        if not available_housing:
            return None
        
        best_home = None
        best_score = -1
        
        for home in available_housing:
            score = self._calculate_residence_score(home, city_state, land_price_system)
            if score > best_score:
                best_score = score
                best_home = home
        
        return best_home
    
    def _calculate_residence_score(self, home: Dict, city_state: Dict, land_price_system) -> float:
        """计算居住地评分"""
        position = home['xy']
        
        # 1. 成本因素（地价影响）
        land_price = land_price_system.get_land_price(position)
        cost_score = 1.0 / (1.0 + land_price / 100)  # 地价越低分数越高
        
        # 2. 便利性因素（距离公共设施）
        convenience_score = self._calculate_convenience_score(position, city_state)
        
        # 3. 质量因素（建筑容量、使用率）
        quality_score = self._calculate_quality_score(home)
        
        # 综合评分
        total_score = (
            self.preference_weight['cost'] * cost_score +
            self.preference_weight['convenience'] * convenience_score +
            self.preference_weight['quality'] * quality_score
        )
        
        return total_score
    
    def _calculate_convenience_score(self, position: List[int], city_state: Dict) -> float:
        """计算便利性评分"""
        public_buildings = city_state.get('public', [])
        commercial_buildings = city_state.get('commercial', [])
        
        total_convenience = 0
        
        # 公共设施便利性
        for building in public_buildings:
            distance = self._calculate_distance(position, building['xy'])
            if distance <= 100:  # 100像素内的便利性
                total_convenience += 1.0 / (1.0 + distance / 50)
        
        # 商业设施便利性
        for building in commercial_buildings:
            distance = self._calculate_distance(position, building['xy'])
            if distance <= 150:  # 150像素内的便利性
                total_convenience += 1.0 / (1.0 + distance / 75)
        
        return min(total_convenience / 10.0, 1.0)  # 归一化到0-1
    
    def _calculate_quality_score(self, home: Dict) -> float:
        """计算质量评分"""
        capacity = home.get('capacity', 200)
        current_usage = home.get('current_usage', 0)
        
        # 使用率适中的建筑质量更高
        usage_ratio = current_usage / capacity if capacity > 0 else 0
        quality_score = 1.0 - abs(usage_ratio - 0.7)  # 70%使用率最佳
        
        return max(quality_score, 0.1)
    
    def _calculate_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
