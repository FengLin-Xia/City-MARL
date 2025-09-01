#!/usr/bin/env python3
"""
滞后替代系统 v2.3
实现住宅→商业替代的滞后逻辑
"""

import numpy as np
import math
from typing import List, Dict, Tuple
import json

class HysteresisSystem:
    """滞后替代系统：住宅→商业替代"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.hysteresis_config = config.get('landuse_hysteresis', {})
        
        # 滞后参数
        self.delta_bid = self.hysteresis_config.get('delta_bid', 0.15)  # 商业评分优势阈值
        self.L_quarters = self.hysteresis_config.get('L_quarters', 2)  # 连续满足季度数
        self.cooldown_quarters = self.hysteresis_config.get('cooldown_quarters', 4)  # 冷却期
        self.res_min_share = self.hysteresis_config.get('res_min_share', 0.35)  # 住宅最小占比
        
        # 系统状态
        self.current_quarter = 0
        self.consecutive_satisfied_quarters = 0
        self.cooldown_counter = 0
        self.conversion_history = []
        self.building_scores = {}  # 记录每个建筑的评分历史
        
        # 替代标记
        self.hysteresis_flags = {}
        
    def update_quarter(self, quarter: int):
        """更新季度"""
        self.current_quarter = quarter
        
    def evaluate_conversion_conditions(self, city_state: Dict, land_price_system) -> Dict:
        """评估替代条件"""
        residential_buildings = city_state.get('residential', [])
        commercial_buildings = city_state.get('commercial', [])
        
        if not residential_buildings:
            return {'should_convert': False, 'reason': 'no_residential_buildings'}
        
        # 检查冷却期
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return {'should_convert': False, 'reason': 'in_cooldown', 'cooldown_remaining': self.cooldown_counter}
        
        # 检查住宅最小占比
        total_buildings = len(residential_buildings) + len(commercial_buildings)
        residential_ratio = len(residential_buildings) / total_buildings if total_buildings > 0 else 0
        
        if residential_ratio <= self.res_min_share:
            return {'should_convert': False, 'reason': 'residential_ratio_too_low', 'current_ratio': residential_ratio}
        
        # 计算建筑评分
        building_scores = self._calculate_building_scores(city_state, land_price_system)
        
        # 检查替代条件
        conversion_candidates = self._identify_conversion_candidates(building_scores)
        
        if not conversion_candidates:
            self.consecutive_satisfied_quarters = 0
            return {'should_convert': False, 'reason': 'no_suitable_candidates'}
        
        # 检查连续满足条件
        self.consecutive_satisfied_quarters += 1
        
        if self.consecutive_satisfied_quarters >= self.L_quarters:
            # 满足替代条件
            self.consecutive_satisfied_quarters = 0
            self.cooldown_counter = self.cooldown_quarters
            
            return {
                'should_convert': True,
                'candidates': conversion_candidates,
                'reason': 'conditions_met',
                'consecutive_quarters': self.L_quarters
            }
        else:
            return {
                'should_convert': False,
                'reason': 'waiting_for_consecutive_quarters',
                'consecutive_quarters': self.consecutive_satisfied_quarters,
                'required_quarters': self.L_quarters
            }
    
    def _calculate_building_scores(self, city_state: Dict, land_price_system) -> Dict:
        """计算建筑评分"""
        building_scores = {}
        
        # 计算住宅建筑评分
        residential_buildings = city_state.get('residential', [])
        for building in residential_buildings:
            score = self._calculate_residential_score(building, city_state, land_price_system)
            building_scores[building['id']] = {
                'type': 'residential',
                'score': score,
                'position': building['xy'],
                'capacity': building.get('capacity', 200),
                'current_usage': building.get('current_usage', 0)
            }
        
        # 计算商业建筑评分
        commercial_buildings = city_state.get('commercial', [])
        for building in commercial_buildings:
            score = self._calculate_commercial_score(building, city_state, land_price_system)
            building_scores[building['id']] = {
                'type': 'commercial',
                'score': score,
                'position': building['xy'],
                'capacity': building.get('capacity', 800),
                'current_usage': building.get('current_usage', 0)
            }
        
        return building_scores
    
    def _calculate_residential_score(self, building: Dict, city_state: Dict, land_price_system) -> float:
        """计算住宅建筑评分"""
        position = building['xy']
        
        # 1. 地价因素（住宅偏好较低地价）
        land_price = land_price_system.get_land_price(position)
        price_score = 1.0 / (1.0 + land_price / 100.0)
        
        # 2. 使用率因素（使用率适中的建筑评分更高）
        capacity = building.get('capacity', 200)
        current_usage = building.get('current_usage', 0)
        usage_ratio = current_usage / capacity if capacity > 0 else 0
        usage_score = 1.0 - abs(usage_ratio - 0.7)  # 70%使用率最佳
        
        # 3. 可达性因素
        accessibility_score = self._calculate_accessibility_score(position, city_state)
        
        # 4. 公共设施接近度
        facility_score = self._calculate_facility_proximity_score(position, city_state)
        
        # 综合评分
        total_score = (
            0.3 * price_score +
            0.3 * usage_score +
            0.2 * accessibility_score +
            0.2 * facility_score
        )
        
        return total_score
    
    def _calculate_commercial_score(self, building: Dict, city_state: Dict, land_price_system) -> float:
        """计算商业建筑评分"""
        position = building['xy']
        
        # 1. 地价因素（商业偏好较高地价）
        land_price = land_price_system.get_land_price(position)
        price_score = land_price / 300.0  # 归一化到0-1
        
        # 2. 使用率因素（使用率高的商业建筑评分更高）
        capacity = building.get('capacity', 800)
        current_usage = building.get('current_usage', 0)
        usage_ratio = current_usage / capacity if capacity > 0 else 0
        usage_score = usage_ratio
        
        # 3. 人流因素（基于热力图）
        traffic_score = self._calculate_traffic_score(position, city_state)
        
        # 4. 竞争因素（周围商业建筑密度）
        competition_score = self._calculate_competition_score(position, city_state)
        
        # 综合评分
        total_score = (
            0.4 * price_score +
            0.3 * usage_score +
            0.2 * traffic_score +
            0.1 * competition_score
        )
        
        return total_score
    
    def _calculate_accessibility_score(self, position: List[int], city_state: Dict) -> float:
        """计算可达性评分"""
        # 基于到交通枢纽的距离
        transport_hubs = city_state.get('transport_hubs', [])
        
        if not transport_hubs:
            return 0.5
        
        min_distance = float('inf')
        for hub in transport_hubs:
            distance = math.sqrt((position[0] - hub[0])**2 + (position[1] - hub[1])**2)
            min_distance = min(min_distance, distance)
        
        # 距离越近，可达性越高
        accessibility = 1.0 / (1.0 + min_distance / 100.0)
        return accessibility
    
    def _calculate_facility_proximity_score(self, position: List[int], city_state: Dict) -> float:
        """计算公共设施接近度评分"""
        public_buildings = city_state.get('public', [])
        
        if not public_buildings:
            return 0.0
        
        total_proximity = 0.0
        for building in public_buildings:
            distance = math.sqrt((position[0] - building['xy'][0])**2 + (position[1] - building['xy'][1])**2)
            if distance <= 100:  # 100像素内的设施
                total_proximity += 1.0 / (1.0 + distance / 50)
        
        return min(total_proximity / len(public_buildings), 1.0)
    
    def _calculate_traffic_score(self, position: List[int], city_state: Dict) -> float:
        """计算人流评分"""
        # 基于周围居民密度
        residents = city_state.get('residents', [])
        
        if not residents:
            return 0.0
        
        nearby_residents = 0
        for resident in residents:
            distance = math.sqrt((position[0] - resident['pos'][0])**2 + (position[1] - resident['pos'][1])**2)
            if distance <= 150:  # 150像素内的居民
                nearby_residents += 1.0 / (1.0 + distance / 75)
        
        # 归一化
        traffic_score = min(nearby_residents / len(residents), 1.0)
        return traffic_score
    
    def _calculate_competition_score(self, position: List[int], city_state: Dict) -> float:
        """计算竞争评分（周围商业建筑密度）"""
        commercial_buildings = city_state.get('commercial', [])
        
        if not commercial_buildings:
            return 0.0
        
        nearby_commercial = 0
        for building in commercial_buildings:
            distance = math.sqrt((position[0] - building['xy'][0])**2 + (position[1] - building['xy'][1])**2)
            if distance <= 100:  # 100像素内的商业建筑
                nearby_commercial += 1.0 / (1.0 + distance / 50)
        
        # 竞争越少越好（1 - 竞争密度）
        competition_score = 1.0 - min(nearby_commercial / len(commercial_buildings), 1.0)
        return competition_score
    
    def _identify_conversion_candidates(self, building_scores: Dict) -> List[Dict]:
        """识别替代候选建筑"""
        candidates = []
        
        # 找出所有住宅建筑
        residential_buildings = {bid: data for bid, data in building_scores.items() 
                               if data['type'] == 'residential'}
        
        # 找出所有商业建筑
        commercial_buildings = {bid: data for bid, data in building_scores.items() 
                              if data['type'] == 'commercial'}
        
        # 计算商业建筑的平均评分
        if commercial_buildings:
            avg_commercial_score = np.mean([data['score'] for data in commercial_buildings.values()])
        else:
            avg_commercial_score = 0.5  # 默认值
        
        # 检查每个住宅建筑
        for building_id, building_data in residential_buildings.items():
            residential_score = building_data['score']
            
            # 检查是否满足替代条件
            if residential_score + self.delta_bid < avg_commercial_score:
                candidates.append({
                    'building_id': building_id,
                    'current_score': residential_score,
                    'target_score': avg_commercial_score,
                    'score_difference': avg_commercial_score - residential_score,
                    'position': building_data['position'],
                    'capacity': building_data['capacity']
                })
        
        # 按评分差异排序（差异越大，优先级越高）
        candidates.sort(key=lambda x: x['score_difference'], reverse=True)
        
        return candidates
    
    def convert_building(self, building_id: str, city_state: Dict) -> Dict:
        """执行建筑替代"""
        # 找到要替代的住宅建筑
        residential_buildings = city_state.get('residential', [])
        target_building = None
        target_index = -1
        
        for i, building in enumerate(residential_buildings):
            if building['id'] == building_id:
                target_building = building
                target_index = i
                break
        
        if not target_building:
            return {'success': False, 'reason': 'building_not_found'}
        
        # 创建新的商业建筑
        new_commercial_building = {
            'id': f'com_conv_{building_id}',
            'type': 'commercial',
            'xy': target_building['xy'],
            'capacity': 800,  # 商业建筑容量
            'current_usage': 0,
            'construction_cost': 1000,
            'revenue_per_person': 20,
            'revenue': 0,
            'converted_from': building_id,
            'conversion_quarter': self.current_quarter
        }
        
        # 移除原住宅建筑
        residential_buildings.pop(target_index)
        
        # 添加到商业建筑列表
        city_state['commercial'].append(new_commercial_building)
        
        # 记录替代历史
        conversion_record = {
            'quarter': self.current_quarter,
            'original_building_id': building_id,
            'new_building_id': new_commercial_building['id'],
            'position': target_building['xy'],
            'original_capacity': target_building.get('capacity', 200),
            'new_capacity': 800,
            'reason': 'hysteresis_conversion'
        }
        
        self.conversion_history.append(conversion_record)
        
        # 更新替代标记
        self.hysteresis_flags[building_id] = {
            'converted': True,
            'quarter': self.current_quarter,
            'new_building_id': new_commercial_building['id']
        }
        
        return {
            'success': True,
            'converted_building': new_commercial_building,
            'conversion_record': conversion_record
        }
    
    def get_conversion_statistics(self) -> Dict:
        """获取替代统计信息"""
        total_conversions = len(self.conversion_history)
        
        # 按季度统计
        quarterly_conversions = {}
        for record in self.conversion_history:
            quarter = record['quarter']
            if quarter not in quarterly_conversions:
                quarterly_conversions[quarter] = 0
            quarterly_conversions[quarter] += 1
        
        return {
            'total_conversions': total_conversions,
            'quarterly_conversions': quarterly_conversions,
            'current_cooldown': self.cooldown_counter,
            'consecutive_quarters': self.consecutive_satisfied_quarters
        }
    
    def save_conversion_data(self, output_dir: str):
        """保存替代数据"""
        conversion_data = {
            'conversion_history': self.conversion_history,
            'hysteresis_flags': self.hysteresis_flags,
            'statistics': self.get_conversion_statistics(),
            'config': self.hysteresis_config
        }
        
        filepath = f"{output_dir}/conversion_events.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversion_data, f, indent=2, ensure_ascii=False)
