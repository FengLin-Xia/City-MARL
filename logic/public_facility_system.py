#!/usr/bin/env python3
"""
公共设施系统 v2.3
实现公共用地的强制介入机制
"""

import numpy as np
import math
from typing import List, Dict, Tuple
import random
import json

class PublicFacilitySystem:
    """公共设施系统：智能公共用地布局"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.facility_config = config.get('public_facility_rules', {})
        self.city_map_size = config.get('city', {}).get('map_size', [256, 256])
        
        # 设施类型配置
        self.school_config = self.facility_config.get('school', {
            'trigger_population': 500,
            'service_radius': 500,
            'coverage_threshold': 0.8,
            'symbol': '🏫'
        })
        
        self.hospital_config = self.facility_config.get('hospital', {
            'trigger_distance_threshold': 800,
            'service_radius': 800,
            'symbol': '🏥'
        })
        
        self.park_config = self.facility_config.get('park', {
            'trigger_building_density': 0.6,
            'service_radius': 300,
            'building_area_px': 100,
            'symbol': '🌳'
        })
        
        self.plaza_config = self.facility_config.get('plaza', {
            'trigger_commercial_density': 0.5,
            'service_radius': 400,
            'commercial_area_px': 150,
            'symbol': '🏛️'
        })
        
        # 系统状态
        self.facility_history = []
        self.trigger_events = []
        
    def evaluate_facility_needs(self, city_state: Dict) -> Dict:
        """评估公共设施需求"""
        needs = {
            'school': self._evaluate_school_need(city_state),
            'hospital': self._evaluate_hospital_need(city_state),
            'park': self._evaluate_park_need(city_state),
            'plaza': self._evaluate_plaza_need(city_state)
        }
        
        return needs
    
    def _evaluate_school_need(self, city_state: Dict) -> Dict:
        """评估学校需求"""
        residents = city_state.get('residents', [])
        public_buildings = city_state.get('public', [])
        
        # 统计现有学校
        existing_schools = [b for b in public_buildings if b.get('facility_type') == 'school']
        
        # 检查人口触发条件
        population = len(residents)
        trigger_population = self.school_config['trigger_population']
        
        # 计算覆盖率
        covered_residents = 0
        for resident in residents:
            resident_pos = resident['pos']
            for school in existing_schools:
                distance = self._calculate_distance(resident_pos, school['xy'])
                if distance <= self.school_config['service_radius']:
                    covered_residents += 1
                    break
        
        coverage_ratio = covered_residents / population if population > 0 else 0
        
        # 判断是否需要建设
        coverage_threshold = float(self.school_config.get('coverage_threshold', 0.8))
        need_school = (population >= trigger_population and coverage_ratio < coverage_threshold)
        
        return {
            'needed': need_school,
            'reason': 'population_threshold' if population >= trigger_population else 'coverage_insufficient',
            'population': population,
            'trigger_population': trigger_population,
            'coverage_ratio': coverage_ratio,
            'existing_schools': len(existing_schools)
        }
    
    def _evaluate_hospital_need(self, city_state: Dict) -> Dict:
        """评估医院需求"""
        residents = city_state.get('residents', [])
        public_buildings = city_state.get('public', [])
        
        # 统计现有医院
        existing_hospitals = [b for b in public_buildings if b.get('facility_type') == 'hospital']
        
        # 计算平均可达性
        total_accessibility = 0
        for resident in residents:
            resident_pos = resident['pos']
            min_distance = float('inf')
            
            for hospital in existing_hospitals:
                distance = self._calculate_distance(resident_pos, hospital['xy'])
                min_distance = min(min_distance, distance)
            
            if min_distance == float('inf'):
                min_distance = self.hospital_config['trigger_distance_threshold']
            
            total_accessibility += min_distance
        
        avg_accessibility = total_accessibility / len(residents) if residents else 0
        
        # 判断是否需要建设
        need_hospital = avg_accessibility > self.hospital_config['trigger_distance_threshold']
        
        return {
            'needed': need_hospital,
            'reason': 'accessibility_poor',
            'avg_accessibility': avg_accessibility,
            'threshold': self.hospital_config['trigger_distance_threshold'],
            'existing_hospitals': len(existing_hospitals)
        }
    
    def _evaluate_park_need(self, city_state: Dict) -> Dict:
        """评估公园需求"""
        all_buildings = []
        all_buildings.extend(city_state.get('public', []))
        all_buildings.extend(city_state.get('residential', []))
        all_buildings.extend(city_state.get('commercial', []))
        
        public_buildings = city_state.get('public', [])
        existing_parks = [b for b in public_buildings if b.get('facility_type') == 'park']
        
        # 计算建筑密度
        map_size = self.city_map_size
        total_area = map_size[0] * map_size[1]
        building_area_px = float(self.park_config.get('building_area_px', 100))
        building_area = len(all_buildings) * building_area_px
        building_density = building_area / total_area
        
        # 判断是否需要建设
        need_park = building_density > self.park_config['trigger_building_density']
        
        return {
            'needed': need_park,
            'reason': 'building_density_high',
            'building_density': building_density,
            'threshold': self.park_config['trigger_building_density'],
            'existing_parks': len(existing_parks)
        }
    
    def _evaluate_plaza_need(self, city_state: Dict) -> Dict:
        """评估广场需求"""
        commercial_buildings = city_state.get('commercial', [])
        public_buildings = city_state.get('public', [])
        existing_plazas = [b for b in public_buildings if b.get('facility_type') == 'plaza']
        
        # 计算商业密度
        map_size = self.city_map_size
        total_area = map_size[0] * map_size[1]
        commercial_area_px = float(self.plaza_config.get('commercial_area_px', 150))
        commercial_area = len(commercial_buildings) * commercial_area_px
        commercial_density = commercial_area / total_area
        
        # 判断是否需要建设
        need_plaza = commercial_density > self.plaza_config['trigger_commercial_density']
        
        return {
            'needed': need_plaza,
            'reason': 'commercial_density_high',
            'commercial_density': commercial_density,
            'threshold': self.plaza_config['trigger_commercial_density'],
            'existing_plazas': len(existing_plazas)
        }
    
    def generate_facilities(self, city_state: Dict, facility_needs: Dict) -> List[Dict]:
        """生成公共设施"""
        new_facilities = []
        
        # 生成学校
        if facility_needs['school']['needed']:
            school = self._generate_school(city_state)
            if school:
                new_facilities.append(school)
        
        # 生成医院
        if facility_needs['hospital']['needed']:
            hospital = self._generate_hospital(city_state)
            if hospital:
                new_facilities.append(hospital)
        
        # 生成公园
        if facility_needs['park']['needed']:
            park = self._generate_park(city_state)
            if park:
                new_facilities.append(park)
        
        # 生成广场
        if facility_needs['plaza']['needed']:
            plaza = self._generate_plaza(city_state)
            if plaza:
                new_facilities.append(plaza)
        
        return new_facilities
    
    def _generate_school(self, city_state: Dict) -> Dict:
        """生成学校"""
        residents = city_state.get('residents', [])
        public_buildings = city_state.get('public', [])
        
        # 找到需求最强烈的区域
        best_position = self._find_best_school_position(residents, public_buildings)
        
        if best_position:
            school = {
                'id': f'pub_school_{len(public_buildings) + 1}',
                'type': 'public',
                'facility_type': 'school',
                'xy': best_position,
                'capacity': 600,
                'current_usage': 0,
                'service_radius': self.school_config['service_radius'],
                'construction_cost': 2000,
                'revenue': 0,
                'symbol': self.school_config['symbol']
            }
            
            # 记录触发事件
            self.trigger_events.append({
                'facility_type': 'school',
                'position': best_position,
                'reason': 'population_threshold',
                'population': len(residents)
            })
            
            return school
        
        return None
    
    def _generate_hospital(self, city_state: Dict) -> Dict:
        """生成医院"""
        residents = city_state.get('residents', [])
        public_buildings = city_state.get('public', [])
        
        # 找到可达性最差的区域
        best_position = self._find_best_hospital_position(residents, public_buildings)
        
        if best_position:
            hospital = {
                'id': f'pub_hospital_{len(public_buildings) + 1}',
                'type': 'public',
                'facility_type': 'hospital',
                'xy': best_position,
                'capacity': 300,
                'current_usage': 0,
                'service_radius': self.hospital_config['service_radius'],
                'construction_cost': 3000,
                'revenue': 0,
                'symbol': self.hospital_config['symbol']
            }
            
            # 记录触发事件
            self.trigger_events.append({
                'facility_type': 'hospital',
                'position': best_position,
                'reason': 'accessibility_poor'
            })
            
            return hospital
        
        return None
    
    def _generate_park(self, city_state: Dict) -> Dict:
        """生成公园"""
        all_buildings = []
        all_buildings.extend(city_state.get('public', []))
        all_buildings.extend(city_state.get('residential', []))
        all_buildings.extend(city_state.get('commercial', []))
        
        # 找到建筑密度最高的区域
        best_position = self._find_best_park_position(all_buildings)
        
        if best_position:
            park = {
                'id': f'pub_park_{len(city_state.get("public", [])) + 1}',
                'type': 'public',
                'facility_type': 'park',
                'xy': best_position,
                'capacity': 1000,
                'current_usage': 0,
                'service_radius': self.park_config['service_radius'],
                'construction_cost': 1500,
                'revenue': 0,
                'symbol': self.park_config['symbol']
            }
            
            # 记录触发事件
            self.trigger_events.append({
                'facility_type': 'park',
                'position': best_position,
                'reason': 'building_density_high'
            })
            
            return park
        
        return None
    
    def _generate_plaza(self, city_state: Dict) -> Dict:
        """生成广场"""
        commercial_buildings = city_state.get('commercial', [])
        
        # 找到商业密度最高的区域
        best_position = self._find_best_plaza_position(commercial_buildings)
        
        if best_position:
            plaza = {
                'id': f'pub_plaza_{len(city_state.get("public", [])) + 1}',
                'type': 'public',
                'facility_type': 'plaza',
                'xy': best_position,
                'capacity': 800,
                'current_usage': 0,
                'service_radius': self.plaza_config['service_radius'],
                'construction_cost': 2500,
                'revenue': 0,
                'symbol': self.plaza_config['symbol']
            }
            
            # 记录触发事件
            self.trigger_events.append({
                'facility_type': 'plaza',
                'position': best_position,
                'reason': 'commercial_density_high'
            })
            
            return plaza
        
        return None
    
    def _find_best_school_position(self, residents: List[Dict], public_buildings: List[Dict]) -> List[int]:
        """找到最佳学校位置"""
        # 找到未被学校覆盖的居民聚集区域
        width, height = self.city_map_size[0], self.city_map_size[1]
        uncovered_demand = np.zeros((height, width))
        
        for resident in residents:
            resident_pos = resident['pos']
            x, y = int(resident_pos[0]), int(resident_pos[1])
            
            # 检查是否已被学校覆盖
            is_covered = False
            for building in public_buildings:
                if building.get('facility_type') == 'school':
                    distance = self._calculate_distance(resident_pos, building['xy'])
                    if distance <= self.school_config['service_radius']:
                        is_covered = True
                        break
            
            if not is_covered:
                # 在需求地图上增加权重
                for dy in range(-50, 51):
                    for dx in range(-50, 51):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance <= 50:
                                uncovered_demand[ny, nx] += 1.0 / (1.0 + distance)
        
        # 找到需求最高的位置
        if np.max(uncovered_demand) > 0:
            best_positions = np.where(uncovered_demand == np.max(uncovered_demand))
            y, x = best_positions[0][0], best_positions[1][0]
            return [int(x), int(y)]
        
        return None
    
    def _find_best_hospital_position(self, residents: List[Dict], public_buildings: List[Dict]) -> List[int]:
        """找到最佳医院位置"""
        # 找到医疗可达性最差的区域
        width, height = self.city_map_size[0], self.city_map_size[1]
        accessibility_map = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):
                position = [x, y]
                min_distance = float('inf')
                
                # 计算到最近医院的距离
                for building in public_buildings:
                    if building.get('facility_type') == 'hospital':
                        distance = self._calculate_distance(position, building['xy'])
                        min_distance = min(min_distance, distance)
                
                if min_distance == float('inf'):
                    min_distance = 1000  # 默认值
                
                accessibility_map[y, x] = min_distance
        
        # 找到可达性最差的位置
        worst_positions = np.where(accessibility_map == np.max(accessibility_map))
        y, x = worst_positions[0][0], worst_positions[1][0]
        return [int(x), int(y)]
    
    def _find_best_park_position(self, all_buildings: List[Dict]) -> List[int]:
        """找到最佳公园位置"""
        # 计算建筑密度图
        width, height = self.city_map_size[0], self.city_map_size[1]
        density_map = np.zeros((height, width))
        
        for building in all_buildings:
            building_pos = building['xy']
            x, y = int(building_pos[0]), int(building_pos[1])
            
            # 在建筑周围增加密度
            for dy in range(-30, 31):
                for dx in range(-30, 31):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        distance = math.sqrt(dx*dx + dy*dy)
                        if distance <= 30:
                            density_map[ny, nx] += 1.0 / (1.0 + distance)
        
        # 找到密度最高的位置
        if np.max(density_map) > 0:
            best_positions = np.where(density_map == np.max(density_map))
            y, x = best_positions[0][0], best_positions[1][0]
            return [int(x), int(y)]
        
        return None
    
    def _find_best_plaza_position(self, commercial_buildings: List[Dict]) -> List[int]:
        """找到最佳广场位置"""
        # 计算商业密度图
        width, height = self.city_map_size[0], self.city_map_size[1]
        commercial_density = np.zeros((height, width))
        
        for building in commercial_buildings:
            building_pos = building['xy']
            x, y = int(building_pos[0]), int(building_pos[1])
            
            # 在商业建筑周围增加密度
            for dy in range(-40, 41):
                for dx in range(-40, 41):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        distance = math.sqrt(dx*dx + dy*dy)
                        if distance <= 40:
                            commercial_density[ny, nx] += 1.0 / (1.0 + distance)
        
        # 找到商业密度最高的位置
        if np.max(commercial_density) > 0:
            best_positions = np.where(commercial_density == np.max(commercial_density))
            y, x = best_positions[0][0], best_positions[1][0]
            return [int(x), int(y)]
        
        return None
    
    def _calculate_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_facility_statistics(self, city_state: Dict) -> Dict:
        """获取公共设施统计信息"""
        public_buildings = city_state.get('public', [])
        
        facility_counts = {
            'school': len([b for b in public_buildings if b.get('facility_type') == 'school']),
            'hospital': len([b for b in public_buildings if b.get('facility_type') == 'hospital']),
            'park': len([b for b in public_buildings if b.get('facility_type') == 'park']),
            'plaza': len([b for b in public_buildings if b.get('facility_type') == 'plaza'])
        }
        
        return {
            'total_facilities': len(public_buildings),
            'facility_counts': facility_counts,
            'trigger_events': len(self.trigger_events)
        }
    
    def save_facility_data(self, output_dir: str):
        """保存公共设施数据"""
        facility_data = {
            'trigger_events': self.trigger_events,
            'facility_history': self.facility_history,
            'config': self.facility_config
        }
        
        filepath = f"{output_dir}/facility_events.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(facility_data, f, indent=2, ensure_ascii=False)
