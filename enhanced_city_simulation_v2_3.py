#!/usr/bin/env python3
"""
增强城市模拟系统 v2.3
基于高斯核地价场的城市演化模拟
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import random
import math
import time

# 导入v2.3新模块
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem
from logic.hysteresis_system import HysteresisSystem
from logic.public_facility_system import PublicFacilitySystem

# 导入现有模块
from logic.enhanced_agents import GovernmentAgent, BusinessAgent, ResidentAgent
from logic.output_system import OutputSystem
from logic.placement import PlacementLogic
from logic.schedule import ScheduleLogic
from logic.move import MoveLogic
from logic.trajectory_system import TrajectorySystem
from viz.ide import CityVisualizer

class EnhancedCitySimulationV2_3:
    """增强城市模拟系统 v2.3"""
    
    def __init__(self):
        """初始化模拟系统"""
        # 加载v2.3配置
        self.city_config = self._load_config('configs/city_config_v2_3.json')
        self.building_config = self._load_config('configs/building_config.json')
        self.agent_config = self._load_config('configs/agent_config.json')
        
        # 初始化v2.3新系统
        self.land_price_system = GaussianLandPriceSystem(self.city_config)
        self.isocontour_system = IsocontourBuildingSystem(self.city_config)
        self.hysteresis_system = HysteresisSystem(self.city_config)
        self.public_facility_system = PublicFacilitySystem(self.city_config)
        
        # 初始化现有系统
        self.government_agent = GovernmentAgent(self.agent_config['government_agent'])
        business_config = self.agent_config['business_agent'].copy()
        if 'building_growth' in self.building_config:
            business_config.update(self.building_config['building_growth'])
        self.business_agent = BusinessAgent(business_config)
        self.resident_agent = ResidentAgent(self.agent_config['resident_agent'])
        
        # 初始化输出系统
        self.output_system = OutputSystem('enhanced_simulation_v2_3_output')
        self.output_dir = 'enhanced_simulation_v2_3_output'
        
        # 初始化现有逻辑模块
        self.placement_logic = PlacementLogic()
        self.schedule_logic = ScheduleLogic()
        self.move_logic = MoveLogic()
        
        # 初始化可视化器
        self.visualizer = CityVisualizer()
        
        # 模拟状态
        self.current_month = 0
        self.current_quarter = 0
        self.current_year = 0
        self.city_state = {}
        self.monthly_stats = []
        self.quarterly_stats = []
        self.yearly_stats = []
        
        # 居民-住宅关系映射
        self.resident_homes = {}
        
        # 时间系统配置
        self.time_config = self.city_config.get('time_system', {})
        self.update_frequencies = self.time_config.get('update_frequencies', {})
        self.quarterly_growth = self.time_config.get('quarterly_building_growth', {})
        
        # 可视化配置
        self.viz_config = self.city_config.get('visualization_config', {})
        
        # 逐层填满建筑生成系统
        self.progressive_growth_config = self.city_config.get('progressive_growth', {
            'enabled': True,
            'layer_activation_threshold': 0.8,
            'layer_delay_threshold': 0.3,
            'max_quarters_per_layer': 4,
            'growth_animation': True
        })
        
        # 当前激活层状态
        self.active_layers = {
            'commercial': {'current_layer': 0, 'quarters_in_layer': 0},
            'residential': {'current_layer': 0, 'quarters_in_layer': 0}
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告：配置文件 {config_path} 不存在，使用默认配置")
            return {}
    
    def initialize_simulation(self):
        """初始化模拟"""
        print("🏙️ 初始化增强城市模拟系统 v2.3...")
        
        # 获取配置
        map_size = self.city_config.get('map_config', {}).get('map_size', [256, 256])
        transport_hubs = self.city_config.get('transport_config', {}).get('transport_hubs', [[40, 128], [216, 128]])
        
        # 初始化高斯核地价场系统
        self.land_price_system.initialize_system(transport_hubs, map_size)
        
        # 初始化几何等距等值线建筑系统
        land_price_field = self.land_price_system.get_land_price_field()
        self.isocontour_system.initialize_system(land_price_field, transport_hubs, map_size)
        
        # 初始化城市状态
        self.city_state = {
            'core_point': [128, 128],
            'trunk_road': transport_hubs,
            'public': [],
            'residential': [],
            'commercial': [],
            'residents': [],
            'land_price_field': land_price_field,
            'land_price_stats': self.land_price_system.get_land_price_stats(),
            'trajectory_system': None
        }
        
        # 初始化轨迹系统
        self.trajectory_system = TrajectorySystem(map_size, self.building_config)
        self.city_state['trajectory_system'] = self.trajectory_system
        
        # 创建初始居民
        initial_population = self.city_config.get('simulation_config', {}).get('initial_population', 100)
        self._create_initial_residents(initial_population)
        
        print(f"✅ v2.3初始化完成：{initial_population} 个初始居民")
        print(f"📊 高斯核地价场系统：连续分布，平滑等高线，自然演化")
        print(f"🏗️ 几何等距等值线：marching squares算法，等弧长采样")
        print(f"📏 分带系统：基于主干道法向距离，物理单位转换")
        print(f"🔄 滞后替代：住宅→商业，冷却期机制")
        print(f"🏛️ 公共设施：智能触发，符号显示")
    
    def _create_initial_residents(self, count: int):
        """创建初始居民"""
        for i in range(count):
            resident = {
                'id': f'agent_{i+1}',
                'pos': [random.randint(50, 206), random.randint(50, 206)],
                'home': None,
                'workplace': None,
                'income': random.randint(3000, 8000),
                'housing_cost': 0,
                'transport_cost': 0,
                'current_plan_index': 0,
                'plan': []
            }
            self.city_state['residents'].append(resident)
        
        # 为居民分配住宅
        self._assign_residents_to_homes()
        
        # 为居民分配日程
        self.schedule_logic.assign_daily_plans(
            self.city_state['residents'], 
            self.city_state.get('commercial', []), 
            self.city_state.get('public', [])
        )
    
    def _assign_residents_to_homes(self):
        """为居民分配住宅"""
        residents = self.city_state['residents']
        residential_buildings = self.city_state['residential']
        
        if not residential_buildings:
            # 如果没有住宅建筑，创建一些初始住宅
            self._create_initial_housing()
            residential_buildings = self.city_state['residential']
        
        # 为每个居民分配住宅
        for resident in residents:
            if resident['home'] is None:
                best_home = self.resident_agent.choose_residence(
                    residential_buildings, self.city_state, self.land_price_system
                )
                if best_home:
                    resident['home'] = best_home['id']
                    best_home['current_usage'] = best_home.get('current_usage', 0) + 1
                    self.resident_homes[resident['id']] = best_home['id']
    
    def _create_initial_housing(self):
        """创建初始住宅"""
        trunk_center_y = 128
        for i in range(5):  # 创建5个初始住宅
            x = random.randint(60, 196)
            y = trunk_center_y + random.randint(-40, 40)
            
            home = {
                'id': f'res_{i+1}',
                'type': 'residential',
                'xy': [x, y],
                'capacity': 200,
                'current_usage': 0,
                'construction_cost': 500,
                'revenue_per_person': 10,
                'revenue': 0
            }
            self.city_state['residential'].append(home)
    
    def run_simulation(self):
        """运行模拟"""
        simulation_months = self.city_config.get('simulation_config', {}).get('simulation_months', 24)
        render_every_month = self.city_config.get('simulation_config', {}).get('render_every_month', 1)
        
        print(f"🚀 开始运行 {simulation_months} 个月模拟 (v2.3)...")
        
        for month in range(simulation_months):
            self.current_month = month
            self.current_quarter = month // 3
            self.current_year = month // 12
            
            # 更新滞后系统季度
            self.hysteresis_system.update_quarter(self.current_quarter)
            
            # 每月更新
            self._monthly_update()
            
            # 季度更新
            if month % 3 == 0:
                self._quarterly_update()
            
            # 年度更新
            if month % 12 == 0:
                self._yearly_update()
            
            # 定期渲染
            if month % render_every_month == 0:
                self._render_frame(month)
            
            # 定期输出
            if month % 3 == 0:
                self._save_periodic_outputs(month)
            
            # 进度显示
            if month % 6 == 0:
                self._print_progress(month)
        
        # 最终输出
        self._save_final_outputs(simulation_months)
        print("✅ v2.3模拟完成！")
    
    def _monthly_update(self):
        """每月更新"""
        # 1. 更新轨迹系统
        self._update_trajectories()
        
        # 2. 应用热力图衰减
        self.trajectory_system.apply_decay()
        
        # 3. 居民增长
        self._spawn_new_residents()
        
        # 4. 更新建筑使用情况
        self._update_building_usage()
        
        # 5. 计算统计信息
        self._calculate_monthly_stats()
        
        # 6. 保存月度输出（包括地价场帧）
        self._save_monthly_outputs(self.current_month)
    
    def _quarterly_update(self):
        """季度更新"""
        print(f"📅 第 {self.current_quarter} 季度更新...")
        
        # 1. 等值线建筑生成
        self._generate_isocontour_buildings()
        
        # 2. 滞后替代评估
        self._evaluate_hysteresis_conversion()
        
        # 3. 公共设施评估
        self._evaluate_public_facilities()
        
        # 4. 计算季度统计
        self._calculate_quarterly_stats()
    
    def _yearly_update(self):
        """年度更新"""
        print(f"📅 第 {self.current_year} 年更新...")
        
        # 1. 高斯核地价场演化
        self.land_price_system.update_land_price_field(self.current_month, self.city_state)
        
        # 2. 更新城市状态中的地价场
        self.city_state['land_price_field'] = self.land_price_system.get_land_price_field()
        self.city_state['land_price_stats'] = self.land_price_system.get_land_price_stats()
        
        # 3. 更新几何等距等值线系统
        self.isocontour_system.initialize_system(
            self.city_state['land_price_field'], 
            self.city_state['trunk_road'], 
            [256, 256]
        )
        
        # 4. 计算年度统计
        self._calculate_yearly_stats()
    
    def _update_trajectories(self):
        """更新轨迹系统"""
        # 为居民分配工作地点
        self._assign_workplaces()
        
        # 更新轨迹热力图
        self.trajectory_system.update_trajectories(self.city_state['residents'], self.city_state)
    
    def _assign_workplaces(self):
        """为居民分配工作地点"""
        commercial_buildings = self.city_state.get('commercial', [])
        if not commercial_buildings:
            return
        
        # 统计每个商业建筑的使用情况
        building_usage = {building['id']: 0 for building in commercial_buildings}
        
        # 为已有工作地点的居民统计使用情况
        for resident in self.city_state['residents']:
            if resident.get('workplace'):
                building_usage[resident['workplace']] += 1
        
        # 为没有工作地点的居民分配工作
        for resident in self.city_state['residents']:
            if not resident.get('workplace'):
                best_workplace = self._select_best_workplace_balanced(resident, commercial_buildings, building_usage)
                if best_workplace:
                    resident['workplace'] = best_workplace['id']
                    building_usage[best_workplace['id']] += 1
    
    def _select_best_workplace_balanced(self, resident: Dict, commercial_buildings: List[Dict], building_usage: Dict) -> Dict:
        """选择最佳工作地点（平衡使用率版）"""
        if not commercial_buildings:
            return None
        
        # 获取居民住宅位置
        home_pos = None
        for building in self.city_state.get('residential', []):
            if building['id'] == resident.get('home'):
                home_pos = building['xy']
                break
        
        if not home_pos:
            return commercial_buildings[0]
        
        # 计算每个商业建筑的评分（距离 + 使用率）
        best_score = float('inf')
        best_workplace = None
        
        for building in commercial_buildings:
            # 距离评分（越近越好）
            distance = self._calculate_distance(home_pos, building['xy'])
            distance_score = distance / 100.0  # 归一化到0-1
            
            # 使用率评分（使用率越低越好）
            usage = building_usage.get(building['id'], 0)
            capacity = building.get('capacity', 800)
            usage_ratio = usage / capacity if capacity > 0 else 0
            usage_score = usage_ratio
            
            # 综合评分（距离权重0.6，使用率权重0.4）
            total_score = 0.6 * distance_score + 0.4 * usage_score
            
            if total_score < best_score:
                best_score = total_score
                best_workplace = building
        
        return best_workplace
    
    def _generate_isocontour_buildings(self):
        """生成几何等距等值线建筑（逐层填满逻辑）"""
        if not self.progressive_growth_config.get('enabled', True):
            # 如果未启用逐层填满，使用原有逻辑
            self._generate_isocontour_buildings_legacy()
            return
        
        print(f"🏗️ 第 {self.current_quarter} 季度：逐层填满建筑生成...")
        
        # 获取季度建筑增长目标
        residential_target = random.randint(*self.quarterly_growth.get('residential', [10, 20]))
        commercial_target = random.randint(*self.quarterly_growth.get('commercial', [5, 12]))
        
        # 检查并更新激活层
        self._update_active_layers()
        
        # 生成住宅建筑（逐层填满）
        new_residential = self._generate_residential_buildings_progressive(residential_target)
        
        # 生成商业建筑（逐层填满）
        new_commercial = self._generate_commercial_buildings_progressive(commercial_target)
        
        # 添加到城市状态
        self.city_state['residential'].extend(new_residential)
        self.city_state['commercial'].extend(new_commercial)
        
        if new_residential or new_commercial:
            print(f"✅ 生成完成：{len(new_residential)} 个住宅，{len(new_commercial)} 个商业建筑")
            
            # 输出建筑位置信息
            self._output_building_positions()
            
            # 输出分带统计
            zone_stats = self.isocontour_system.get_zone_statistics(self.city_state)
            print(f"📏 分带配置：前排区域{zone_stats['front_zone_buildings']}个建筑，住宅带{zone_stats['residential_zone_buildings']}个建筑")
            
            # 输出回退统计
            fallback_stats = self.isocontour_system.get_fallback_statistics()
            if fallback_stats['total_events'] > 0:
                print(f"🔄 分位数回退：{fallback_stats['total_events']} 次")
            
            # 输出当前激活层状态
            print(f"🌱 当前激活层：商业第{self.active_layers['commercial']['current_layer']}层，住宅第{self.active_layers['residential']['current_layer']}层")
    
    def _update_active_layers(self):
        """更新当前激活层状态"""
        for building_type in ['commercial', 'residential']:
            current_layer = self.active_layers[building_type]['current_layer']
            quarters_in_layer = self.active_layers[building_type]['quarters_in_layer']
            
            # 检查当前层密度
            layer_density = self._calculate_layer_density(building_type, current_layer)
            
            # 判断是否激活下一层
            should_activate_next = (
                layer_density >= self.progressive_growth_config['layer_activation_threshold'] or
                quarters_in_layer >= self.progressive_growth_config['max_quarters_per_layer']
            )
            
            if should_activate_next and current_layer < 3:  # 最多激活4层（0,1,2,3）
                self.active_layers[building_type]['current_layer'] += 1
                self.active_layers[building_type]['quarters_in_layer'] = 0
                print(f"🎯 {building_type}建筑：激活第{self.active_layers[building_type]['current_layer']}层")
            else:
                self.active_layers[building_type]['quarters_in_layer'] += 1
    
    def _calculate_layer_density(self, building_type: str, layer_index: int) -> float:
        """计算指定层的建筑密度"""
        if layer_index == 0:
            # 第0层：检查最内圈等值线
            return self._calculate_innermost_layer_density(building_type)
        else:
            # 其他层：检查当前层及之前所有层的综合密度
            return self._calculate_cumulative_layer_density(building_type, layer_index)
    
    def _calculate_innermost_layer_density(self, building_type: str) -> float:
        """计算最内圈层的建筑密度"""
        # 获取最内圈等值线的理论最大建筑数
        max_buildings = self._get_theoretical_max_buildings(building_type, 0)
        
        if max_buildings == 0:
            return 0.0
        
        # 统计当前已放置的建筑数量
        current_buildings = self._count_buildings_in_layer(building_type, 0)
        
        return current_buildings / max_buildings
    
    def _calculate_cumulative_layer_density(self, building_type: str, layer_index: int) -> float:
        """计算累积层密度（从第0层到当前层）"""
        total_max_buildings = 0
        total_current_buildings = 0
        
        for layer in range(layer_index + 1):
            max_buildings = self._get_theoretical_max_buildings(building_type, layer)
            current_buildings = self._count_buildings_in_layer(building_type, layer)
            
            total_max_buildings += max_buildings
            total_current_buildings += current_buildings
        
        if total_max_buildings == 0:
            return 0.0
        
        return total_current_buildings / total_max_buildings
    
    def _get_theoretical_max_buildings(self, building_type: str, layer_index: int) -> int:
        """获取指定层的理论最大建筑数"""
        # 这里需要根据等值线配置计算理论最大建筑数
        # 简化实现：基于弧长和建筑间距
        if building_type == 'commercial':
            # 商业建筑：弧长间距25-35m
            arc_spacing = 30  # 平均30m
        else:  # residential
            # 住宅建筑：弧长间距35-55m
            arc_spacing = 45  # 平均45m
        
        # 估算等值线弧长（简化：假设为圆形等值线）
        # 实际应该从等值线系统获取真实弧长
        estimated_arc_length = 200 + layer_index * 100  # 每层增加100m
        
        return max(1, int(estimated_arc_length / arc_spacing))
    
    def _count_buildings_in_layer(self, building_type: str, layer_index: int) -> int:
        """统计指定层的建筑数量"""
        # 这里需要根据建筑的实际地价值判断属于哪一层
        # 简化实现：基于建筑到核心的距离
        buildings = self.city_state.get(building_type, [])
        core_point = self.city_state['core_point']
        
        count = 0
        for building in buildings:
            if 'land_price_value' in building:
                # 根据地价值判断层数
                land_price_value = building['land_price_value']
                if self._is_building_in_layer(building_type, land_price_value, layer_index):
                    count += 1
        
        return count
    
    def _is_building_in_layer(self, building_type: str, land_price_value: float, layer_index: int) -> bool:
        """判断建筑是否属于指定层"""
        if building_type == 'commercial':
            # 商业建筑等值线序列：P₀=0.85, P₁=0.78, P₂=0.71, P₃=0.64
            layer_values = [0.85, 0.78, 0.71, 0.64]
        else:  # residential
            # 住宅建筑等值线序列：P₀=0.55, P₁=0.40, P₂=0.29, P₃=0.21
            layer_values = [0.55, 0.40, 0.29, 0.21]
        
        if layer_index >= len(layer_values):
            return False
        
        # 允许一定的误差范围（±5%）
        target_value = layer_values[layer_index]
        tolerance = target_value * 0.05
        
        return abs(land_price_value - target_value) <= tolerance
    
    def _generate_residential_buildings_progressive(self, target_count: int) -> List[Dict]:
        """渐进式生成住宅建筑"""
        current_layer = self.active_layers['residential']['current_layer']
        
        # 优先在当前激活层生成建筑
        new_buildings = []
        
        # 尝试在当前层生成建筑
        layer_buildings = self.isocontour_system.generate_residential_buildings(
            self.city_state, target_count, target_layer=current_layer
        )
        
        if layer_buildings:
            new_buildings.extend(layer_buildings)
            print(f"🏠 住宅建筑：第{current_layer}层生成{len(layer_buildings)}个")
        
        # 如果当前层生成不足，检查是否可以激活下一层
        if len(new_buildings) < target_count and current_layer < 3:
            remaining_count = target_count - len(new_buildings)
            next_layer_buildings = self.isocontour_system.generate_residential_buildings(
                self.city_state, remaining_count, target_layer=current_layer + 1
            )
            
            if next_layer_buildings:
                new_buildings.extend(next_layer_buildings)
                print(f"🏠 住宅建筑：第{current_layer + 1}层生成{len(next_layer_buildings)}个")
        
        return new_buildings
    
    def _generate_commercial_buildings_progressive(self, target_count: int) -> List[Dict]:
        """渐进式生成商业建筑"""
        current_layer = self.active_layers['commercial']['current_layer']
        
        # 优先在当前激活层生成建筑
        new_buildings = []
        
        # 尝试在当前层生成建筑
        layer_buildings = self.isocontour_system.generate_commercial_buildings(
            self.city_state, target_count, target_layer=current_layer
        )
        
        if layer_buildings:
            new_buildings.extend(layer_buildings)
            print(f"🏢 商业建筑：第{current_layer}层生成{len(layer_buildings)}个")
        
        # 如果当前层生成不足，检查是否可以激活下一层
        if len(new_buildings) < target_count and current_layer < 3:
            remaining_count = target_count - len(new_buildings)
            next_layer_buildings = self.isocontour_system.generate_commercial_buildings(
                self.city_state, remaining_count, target_layer=current_layer + 1
            )
            
            if next_layer_buildings:
                new_buildings.extend(next_layer_buildings)
                print(f"🏢 商业建筑：第{current_layer + 1}层生成{len(next_layer_buildings)}个")
        
        return new_buildings
    
    def _generate_isocontour_buildings_legacy(self):
        """原有的等值线建筑生成逻辑（作为备选）"""
        # 获取季度建筑增长目标
        residential_target = random.randint(*self.quarterly_growth.get('residential', [10, 20]))
        commercial_target = random.randint(*self.quarterly_growth.get('commercial', [5, 12]))
        
        # 生成住宅建筑
        new_residential = self.isocontour_system.generate_residential_buildings(
            self.city_state, residential_target
        )
        
        # 生成商业建筑
        new_commercial = self.isocontour_system.generate_commercial_buildings(
            self.city_state, commercial_target
        )
        
        # 添加到城市状态
        self.city_state['residential'].extend(new_residential)
        self.city_state['commercial'].extend(new_commercial)
        
        if new_residential or new_commercial:
            print(f"🏗️ 第 {self.current_quarter} 季度：生成 {len(new_residential)} 个住宅，{len(new_commercial)} 个商业建筑")
            
            # 输出建筑位置信息
            self._output_building_positions()
            
            # 输出分带统计
            zone_stats = self.isocontour_system.get_zone_statistics(self.city_state)
            print(f"📏 分带配置：前排区域{zone_stats['front_zone_buildings']}个建筑，住宅带{zone_stats['residential_zone_buildings']}个建筑")
            
            # 输出回退统计
            fallback_stats = self.isocontour_system.get_fallback_statistics()
            if fallback_stats['total_events'] > 0:
                print(f"🔄 分位数回退：{fallback_stats['total_events']} 次")
    
    def _output_building_positions(self):
        """输出建筑位置和颜色信息的JSON"""
        building_data = {
            'timestamp': f'month_{self.current_month:02d}',
            'buildings': []
        }
        
        # 添加住宅建筑
        for building in self.city_state['residential']:
            building_data['buildings'].append({
                'id': building['id'],
                'type': 'residential',
                'position': building['xy'],
                'color': '#F6C344',  # 黄色
                'land_price_value': building.get('land_price_value', 0.0)
            })
        
        # 添加商业建筑
        for building in self.city_state['commercial']:
            building_data['buildings'].append({
                'id': building['id'],
                'type': 'commercial',
                'position': building['xy'],
                'color': '#FD7E14',  # 橙色
                'land_price_value': building.get('land_price_value', 0.0)
            })
        
        # 添加公共建筑
        for building in self.city_state['public']:
            building_data['buildings'].append({
                'id': building['id'],
                'type': 'public',
                'position': building['xy'],
                'color': '#22A6B3',  # 青色
                'land_price_value': building.get('land_price_value', 0.0)
            })
        
        # 保存到文件
        output_file = f"{self.output_dir}/building_positions_month_{self.current_month:02d}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(building_data, f, indent=2, ensure_ascii=False)
    
    def _evaluate_hysteresis_conversion(self):
        """评估滞后替代"""
        conversion_result = self.hysteresis_system.evaluate_conversion_conditions(
            self.city_state, self.land_price_system
        )
        
        if conversion_result['should_convert']:
            # 执行替代
            candidates = conversion_result['candidates']
            if candidates:
                # 选择评分差异最大的候选建筑
                best_candidate = candidates[0]
                conversion_result = self.hysteresis_system.convert_building(
                    best_candidate['building_id'], self.city_state
                )
                
                if conversion_result['success']:
                    print(f"🔄 第 {self.current_quarter} 季度：住宅 {best_candidate['building_id']} 转换为商业建筑")
    
    def _evaluate_public_facilities(self):
        """评估公共设施需求"""
        facility_needs = self.public_facility_system.evaluate_facility_needs(self.city_state)
        
        # 生成需要的公共设施
        new_facilities = self.public_facility_system.generate_facilities(self.city_state, facility_needs)
        
        if new_facilities:
            self.city_state['public'].extend(new_facilities)
            facility_types = [f.get('facility_type', 'unknown') for f in new_facilities]
            print(f"🏛️ 第 {self.current_quarter} 季度：新增公共设施 {facility_types}")
    
    def _spawn_new_residents(self):
        """生成新居民"""
        # 每月增长
        growth_rate_range = self.city_config.get('simulation_config', {}).get('monthly_growth_rate', [0.08, 0.12])
        growth_rate = random.uniform(growth_rate_range[0], growth_rate_range[1])
        current_population = len(self.city_state['residents'])
        
        # 计算新居民数量
        new_residents_count = int(current_population * growth_rate)
        
        # 考虑住宅容量限制
        total_capacity = sum(building.get('capacity', 200) for building in self.city_state['residential'])
        max_population = int(total_capacity * self.city_config.get('simulation_config', {}).get('max_population_density', 0.8))
        
        if len(self.city_state['residents']) + new_residents_count > max_population:
            new_residents_count = max(0, max_population - len(self.city_state['residents']))
        
        # 创建新居民
        for i in range(new_residents_count):
            resident = {
                'id': f'agent_{len(self.city_state["residents"]) + 1}',
                'pos': [random.randint(50, 206), random.randint(50, 206)],
                'home': None,
                'workplace': None,
                'income': random.randint(3000, 8000),
                'housing_cost': 0,
                'transport_cost': 0,
                'current_plan_index': 0,
                'plan': []
            }
            self.city_state['residents'].append(resident)
        
        # 为新居民分配住宅
        if new_residents_count > 0:
            self._assign_residents_to_homes()
            print(f"👥 第 {self.current_month} 个月：新增 {new_residents_count} 个居民")
    
    def _update_building_usage(self):
        """更新建筑使用情况"""
        # 重置使用情况
        for building in self.city_state['residential']:
            building['current_usage'] = 0
        
        # 统计实际使用情况
        for resident in self.city_state['residents']:
            if resident['home']:
                for building in self.city_state['residential']:
                    if building['id'] == resident['home']:
                        building['current_usage'] += 1
                        break
    
    def _calculate_monthly_stats(self):
        """计算每月统计"""
        stats = {
            'month': self.current_month,
            'quarter': self.current_quarter,
            'year': self.current_year,
            'population': len(self.city_state['residents']),
            'public_buildings': len(self.city_state['public']),
            'residential_buildings': len(self.city_state['residential']),
            'commercial_buildings': len(self.city_state['commercial']),
            'total_buildings': len(self.city_state['public']) + len(self.city_state['residential']) + len(self.city_state['commercial']),
            'land_price_stats': self.city_state['land_price_stats'],
            'trajectory_stats': self.trajectory_system.get_trajectory_stats(),
            'land_price_evolution_stage': self.land_price_system._get_evolution_stage(self.current_month)
        }
        
        self.monthly_stats.append(stats)
    
    def _calculate_quarterly_stats(self):
        """计算季度统计"""
        stats = {
            'quarter': self.current_quarter,
            'year': self.current_year,
            'hysteresis_stats': self.hysteresis_system.get_conversion_statistics(),
            'facility_stats': self.public_facility_system.get_facility_statistics(self.city_state),
            'zone_stats': self.isocontour_system.get_zone_statistics(self.city_state)
        }
        
        self.quarterly_stats.append(stats)
    
    def _calculate_yearly_stats(self):
        """计算年度统计"""
        stats = {
            'year': self.current_year,
            'land_price_evolution': self.land_price_system.get_evolution_history()[-1] if self.land_price_system.get_evolution_history() else None,
            'land_price_stats': self.city_state['land_price_stats']
        }
        
        self.yearly_stats.append(stats)
    
    def _print_progress(self, month: int):
        """打印进度"""
        total_buildings = len(self.city_state['public']) + len(self.city_state['residential']) + len(self.city_state['commercial'])
        evolution_stage = self.land_price_system._get_evolution_stage(month)
        
        print(f"📊 第 {month} 个月：人口 {len(self.city_state['residents'])}，建筑 {total_buildings}")
        print(f"   地价场演化阶段：{evolution_stage['name']} (σ_hub={evolution_stage['hub_sigma']:.1f}, σ_road={evolution_stage['road_sigma']:.1f})")
        print(f"   滞后替代：{self.hysteresis_system.get_conversion_statistics()['total_conversions']} 次")
        print(f"   公共设施：{self.public_facility_system.get_facility_statistics(self.city_state)['total_facilities']} 个")
        
        # 输出几何等距等值线统计
        zone_stats = self.isocontour_system.get_zone_statistics(self.city_state)
        fallback_stats = self.isocontour_system.get_fallback_statistics()
        print(f"   📏 分带配置：前排区域{zone_stats['front_zone_buildings']}个建筑，住宅带{zone_stats['residential_zone_buildings']}个建筑")
        print(f"   🔄 回退：{fallback_stats['total_events']} 次")
        
        # 输出逐层填满建筑生成系统状态
        if self.progressive_growth_config.get('enabled', True):
            commercial_layer = self.active_layers['commercial']['current_layer']
            residential_layer = self.active_layers['residential']['current_layer']
            commercial_quarters = self.active_layers['commercial']['quarters_in_layer']
            residential_quarters = self.active_layers['residential']['quarters_in_layer']
            
            print(f"   🌱 逐层填满：商业第{commercial_layer}层({commercial_quarters}季度)，住宅第{residential_layer}层({residential_quarters}季度)")
            
            # 显示各层密度
            for building_type in ['commercial', 'residential']:
                current_layer = self.active_layers[building_type]['current_layer']
                layer_density = self._calculate_layer_density(building_type, current_layer)
                print(f"      {building_type}第{current_layer}层密度：{layer_density:.1%}")
    
    def _render_frame(self, month: int):
        """渲染帧"""
        # 准备渲染数据
        hubs = [{'id': 'A', 'xy': [40, 128]}, {'id': 'B', 'xy': [216, 128]}]
        trunk = self.city_state['trunk_road']
        public_pois = self.city_state['public']
        residential_pois = self.city_state['residential']
        retail_pois = self.city_state['commercial']
        agents = self.city_state['residents']
        
        # 获取轨迹热力图数据
        heatmap_data = self.trajectory_system.get_heatmap_data()
        combined_heatmap = heatmap_data['combined_heatmap']
        
        # 获取几何等距等值线数据用于可视化
        contour_data = self.isocontour_system.get_contour_data_for_visualization()
        
        # 渲染并保存
        self.visualizer.render_layers(
            hubs=hubs,
            trunk=trunk,
            public_pois=public_pois,
            residential_pois=residential_pois,
            retail_pois=retail_pois,
            heat_map=combined_heatmap,
            agents=agents,
            show_agents=False,
            contour_data=contour_data  # 传递等值线数据
        )
        self.visualizer.save_frame(f'enhanced_simulation_v2_3_output/images/month_{month:02d}.png')
    
    def _save_periodic_outputs(self, month: int):
        """保存定期输出"""
        # 保存城市状态
        self.output_system.save_city_state_output(self.city_state, month)
        
        # 保存坐标信息
        self.output_system.save_coordinates_output(self.city_state, month)
        
        # 保存轨迹数据
        self._save_trajectory_data(month)
        
        # 保存地价场数据
        self.land_price_system.save_land_price_frame(month, 'enhanced_simulation_v2_3_output')
        
        # 保存滞后替代数据
        self.hysteresis_system.save_conversion_data('enhanced_simulation_v2_3_output')
        
        # 保存公共设施数据
        self.public_facility_system.save_facility_data('enhanced_simulation_v2_3_output')
    
    def _save_trajectory_data(self, month: int):
        """保存轨迹数据"""
        heatmap_data = self.trajectory_system.get_heatmap_data()
        
        # 转换numpy数组为列表
        trajectory_data = {
            'month': month,
            'heatmap_data': {
                'commute_heatmap': heatmap_data['commute_heatmap'].tolist(),
                'commercial_heatmap': heatmap_data['commercial_heatmap'].tolist(),
                'combined_heatmap': heatmap_data['combined_heatmap'].tolist(),
                'trajectory_types': heatmap_data['trajectory_types']
            },
            'trajectory_stats': self.trajectory_system.get_trajectory_stats()
        }
        
        filepath = self.output_system.output_dir / 'trajectory_data.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
    
    def _save_monthly_outputs(self, month: int):
        """保存月度输出"""
        # 保存地价场帧
        self.land_price_system.save_land_price_frame(month, 'enhanced_simulation_v2_3_output')
        
        # 保存建筑位置
        self._output_building_positions()
        
        print(f"💾 第 {month} 个月输出已保存")
    
    def _save_final_outputs(self, simulation_months: int):
        """保存最终输出"""
        # 保存所有统计数据
        self.output_system.save_daily_stats(self.monthly_stats)
        
        # 保存季度和年度统计
        self._save_quarterly_stats()
        self._save_yearly_stats()
        
        # 保存建筑分布
        building_distribution = {
            'public': len(self.city_state['public']),
            'residential': len(self.city_state['residential']),
            'commercial': len(self.city_state['commercial'])
        }
        self.output_system.save_building_distribution(building_distribution)
        
        # 保存最终总结
        self.output_system.save_final_summary(self.city_state, simulation_months)
        
        print("📊 所有v2.3输出文件已保存到 enhanced_simulation_v2_3_output/ 目录")
    
    def _save_quarterly_stats(self):
        """保存季度统计"""
        filepath = self.output_system.output_dir / 'quarterly_stats.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.quarterly_stats, f, indent=2, ensure_ascii=False)
    
    def _save_yearly_stats(self):
        """保存年度统计"""
        filepath = self.output_system.output_dir / 'yearly_stats.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.yearly_stats, f, indent=2, ensure_ascii=False)
    
    def _calculate_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def main():
    """主函数"""
    print("🏙️ 增强城市模拟系统 v2.3")
    print("=" * 60)
    print("🎯 新特性：")
    print("  • 高斯核地价场系统（连续分布，平滑等高线）")
    print("  • 几何等距等值线建筑生成（marching squares）")
    print("  • 基于主干道法向距离的分带系统")
    print("  • 滞后替代机制（住宅→商业）")
    print("  • 智能公共设施（触发机制）")
    print("  • 时间分层系统（年/季/月）")
    print("  • 单位系统一致性（meters_per_pixel）")
    print("  • 逐层填满建筑生成（渐进式城市生长）")
    print("=" * 60)
    
    # 创建并运行模拟
    simulation = EnhancedCitySimulationV2_3()
    simulation.initialize_simulation()
    simulation.run_simulation()
    
    print("\n🎉 v2.3模拟完成！")
    print("📁 输出文件保存在 enhanced_simulation_v2_3_output/ 目录")
    print("📊 查看 final_summary.json 了解模拟结果")

if __name__ == "__main__":
    main()
