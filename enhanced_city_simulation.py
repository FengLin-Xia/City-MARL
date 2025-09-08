#!/usr/bin/env python3
"""
增强的城市模拟系统
集成地价驱动的多智能体城市模拟
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import random
import math

# 导入自定义模块
from logic.land_price_system import LandPriceSystem
from logic.enhanced_agents import GovernmentAgent, BusinessAgent, ResidentAgent
from logic.output_system import OutputSystem
from logic.placement import PlacementLogic
from logic.schedule import ScheduleLogic
from logic.move import MoveLogic
from logic.trajectory_system import TrajectorySystem
from viz.ide import CityVisualizer

class EnhancedCitySimulation:
    """增强的城市模拟系统"""
    
    def __init__(self):
        """初始化模拟系统"""
        # 加载配置
        self.city_config = self._load_config('configs/city_config.json')
        self.building_config = self._load_config('configs/building_config.json')
        self.agent_config = self._load_config('configs/agent_config.json')
        
        # 初始化系统组件
        self.land_price_system = LandPriceSystem(self.building_config)
        self.government_agent = GovernmentAgent(self.agent_config['government_agent'])
        # 合并业务代理配置和建筑增长配置
        business_config = self.agent_config['business_agent'].copy()
        if 'building_growth' in self.building_config:
            business_config.update(self.building_config['building_growth'])
        self.business_agent = BusinessAgent(business_config)
        self.resident_agent = ResidentAgent(self.agent_config['resident_agent'])
        
        # 初始化输出系统
        self.output_system = OutputSystem('enhanced_simulation_output')
        
        # 初始化现有逻辑模块
        self.placement_logic = PlacementLogic()
        self.schedule_logic = ScheduleLogic()
        self.move_logic = MoveLogic()
        
        # 初始化可视化器
        self.visualizer = CityVisualizer()
        
        # 模拟状态
        self.current_month = 0
        self.city_state = {}
        self.monthly_stats = []
        self.land_price_history = []
        self.population_history = []
        
        # 居民-住宅关系映射
        self.resident_homes = {}
        
        # Logistic增长参数
        self.building_growth_config = self.building_config.get('building_growth', {})
        self.growth_params = self.building_growth_config.get('params', {'K': 80, 'r': 0.4, 't0': 12})
        self.min_new_per_month = self.building_growth_config.get('min_new_per_month', 2)
        self.max_new_per_month = self.building_growth_config.get('max_new_per_month', 7)
        
        # 分批闪现参数
        self.batch_rendering = self.building_growth_config.get('batch_rendering', {})
        self.batch_enabled = self.batch_rendering.get('enabled', True)
        self.batches_per_month = self.batch_rendering.get('batches_per_month', 3)
        self.batch_interval = self.batch_rendering.get('interval_seconds', 1.5)
        
        # 建筑增长历史
        self.building_growth_history = []
        self.total_buildings_target = 0
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告：配置文件 {config_path} 不存在，使用默认配置")
            return {}
    
    def _calculate_logistic_growth(self, month: int) -> int:
        """计算Logistic增长函数，返回第month月应该有的累计建筑数量"""
        K = self.growth_params.get('K', 80)  # 最大建筑容量
        r = self.growth_params.get('r', 0.4)  # 增长速率
        t0 = self.growth_params.get('t0', 12)  # 拐点位置
        
        # Logistic函数: N(t) = K / (1 + e^(-r(t-t0)))
        N_t = K / (1 + math.exp(-r * (month - t0)))
        return int(N_t)
    
    def _calculate_monthly_new_buildings(self, month: int) -> int:
        """计算第month月应该新增的建筑数量"""
        # 计算当前月累计建筑数量
        current_total = self._calculate_logistic_growth(month)
        
        if month == 0:
            # 第0个月返回初始建筑数量
            return max(self.min_new_per_month, min(self.max_new_per_month, current_total))
        
        # 计算上个月累计建筑数量
        previous_total = self._calculate_logistic_growth(month - 1)
        
        # 新增数量
        new_buildings = current_total - previous_total
        
        # 应用最小和最大限制
        new_buildings = max(self.min_new_per_month, min(self.max_new_per_month, new_buildings))
        
        return new_buildings
    
    def _get_building_type_distribution(self, total_new: int) -> Dict[str, int]:
        """根据建筑类型分布比例分配新增建筑数量"""
        residential_ratio = self.building_growth_config.get('residential_ratio', 0.5)
        commercial_ratio = self.building_growth_config.get('commercial_ratio', 0.3)
        public_ratio = self.building_growth_config.get('public_ratio', 0.2)
        
        # 确保比例总和为1
        total_ratio = residential_ratio + commercial_ratio + public_ratio
        if total_ratio != 1.0:
            residential_ratio /= total_ratio
            commercial_ratio /= total_ratio
            public_ratio /= total_ratio
        
        # 当建筑数量很少时，确保每种类型至少有机会得到建筑
        if total_new <= 3:
            # 对于少量建筑，优先考虑商业和住宅
            if total_new == 1:
                # 1个建筑：优先给商业
                distribution = {'residential': 0, 'commercial': 1, 'public': 0}
            elif total_new == 2:
                # 2个建筑：1个住宅，1个商业
                distribution = {'residential': 1, 'commercial': 1, 'public': 0}
            elif total_new == 3:
                # 3个建筑：1个住宅，1个商业，1个公共
                distribution = {'residential': 1, 'commercial': 1, 'public': 1}
        else:
            # 正常比例分配
            distribution = {
                'residential': int(total_new * residential_ratio),
                'commercial': int(total_new * commercial_ratio),
                'public': int(total_new * public_ratio)
            }
            
            # 处理舍入误差
            remaining = total_new - sum(distribution.values())
            if remaining > 0:
                # 优先分配给商业建筑
                distribution['commercial'] += remaining
        
        return distribution
    
    def initialize_simulation(self):
        """初始化模拟"""
        print("🏙️ 初始化增强城市模拟系统...")
        
        # 初始化地价系统
        map_size = self.city_config.get('map_size', [256, 256])
        transport_hubs = self.city_config.get('trunk_road', [[40, 128], [216, 128]])
        self.land_price_system.initialize_land_prices(map_size, transport_hubs)
        
        # 初始化城市状态
        self.city_state = {
            'core_point': [128, 128],  # 保留地图中心点作为参考
            'trunk_road': transport_hubs,
            'public': [],
            'residential': [],
            'commercial': [],
            'residents': [],
            'land_price_stats': self.land_price_system.get_land_price_stats()
        }
        
        # 初始化轨迹系统
        self.trajectory_system = TrajectorySystem(map_size, self.building_config)
        
        # 创建初始居民
        initial_population = self.city_config.get('initial_population', 100)
        self._create_initial_residents(initial_population)
        
        print(f"✅ 初始化完成：{initial_population} 个初始居民")
        print(f"📈 Logistic增长参数：K={self.growth_params.get('K', 80)}, r={self.growth_params.get('r', 0.4)}, t0={self.growth_params.get('t0', 12)}")
        print(f"🏗️ 分批渲染：{'启用' if self.batch_enabled else '禁用'}, {self.batches_per_month} 批次/月, {self.batch_interval}秒间隔")
    
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
        simulation_months = self.city_config.get('simulation_months', 24)
        render_every_month = self.city_config.get('render_every_month', 1)
        
        print(f"🚀 开始运行 {simulation_months} 个月模拟...")
        
        for month in range(simulation_months):
            self.current_month = month
            
            # 每月更新
            self._monthly_update()
            
            # 定期渲染
            if month % render_every_month == 0:
                self._render_frame(month)
            
            # 定期输出
            if month % 3 == 0:
                self._save_periodic_outputs(month)
            
            # 进度显示
            if month % 6 == 0:
                total_buildings = len(self.city_state['public']) + len(self.city_state['residential']) + len(self.city_state['commercial'])
                target_total = self._calculate_logistic_growth(month)
                print(f"📅 第 {month} 个月：人口 {len(self.city_state['residents'])}，建筑 {total_buildings}/{target_total} (目标)")
        
        # 最终输出
        self._save_final_outputs(simulation_months)
        print("✅ 模拟完成！")
    
    def _monthly_update(self):
        """每月更新"""
        # 1. 更新轨迹系统
        self._update_trajectories()
        
        # 2. 应用热力图衰减
        self.trajectory_system.apply_decay()
        
        # 3. 更新地价
        self.land_price_system.update_land_prices(self.city_state)
        self.city_state['land_price_stats'] = self.land_price_system.get_land_price_stats()
        
        # 4. 智能体决策
        self._agent_decisions()
        
        # 5. 居民增长
        self._spawn_new_residents()
        
        # 6. 更新建筑使用情况
        self._update_building_usage()
        
        # 7. 计算统计信息
        self._calculate_monthly_stats()
    
    def _update_trajectories(self):
        """更新轨迹系统"""
        # 为居民分配工作地点
        self._assign_workplaces()
        
        # 更新轨迹热力图
        self.trajectory_system.update_trajectories(self.city_state['residents'], self.city_state)
    
    def _assign_workplaces(self):
        """为居民分配工作地点（优化版）"""
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
                # 选择使用率最低且距离合适的商业建筑
                best_workplace = self._select_best_workplace_balanced(resident, commercial_buildings, building_usage)
                if best_workplace:
                    resident['workplace'] = best_workplace['id']
                    building_usage[best_workplace['id']] += 1
    
    def _select_best_workplace(self, resident: Dict, commercial_buildings: List[Dict]) -> Dict:
        """选择最佳工作地点（简单版）"""
        if not commercial_buildings:
            return None
        
        # 简单的选择策略：选择最近的商业建筑
        home_pos = None
        for building in self.city_state.get('residential', []):
            if building['id'] == resident.get('home'):
                home_pos = building['xy']
                break
        
        if not home_pos:
            return commercial_buildings[0]  # 默认选择第一个
        
        # 选择最近的商业建筑
        min_distance = float('inf')
        best_workplace = None
        
        for building in commercial_buildings:
            distance = self._calculate_distance(home_pos, building['xy'])
            if distance < min_distance:
                min_distance = distance
                best_workplace = building
        
        return best_workplace
    
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
    
    def _calculate_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_monthly_stats(self):
        """计算每月统计"""
        # 计算本月建筑增长目标
        monthly_new_buildings = self._calculate_monthly_new_buildings(self.current_month)
        target_total = self._calculate_logistic_growth(self.current_month)
        
        stats = {
            'month': self.current_month,
            'population': len(self.city_state['residents']),
            'public_buildings': len(self.city_state['public']),
            'residential_buildings': len(self.city_state['residential']),
            'commercial_buildings': len(self.city_state['commercial']),
            'total_buildings': len(self.city_state['public']) + len(self.city_state['residential']) + len(self.city_state['commercial']),
            'land_price_stats': self.city_state['land_price_stats'],
            'trajectory_stats': self.trajectory_system.get_trajectory_stats(),
            'building_growth': {
                'target_new': monthly_new_buildings,
                'target_total': target_total,
                'actual_total': len(self.city_state['public']) + len(self.city_state['residential']) + len(self.city_state['commercial'])
            }
        }
        
        self.monthly_stats.append(stats)
        self.population_history.append(stats['population'])
        self.land_price_history.append(stats['land_price_stats'])
    
    def _agent_decisions(self):
        """智能体决策（Logistic增长 + 分批闪现）"""
        # 计算本月应该新增的建筑数量
        monthly_new_buildings = self._calculate_monthly_new_buildings(self.current_month)
        
        if monthly_new_buildings > 0:
            # 根据建筑类型分布分配新增建筑
            building_distribution = self._get_building_type_distribution(monthly_new_buildings)
            
            # 分批建设建筑
            self._build_buildings_in_batches(building_distribution)
            
            # 记录建筑增长历史
            self.building_growth_history.append({
                'month': self.current_month,
                'target_new': monthly_new_buildings,
                'actual_new': sum(building_distribution.values()),
                'distribution': building_distribution
            })
    
    def _build_buildings_in_batches(self, building_distribution: Dict[str, int]):
        """分批建设建筑"""
        total_new = sum(building_distribution.values())
        if total_new == 0:
            return
        
        # 计算每批次的建筑数量
        buildings_per_batch = max(1, total_new // self.batches_per_month)
        
        # 准备所有要建设的建筑
        all_new_buildings = []
        
        # 政府建筑（公共设施）
        for i in range(building_distribution['public']):
            new_public = self.government_agent.make_decisions(self.city_state, self.land_price_system)
            if new_public:
                all_new_buildings.extend(new_public)
        
        # 企业建筑（住宅和商业）
        land_price_matrix = self.land_price_system.get_land_price_matrix()
        heatmap_data = self.trajectory_system.get_heatmap_data()
        
        # 住宅建筑
        residential_built = 0
        for i in range(building_distribution['residential']):
            new_residential = self.business_agent._decide_residential_development_enhanced(
                self.city_state, self.land_price_system, land_price_matrix, heatmap_data
            )
            if new_residential:
                all_new_buildings.extend(new_residential)
                residential_built += len(new_residential)
        
        # 商业建筑 - 确保能够建设
        commercial_built = 0
        for i in range(building_distribution['commercial']):
            new_commercial = self.business_agent._decide_commercial_development_enhanced(
                self.city_state, self.land_price_system, land_price_matrix, heatmap_data
            )
            if new_commercial:
                all_new_buildings.extend(new_commercial)
                commercial_built += len(new_commercial)
            else:
                # 如果无法建设商业建筑，尝试在更宽松的条件下建设
                print(f"⚠️ 第 {self.current_month} 个月：商业建筑建设失败，尝试备用方案")
                # 可以在这里添加备用建设逻辑
        
        # 分批添加到城市状态
        if self.batch_enabled and len(all_new_buildings) > 1:
            # 分批添加
            for i in range(0, len(all_new_buildings), buildings_per_batch):
                batch = all_new_buildings[i:i + buildings_per_batch]
                self._add_buildings_to_city_state(batch)
                
                # 渲染当前批次
                self._render_batch_frame(i // buildings_per_batch + 1)
                
                # 等待间隔（在实际运行中，这里只是记录，不实际等待）
                if i + buildings_per_batch < len(all_new_buildings):
                    print(f"⏳ 批次 {i // buildings_per_batch + 1} 完成，等待 {self.batch_interval} 秒...")
        else:
            # 一次性添加所有建筑
            self._add_buildings_to_city_state(all_new_buildings)
        
        # 输出建设信息
        if all_new_buildings:
            public_count = len([b for b in all_new_buildings if b['type'] == 'public'])
            residential_count = len([b for b in all_new_buildings if b['type'] == 'residential'])
            commercial_count = len([b for b in all_new_buildings if b['type'] == 'commercial'])
            
            print(f"🏗️ 第 {self.current_month} 个月：建设了 {public_count} 个公共设施，{residential_count} 个住宅，{commercial_count} 个商业建筑")
            
            # 如果有新的商业建筑，立即为居民分配工作
            if commercial_count > 0:
                self._assign_workplaces()
                working_residents = sum(1 for r in self.city_state['residents'] if r.get('workplace'))
                print(f"💼 工作分配完成：{working_residents} 个居民有工作地点")
    
    def _add_buildings_to_city_state(self, new_buildings: List[Dict]):
        """将新建筑添加到城市状态"""
        for building in new_buildings:
            if building['type'] == 'public':
                self.city_state['public'].append(building)
            elif building['type'] == 'residential':
                self.city_state['residential'].append(building)
            elif building['type'] == 'commercial':
                self.city_state['commercial'].append(building)
    
    def _render_batch_frame(self, batch_number: int):
        """渲染批次帧"""
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
        
        # 渲染并保存批次帧
        self.visualizer.render_layers(
            hubs=hubs,
            trunk=trunk,
            public_pois=public_pois,
            residential_pois=residential_pois,
            retail_pois=retail_pois,
            heat_map=combined_heatmap,
            agents=agents,
            show_agents=False
        )
        
        # 保存批次帧（使用特殊命名）
        batch_filename = f'enhanced_simulation_output/images/month_{self.current_month:02d}_batch_{batch_number:02d}.png'
        self.visualizer.save_frame(batch_filename)
    
    def _spawn_new_residents(self):
        """生成新居民"""
        # 每月增长
        growth_rate_range = self.city_config.get('monthly_growth_rate', [0.08, 0.12])
        growth_rate = random.uniform(growth_rate_range[0], growth_rate_range[1])
        current_population = len(self.city_state['residents'])
        
        # 计算新居民数量
        new_residents_count = int(current_population * growth_rate)
        
        # 考虑住宅容量限制
        total_capacity = sum(building.get('capacity', 200) for building in self.city_state['residential'])
        max_population = int(total_capacity * self.city_config.get('max_population_density', 0.8))
        
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
        
        # 渲染并保存
        self.visualizer.render_layers(
            hubs=hubs,
            trunk=trunk,
            public_pois=public_pois,
            residential_pois=residential_pois,
            retail_pois=retail_pois,
            heat_map=combined_heatmap,
            agents=agents,
            show_agents=False
        )
        self.visualizer.save_frame(f'enhanced_simulation_output/images/month_{month:02d}.png')
    
    def _save_periodic_outputs(self, month: int):
        """保存定期输出"""
        # 保存城市状态
        self.output_system.save_city_state_output(self.city_state, month)
        
        # 保存坐标信息
        self.output_system.save_coordinates_output(self.city_state, month)
        
        # 保存轨迹数据
        self._save_trajectory_data(month)
        
        # 保存可视化配置
        self.output_system.save_visualization_config(self.building_config)
    
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
    
    def _save_final_outputs(self, simulation_months: int):
        """保存最终输出"""
        # 保存所有统计数据
        self.output_system.save_daily_stats(self.monthly_stats)  # 复用现有方法
        self.output_system.save_land_price_evolution(self.land_price_history)
        
        # 保存建筑分布
        building_distribution = {
            'public': len(self.city_state['public']),
            'residential': len(self.city_state['residential']),
            'commercial': len(self.city_state['commercial'])
        }
        self.output_system.save_building_distribution(building_distribution)
        
        # 保存建筑增长历史
        self._save_building_growth_history(simulation_months)
        
        # 保存最终总结
        self.output_system.save_final_summary(self.city_state, simulation_months)
        
        print("📊 所有输出文件已保存到 enhanced_simulation_output/ 目录")
    
    def _save_building_growth_history(self, simulation_months: int):
        """保存建筑增长历史"""
        # 计算Logistic增长曲线数据
        logistic_curve = []
        for month in range(simulation_months + 1):
            target_total = self._calculate_logistic_growth(month)
            monthly_new = self._calculate_monthly_new_buildings(month)
            logistic_curve.append({
                'month': month,
                'target_total': target_total,
                'monthly_new': monthly_new
            })
        
        # 保存建筑增长历史
        growth_history = {
            'logistic_curve': logistic_curve,
            'actual_growth': self.building_growth_history,
            'growth_params': self.growth_params,
            'batch_config': self.batch_rendering
        }
        
        filepath = self.output_system.output_dir / 'building_growth_history.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(growth_history, f, indent=2, ensure_ascii=False)

def main():
    """主函数"""
    print("🏙️ 增强城市模拟系统 v2.0")
    print("=" * 50)
    
    # 创建并运行模拟
    simulation = EnhancedCitySimulation()
    simulation.initialize_simulation()
    simulation.run_simulation()
    
    print("\n🎉 模拟完成！")
    print("📁 输出文件保存在 enhanced_simulation_output/ 目录")
    print("📊 查看 final_summary.json 了解模拟结果")

if __name__ == "__main__":
    main()
