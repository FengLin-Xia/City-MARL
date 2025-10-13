#!/usr/bin/env python3
"""
改进的城市模拟器
解决居民生成和住宅关系问题
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 导入现有模块
from logic.placement import PlacementLogic
from logic.schedule import ScheduleLogic
from logic.move import MoveLogic
from viz.ide import CityVisualizer

class ImprovedCitySimulation:
    def __init__(self):
        """初始化改进的城市模拟器"""
        # 时间参数
        self.days = 365
        self.current_day = 0
        
        # 居民生成参数（更现实）
        self.monthly_growth_rate = 0.05  # 每月5%增长率
        self.initial_population = 100    # 初始100人
        self.max_population_density = 0.8  # 最大人口密度80%
        
        # 移动参数
        self.movement_speed = 4
        self.movement_mode = "linear"
        
        # 热力图参数
        self.heat_evaporation = 0.995
        self.heat_map = np.zeros((256, 256))
        
        # 统计跟踪
        self.daily_stats = []
        self.poi_evolution = []
        self.population_history = []
        
        # 初始化逻辑模块
        self.placement_logic = PlacementLogic()
        self.schedule_logic = ScheduleLogic()
        self.move_logic = MoveLogic()
        self.visualizer = CityVisualizer()
        
        # 居民管理
        self.residents = []
        self.resident_homes = {}  # 居民ID -> 住宅POI ID
        
        # 输出设置
        self.output_dir = Path('improved_simulation_output')
        self.output_dir.mkdir(exist_ok=True)
        
        # 渲染频率
        self.render_every_n_days = 5  # 每5天渲染一次
        
    def load_data(self):
        """加载初始数据"""
        try:
            with open('data/poi_example.json', 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # 初始化住宅POI的人口计数
            for res in self.data['residential']:
                res['current_population'] = 0
                res['residents'] = []  # 存储居住的居民ID
            
            print("✅ 数据加载成功")
            return True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def calculate_monthly_growth(self):
        """计算月度人口增长"""
        current_month = self.current_day // 30
        if current_month == 0:
            return self.initial_population
        
        # 基于当前人口和增长率计算
        current_population = len(self.residents)
        monthly_growth = int(current_population * self.monthly_growth_rate)
        
        # 限制增长，避免过度拥挤
        total_capacity = sum(res.get('capacity', 200) for res in self.data['residential'])
        max_allowed = int(total_capacity * self.max_population_density)
        
        if current_population + monthly_growth > max_allowed:
            monthly_growth = max(0, max_allowed - current_population)
        
        return monthly_growth
    
    def spawn_residents(self):
        """生成新居民（改进版）"""
        # 每30天（一个月）生成一次新居民
        if self.current_day % 30 == 0:
            monthly_growth = self.calculate_monthly_growth()
            
            if monthly_growth > 0:
                new_residents = self._create_new_residents(monthly_growth)
                
                # 为新居民分配计划
                self.schedule_logic.assign_daily_plans(
                    new_residents, 
                    self.data['retail'], 
                    self.data['public']
                )
                
                self.residents.extend(new_residents)
                
                print(f"📈 第{self.current_day}天（第{self.current_day//30}月）: "
                      f"新增{len(new_residents)}居民，总居民数: {len(self.residents)}")
    
    def _create_new_residents(self, count: int) -> List[Dict]:
        """创建新居民并分配到住宅"""
        new_residents = []
        
        for i in range(count):
            # 找到最适合的住宅POI
            best_home = self._find_best_home()
            
            if best_home:
                resident = {
                    "id": f"agent_{len(self.residents) + i + 1}",
                    "pos": best_home['xy'].copy(),
                    "home": best_home['id'],
                    "plan": [],
                    "current_plan_index": 0,
                    "target": best_home['xy'].copy(),
                    "current_activity": "home",
                    "move_in_day": self.current_day
                }
                
                new_residents.append(resident)
                
                # 更新住宅POI
                best_home['current_population'] += 1
                best_home['residents'].append(resident['id'])
                self.resident_homes[resident['id']] = best_home['id']
        
        return new_residents
    
    def _find_best_home(self) -> Dict:
        """找到最适合的住宅POI"""
        available_homes = []
        
        for res in self.data['residential']:
            capacity = res.get('capacity', 200)
            current_pop = res.get('current_population', 0)
            
            if current_pop < capacity:
                # 计算可用空间
                available_space = capacity - current_pop
                # 计算到主干道的距离（偏好靠近主干道）
                trunk_distance = abs(res['xy'][1] - 128)  # 主干道在y=128
                
                available_homes.append({
                    'poi': res,
                    'available_space': available_space,
                    'trunk_distance': trunk_distance,
                    'score': available_space - trunk_distance * 0.1  # 评分：空间优先，距离次之
                })
        
        if not available_homes:
            return None
        
        # 选择评分最高的住宅
        best_home = max(available_homes, key=lambda x: x['score'])
        return best_home['poi']
    
    def move_residents(self):
        """移动居民"""
        for resident in self.residents:
            if resident['target'] is None:
                # 分配新目标
                resident['target'] = self.schedule_logic.get_next_target(
                    resident, 
                    self.data
                )
            
            if resident['target']:
                # 移动居民
                new_pos = self.move_logic.move_towards(
                    resident['pos'], 
                    resident['target'], 
                    self.movement_speed, 
                    self.movement_mode
                )
                
                # 更新位置
                resident['pos'] = new_pos
                
                # 更新热力图
                x, y = int(new_pos[0]), int(new_pos[1])
                if 0 <= x < 256 and 0 <= y < 256:
                    self.heat_map[y, x] += 1.0
                
                # 检查是否到达目标
                if self.move_logic.reached(new_pos, resident['target']):
                    resident['target'] = None
                    # 推进计划
                    self.schedule_logic.advance_plan(resident)
    
    def update_heat_map(self):
        """更新热力图"""
        self.heat_map *= self.heat_evaporation
    
    def calculate_stats(self):
        """计算每日统计"""
        # 基础统计
        stats = {
            'day': self.current_day,
            'total_residents': len(self.residents),
            'public_pois': len(self.data['public']),
            'residential_pois': len(self.data['residential']),
            'retail_pois': len(self.data['retail']),
            'heat_sum': np.sum(self.heat_map),
            'heat_max': np.max(self.heat_map),
            'heat_mean': np.mean(self.heat_map),
            # 添加POI数据供placement_logic使用
            'public': self.data['public'],
            'residential': self.data['residential'],
            'retail': self.data['retail'],
            'hubs': self.data['hubs']
        }
        
        # 计算住宅使用率统计
        self._calculate_residential_stats(stats)
        
        # 计算商业使用率统计
        self._calculate_retail_stats(stats)
        
        # 计算覆盖率统计
        self._calculate_coverage_stats(stats)
        
        self.daily_stats.append(stats)
        
        # 记录POI演化
        poi_state = {
            'day': stats['day'],
            'public': len(self.data['public']),
            'residential': len(self.data['residential']),
            'retail': len(self.data['retail'])
        }
        self.poi_evolution.append(poi_state)
        
        # 记录人口历史
        self.population_history.append({
            'day': self.current_day,
            'population': len(self.residents),
            'month': self.current_day // 30
        })
        
        return stats
    
    def _calculate_residential_stats(self, stats):
        """计算住宅使用率统计"""
        total_capacity = sum(res.get('capacity', 200) for res in self.data['residential'])
        total_population = sum(res.get('current_population', 0) for res in self.data['residential'])
        
        overall_usage = total_population / total_capacity if total_capacity > 0 else 0
        
        for res in self.data['residential']:
            capacity = res.get('capacity', 200)
            current_pop = res.get('current_population', 0)
            usage_ratio = current_pop / capacity if capacity > 0 else 0
            
            res['usage_ratio'] = usage_ratio
            
            # 更新连续高使用天数
            if usage_ratio > 0.6:
                res['consecutive_high_usage'] = res.get('consecutive_high_usage', 0) + 1
            else:
                res['consecutive_high_usage'] = 0
        
        stats['residential_usage'] = overall_usage
        stats['residential_capacity'] = total_capacity
        stats['residential_population'] = total_population
    
    def _calculate_retail_stats(self, stats):
        """计算商业使用率统计"""
        total_capacity = sum(ret.get('capacity', 800) for ret in self.data['retail'])
        # 假设所有居民都会访问商业设施
        total_visitors = len(self.residents)
        
        overall_usage = total_visitors / total_capacity if total_capacity > 0 else 0
        
        for ret in self.data['retail']:
            capacity = ret.get('capacity', 800)
            # 简化：假设访问量按容量比例分配
            estimated_visitors = int(total_visitors * (capacity / total_capacity)) if total_capacity > 0 else 0
            usage_ratio = estimated_visitors / capacity if capacity > 0 else 0
            
            ret['usage_ratio'] = usage_ratio
            
            # 更新连续高使用天数
            if usage_ratio > 0.6:
                ret['consecutive_high_usage'] = ret.get('consecutive_high_usage', 0) + 1
            else:
                ret['consecutive_high_usage'] = 0
        
        stats['retail_usage'] = overall_usage
        stats['retail_capacity'] = total_capacity
        stats['retail_visitors'] = total_visitors
    
    def _calculate_coverage_stats(self, stats):
        """计算覆盖率统计"""
        # 简化的覆盖率计算
        total_residents = len(self.residents)
        if total_residents == 0:
            stats['school_coverage'] = 0
            stats['clinic_coverage'] = 0
            stats['avg_school_time'] = 0
            stats['avg_clinic_time'] = 0
            return
        
        # 计算到最近学校和诊所的距离
        school_distances = []
        clinic_distances = []
        
        for resident in self.residents:
            home_pos = resident['pos']
            
            # 找到最近的学校
            schools = [p for p in self.data['public'] if p['type'] == 'school']
            if schools:
                min_school_dist = min(self._calculate_distance(home_pos, school['xy']) for school in schools)
                school_distances.append(min_school_dist)
            
            # 找到最近的诊所
            clinics = [p for p in self.data['public'] if p['type'] == 'clinic']
            if clinics:
                min_clinic_dist = min(self._calculate_distance(home_pos, clinic['xy']) for clinic in clinics)
                clinic_distances.append(min_clinic_dist)
        
        # 计算统计值
        stats['avg_school_time'] = np.mean(school_distances) if school_distances else 0
        stats['school_coverage'] = len(school_distances) / total_residents if total_residents > 0 else 0
        stats['avg_clinic_time'] = np.mean(clinic_distances) if clinic_distances else 0
        stats['clinic_coverage'] = len(clinic_distances) / total_residents if total_residents > 0 else 0
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点间距离"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def update_pois(self, stats):
        """更新POI（政府和企业决策）"""
        # 政府添加公共设施
        new_public = self.placement_logic.gov_add_public(stats, self.data['trunk'])
        self.data['public'].extend(new_public)
        
        # 企业更新（住宅和商业）
        new_residential, new_retail = self.placement_logic.firm_update(stats)
        
        # 添加新住宅POI
        for res in new_residential:
            res['current_population'] = 0
            res['residents'] = []
            self.data['residential'].append(res)
        
        # 添加新商业POI
        for ret in new_retail:
            ret['current_population'] = 0
            self.data['retail'].append(ret)
    
    def render_frame(self):
        """渲染当前帧"""
        if self.current_day % self.render_every_n_days == 0:
            self.visualizer.render_layers(
                self.data['hubs'],
                self.data['trunk'],
                self.data['public'],
                self.data['residential'],
                self.data['retail'],
                self.heat_map,
                self.residents
            )
            
            # 保存图片
            filename = f"improved_day_{self.current_day:03d}.png"
            filepath = self.output_dir / filename
            self.visualizer.save_frame(str(filepath))
            print(f"📸 保存图片: {filename}")
    
    def save_data(self):
        """保存数据"""
        # 保存每日统计
        with open(self.output_dir / 'daily_stats.json', 'w', encoding='utf-8') as f:
            json.dump(self.daily_stats, f, indent=2, ensure_ascii=False)
        
        # 保存POI演化
        with open(self.output_dir / 'poi_evolution.json', 'w', encoding='utf-8') as f:
            json.dump(self.poi_evolution, f, indent=2, ensure_ascii=False)
        
        # 保存人口历史
        with open(self.output_dir / 'population_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.population_history, f, indent=2, ensure_ascii=False)
        
        # 保存最终热力图
        np.save(self.output_dir / 'final_heatmap.npy', self.heat_map)
        
        print("💾 数据保存完成")
    
    def print_summary(self):
        """打印总结"""
        print("\n" + "="*50)
        print("🏙️ 改进城市模拟总结")
        print("="*50)
        print(f"📅 模拟天数: {self.days}")
        print(f"👥 最终人口: {len(self.residents)}")
        print(f"🏠 住宅设施: {len(self.data['residential'])}")
        print(f"🛒 商业设施: {len(self.data['retail'])}")
        print(f"🏛️ 公共设施: {len(self.data['public'])}")
        
        # 人口增长分析
        if self.population_history:
            initial_pop = self.population_history[0]['population']
            final_pop = self.population_history[-1]['population']
            growth_rate = (final_pop - initial_pop) / initial_pop * 100 if initial_pop > 0 else 0
            print(f"📈 人口增长率: {growth_rate:.1f}%")
        
        print("="*50)
    
    def run_simulation(self):
        """运行模拟"""
        print("🚀 开始改进城市模拟")
        print(f"📅 模拟天数: {self.days}")
        print(f"👥 初始人口: {self.initial_population}")
        print(f"📊 月度增长率: {self.monthly_growth_rate*100:.1f}%")
        
        if not self.load_data():
            return
        
        # 生成初始居民
        initial_residents = self._create_new_residents(self.initial_population)
        self.residents.extend(initial_residents)
        
        # 为初始居民分配计划
        self.schedule_logic.assign_daily_plans(
            self.residents, 
            self.data['retail'], 
            self.data['public']
        )
        
        print(f"🏠 初始居民: {len(self.residents)}人")
        
        # 主循环
        for day in range(self.days):
            self.current_day = day
            
            # 生成新居民（每月一次）
            self.spawn_residents()
            
            # 移动居民
            self.move_residents()
            
            # 更新热力图
            self.update_heat_map()
            
            # 计算统计
            stats = self.calculate_stats()
            
            # 更新POI
            self.update_pois(stats)
            
            # 渲染帧
            self.render_frame()
            
            # 打印进度
            if day % 30 == 0:
                print(f"📅 第{day}天 - 人口: {len(self.residents)}, "
                      f"住宅: {len(self.data['residential'])}, "
                      f"商业: {len(self.data['retail'])}")
        
        # 保存数据
        self.save_data()
        
        # 打印总结
        self.print_summary()
        
        print("✅ 改进城市模拟完成！")

def main():
    """主函数"""
    simulation = ImprovedCitySimulation()
    simulation.run_simulation()

if __name__ == "__main__":
    main()
