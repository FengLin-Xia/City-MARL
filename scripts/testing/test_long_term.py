#!/usr/bin/env python3
"""
测试长期训练 - 20天版本
"""

import json
import numpy as np
from pathlib import Path
from logic.placement import PlacementLogic
from logic.schedule import ScheduleLogic
from logic.move import MoveLogic
from viz.ide import CityVisualizer

class TestLongTermSimulation:
    def __init__(self):
        # 测试参数
        self.days = 365  # 365天测试
        self.steps_per_day = 144
        self.daily_residents = 30  # 每天30人
        self.movement_speed = 4
        self.movement_mode = "linear"
        
        # 热力图参数
        self.heat_evaporation = 0.995
        self.heat_map = np.zeros((256, 256))
        
        # 统计跟踪
        self.daily_stats = []
        self.poi_evolution = []
        
        # 初始化逻辑模块
        self.placement_logic = PlacementLogic()
        self.schedule_logic = ScheduleLogic()
        self.move_logic = MoveLogic()
        self.visualizer = CityVisualizer()
        
        # 居民列表
        self.residents = []
        
        # 输出设置
        self.output_dir = Path('test_long_term_output')
        self.output_dir.mkdir(exist_ok=True)
        
        # 渲染频率
        self.render_every_n_days = 2  # 每2天渲染一次
        
    def load_data(self):
        """加载初始数据"""
        try:
            with open('data/poi_example.json', 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print("✅ 数据加载成功")
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
        return True
    
    def spawn_residents(self):
        """生成新居民"""
        new_residents = self.schedule_logic.spawn_residents(
            self.data['residential'], 
            self.daily_residents
        )
        self.residents.extend(new_residents)
        print(f"📈 第{len(self.residents)}天: 新增{len(new_residents)}居民，总居民数: {len(self.residents)}")
    
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
    
    def update_heat_map(self):
        """更新热力图"""
        self.heat_map *= self.heat_evaporation
    
    def calculate_stats(self):
        """计算每日统计"""
        # 基础统计
        stats = {
            'day': len(self.daily_stats) + 1,
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
        
        # 计算使用率统计
        self._calculate_usage_stats(stats)
        
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
        
        return stats
    
    def _calculate_usage_stats(self, stats):
        """计算POI使用率统计"""
        # 计算住宅使用率
        total_residential_capacity = sum(res.get('capacity', 200) for res in self.data['residential'])
        residential_usage_ratio = len(self.residents) / total_residential_capacity if total_residential_capacity > 0 else 0
        
        for res in self.data['residential']:
            res['usage_ratio'] = residential_usage_ratio
            # 更新连续高使用天数（使用新的阈值0.6）
            if residential_usage_ratio > 0.6:
                res['consecutive_high_usage'] = res.get('consecutive_high_usage', 0) + 1
            else:
                res['consecutive_high_usage'] = 0
        
        # 计算商业使用率（简化：假设所有居民都会访问商业设施）
        total_retail_capacity = sum(ret.get('capacity', 800) for ret in self.data['retail'])
        retail_usage_ratio = len(self.residents) / total_retail_capacity if total_retail_capacity > 0 else 0
        
        for ret in self.data['retail']:
            ret['usage_ratio'] = retail_usage_ratio
            # 更新连续高使用天数（使用新的阈值0.6）
            if retail_usage_ratio > 0.6:
                ret['consecutive_high_usage'] = ret.get('consecutive_high_usage', 0) + 1
            else:
                ret['consecutive_high_usage'] = 0
    
    def _calculate_coverage_stats(self, stats):
        """计算覆盖率统计"""
        # 简化的覆盖率计算：基于距离
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
            resident_pos = resident['pos']
            
            # 找到最近的学校
            min_school_dist = float('inf')
            for poi in self.data['public']:
                if poi.get('type') == 'school':
                    dist = np.linalg.norm(np.array(resident_pos) - np.array(poi['xy']))
                    min_school_dist = min(min_school_dist, dist)
            
            if min_school_dist != float('inf'):
                school_distances.append(min_school_dist)
            
            # 找到最近的诊所
            min_clinic_dist = float('inf')
            for poi in self.data['public']:
                if poi.get('type') == 'clinic':
                    dist = np.linalg.norm(np.array(resident_pos) - np.array(poi['xy']))
                    min_clinic_dist = min(min_clinic_dist, dist)
            
            if min_clinic_dist != float('inf'):
                clinic_distances.append(min_clinic_dist)
        
        # 计算平均时间和覆盖率
        if school_distances:
            stats['avg_school_time'] = np.mean(school_distances) / 4.0  # 假设4px/step的速度
            stats['school_coverage'] = len(school_distances) / total_residents
        else:
            stats['avg_school_time'] = 100  # 很大的值表示没有覆盖
            stats['school_coverage'] = 0
        
        if clinic_distances:
            stats['avg_clinic_time'] = np.mean(clinic_distances) / 4.0
            stats['clinic_coverage'] = len(clinic_distances) / total_residents
        else:
            stats['avg_clinic_time'] = 100
            stats['clinic_coverage'] = 0
    
    def update_pois(self, stats):
        """更新POI"""
        # 政府补点
        new_public = self.placement_logic.gov_add_public(
            stats, 
            self.data['trunk']
        )
        if new_public:
            self.data['public'].extend(new_public)
            print(f"🏛️ 第{stats['day']}天: 政府新增{len(new_public)}个公共设施")
        
        # 企业扩容
        residential_updates, retail_updates = self.placement_logic.firm_update(stats)
        
        if residential_updates:
            self.data['residential'].extend(residential_updates)
            print(f"🏠 第{stats['day']}天: 企业新增{len(residential_updates)}个住宅设施")
        
        if retail_updates:
            self.data['retail'].extend(retail_updates)
            print(f"🛒 第{stats['day']}天: 企业新增{len(retail_updates)}个零售设施")
    
    def render_frame(self, day):
        """渲染帧"""
        if day % self.render_every_n_days == 0:
            self.visualizer.render_layers(
                self.data['hubs'],
                self.data['trunk'],
                self.data['public'],
                self.data['residential'],
                self.data['retail'],
                self.heat_map,
                self.residents
            )
            
            filename = f"test_day_{day:02d}.png"
            filepath = self.output_dir / filename
            self.visualizer.save_frame(str(filepath))
            print(f"📸 第{day}天: 保存图片 {filename}")
    
    def save_statistics(self):
        """保存统计数据"""
        # 保存每日统计
        stats_file = self.output_dir / 'daily_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.daily_stats, f, indent=2, ensure_ascii=False)
        
        # 保存POI演化
        evolution_file = self.output_dir / 'poi_evolution.json'
        with open(evolution_file, 'w', encoding='utf-8') as f:
            json.dump(self.poi_evolution, f, indent=2, ensure_ascii=False)
        
        # 保存最终热力图
        heat_file = self.output_dir / 'final_heatmap.npy'
        np.save(heat_file, self.heat_map)
        
        print(f"📊 统计数据已保存到 {self.output_dir}")
    
    def print_summary(self):
        """打印仿真总结"""
        print("\n" + "="*60)
        print("🏙️ 测试长期城市仿真总结")
        print("="*60)
        
        final_stats = self.daily_stats[-1]
        print(f"📅 仿真天数: {self.days}")
        print(f"👥 最终居民数: {final_stats['total_residents']}")
        print(f"🏛️ 公共设施: {final_stats['public_pois']}")
        print(f"🏠 住宅设施: {final_stats['residential_pois']}")
        print(f"🛒 零售设施: {final_stats['retail_pois']}")
        print(f"🔥 热力图总和: {final_stats['heat_sum']:.2f}")
        print(f"🔥 热力图最大值: {final_stats['heat_max']:.2f}")
        
        # 分析发展趋势
        if len(self.poi_evolution) > 5:
            early_public = self.poi_evolution[5]['public']
            late_public = self.poi_evolution[-1]['public']
            public_growth = late_public - early_public
            
            early_residential = self.poi_evolution[5]['residential']
            late_residential = self.poi_evolution[-1]['residential']
            residential_growth = late_residential - early_residential
            
            print(f"\n📈 发展趋势分析:")
            print(f"   公共设施增长: +{public_growth}")
            print(f"   住宅设施增长: +{residential_growth}")
            
            if public_growth > 2:
                print("   🎯 城市公共服务发展良好")
            if residential_growth > 3:
                print("   🏘️ 城市居住功能显著扩张")
    
    def run_simulation(self):
        """运行测试仿真"""
        print("🚀 开始测试长期城市仿真...")
        print(f"📅 仿真天数: {self.days}")
        print(f"👥 每日新居民: {self.daily_residents}")
        print(f"📸 渲染频率: 每{self.render_every_n_days}天")
        
        if not self.load_data():
            return
        
        for day in range(1, self.days + 1):
            print(f"\n📅 第{day}天开始...")
            
            # 生成新居民
            self.spawn_residents()
            
            # 分配日程
            self.schedule_logic.assign_daily_plans(
                self.residents,
                self.data['retail'],
                self.data['public']
            )
            
            # 每日步数循环
            for step in range(self.steps_per_day):
                self.move_residents()
                self.update_heat_map()
            
            # 计算统计
            stats = self.calculate_stats()
            
            # 更新POI
            self.update_pois(stats)
            
            # 渲染帧
            self.render_frame(day)
        
        # 保存最终结果
        self.save_statistics()
        self.print_summary()
        
        print(f"\n✅ 测试长期仿真完成！结果保存在 {self.output_dir}")

if __name__ == "__main__":
    simulation = TestLongTermSimulation()
    simulation.run_simulation()
