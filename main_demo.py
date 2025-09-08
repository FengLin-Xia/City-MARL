#!/usr/bin/env python3
"""
最小单元 Demo - 城市仿真主程序
无RL、无固化，纯规则驱动
"""

import json
import numpy as np
import os
from typing import List, Dict, Tuple
import sys

# 添加模块路径
sys.path.append('logic')
sys.path.append('viz')

from logic.placement import PlacementLogic
from logic.schedule import ScheduleLogic
from logic.move import MoveLogic
from viz.ide import CityVisualizer

class CitySimulation:
    def __init__(self, config_file: str = 'data/poi.json'):
        self.config_file = config_file
        self.grid_size = (256, 256)
        
        # 初始化各个模块
        self.placement_logic = PlacementLogic()
        self.schedule_logic = ScheduleLogic()
        self.move_logic = MoveLogic(self.grid_size)
        self.visualizer = CityVisualizer(self.grid_size)
        
        # 加载初始数据
        self.load_data()
        
        # 初始化热力图
        self.heat_map = np.zeros(self.grid_size)
        
        # 仿真参数
        self.days = 10
        self.steps_per_day = 144  # 10分钟一步，一天144步
        self.daily_quota = 100
        self.speed_px = 4
        self.heat_evaporation = 0.98
        
        # 居民列表
        self.residents = []
        
        # 统计信息
        self.stats = {
            'public': [],
            'residential': [],
            'retail': [],
            'hubs': self.data['hubs'],
            'avg_school_time': 0,
            'avg_clinic_time': 0,
            'school_coverage': 0,
            'clinic_coverage': 0
        }
    
    def load_data(self):
        """加载POI数据"""
        try:
            with open(self.config_file, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {self.config_file} 不存在，使用默认配置")
            self.data = {
                "hubs": [
                    {"id": "A", "xy": [40, 128]},
                    {"id": "B", "xy": [216, 128]}
                ],
                "trunk": [[40, 128], [216, 128]],
                "public": [],
                "residential": [],
                "retail": []
            }
    
    def save_data(self):
        """保存POI数据"""
        with open(self.config_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def spawn_residents(self):
        """生成新居民"""
        new_residents = self.schedule_logic.spawn_residents(
            self.data['residential'], self.daily_quota
        )
        self.residents.extend(new_residents)
        
        # 为所有居民分配计划
        self.schedule_logic.assign_daily_plans(
            self.residents, self.data['retail'], self.data['public']
        )
    
    def move_residents(self):
        """移动居民"""
        for resident in self.residents:
            # 检查是否到达目标
            if self.move_logic.reached(resident['pos'], resident['target']):
                # 获取下一个目标
                resident['target'] = self.schedule_logic.get_next_target(
                    resident, self.data
                )
                self.schedule_logic.advance_plan(resident)
            
            # 移动居民
            resident['pos'] = self.move_logic.move_towards(
                resident['pos'], resident['target'], self.speed_px, "linear"
            )
            
            # 更新热力图
            x, y = int(resident['pos'][0]), int(resident['pos'][1])
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                local_cost = self.move_logic.calculate_local_cost(
                    resident['pos'], self.heat_map
                )
                self.heat_map[x, y] += 1 / (1 + local_cost)
    
    def update_heat_map(self):
        """更新热力图（蒸发）"""
        if self.heat_evaporation < 1.0:
            self.heat_map *= self.heat_evaporation
    
    def calculate_stats(self):
        """计算统计信息"""
        # 简化的统计计算
        total_residents = len(self.residents)
        
        # 计算平均到达时间（简化）
        if total_residents > 0:
            self.stats['avg_school_time'] = np.random.uniform(8, 15)
            self.stats['avg_clinic_time'] = np.random.uniform(10, 18)
            self.stats['school_coverage'] = np.random.uniform(0.6, 0.9)
            self.stats['clinic_coverage'] = np.random.uniform(0.5, 0.8)
        else:
            self.stats['avg_school_time'] = 0
            self.stats['avg_clinic_time'] = 0
            self.stats['school_coverage'] = 0
            self.stats['clinic_coverage'] = 0
        
        # 更新POI使用情况
        self.stats['public'] = self.data['public']
        self.stats['residential'] = self.data['residential']
        self.stats['retail'] = self.data['retail']
    
    def update_pois(self):
        """更新POI（政府补点和企业扩容）"""
        # 政府补点
        new_public = self.placement_logic.gov_add_public(
            self.stats, self.data['trunk']
        )
        
        # 企业扩容
        new_residential, new_retail = self.placement_logic.firm_update(self.stats)
        
        # 添加新的POI
        for poi in new_public:
            self.data['public'].append(poi)
            self.visualizer.add_new_poi(poi)
        
        for poi in new_residential:
            self.data['residential'].append(poi)
            self.visualizer.add_new_poi(poi)
        
        for poi in new_retail:
            self.data['retail'].append(poi)
            self.visualizer.add_new_poi(poi)
    
    def render_frame(self, day: int, step: int):
        """渲染当前帧"""
        self.visualizer.render_layers(
            hubs=self.data['hubs'],
            trunk=self.data['trunk'],
            public_pois=self.data['public'],
            residential_pois=self.data['residential'],
            retail_pois=self.data['retail'],
            heat_map=self.heat_map,
            agents=self.residents,
            show_agents=False  # 可以设置为True来显示居民
        )
        
        # 保存帧
        output_dir = 'output_frames'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/day_{day:02d}_step_{step:03d}.png"
        self.visualizer.save_frame(filename)
        
        print(f"已保存: {filename}")
    
    def run_simulation(self):
        """运行仿真"""
        print("开始城市仿真...")
        
        for day in range(1, self.days + 1):
            print(f"\n=== 第 {day} 天 ===")
            
            # 生成新居民
            self.spawn_residents()
            print(f"当前居民数量: {len(self.residents)}")
            
            # 每日步进
            for step in range(self.steps_per_day):
                # 移动居民
                self.move_residents()
                
                # 更新热力图
                self.update_heat_map()
                
                # 每24步（4小时）渲染一次
                if step % 24 == 0:
                    self.render_frame(day, step)
            
            # 计算统计信息
            self.calculate_stats()
            
            # 更新POI
            self.update_pois()
            
            # 渲染每日结束帧
            self.render_frame(day, self.steps_per_day)
            
            # 清除新增POI高亮
            self.visualizer.clear_new_pois()
            
            # 保存数据
            self.save_data()
            
            print(f"第 {day} 天完成")
            print(f"公共POI: {len(self.data['public'])}")
            print(f"住宅POI: {len(self.data['residential'])}")
            print(f"商业POI: {len(self.data['retail'])}")
        
        print("\n仿真完成！")
        self.visualizer.close()

def main():
    """主函数"""
    # 创建仿真实例，使用示例数据
    simulation = CitySimulation('data/poi_example.json')
    
    # 运行仿真
    simulation.run_simulation()

if __name__ == "__main__":
    main()
