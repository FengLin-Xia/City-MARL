#!/usr/bin/env python3
"""
Simple Fixed Visualizer for v2.3
简化版本的可视化器，避免动画警告
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import os
import glob

class SimpleFixedVisualizer:
    """简化版本建筑可视化器"""
    
    def __init__(self):
        self.output_dir = 'enhanced_simulation_v2_3_output'
        self.building_data = {}
        self.sdf_data = {}
        
        # 地图配置
        self.map_size = [256, 256]
        self.trunk_road = [[40, 128], [216, 128]]
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载数据"""
        # 加载建筑位置数据
        building_files = glob.glob(f'{self.output_dir}/building_positions_month_*.json')
        
        for file_path in building_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    month_str = data['timestamp']  # 格式: 'month_00', 'month_03', ...
                    # 提取数字部分
                    month_num = int(month_str.split('_')[1])
                    self.building_data[month_num] = data
            except Exception as e:
                print(f"Failed to load building data {file_path}: {e}")
        
        # 加载SDF数据
        sdf_files = glob.glob(f'{self.output_dir}/sdf_field_month_*.json')
        
        for file_path in sdf_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    month = data['month']  # 格式: 0, 3, 6, ...
                    self.sdf_data[month] = data
            except Exception as e:
                print(f"Failed to load SDF data {file_path}: {e}")
        
        print(f"Loaded building data for {len(self.building_data)} months")
        print(f"Loaded SDF data for {len(self.sdf_data)} months")
    
    def show_month(self, month: int):
        """显示指定月份的数据"""
        if month not in self.building_data:
            print(f"No building data for month {month}")
            return
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 设置坐标轴
        ax.set_xlim(0, self.map_size[0])
        ax.set_ylim(0, self.map_size[1])
        ax.set_aspect('equal')
        
        # 绘制主干道
        x_coords = [self.trunk_road[0][0], self.trunk_road[1][0]]
        y_coords = [self.trunk_road[0][1], self.trunk_road[1][1]]
        ax.plot(x_coords, y_coords, 'k-', linewidth=3, alpha=0.7, label='Trunk Road')
        
        # 绘制交通枢纽
        for i, hub in enumerate(self.trunk_road):
            ax.plot(hub[0], hub[1], 'o', markersize=10, color='blue', 
                    markeredgecolor='black', markeredgewidth=2, label=f'Hub {chr(65+i)}' if i == 0 else "")
        
        # 绘制等值线（如果有SDF数据）
        if month in self.sdf_data:
            sdf_data = self.sdf_data[month]
            sdf_field = np.array(sdf_data['sdf_field'])
            
            # 创建坐标网格
            y_coords = np.arange(sdf_field.shape[0])
            x_coords = np.arange(sdf_field.shape[1])
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # 绘制商业等值线（红色虚线）
            commercial_levels = [0.85, 0.70, 0.55]
            for level in commercial_levels:
                try:
                    if np.min(sdf_field) <= level <= np.max(sdf_field):
                        contour = ax.contour(X, Y, sdf_field, levels=[level], colors='red', 
                                           linestyles='dashed', alpha=0.6, linewidths=1)
                except Exception as e:
                    print(f"Failed to draw commercial contour at level {level}: {e}")
            
            # 绘制住宅等值线（蓝色虚线）
            residential_levels = [0.55, 0.40, 0.25]
            for level in residential_levels:
                try:
                    if np.min(sdf_field) <= level <= np.max(sdf_field):
                        contour = ax.contour(X, Y, sdf_field, levels=[level], colors='blue', 
                                           linestyles='dashed', alpha=0.6, linewidths=1)
                except Exception as e:
                    print(f"Failed to draw residential contour at level {level}: {e}")
        
        # 绘制建筑
        data = self.building_data[month]
        buildings = data['buildings']
        
        for building in buildings:
            pos = building['position']
            building_type = building['type']
            sdf_value = building.get('sdf_value', 0.0)
            
            if building_type == 'residential':
                # 住宅建筑：黄色方形
                rect = patches.Rectangle((pos[0]-2, pos[1]-2), 4, 4, 
                                       facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                
            elif building_type == 'commercial':
                # 商业建筑：橙色圆形
                circle = patches.Circle((pos[0], pos[1]), radius=3, 
                                      facecolor='orange', alpha=0.8, edgecolor='black', linewidth=1)
                ax.add_patch(circle)
                
            elif building_type == 'public':
                # 公共建筑：青色三角形
                triangle = patches.RegularPolygon((pos[0], pos[1]), numVertices=3, radius=4,
                                                facecolor='cyan', alpha=0.8, edgecolor='black', linewidth=1)
                ax.add_patch(triangle)
        
        # 设置标题
        ax.set_title(f'Month {month:02d} - Building Distribution', fontsize=14)
        
        # 显示建筑统计
        residential_count = len([b for b in buildings if b['type'] == 'residential'])
        commercial_count = len([b for b in buildings if b['type'] == 'commercial'])
        public_count = len([b for b in buildings if b['type'] == 'public'])
        
        sdf_values = [b.get('sdf_value', 0.0) for b in buildings]
        avg_sdf = np.mean(sdf_values) if sdf_values else 0.0
        
        stats_text = (f'Buildings:\n'
                     f'Residential: {residential_count}\n'
                     f'Commercial: {commercial_count}\n'
                     f'Public: {public_count}\n'
                     f'Avg SDF: {avg_sdf:.3f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 显示图例
        legend_elements = [
            plt.Line2D([0], [0], color='black', linewidth=3, label='Trunk Road'),
            plt.Line2D([0], [0], marker='o', color='blue', markersize=10, label='Transport Hub'),
            patches.Patch(facecolor='yellow', label='Residential'),
            patches.Patch(facecolor='orange', label='Commercial'),
            patches.Patch(facecolor='cyan', label='Public'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Commercial Isocontour'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label='Residential Isocontour')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.show()
    
    def show_all_months(self):
        """显示所有月份的数据"""
        available_months = sorted(self.building_data.keys())
        print(f"Available months: {available_months}")
        
        for month in available_months:
            print(f"\nShowing month {month}...")
            self.show_month(month)
            input("Press Enter to continue to next month...")

def main():
    """主函数"""
    print("🏙️ Simple Fixed Visualizer v2.3")
    print("=" * 50)
    print("🎯 Features:")
    print("  • Fixed data format issues")
    print("  • Simple month-by-month display")
    print("  • No animation warnings")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = SimpleFixedVisualizer()
    
    # 显示可用月份
    available_months = sorted(visualizer.building_data.keys())
    print(f"\nAvailable months: {available_months}")
    
    # 询问用户想看哪个月
    try:
        month = int(input(f"Enter month to view (0-{max(available_months)}): "))
        if month in available_months:
            visualizer.show_month(month)
        else:
            print(f"Month {month} not available. Showing month 0...")
            visualizer.show_month(0)
    except ValueError:
        print("Invalid input. Showing month 0...")
        visualizer.show_month(0)

if __name__ == "__main__":
    main()


