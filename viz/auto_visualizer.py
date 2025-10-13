#!/usr/bin/env python3
"""
自动化可视化器 - 展示逐月城市生长效果
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import time

class AutoVisualizer:
    def __init__(self, output_dir='enhanced_simulation_v2_3_output'):
        self.output_dir = output_dir
        self.building_data = {}
        self.sdf_data = {}
        self.available_months = []
        
        # 颜色配置
        self.colors = {
            'residential': '#F6C344',  # 黄色
            'commercial': '#FD7E14',   # 橙色
            'public': '#0B5ED7',       # 蓝色
            'trunk_road': '#9AA4B2',   # 灰色
            'hub': '#0B5ED7'           # 深蓝色
        }
        
        self.load_data()
    
    def load_data(self):
        """加载所有数据"""
        print("📁 正在加载数据...")
        
        # 加载建筑位置数据
        for filename in os.listdir(self.output_dir):
            if filename.startswith('building_positions_month_') and filename.endswith('.json'):
                month = int(filename.split('_')[2].split('.')[0])
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.building_data[month] = json.load(f)
                self.available_months.append(month)
        
        # 加载SDF场数据
        for filename in os.listdir(self.output_dir):
            if filename.startswith('sdf_field_month_') and filename.endswith('.json'):
                month = int(filename.split('_')[2].split('.')[0])
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.sdf_data[month] = json.load(f)
        
        self.available_months.sort()
        print(f"✅ 加载完成！可用月份: {self.available_months}")
        print(f"📊 建筑数据: {len(self.building_data)} 个月")
        print(f"📊 SDF数据: {len(self.sdf_data)} 个月")
    
    def visualize_month(self, month):
        """可视化指定月份"""
        if month not in self.available_months:
            print(f"❌ 月份 {month} 不可用")
            return
        
        print(f"🎨 正在可视化第 {month} 个月...")
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'City Growth Simulation - Month {month}', fontsize=16)
        
        # 左图：建筑分布
        self._plot_buildings(ax1, month)
        ax1.set_title(f'Building Distribution - Month {month}')
        
        # 右图：SDF场和等值线
        self._plot_sdf_field(ax2, month)
        ax2.set_title(f'SDF Field & Isocontours - Month {month}')
        
        plt.tight_layout()
        plt.show()
        
        # 等待用户查看
        input(f"按回车键继续查看下一个月...")
        plt.close()
    
    def _plot_buildings(self, ax, month):
        """绘制建筑分布"""
        ax.clear()
        
        # 绘制主干道
        trunk_road = [[40, 128], [216, 128]]
        ax.plot([trunk_road[0][0], trunk_road[1][0]], 
                [trunk_road[0][1], trunk_road[1][1]], 
                color=self.colors['trunk_road'], linewidth=3, label='Trunk Road')
        
        # 绘制交通枢纽
        hubs = [[40, 128], [216, 128]]
        for hub in hubs:
            ax.scatter(hub[0], hub[1], c=self.colors['hub'], s=200, marker='o', 
                      edgecolors='black', linewidth=2, label='Transport Hub', zorder=5)
        
        # 绘制建筑
        buildings = self.building_data.get(month, [])
        building_counts = {'residential': 0, 'commercial': 0, 'public': 0}
        
        for building in buildings:
            building_type = building.get('type', 'unknown')
            if building_type in building_counts:
                building_counts[building_type] += 1
                
                pos = building['xy']
                color = self.colors.get(building_type, '#999999')
                
                # 根据建筑类型绘制不同形状
                if building_type == 'residential':
                    rect = patches.Rectangle((pos[0]-2, pos[1]-2), 4, 4, 
                                           facecolor=color, edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
                elif building_type == 'commercial':
                    circle = patches.Circle((pos[0], pos[1]), 3, 
                                          facecolor=color, edgecolor='black', linewidth=0.5)
                    ax.add_patch(circle)
                elif building_type == 'public':
                    # 六边形
                    hexagon = patches.RegularPolygon((pos[0], pos[1]), 6, radius=3,
                                                   facecolor=color, edgecolor='black', linewidth=0.5)
                    ax.add_patch(hexagon)
        
        # 设置图形属性
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # 添加图例
        legend_elements = [
            patches.Patch(color=self.colors['residential'], label=f'Residential ({building_counts["residential"]})'),
            patches.Patch(color=self.colors['commercial'], label=f'Commercial ({building_counts["commercial"]})'),
            patches.Patch(color=self.colors['public'], label=f'Public ({building_counts["public"]})')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 添加统计信息
        total_buildings = sum(building_counts.values())
        ax.text(0.02, 0.98, f'Total Buildings: {total_buildings}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_sdf_field(self, ax, month):
        """绘制SDF场和等值线"""
        ax.clear()
        
        if month not in self.sdf_data:
            ax.text(0.5, 0.5, 'SDF data not available', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # 加载SDF场
        sdf_field = np.array(self.sdf_data[month])
        
        # 绘制SDF场热力图
        im = ax.imshow(sdf_field, cmap='viridis', origin='lower', 
                       extent=[0, 256, 0, 256], alpha=0.7)
        
        # 绘制等值线
        levels = np.linspace(np.min(sdf_field), np.max(sdf_field), 10)
        contours = ax.contour(sdf_field, levels=levels, colors='white', 
                             linewidths=0.5, alpha=0.8)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        # 绘制主干道
        trunk_road = [[40, 128], [216, 128]]
        ax.plot([trunk_road[0][0], trunk_road[1][0]], 
                [trunk_road[0][1], trunk_road[1][1]], 
                color='red', linewidth=2, label='Trunk Road')
        
        # 设置图形属性
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.set_aspect('equal')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'SDF Field & Isocontours')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('SDF Value')
        
        # 添加统计信息
        ax.text(0.02, 0.98, f'SDF Range: [{np.min(sdf_field):.3f}, {np.max(sdf_field):.3f}]', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def run_auto_visualization(self):
        """运行自动可视化"""
        print("🚀 开始自动可视化...")
        print("=" * 50)
        
        for month in self.available_months:
            print(f"\n📅 正在显示第 {month} 个月...")
            self.visualize_month(month)
        
        print("\n🎉 可视化完成！")
    
    def show_growth_summary(self):
        """显示生长摘要"""
        print("\n📊 城市生长摘要:")
        print("=" * 30)
        
        for month in self.available_months:
            buildings = self.building_data.get(month, [])
            building_counts = {'residential': 0, 'commercial': 0, 'public': 0}
            
            for building in buildings:
                building_type = building.get('type', 'unknown')
                if building_type in building_counts:
                    building_counts[building_type] += 1
            
            total = sum(building_counts.values())
            print(f"第 {month:2d} 个月: 住宅 {building_counts['residential']:2d}, "
                  f"商业 {building_counts['commercial']:2d}, "
                  f"公共 {building_counts['public']:2d}, "
                  f"总计 {total:2d}")

def main():
    """主函数"""
    visualizer = AutoVisualizer()
    
    # 显示生长摘要
    visualizer.show_growth_summary()
    
    # 询问是否运行自动可视化
    response = input("\n是否运行自动可视化？(y/n): ").lower().strip()
    
    if response in ['y', 'yes', '是']:
        visualizer.run_auto_visualization()
    else:
        print("👋 再见！")

if __name__ == "__main__":
    main()


