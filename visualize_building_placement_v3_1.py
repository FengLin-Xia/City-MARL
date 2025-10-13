#!/usr/bin/env python3
"""
建筑放置位置逐帧播放可视化脚本 v3.1
专门用于显示建筑放置位置的动态变化
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from pathlib import Path
import time

class BuildingPlacementVisualizer:
    """建筑放置位置可视化器"""
    
    def __init__(self, output_dir="enhanced_simulation_v3_1_output"):
        self.output_dir = output_dir
        self.simplified_dir = os.path.join(output_dir, "simplified")
        
        # 建筑类型颜色映射
        self.building_colors = {
            0: '#4CAF50',  # 住宅 - 绿色
            1: '#2196F3',  # 商业 - 蓝色
            2: '#FF9800',  # 工业 - 橙色
            3: '#9C27B0',  # 公共 - 紫色
            4: '#F44336'   # 其他 - 红色
        }
        
        # 建筑类型标签
        self.building_labels = {
            0: '住宅',
            1: '商业', 
            2: '工业',
            3: '公共',
            4: '其他'
        }
        
        # 地图配置
        self.map_size = [110, 110]
        self.transport_hubs = [[20, 55], [90, 55], [67, 94]]
        
        # 数据存储
        self.monthly_buildings = {}
        self.max_month = 0
        
        print("🏗️ 建筑放置位置可视化器 v3.1 初始化完成")
    
    def load_simulation_data(self):
        """加载模拟数据"""
        print("📂 加载模拟数据...")
        
        # 查找所有可用的月份数据
        available_months = []
        
        # 检查完整建筑位置文件
        for file in os.listdir(self.output_dir):
            if file.startswith("building_positions_month_") and file.endswith(".json"):
                try:
                    month_str = file.replace("building_positions_month_", "").replace(".json", "")
                    month = int(month_str)
                    available_months.append(month)
                except ValueError:
                    continue
        
        if not available_months:
            print("❌ 未找到模拟数据文件")
            return False
        
        available_months.sort()
        self.max_month = max(available_months)
        
        print(f"📊 找到 {len(available_months)} 个月份的数据 (0-{self.max_month})")
        
        # 加载每个月份的建筑数据
        for month in available_months:
            self._load_month_buildings(month)
        
        print(f"✅ 数据加载完成：{len(self.monthly_buildings)} 个月份")
        return True
    
    def _load_month_buildings(self, month):
        """加载指定月份的建筑数据"""
        json_file = os.path.join(self.output_dir, f"building_positions_month_{month:02d}.json")
        
        buildings = []
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 解析完整JSON格式
                for building_data in data.get('buildings', []):
                    building = self._parse_building_json(building_data)
                    if building:
                        buildings.append(building)
                
            except Exception as e:
                print(f"⚠️ 加载第 {month} 个月数据时出错: {e}")
        
        self.monthly_buildings[month] = buildings
        
        if month % 6 == 0:  # 每6个月打印一次进度
            print(f"  📅 第 {month} 个月：{len(buildings)} 个建筑")
    
    def _parse_building_json(self, building_data):
        """解析建筑JSON数据"""
        try:
            # 获取建筑类型
            building_type = building_data.get('type', 'unknown').lower()
            
            # 类型映射
            type_mapping = {
                'residential': 0,
                'commercial': 1,
                'industrial': 2,
                'public': 3
            }
            
            type_id = type_mapping.get(building_type, 4)
            
            # 获取位置
            position = building_data.get('position', [0, 0])
            if len(position) >= 2:
                return {
                    'type': type_id,
                    'x': float(position[0]),
                    'y': float(position[1]),
                    'z': 0.0
                }
        except (ValueError, KeyError) as e:
            return None
        
        return None
    
    def create_animation(self, save_gif=True, show_plot=True):
        """创建动画"""
        if not self.monthly_buildings:
            print("❌ 没有数据可以可视化")
            return
        
        print("🎬 创建建筑放置动画...")
        
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle('建筑放置位置动态演化 v3.1', fontsize=16, fontweight='bold')
        
        # 设置地图范围
        ax.set_xlim(0, self.map_size[0])
        ax.set_ylim(0, self.map_size[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        
        # 绘制交通枢纽
        self._draw_transport_hubs(ax)
        
        # 创建图例
        self._create_legend(ax)
        
        # 初始化空散点图
        scatter = ax.scatter([], [], s=50, alpha=0.7)
        
        # 创建文本显示月份
        month_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=14, fontweight='bold', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           verticalalignment='top')
        
        # 创建建筑统计文本
        stats_text = ax.text(0.02, 0.88, '', transform=ax.transAxes,
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                           verticalalignment='top')
        
        def animate(frame):
            """动画帧函数"""
            month = frame
            
            if month in self.monthly_buildings:
                buildings = self.monthly_buildings[month]
                
                if buildings:
                    # 准备数据
                    x_coords = [b['x'] for b in buildings]
                    y_coords = [b['y'] for b in buildings]
                    colors = [self.building_colors.get(b['type'], '#F44336') for b in buildings]
                    
                    # 更新散点图
                    scatter.set_offsets(np.column_stack([x_coords, y_coords]))
                    scatter.set_color(colors)
                    scatter.set_sizes([50] * len(buildings))
                else:
                    # 没有建筑时清空
                    scatter.set_offsets(np.empty((0, 2)))
                    scatter.set_color([])
                    scatter.set_sizes([])
                
                # 更新月份文本
                month_text.set_text(f'第 {month} 个月')
                
                # 更新统计信息
                if buildings:
                    stats = self._calculate_building_stats(buildings)
                    stats_str = f"总建筑: {len(buildings)}\n"
                    for type_id, count in stats.items():
                        if count > 0:
                            stats_str += f"{self.building_labels[type_id]}: {count}\n"
                    stats_text.set_text(stats_str.rstrip())
                else:
                    stats_text.set_text("总建筑: 0")
            
            return scatter, month_text, stats_text
        
        # 创建动画
        frames = list(range(self.max_month + 1))
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                     interval=800, blit=False, repeat=True)
        
        # 保存GIF
        if save_gif:
            gif_path = "building_placement_animation_v3_1.gif"
            print(f"💾 保存动画到 {gif_path}...")
            anim.save(gif_path, writer='pillow', fps=1.25, dpi=100)
            print(f"✅ 动画已保存到 {gif_path}")
        
        # 显示动画
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return anim
    
    def _draw_transport_hubs(self, ax):
        """绘制交通枢纽"""
        for i, hub in enumerate(self.transport_hubs):
            x, y = hub
            circle = Circle((x, y), 3, color='red', alpha=0.8, zorder=10)
            ax.add_patch(circle)
            ax.text(x, y-6, f'Hub{i+1}', ha='center', va='top', 
                   fontsize=10, fontweight='bold', color='red')
    
    def _create_legend(self, ax):
        """创建图例"""
        legend_elements = []
        for type_id, color in self.building_colors.items():
            if type_id in self.building_labels:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=8,
                                                label=self.building_labels[type_id]))
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    def _calculate_building_stats(self, buildings):
        """计算建筑统计"""
        stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for building in buildings:
            type_id = building['type']
            if type_id in stats:
                stats[type_id] += 1
        return stats
    
    def create_static_plots(self, months_to_show=None):
        """创建静态对比图"""
        if not self.monthly_buildings:
            print("❌ 没有数据可以可视化")
            return
        
        if months_to_show is None:
            # 默认显示关键月份
            total_months = len(self.monthly_buildings)
            if total_months <= 4:
                months_to_show = list(self.monthly_buildings.keys())
            else:
                # 选择开始、中间、结束的月份
                sorted_months = sorted(self.monthly_buildings.keys())
                months_to_show = [
                    sorted_months[0],  # 开始
                    sorted_months[len(sorted_months)//3],  # 1/3
                    sorted_months[2*len(sorted_months)//3],  # 2/3
                    sorted_months[-1]  # 结束
                ]
        
        print(f"📊 创建静态对比图：月份 {months_to_show}")
        
        n_plots = len(months_to_show)
        cols = min(4, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('建筑放置位置关键月份对比 v3.1', fontsize=16, fontweight='bold')
        
        for i, month in enumerate(months_to_show):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # 设置子图
            ax.set_xlim(0, self.map_size[0])
            ax.set_ylim(0, self.map_size[1])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'第 {month} 个月', fontsize=12, fontweight='bold')
            
            # 绘制交通枢纽
            self._draw_transport_hubs(ax)
            
            # 绘制建筑
            if month in self.monthly_buildings:
                buildings = self.monthly_buildings[month]
                if buildings:
                    for building in buildings:
                        color = self.building_colors.get(building['type'], '#F44336')
                        ax.scatter(building['x'], building['y'], 
                                 c=color, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # 添加建筑数量标注
            if month in self.monthly_buildings:
                count = len(self.monthly_buildings[month])
                ax.text(0.02, 0.98, f'建筑数: {count}', transform=ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       verticalalignment='top')
        
        # 隐藏多余的子图
        for i in range(n_plots, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        static_path = "building_placement_comparison_v3_1.png"
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        print(f"💾 静态对比图已保存到 {static_path}")
        
        plt.show()

def main():
    """主函数"""
    print("🏗️ 建筑放置位置可视化器 v3.1")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = BuildingPlacementVisualizer()
    
    # 加载数据
    if not visualizer.load_simulation_data():
        return
    
    print("\n🎬 选择可视化模式:")
    print("1. 逐帧动画播放")
    print("2. 静态对比图")
    print("3. 两者都生成")
    
    try:
        choice = input("\n请选择 (1/2/3): ").strip()
        
        if choice == "1":
            visualizer.create_animation(save_gif=True, show_plot=True)
        elif choice == "2":
            visualizer.create_static_plots()
        elif choice == "3":
            visualizer.create_animation(save_gif=True, show_plot=False)
            visualizer.create_static_plots()
        else:
            print("❌ 无效选择，默认生成动画")
            visualizer.create_animation(save_gif=True, show_plot=True)
    
    except KeyboardInterrupt:
        print("\n👋 用户取消操作")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    
    print("\n✅ 可视化完成！")

if __name__ == "__main__":
    main()
