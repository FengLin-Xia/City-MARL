#!/usr/bin/env python3
"""
增强城市模拟系统 v3.3 可视化脚本
展示地价场驱动的建筑生长过程
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os
import glob
from typing import List, Dict, Tuple, Optional
import argparse

class CityEvolutionVisualizerV3_3:
    """城市演化可视化器 v3.3"""
    
    def __init__(self, output_dir: str = 'enhanced_simulation_v3_3_output'):
        self.output_dir = output_dir
        self.building_frames = []
        self.land_price_frames = []
        self.max_month = 0
        
        # 可视化配置
        self.building_colors = {
            'residential': '#2E8B57',    # 海绿色
            'commercial': '#FF6347',     # 番茄红
            'industrial': '#4682B4',     # 钢蓝色
            'public': '#9370DB'          # 中紫色
        }
        
        self.building_markers = {
            'residential': 'o',
            'commercial': 's',
            'industrial': '^',
            'public': 'D'
        }
        
        self.building_sizes = {
            'residential': 30,
            'commercial': 40,
            'industrial': 50,
            'public': 60
        }
        
        print("✅ 城市演化可视化器v3.3初始化完成")
    
    def load_simulation_data(self):
        """加载模拟数据"""
        print("📂 加载模拟数据...")
        
        # 加载建筑数据
        self._load_building_frames()
        
        # 加载地价场数据
        self._load_land_price_frames()
        
        print(f"✅ 数据加载完成，共 {self.max_month + 1} 个月的数据")
    
    def _load_building_frames(self):
        """加载建筑帧数据"""
        # 首先加载第0个月的完整状态
        month_0_file = os.path.join(self.output_dir, 'building_positions_month_00.json')
        if os.path.exists(month_0_file):
            with open(month_0_file, 'r') as f:
                data = json.load(f)
                self.building_frames.append(data['buildings'])
                print(f"  📊 加载第0个月完整建筑状态")
        else:
            print("  ⚠️ 未找到第0个月建筑数据")
            return
        
        # 加载增量数据
        delta_files = sorted(glob.glob(os.path.join(self.output_dir, 'building_delta_month_*.json')))
        
        # 找到最大月份数
        max_month = 0
        for delta_file in delta_files:
            with open(delta_file, 'r') as f:
                delta_data = json.load(f)
                month = delta_data['month']
                max_month = max(max_month, month)
        
        # 为每个月创建帧
        for month in range(1, max_month + 1):
            # 复制前一帧的状态
            prev_buildings = self.building_frames[month - 1].copy()
            
            # 查找该月的增量数据
            delta_file = os.path.join(self.output_dir, f'building_delta_month_{month:02d}.json')
            if os.path.exists(delta_file):
                with open(delta_file, 'r') as f:
                    delta_data = json.load(f)
                    new_buildings = delta_data['new_buildings']
                    
                    # 添加新建筑
                    for building in new_buildings:
                        building_type = building['building_type']
                        if building_type in prev_buildings:
                            prev_buildings[building_type].append(building)
                    
                    print(f"  📊 加载第{month}个月增量建筑数据 ({len(new_buildings)}个新建筑)")
            else:
                print(f"  📊 第{month}个月无增量数据")
            
            self.building_frames.append(prev_buildings)
        
        self.max_month = len(self.building_frames) - 1
    
    def _load_land_price_frames(self):
        """加载地价场帧数据"""
        land_price_files = sorted(glob.glob(os.path.join(self.output_dir, 'land_price_field_month_*.npy')))
        
        for land_price_file in land_price_files:
            land_price_field = np.load(land_price_file)
            self.land_price_frames.append(land_price_field)
        
        print(f"  📊 加载了 {len(self.land_price_frames)} 个地价场帧")
    
    def create_window_animation(self, fps: int = 2):
        """创建窗口动画"""
        print("🎬 创建窗口动画...")
        
        # 创建图形和子图
        fig, (ax_main, ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 动画状态
        self.animation_running = True
        self.current_frame = 0
        
        # 初始化动画
        def animate(frame):
            if frame >= len(self.building_frames) or not self.animation_running:
                return
            
            # 清除子图
            ax_main.clear()
            ax_stats.clear()
            
            # 绘制主图
            self._plot_main_frame(ax_main, frame)
            
            # 绘制建筑统计
            self._plot_building_stats(ax_stats, frame)
            
            self.current_frame = frame
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=self.max_month + 1, 
                                     interval=1000//fps, repeat=True, blit=False)
        
        # 添加键盘控制
        def on_key(event):
            if event.key == ' ':  # 空格键暂停/播放
                self.animation_running = not self.animation_running
                print("▶️ 播放" if self.animation_running else "⏸️ 暂停")
            elif event.key == 'r':  # R键重置
                self.current_frame = 0
                self.animation_running = False
                animate(0)
                print("🔄 重置到第0帧")
            elif event.key == 'left':  # 左箭头键上一帧
                self.current_frame = max(0, self.current_frame - 1)
                self.animation_running = False
                animate(self.current_frame)
            elif event.key == 'right':  # 右箭头键下一帧
                self.current_frame = min(self.max_month, self.current_frame + 1)
                self.animation_running = False
                animate(self.current_frame)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # 显示窗口
        plt.tight_layout()
        
        # 添加说明文本
        fig.text(0.5, 0.02, '控制说明: 空格键=播放/暂停, R键=重置, 左右箭头=单帧控制', 
                fontsize=10, ha='center')
        
        print("✅ 窗口动画已创建")
        print("   控制说明:")
        print("   - 空格键: 播放/暂停")
        print("   - R键: 重置到第0帧")
        print("   - 左右箭头: 单帧控制")
        
        plt.show()
    
    def _plot_main_frame(self, ax, frame: int):
        """绘制主帧"""
        if frame >= len(self.building_frames):
            return
        
        buildings = self.building_frames[frame]
        
        # 绘制地价场背景
        if frame < len(self.land_price_frames):
            land_price_field = self.land_price_frames[frame]
            im = ax.imshow(land_price_field, cmap='YlOrRd', alpha=0.6, 
                          extent=[0, land_price_field.shape[1], 0, land_price_field.shape[0]])
        
        # 绘制建筑
        total_buildings = 0
        for building_type, building_list in buildings.items():
            if not building_list:
                continue
            
            x_coords = [building['xy'][0] for building in building_list]
            y_coords = [building['xy'][1] for building in building_list]
            
            color = self.building_colors.get(building_type, '#666666')
            marker = self.building_markers.get(building_type, 'o')
            size = self.building_sizes.get(building_type, 20)
            
            ax.scatter(x_coords, y_coords, c=color, marker=marker, s=size, 
                      alpha=0.8, edgecolors='black', linewidth=0.5, 
                      label=f'{building_type} ({len(building_list)})')
            total_buildings += len(building_list)
        
        # 绘制政府骨架
        self._plot_government_backbone(ax)
        
        # 绘制等值线
        if frame < len(self.land_price_frames):
            self._plot_land_price_contours(ax, frame)
        
        # 设置图例
        ax.legend(loc='upper right', fontsize=10)
        
        # 设置标题
        ax.set_title(f'城市演化 - 第{frame}个月 (总计{total_buildings}个建筑)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X坐标 (像素)')
        ax.set_ylabel('Y坐标 (像素)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        ax.set_xlim(0, 110)
        ax.set_ylim(0, 110)
    
    def _plot_government_backbone(self, ax):
        """绘制政府骨架"""
        # 主干道
        ax.axhline(y=55, color='black', linewidth=3, alpha=0.8, label='主干道')
        
        # 商业枢纽
        ax.scatter(37, 55, c='red', marker='*', s=200, alpha=0.9, 
                  edgecolors='black', linewidth=2, label='商业枢纽')
        
        # 工业枢纽
        ax.scatter(73, 55, c='blue', marker='*', s=200, alpha=0.9, 
                  edgecolors='black', linewidth=2, label='工业枢纽')
    
    def _plot_land_price_contours(self, ax, frame: int):
        """绘制地价等值线"""
        if frame >= len(self.land_price_frames):
            return
        
        land_price_field = self.land_price_frames[frame]
        
        # 绘制等值线
        contours = ax.contour(land_price_field, levels=[0.2, 0.4, 0.6, 0.8], 
                             colors=['white', 'yellow', 'orange', 'red'], 
                             linewidths=1, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    def _plot_building_stats(self, ax, frame: int):
        """绘制建筑统计"""
        if frame >= len(self.building_frames):
            return
        
        buildings = self.building_frames[frame]
        
        building_types = []
        building_counts = []
        colors = []
        
        for building_type, building_list in buildings.items():
            if building_type != 'public':  # 排除公共建筑
                building_types.append(building_type)
                building_counts.append(len(building_list))
                colors.append(self.building_colors.get(building_type, '#666666'))
        
        if building_types:
            bars = ax.bar(building_types, building_counts, color=colors, alpha=0.7)
            ax.set_ylabel('建筑数量')
            ax.set_title(f'建筑统计 - 第{frame}个月')
            
            # 在柱状图上显示数值
            for bar, count in zip(bars, building_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # 设置y轴范围
            max_count = max(building_counts) if building_counts else 1
            ax.set_ylim(0, max_count * 1.2)
    
    def create_animation(self, output_file: str = 'city_evolution_v3_3.gif', 
                        fps: int = 2, dpi: int = 100):
        """创建GIF动画"""
        print(f"🎬 创建GIF动画: {output_file}")
        
        # 创建图形和子图
        fig, (ax_main, ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 初始化动画
        def animate(frame):
            if frame >= len(self.building_frames):
                return
            
            # 清除子图
            ax_main.clear()
            ax_stats.clear()
            
            # 绘制主图
            self._plot_main_frame(ax_main, frame)
            
            # 绘制建筑统计
            self._plot_building_stats(ax_stats, frame)
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=self.max_month + 1, 
                                     interval=1000//fps, repeat=True, blit=False)
        
        # 保存动画
        anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
        print(f"✅ GIF动画已保存: {output_file}")
        
        plt.close()
    
    def create_comparison_plot(self, months: List[int] = [0, 6, 12, 18, 23]):
        """创建对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, month in enumerate(months):
            if month > self.max_month:
                continue
            
            ax = axes[i]
            
            # 绘制主帧
            self._plot_main_frame(ax, month)
            ax.set_title(f'第{month}个月', fontsize=12, fontweight='bold')
        
        # 隐藏多余的子图
        for i in range(len(months), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('city_evolution_comparison_v3_3.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ 对比图已保存: city_evolution_comparison_v3_3.png")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强城市模拟系统 v3.3 可视化')
    parser.add_argument('--output_dir', type=str, default='enhanced_simulation_v3_3_output',
                       help='模拟输出目录')
    parser.add_argument('--window_mode', action='store_true',
                       help='启动窗口动画模式')
    parser.add_argument('--gif', action='store_true',
                       help='创建GIF动画')
    parser.add_argument('--comparison', action='store_true',
                       help='创建对比图')
    
    args = parser.parse_args()
    
    print("🎬 增强城市模拟系统 v3.3 可视化")
    print("   展示地价场驱动的建筑生长过程")
    
    # 创建可视化器
    visualizer = CityEvolutionVisualizerV3_3(args.output_dir)
    
    # 加载数据
    visualizer.load_simulation_data()
    
    if visualizer.max_month < 0:
        print("❌ 未找到模拟数据，请先运行 enhanced_city_simulation_v3_3.py")
        return
    
    # 根据参数选择输出模式
    if args.window_mode:
        print("\n🎬 启动窗口动画...")
        visualizer.create_window_animation(fps=2)
    
    if args.gif:
        print("\n🎬 创建GIF动画...")
        visualizer.create_animation('city_evolution_v3_3.gif', fps=2)
        print("✅ GIF动画已保存: city_evolution_v3_3.gif")
    
    if args.comparison:
        print("\n📊 创建对比图...")
        visualizer.create_comparison_plot()
        print("✅ 对比图已保存: city_evolution_comparison_v3_3.png")
    
    # 如果没有指定任何模式，默认启动窗口动画
    if not any([args.window_mode, args.gif, args.comparison]):
        print("\n🎬 启动窗口动画...")
        visualizer.create_window_animation(fps=2)
    
    print("\n🎉 可视化完成！")

if __name__ == "__main__":
    main()