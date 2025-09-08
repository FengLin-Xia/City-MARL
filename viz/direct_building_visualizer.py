#!/usr/bin/env python3
"""
Direct Building Visualizer for v2.3
直接显示建筑数据和等值线，不依赖PNG图片
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
import numpy as np
import json
import os
import glob
import math

class DirectBuildingVisualizer:
    """直接建筑可视化器"""
    
    def __init__(self):
        self.output_dir = 'enhanced_simulation_v2_3_output'
        self.current_frame = 0
        self.total_frames = 0
        self.building_data = {}
        self.sdf_data = {}
        
        # 地图配置
        self.map_size = [256, 256]
        self.trunk_road = [[40, 128], [216, 128]]
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.setup_ui()
        self.load_data()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主标题
        self.fig.suptitle('Enhanced City Simulation v2.3 - Direct Building Visualization', 
                         fontsize=16, fontweight='bold')
        
        # 控制按钮
        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.04])
        ax_play = plt.axes([0.25, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.4, 0.05, 0.1, 0.04])
        ax_analyze = plt.axes([0.55, 0.05, 0.15, 0.04])
        
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_play = Button(ax_play, 'Play/Pause')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_analyze = Button(ax_analyze, 'Analyze Buildings')
        
        # 绑定事件
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_analyze.on_clicked(self.analyze_buildings)
        
        # 帧滑块
        ax_slider = plt.axes([0.1, 0.12, 0.6, 0.02])
        self.slider = Slider(ax_slider, 'Frame', 0, 1, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)
        
        # 状态显示
        self.status_text = self.fig.text(0.75, 0.12, '', fontsize=10)
        
        # 播放状态
        self.is_playing = False
        self.animation = None
        
    def load_data(self):
        """加载数据"""
        # 加载建筑位置数据
        self.load_building_positions()
        
        # 加载SDF数据
        self.load_sdf_data()
        
        # 显示第一帧
        if self.total_frames > 0:
            self.show_frame(0)
    
    def load_building_positions(self):
        """加载建筑位置数据"""
        building_files = glob.glob(f'{self.output_dir}/building_positions_month_*.json')
        
        for file_path in building_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    month = data['timestamp']
                    self.building_data[month] = data
            except Exception as e:
                print(f"Failed to load building data {file_path}: {e}")
        
        self.total_frames = len(self.building_data)
        print(f"Loaded building data for {self.total_frames} months")
        
        # 更新滑块范围
        if self.total_frames > 0:
            self.slider.valmax = self.total_frames - 1
            self.slider.valstep = 1
    
    def load_sdf_data(self):
        """加载SDF数据"""
        sdf_files = glob.glob(f'{self.output_dir}/sdf_field_month_*.json')
        
        for file_path in sdf_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    month = data['month']
                    self.sdf_data[month] = data
            except Exception as e:
                print(f"Failed to load SDF data {file_path}: {e}")
        
        print(f"Loaded SDF data for {len(self.sdf_data)} months")
    
    def show_frame(self, frame_index: int):
        """显示指定帧"""
        if frame_index < 0 or frame_index >= self.total_frames:
            return
        
        self.current_frame = frame_index
        
        # 清除当前图像
        self.ax.clear()
        
        # 设置坐标轴
        self.ax.set_xlim(0, self.map_size[0])
        self.ax.set_ylim(0, self.map_size[1])
        self.ax.set_aspect('equal')
        
        # 绘制主干道
        self.draw_trunk_road()
        
        # 绘制交通枢纽
        self.draw_transport_hubs()
        
        # 绘制建筑
        self.draw_buildings(frame_index)
        
        # 绘制等值线（如果有SDF数据）
        self.draw_isocontours(frame_index)
        
        # 设置标题
        month = frame_index
        self.ax.set_title(f'Month {month:02d} - Direct Building Visualization', fontsize=14)
        
        # 显示建筑统计
        self.show_building_stats(month)
        
        # 显示图例
        self.show_legend()
        
        # 更新状态
        self.update_status()
        
        # 更新滑块
        self.slider.set_val(frame_index)
        
        # 刷新显示
        self.fig.canvas.draw_idle()
    
    def draw_trunk_road(self):
        """绘制主干道"""
        x_coords = [self.trunk_road[0][0], self.trunk_road[1][0]]
        y_coords = [self.trunk_road[0][1], self.trunk_road[1][1]]
        self.ax.plot(x_coords, y_coords, 'k-', linewidth=3, alpha=0.7, label='Trunk Road')
    
    def draw_transport_hubs(self):
        """绘制交通枢纽"""
        for i, hub in enumerate(self.trunk_road):
            self.ax.plot(hub[0], hub[1], 'o', markersize=10, color='blue', 
                        markeredgecolor='black', markeredgewidth=2, label=f'Hub {chr(65+i)}' if i == 0 else "")
    
    def draw_buildings(self, frame_index: int):
        """绘制建筑"""
        month_key = f'month_{frame_index:02d}'
        if month_key not in self.building_data:
            return
        
        data = self.building_data[month_key]
        buildings = data['buildings']
        
        for building in buildings:
            pos = building['position']
            building_type = building['type']
            sdf_value = building.get('sdf_value', 0.0)
            
            if building_type == 'residential':
                # 住宅建筑：黄色方形
                rect = patches.Rectangle((pos[0]-2, pos[1]-2), 4, 4, 
                                       color='yellow', alpha=0.8, edgecolor='black')
                self.ax.add_patch(rect)
                
            elif building_type == 'commercial':
                # 商业建筑：橙色圆形
                circle = patches.Circle((pos[0], pos[1]), radius=3, 
                                      color='orange', alpha=0.8, edgecolor='black')
                self.ax.add_patch(circle)
                
            elif building_type == 'public':
                # 公共建筑：青色三角形
                triangle = patches.RegularPolygon((pos[0], pos[1]), numVertices=3, radius=4,
                                                color='cyan', alpha=0.8, edgecolor='black')
                self.ax.add_patch(triangle)
    
    def draw_isocontours(self, frame_index: int):
        """绘制等值线"""
        if frame_index not in self.sdf_data:
            return
        
        sdf_data = self.sdf_data[frame_index]
        sdf_field = np.array(sdf_data['sdf_field'])
        
        # 绘制商业等值线（红色虚线）
        commercial_levels = [0.85, 0.70, 0.55]
        for level in commercial_levels:
            try:
                contour = plt.contour(sdf_field, levels=[level], colors='red', 
                                    linestyles='dashed', alpha=0.6, linewidths=1)
            except:
                pass
        
        # 绘制住宅等值线（蓝色虚线）
        residential_levels = [0.55, 0.40, 0.25]
        for level in residential_levels:
            try:
                contour = plt.contour(sdf_field, levels=[level], colors='blue', 
                                    linestyles='dashed', alpha=0.6, linewidths=1)
            except:
                pass
    
    def show_building_stats(self, month: int):
        """显示建筑统计"""
        month_key = f'month_{month:02d}'
        if month_key in self.building_data:
            data = self.building_data[month_key]
            buildings = data['buildings']
            
            # 统计建筑类型
            residential_count = len([b for b in buildings if b['type'] == 'residential'])
            commercial_count = len([b for b in buildings if b['type'] == 'commercial'])
            public_count = len([b for b in buildings if b['type'] == 'public'])
            
            # 计算平均SDF值
            sdf_values = [b.get('sdf_value', 0.0) for b in buildings]
            avg_sdf = np.mean(sdf_values) if sdf_values else 0.0
            
            # 显示统计信息
            stats_text = (f'Buildings:\n'
                         f'Residential: {residential_count}\n'
                         f'Commercial: {commercial_count}\n'
                         f'Public: {public_count}\n'
                         f'Avg SDF: {avg_sdf:.3f}')
            
            self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def show_legend(self):
        """显示图例"""
        # 创建图例
        legend_elements = [
            plt.Line2D([0], [0], color='black', linewidth=3, label='Trunk Road'),
            plt.Line2D([0], [0], marker='o', color='blue', markersize=10, label='Transport Hub'),
            patches.Patch(color='yellow', label='Residential'),
            patches.Patch(color='orange', label='Commercial'),
            patches.Patch(color='cyan', label='Public'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Commercial Isocontour'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label='Residential Isocontour')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    def update_status(self):
        """更新状态显示"""
        if self.total_frames > 0:
            month = self.current_frame
            status = f'Current: Month {month:02d} / {self.total_frames-1}'
            self.status_text.set_text(status)
    
    def prev_frame(self, event):
        """上一帧"""
        if self.total_frames > 0:
            new_frame = max(0, self.current_frame - 1)
            self.show_frame(new_frame)
    
    def next_frame(self, event):
        """下一帧"""
        if self.total_frames > 0:
            new_frame = min(self.total_frames - 1, self.current_frame + 1)
            self.show_frame(new_frame)
    
    def toggle_play(self, event):
        """切换播放/暂停"""
        if self.is_playing:
            self.stop_animation()
        else:
            self.start_animation()
    
    def start_animation(self):
        """开始动画"""
        if self.total_frames <= 1:
            return
        
        self.is_playing = True
        self.btn_play.label.set_text('Pause')
        
        # 创建动画
        def animate(frame):
            self.show_frame(frame)
            return []
        
        import matplotlib.animation as animation
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=self.total_frames,
            interval=1000, repeat=True, blit=False
        )
    
    def stop_animation(self):
        """停止动画"""
        self.is_playing = False
        self.btn_play.label.set_text('Play')
        
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
    
    def on_slider_change(self, val):
        """滑块值改变"""
        frame_index = int(val)
        if frame_index != self.current_frame:
            self.show_frame(frame_index)
    
    def analyze_buildings(self, event):
        """分析建筑分布"""
        if not self.building_data:
            print("No building data available for analysis")
            return
        
        # 创建分析窗口
        self.create_analysis_window()
    
    def create_analysis_window(self):
        """创建分析窗口"""
        analysis_fig, analysis_axes = plt.subplots(2, 2, figsize=(16, 12))
        analysis_fig.suptitle('Building Distribution Analysis - v2.3', fontsize=16, fontweight='bold')
        
        # 1. 建筑类型分布
        ax1 = analysis_axes[0, 0]
        self.plot_building_types(ax1)
        
        # 2. 建筑位置分布
        ax2 = analysis_axes[0, 1]
        self.plot_building_positions(ax2)
        
        # 3. SDF值分布
        ax3 = analysis_axes[1, 0]
        self.plot_sdf_distribution(ax3)
        
        # 4. 建筑增长趋势
        ax4 = analysis_axes[1, 1]
        self.plot_growth_trend(ax4)
        
        plt.tight_layout()
        plt.show()
    
    def plot_building_types(self, ax):
        """绘制建筑类型分布"""
        # 统计所有月份的建筑类型
        all_residential = []
        all_commercial = []
        all_public = []
        
        for month_key, data in self.building_data.items():
            buildings = data['buildings']
            residential_count = len([b for b in buildings if b['type'] == 'residential'])
            commercial_count = len([b for b in buildings if b['type'] == 'commercial'])
            public_count = len([b for b in buildings if b['type'] == 'public'])
            
            all_residential.append(residential_count)
            all_commercial.append(commercial_count)
            all_public.append(public_count)
        
        months = list(range(len(all_residential)))
        
        ax.plot(months, all_residential, 'o-', color='yellow', label='Residential', linewidth=2)
        ax.plot(months, all_commercial, 's-', color='orange', label='Commercial', linewidth=2)
        ax.plot(months, all_public, '^-', color='cyan', label='Public', linewidth=2)
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Buildings')
        ax.set_title('Building Type Distribution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_building_positions(self, ax):
        """绘制建筑位置分布"""
        # 获取最后一帧的建筑位置
        last_month = max(self.building_data.keys())
        data = self.building_data[last_month]
        buildings = data['buildings']
        
        # 绘制建筑位置
        for building in buildings:
            pos = building['position']
            building_type = building['type']
            
            if building_type == 'residential':
                ax.plot(pos[0], pos[1], 's', color='yellow', markersize=4, alpha=0.7)
            elif building_type == 'commercial':
                ax.plot(pos[0], pos[1], 'o', color='orange', markersize=4, alpha=0.7)
            elif building_type == 'public':
                ax.plot(pos[0], pos[1], '^', color='cyan', markersize=6, alpha=0.8)
        
        # 绘制主干道
        x_coords = [self.trunk_road[0][0], self.trunk_road[1][0]]
        y_coords = [self.trunk_road[0][1], self.trunk_road[1][1]]
        ax.plot(x_coords, y_coords, 'k-', linewidth=2, alpha=0.7)
        
        # 绘制交通枢纽
        for hub in self.trunk_road:
            ax.plot(hub[0], hub[1], 'o', markersize=8, color='blue', 
                   markeredgecolor='black', markeredgewidth=2)
        
        ax.set_xlim(0, self.map_size[0])
        ax.set_ylim(0, self.map_size[1])
        ax.set_aspect('equal')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Final Building Distribution')
        ax.grid(True, alpha=0.3)
    
    def plot_sdf_distribution(self, ax):
        """绘制SDF值分布"""
        # 收集所有SDF值
        all_sdf_values = []
        for month_key, data in self.building_data.items():
            buildings = data['buildings']
            sdf_values = [b.get('sdf_value', 0.0) for b in buildings]
            all_sdf_values.extend(sdf_values)
        
        if all_sdf_values:
            ax.hist(all_sdf_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('SDF Value')
            ax.set_ylabel('Frequency')
            ax.set_title('SDF Value Distribution')
            ax.grid(True, alpha=0.3)
    
    def plot_growth_trend(self, ax):
        """绘制增长趋势"""
        # 统计总建筑数量
        total_buildings = []
        for month_key, data in self.building_data.items():
            total_buildings.append(len(data['buildings']))
        
        months = list(range(len(total_buildings)))
        
        ax.plot(months, total_buildings, 'o-', color='green', linewidth=2, markersize=6)
        ax.set_xlabel('Month')
        ax.set_ylabel('Total Buildings')
        ax.set_title('Building Growth Trend')
        ax.grid(True, alpha=0.3)

def main():
    """主函数"""
    print("🏙️ Direct Building Visualizer v2.3")
    print("=" * 50)
    print("🎯 Features:")
    print("  • Direct building visualization")
    print("  • Isocontour display")
    print("  • Building distribution analysis")
    print("  • Interactive playback controls")
    print("=" * 50)
    
    # 创建并显示可视化器
    visualizer = DirectBuildingVisualizer()
    plt.show()

if __name__ == "__main__":
    main()


