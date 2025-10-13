#!/usr/bin/env python3
"""
v2.3 模拟结果可视化播放器
支持等值线显示和建筑位置分析
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import glob

class V2_3Visualizer:
    """v2.3模拟结果可视化播放器"""
    
    def __init__(self):
        self.output_dir = 'enhanced_simulation_v2_3_output'
        self.images_dir = f'{self.output_dir}/images'
        self.current_frame = 0
        self.total_frames = 0
        self.image_files = []
        self.building_data = {}
        
        # 创建图形界面
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.setup_ui()
        self.load_data()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主标题
        self.fig.suptitle('Enhanced City Simulation v2.3 - 等值线建筑生成可视化', 
                         fontsize=16, fontweight='bold')
        
        # 控制按钮
        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.04])
        ax_play = plt.axes([0.25, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.4, 0.05, 0.1, 0.04])
        ax_analyze = plt.axes([0.55, 0.05, 0.15, 0.04])
        
        self.btn_prev = Button(ax_prev, '上一帧')
        self.btn_play = Button(ax_play, '播放/暂停')
        self.btn_next = Button(ax_next, '下一帧')
        self.btn_analyze = Button(ax_analyze, '分析建筑分布')
        
        # 绑定事件
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_analyze.on_clicked(self.analyze_buildings)
        
        # 帧数滑块
        ax_slider = plt.axes([0.1, 0.12, 0.6, 0.02])
        self.slider = Slider(ax_slider, '帧数', 0, 1, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)
        
        # 状态显示
        self.status_text = self.fig.text(0.75, 0.12, '', fontsize=10)
        
        # 播放状态
        self.is_playing = False
        self.animation = None
        
    def load_data(self):
        """加载数据"""
        # 加载图像文件
        if os.path.exists(self.images_dir):
            self.image_files = sorted(glob.glob(f'{self.images_dir}/month_*.png'))
            self.total_frames = len(self.image_files)
            print(f"📁 找到 {self.total_frames} 个图像文件")
            
            # 更新滑块范围
            if self.total_frames > 0:
                self.slider.valmax = self.total_frames - 1
                self.slider.valstep = 1
        else:
            print(f"❌ 图像目录不存在: {self.images_dir}")
            return
        
        # 加载建筑位置数据
        self.load_building_positions()
        
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
                print(f"⚠️ 加载建筑数据失败 {file_path}: {e}")
        
        print(f"📊 加载了 {len(self.building_data)} 个月份的建筑数据")
    
    def show_frame(self, frame_index: int):
        """显示指定帧"""
        if frame_index < 0 or frame_index >= self.total_frames:
            return
        
        self.current_frame = frame_index
        
        # 清除当前图像
        self.ax.clear()
        
        # 加载并显示图像
        if os.path.exists(self.image_files[frame_index]):
            img = plt.imread(self.image_files[frame_index])
            self.ax.imshow(img)
            
            # 设置标题
            month = frame_index
            self.ax.set_title(f'Month {month:02d} - 等值线建筑生成可视化', fontsize=14)
            
            # 显示建筑统计
            self.show_building_stats(month)
            
            # 显示等值线信息
            self.show_isocontour_info(month)
        
        # 更新状态
        self.update_status()
        
        # 更新滑块
        self.slider.set_val(frame_index)
        
        # 刷新显示
        self.fig.canvas.draw_idle()
    
    def show_building_stats(self, month: int):
        """显示建筑统计信息"""
        month_key = f'month_{month:02d}'
        if month_key in self.building_data:
            data = self.building_data[month_key]
            buildings = data['buildings']
            
            # 统计各类型建筑
            residential_count = len([b for b in buildings if b['type'] == 'residential'])
            commercial_count = len([b for b in buildings if b['type'] == 'commercial'])
            public_count = len([b for b in buildings if b['type'] == 'public'])
            
            # 显示统计信息
            stats_text = f'建筑统计:\n住宅: {residential_count}\n商业: {commercial_count}\n公共: {public_count}'
            self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def show_isocontour_info(self, month: int):
        """显示等值线信息"""
        # 显示等值线说明
        info_text = ('等值线说明:\n'
                    '🔴 红色虚线: 商业建筑等值线\n'
                    '🔵 蓝色虚线: 住宅建筑等值线\n'
                    '🏢 橙色圆点: 商业建筑\n'
                    '🏠 黄色方块: 住宅建筑\n'
                    '🏛️ 青色标记: 公共设施')
        
        self.ax.text(0.98, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def update_status(self):
        """更新状态显示"""
        if self.total_frames > 0:
            month = self.current_frame
            status = f'当前: Month {month:02d} / {self.total_frames-1}'
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
        self.btn_play.label.set_text('暂停')
        
        # 创建动画
        def animate(frame):
            self.show_frame(frame)
            return []
        
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=self.total_frames,
            interval=1000, repeat=True, blit=False
        )
    
    def stop_animation(self):
        """停止动画"""
        self.is_playing = False
        self.btn_play.label.set_text('播放')
        
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
            print("❌ 没有建筑数据可供分析")
            return
        
        # 创建分析窗口
        self.create_analysis_window()
    
    def create_analysis_window(self):
        """创建分析窗口"""
        analysis_fig, analysis_axes = plt.subplots(2, 2, figsize=(16, 12))
        analysis_fig.suptitle('建筑分布分析 - v2.3', fontsize=16, fontweight='bold')
        
        # 1. 建筑类型分布
        ax1 = analysis_axes[0, 0]
        self.plot_building_types(ax1)
        
        # 2. 建筑位置分布
        ax2 = analysis_axes[0, 1]
        self.plot_building_positions(ax2)
        
        # 3. SDF值分布
        ax3 = analysis_axes[1, 0]
        self.plot_sdf_distribution(ax3)
        
        # 4. 月度增长趋势
        ax4 = analysis_axes[1, 1]
        self.plot_growth_trend(ax4)
        
        plt.tight_layout()
        plt.show()
    
    def plot_building_types(self, ax):
        """绘制建筑类型分布"""
        # 统计所有建筑
        total_residential = 0
        total_commercial = 0
        total_public = 0
        
        for month_data in self.building_data.values():
            for building in month_data['buildings']:
                if building['type'] == 'residential':
                    total_residential += 1
                elif building['type'] == 'commercial':
                    total_commercial += 1
                elif building['type'] == 'public':
                    total_public += 1
        
        # 绘制饼图
        labels = ['住宅', '商业', '公共']
        sizes = [total_residential, total_commercial, total_public]
        colors = ['#F6C344', '#FD7E14', '#22A6B3']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('建筑类型分布')
    
    def plot_building_positions(self, ax):
        """绘制建筑位置分布"""
        # 获取最新月份的数据
        latest_month = max(self.building_data.keys())
        data = self.building_data[latest_month]
        
        # 绘制建筑位置
        for building in data['buildings']:
            pos = building['position']
            if building['type'] == 'residential':
                ax.scatter(pos[0], pos[1], c='yellow', s=50, marker='s', label='住宅', alpha=0.7)
            elif building['type'] == 'commercial':
                ax.scatter(pos[0], pos[1], c='orange', s=40, marker='o', label='商业', alpha=0.7)
            elif building['type'] == 'public':
                ax.scatter(pos[0], pos[1], c='cyan', s=60, marker='^', label='公共', alpha=0.7)
        
        # 绘制交通枢纽
        ax.scatter([40, 216], [128, 128], c='blue', s=200, marker='s', label='交通枢纽', alpha=0.8)
        
        # 绘制主干道
        ax.plot([40, 216], [128, 128], 'gray', linewidth=3, alpha=0.5, label='主干道')
        
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_title(f'建筑位置分布 ({latest_month})')
        
        # 只显示一次图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    def plot_sdf_distribution(self, ax):
        """绘制SDF值分布"""
        # 收集所有建筑的SDF值
        sdf_values = []
        for month_data in self.building_data.values():
            for building in month_data['buildings']:
                if 'sdf_value' in building and building['sdf_value'] > 0:
                    sdf_values.append(building['sdf_value'])
        
        if sdf_values:
            ax.hist(sdf_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('SDF值')
            ax.set_ylabel('建筑数量')
            ax.set_title('建筑SDF值分布')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无SDF数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('建筑SDF值分布')
    
    def plot_growth_trend(self, ax):
        """绘制月度增长趋势"""
        months = []
        residential_counts = []
        commercial_counts = []
        public_counts = []
        
        for month_key in sorted(self.building_data.keys()):
            month_num = int(month_key.split('_')[1])
            months.append(month_num)
            
            data = self.building_data[month_key]
            buildings = data['buildings']
            
            residential_counts.append(len([b for b in buildings if b['type'] == 'residential']))
            commercial_counts.append(len([b for b in buildings if b['type'] == 'commercial']))
            public_counts.append(len([b for b in buildings if b['type'] == 'public']))
        
        if months:
            ax.plot(months, residential_counts, 'o-', label='住宅', color='#F6C344', linewidth=2)
            ax.plot(months, commercial_counts, 's-', label='商业', color='#FD7E14', linewidth=2)
            ax.plot(months, public_counts, '^-', label='公共', color='#22A6B3', linewidth=2)
            
            ax.set_xlabel('月份')
            ax.set_ylabel('建筑数量')
            ax.set_title('建筑增长趋势')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无增长数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('建筑增长趋势')
    
    def run(self):
        """运行可视化器"""
        print("🎬 v2.3可视化播放器启动")
        print("📊 支持等值线显示和建筑分布分析")
        print("🎮 使用按钮控制播放，滑块快速跳转")
        
        plt.show()

def main():
    """主函数"""
    visualizer = V2_3Visualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
