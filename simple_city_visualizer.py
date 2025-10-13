#!/usr/bin/env python3
"""
简化城市可视化器
只显示最终城市布局，支持逐帧播放
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation

# 设置matplotlib使用英文，避免中文字体问题
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class SimpleCityVisualizer:
    """简化城市可视化器"""
    
    def __init__(self):
        self.output_dir = Path('enhanced_simulation_output')
        self.load_data()
    
    def load_data(self):
        """加载模拟数据"""
        # 加载最终总结
        with open(self.output_dir / 'final_summary.json', 'r', encoding='utf-8') as f:
            self.final_summary = json.load(f)
        
        # 加载城市状态
        with open(self.output_dir / 'city_state_output.json', 'r', encoding='utf-8') as f:
            self.city_state = json.load(f)
    
    def plot_final_city_layout(self):
        """绘制最终城市布局"""
        # 创建图形，使用更合理的尺寸
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 设置背景
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制主干道
        trunk_road = [[40, 128], [216, 128]]
        ax.plot([trunk_road[0][0], trunk_road[1][0]], [trunk_road[0][1], trunk_road[1][1]], 
                'k-', linewidth=10, alpha=0.8, label='Main Road')
        
        # 绘制交通枢纽
        ax.scatter(40, 128, s=300, c='darkblue', marker='o', 
                  label='Transport Hub A', zorder=10, edgecolors='black', linewidth=2)
        ax.scatter(216, 128, s=300, c='darkblue', marker='o', 
                  label='Transport Hub B', zorder=10, edgecolors='black', linewidth=2)
        
        # 绘制核心点
        core_point = [128, 128]
        ax.scatter(core_point[0], core_point[1], s=250, c='red', marker='*', 
                  label='City Core', zorder=10, edgecolors='black', linewidth=2)
        
        # 绘制建筑
        buildings = self.city_state['buildings']
        
        # 公共建筑
        for building in buildings['public']:
            ax.scatter(building['xy'][0], building['xy'][1], s=200, c='blue', 
                      marker='s', label='Public Buildings' if building == buildings['public'][0] else "",
                      edgecolors='black', linewidth=1.5, zorder=8)
        
        # 住宅建筑
        for building in buildings['residential']:
            ax.scatter(building['xy'][0], building['xy'][1], s=150, c='green', 
                      marker='o', label='Residential Buildings' if building == buildings['residential'][0] else "",
                      edgecolors='black', linewidth=1.5, zorder=8)
        
        # 商业建筑
        for building in buildings['commercial']:
            ax.scatter(building['xy'][0], building['xy'][1], s=180, c='orange', 
                      marker='^', label='Commercial Buildings' if building == buildings['commercial'][0] else "",
                      edgecolors='black', linewidth=1.5, zorder=8)
        
        # 绘制居民分布
        residents = self.city_state['residents']
        resident_x = [r['pos'][0] for r in residents]
        resident_y = [r['pos'][1] for r in residents]
        ax.scatter(resident_x, resident_y, s=30, c='purple', alpha=0.7, 
                  label=f'Residents ({len(residents)} people)', zorder=6)
        
        # 添加地价热力图
        self._add_land_price_heatmap(ax)
        
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('Enhanced City Simulation - Final City Layout', fontsize=16, fontweight='bold')
        
        # 简化图例，放在图形外部
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # 添加统计信息
        stats_text = f"""City Statistics:
Population: {len(residents)} people
Public Buildings: {len(buildings['public'])} units
Residential Buildings: {len(buildings['residential'])} units
Commercial Buildings: {len(buildings['commercial'])} units"""
        
        ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 调整布局
        plt.subplots_adjust(right=0.8)
        plt.savefig('enhanced_simulation_output/final_city_layout_enhanced.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _add_land_price_heatmap(self, ax):
        """添加地价热力图"""
        # 简化的地价热力图
        x = np.linspace(0, 256, 50)
        y = np.linspace(0, 256, 50)
        X, Y = np.meshgrid(x, y)
        
        # 基于距离交通枢纽的地价计算
        transport_hubs = [[40, 128], [216, 128]]  # 交通枢纽点
        core_point = [128, 128]  # 城市核心点
        
        # 计算到最近交通枢纽的距离
        min_hub_distance = np.full_like(X, float('inf'))
        for hub in transport_hubs:
            distance = np.sqrt((X - hub[0])**2 + (Y - hub[1])**2)
            min_hub_distance = np.minimum(min_hub_distance, distance)
        
        # 计算到核心点的距离
        core_distance = np.sqrt((X - core_point[0])**2 + (Y - core_point[1])**2)
        
        # 计算到主干道的距离
        trunk_distance = np.abs(Y - 128)
        
        # 综合地价计算
        hub_factor = np.exp(-min_hub_distance / 100)
        core_factor = np.exp(-core_distance / 150)
        trunk_factor = np.exp(-trunk_distance / 80)
        
        land_price = 100 * (0.5 * hub_factor + 0.3 * core_factor + 0.2 * trunk_factor) + 50
        
        # 绘制热力图
        im = ax.contourf(X, Y, land_price, levels=20, alpha=0.3, cmap='hot')
        plt.colorbar(im, ax=ax, label='Land Price', shrink=0.8)
    
    def create_frame_animation(self):
        """创建逐帧播放动画"""
        print("🎬 Creating Frame-by-Frame Animation...")
        
        # 获取所有图片文件
        image_files = sorted(self.output_dir.glob('images/day_*.png'))
        
        if not image_files:
            print("❌ No image files found")
            return
        
        print(f"📁 Found {len(image_files)} frame images")
        
        # 创建动画
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        def animate(frame):
            ax.clear()
            ax.axis('off')
            
            # 加载图片
            img = plt.imread(str(image_files[frame]))
            ax.imshow(img)
            
            # 添加标题
            day = int(image_files[frame].stem.split('_')[1])
            ax.set_title(f'City Development - Day {day}', fontsize=16, fontweight='bold')
            
            # 添加进度信息
            progress = f'Frame {frame + 1}/{len(image_files)}'
            ax.text(0.02, 0.98, progress, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 创建动画，设置较慢的播放速度
        anim = animation.FuncAnimation(
            fig, animate, frames=len(image_files),
            interval=1000, repeat=True  # 每帧1秒，更慢的播放速度
        )
        
        # 保存动画
        anim.save('enhanced_simulation_output/city_evolution_frames.gif', writer='pillow', fps=1)
        print("✅ Animation saved: enhanced_simulation_output/city_evolution_frames.gif")
        
        plt.show()
    
    def show_summary(self):
        """显示简要总结"""
        print("📊 Enhanced City Simulation - Final Results")
        print("=" * 50)
        
        summary = self.final_summary['simulation_summary']
        residents = self.city_state['residents']
        buildings = self.city_state['buildings']
        
        print(f"🏙️ Simulation Duration: {summary['total_days']} days")
        print(f"👥 Final Population: {len(residents)} people")
        print(f"🏗️ Total Buildings: {sum(len(buildings[k]) for k in buildings)} units")
        print(f"   - Public Buildings: {len(buildings['public'])} units")
        print(f"   - Residential Buildings: {len(buildings['residential'])} units")
        print(f"   - Commercial Buildings: {len(buildings['commercial'])} units")
        
        land_prices = summary['land_price_summary']
        print(f"\n💰 Land Price Statistics:")
        print(f"   - Average Land Price: {land_prices['avg_price']:.1f}")
        print(f"   - Maximum Land Price: {land_prices['max_price']:.1f}")
        print(f"   - Minimum Land Price: {land_prices['min_price']:.1f}")
    
    def run_visualization(self):
        """运行可视化"""
        print("🎨 Simple City Visualizer")
        print("=" * 40)
        
        # 显示总结
        self.show_summary()
        
        # 绘制最终城市布局
        print("\n🗺️ Generating Final City Layout...")
        self.plot_final_city_layout()
        
        # 询问是否创建逐帧动画
        response = input("\n🎬 Create Frame-by-Frame Animation? (y/n): ").lower()
        if response == 'y':
            self.create_frame_animation()
        
        print("\n✅ Visualization Completed!")

def main():
    """主函数"""
    visualizer = SimpleCityVisualizer()
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
