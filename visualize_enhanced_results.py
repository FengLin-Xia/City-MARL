#!/usr/bin/env python3
"""
增强城市模拟结果可视化
展示地价驱动的城市发展结果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

class EnhancedResultsVisualizer:
    """增强模拟结果可视化器"""
    
    def __init__(self):
        self.output_dir = Path('enhanced_simulation_output')
        self.load_data()
    
    def load_data(self):
        """加载模拟数据"""
        # 加载最终总结
        with open(self.output_dir / 'final_summary.json', 'r', encoding='utf-8') as f:
            self.final_summary = json.load(f)
        
        # 加载每日统计
        with open(self.output_dir / 'daily_stats.json', 'r', encoding='utf-8') as f:
            self.daily_stats = json.load(f)
        
        # 加载地价演化
        with open(self.output_dir / 'land_price_evolution.json', 'r', encoding='utf-8') as f:
            self.land_price_evolution = json.load(f)
        
        # 加载城市状态
        with open(self.output_dir / 'city_state_output.json', 'r', encoding='utf-8') as f:
            self.city_state = json.load(f)
    
    # def plot_population_growth(self):
    #     """绘制人口增长曲线"""
    #     plt.figure(figsize=(12, 8))
    #     
    #     days = [stat['day'] for stat in self.daily_stats]
    #     population = [stat['population'] for stat in self.daily_stats]
    #     
    #     plt.plot(days, population, 'b-', linewidth=2, label='人口数量')
    #     plt.fill_between(days, population, alpha=0.3, color='blue')
    #     
    #     # 标记关键点
    #     plt.scatter(days[0], population[0], color='green', s=100, zorder=5, label=f'初始: {population[0]}人')
    #     plt.scatter(days[-1], population[-1], color='red', s=100, zorder=5, label=f'最终: {population[-1]}人')
    #     
    #     plt.xlabel('天数')
    #     plt.ylabel('人口数量')
    #     plt.title('城市人口增长趋势 (365天)')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     
    #     # 添加增长率信息
    #     growth_rate = (population[-1] - population[0]) / population[0] * 100
    #     plt.text(0.02, 0.98, f'总增长率: {growth_rate:.1f}%', 
    #             transform=plt.gca().transAxes, fontsize=12, 
    #             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    #     
    #     plt.tight_layout()
    #     plt.savefig('enhanced_simulation_output/population_growth.png', dpi=300, bbox_inches='tight')
    #     plt.show()
    
    def plot_building_evolution(self):
        """绘制建筑演化趋势"""
        plt.figure(figsize=(12, 8))
        
        days = [stat['day'] for stat in self.daily_stats]
        public = [stat['public_buildings'] for stat in self.daily_stats]
        residential = [stat['residential_buildings'] for stat in self.daily_stats]
        commercial = [stat['commercial_buildings'] for stat in self.daily_stats]
        
        plt.plot(days, public, 'g-', linewidth=2, label='Public Buildings', marker='o', markersize=4)
        plt.plot(days, residential, 'b-', linewidth=2, label='Residential Buildings', marker='s', markersize=4)
        plt.plot(days, commercial, 'r-', linewidth=2, label='Commercial Buildings', marker='^', markersize=4)
        
        plt.xlabel('Days')
        plt.ylabel('Building Count')
        plt.title('City Building Evolution Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加最终统计
        final_stats = self.final_summary['simulation_summary']['total_buildings']
        plt.text(0.02, 0.98, f'Final Building Distribution:\nPublic: {final_stats["public"]}\nResidential: {final_stats["residential"]}\nCommercial: {final_stats["commercial"]}', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('enhanced_simulation_output/building_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_land_price_evolution(self):
        """绘制地价演化趋势"""
        plt.figure(figsize=(12, 8))
        
        days = [i for i in range(len(self.land_price_evolution))]
        avg_prices = [stats['avg_price'] for stats in self.land_price_evolution]
        max_prices = [stats['max_price'] for stats in self.land_price_evolution]
        min_prices = [stats['min_price'] for stats in self.land_price_evolution]
        
        plt.plot(days, avg_prices, 'b-', linewidth=2, label='Average Land Price')
        plt.plot(days, max_prices, 'r-', linewidth=2, label='Maximum Land Price')
        plt.plot(days, min_prices, 'g-', linewidth=2, label='Minimum Land Price')
        
        plt.fill_between(days, min_prices, max_prices, alpha=0.2, color='gray', label='Land Price Range')
        
        plt.xlabel('Days')
        plt.ylabel('Land Price')
        plt.title('City Land Price Evolution Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加地价统计
        final_prices = self.final_summary['simulation_summary']['land_price_summary']
        plt.text(0.02, 0.98, f'Final Land Price Statistics:\nAverage: {final_prices["avg_price"]:.1f}\nMaximum: {final_prices["max_price"]:.1f}\nMinimum: {final_prices["min_price"]:.1f}', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('enhanced_simulation_output/land_price_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_city_layout(self):
        """绘制最终城市布局"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 设置背景
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制主干道（使用默认值）
        trunk_road = [[40, 128], [216, 128]]
        ax.plot([trunk_road[0][0], trunk_road[1][0]], [trunk_road[0][1], trunk_road[1][1]], 
                'k-', linewidth=8, alpha=0.7, label='Main Road')
        
        # 绘制核心点（使用默认值）
        core_point = [128, 128]
        ax.scatter(core_point[0], core_point[1], s=200, c='red', marker='*', 
                  label='City Core', zorder=10)
        
        # 绘制建筑
        buildings = self.city_state['buildings']
        
        # 公共建筑
        for building in buildings['public']:
            ax.scatter(building['xy'][0], building['xy'][1], s=150, c='blue', 
                      marker='s', label='Public Buildings' if building == buildings['public'][0] else "")
        
        # 住宅建筑
        for building in buildings['residential']:
            ax.scatter(building['xy'][0], building['xy'][1], s=100, c='green', 
                      marker='o', label='Residential Buildings' if building == buildings['residential'][0] else "")
        
        # 商业建筑
        for building in buildings['commercial']:
            ax.scatter(building['xy'][0], building['xy'][1], s=120, c='orange', 
                      marker='^', label='Commercial Buildings' if building == buildings['commercial'][0] else "")
        
        # 绘制居民分布
        residents = self.city_state['residents']
        resident_x = [r['pos'][0] for r in residents]
        resident_y = [r['pos'][1] for r in residents]
        ax.scatter(resident_x, resident_y, s=20, c='purple', alpha=0.6, 
                  label=f'Residents ({len(residents)} people)')
        
        # 添加地价热力图（简化版）
        self._add_land_price_heatmap(ax)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Enhanced City Simulation - Final City Layout')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 添加统计信息
        stats_text = f"""
City Statistics:
Population: {len(residents)} people
Public Buildings: {len(buildings['public'])} units
Residential Buildings: {len(buildings['residential'])} units
Commercial Buildings: {len(buildings['commercial'])} units
Development Stage: {self.final_summary['development_patterns']['development_stage']}
        """
        ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('enhanced_simulation_output/final_city_layout.png', dpi=300, bbox_inches='tight')
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
        plt.colorbar(im, ax=ax, label='Land Price')
    
    def create_animation(self):
        """创建城市发展动画"""
        print("🎬 正在创建城市发展动画...")
        
        # 获取所有图片文件
        image_files = sorted(self.output_dir.glob('images/day_*.png'))
        
        if not image_files:
            print("❌ 没有找到图片文件")
            return
        
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
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(image_files),
            interval=200, repeat=True
        )
        
        # 保存动画
        anim.save('enhanced_simulation_output/city_evolution.gif', writer='pillow', fps=5)
        print("✅ 动画已保存: enhanced_simulation_output/city_evolution.gif")
        
        plt.show()
    
    def show_summary_report(self):
        """显示总结报告"""
        print("📊 Enhanced City Simulation Results Summary")
        print("=" * 50)
        
        summary = self.final_summary['simulation_summary']
        patterns = self.final_summary['development_patterns']
        recommendations = self.final_summary['recommendations']
        
        print(f"🏙️ Simulation Duration: {summary['total_days']} days")
        print(f"👥 Final Population: {summary['final_population']} people")
        print(f"🏗️ Total Buildings: {sum(summary['total_buildings'].values())} units")
        print(f"   - Public Buildings: {summary['total_buildings']['public']} units")
        print(f"   - Residential Buildings: {summary['total_buildings']['residential']} units")
        print(f"   - Commercial Buildings: {summary['total_buildings']['commercial']} units")
        
        print(f"\n💰 Land Price Statistics:")
        land_prices = summary['land_price_summary']
        print(f"   - Average Land Price: {land_prices['avg_price']:.1f}")
        print(f"   - Maximum Land Price: {land_prices['max_price']:.1f}")
        print(f"   - Minimum Land Price: {land_prices['min_price']:.1f}")
        
        print(f"\n🏗️ Development Patterns:")
        print(f"   - Development Stage: {patterns['development_stage']}")
        print(f"   - Public Building Density: {patterns['building_density']['public_density']:.3f}")
        print(f"   - Residential Building Density: {patterns['building_density']['residential_density']:.3f}")
        print(f"   - Commercial Building Density: {patterns['building_density']['commercial_density']:.3f}")
        
        print(f"\n💡 Development Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    def run_all_visualizations(self):
        """运行所有可视化"""
        print("🎨 Starting Enhanced City Simulation Visualization...")
        
        # 显示总结报告
        self.show_summary_report()
        
        # 生成各种图表
        print("\n🏗️ Generating Building Evolution Trend...")
        self.plot_building_evolution()
        
        print("💰 Generating Land Price Evolution Trend...")
        self.plot_land_price_evolution()
        
        print("🗺️ Generating Final City Layout...")
        self.plot_city_layout()
        
        # 询问是否创建动画
        response = input("\n🎬 Create City Development Animation? (y/n): ").lower()
        if response == 'y':
            self.create_animation()
        
        print("\n✅ All Visualizations Completed!")
        print("📁 Output files saved in enhanced_simulation_output/ directory")

def main():
    """主函数"""
    print("🎨 Enhanced City Simulation Results Visualizer")
    print("=" * 40)
    
    visualizer = EnhancedResultsVisualizer()
    visualizer.run_all_visualizations()

if __name__ == "__main__":
    main()
