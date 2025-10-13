#!/usr/bin/env python3
"""
稳定的城市可视化脚本
避免动画窗口崩溃问题
"""

import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import numpy as np
from PIL import Image
import os

class StableCityVisualizer:
    """稳定的城市可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        # 设置matplotlib参数，避免崩溃
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.max_open_warning'] = 0
        
        # 加载数据
        self.load_simulation_data()
        
    def load_simulation_data(self):
        """加载模拟数据"""
        try:
            # 加载最终城市状态
            with open('enhanced_simulation_output/city_state_output.json', 'r', encoding='utf-8') as f:
                self.city_state = json.load(f)
            
            # 加载每日统计数据
            with open('enhanced_simulation_output/daily_stats.json', 'r', encoding='utf-8') as f:
                self.daily_stats = json.load(f)
                
            print("✅ 数据加载成功")
            
        except FileNotFoundError as e:
            print(f"❌ 数据文件未找到: {e}")
            return
    
    def plot_final_city_layout(self):
        """绘制最终城市布局"""
        print("🗺️ 生成最终城市布局...")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 设置背景
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制主干道
        trunk_road = [[40, 128], [216, 128]]
        ax.plot([trunk_road[0][0], trunk_road[1][0]], 
                [trunk_road[0][1], trunk_road[1][1]], 
                color='#9AA4B2', linewidth=10, alpha=0.8, label='Main Road')
        
        # 绘制交通枢纽
        hubs = [{'id': 'A', 'xy': [40, 128]}, {'id': 'B', 'xy': [216, 128]}]
        for hub in hubs:
            x, y = hub['xy']
            circle = plt.Circle((x, y), radius=8, color='#0B5ED7', alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, hub['id'], ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=10)
        
        # 绘制城市核心点
        core_point = [128, 128]
        ax.plot(core_point[0], core_point[1], 'o', color='#0B5ED7', 
               markersize=15, label='City Core')
        
        # 绘制建筑
        buildings = self.city_state['buildings']
        
        # 公共建筑
        for building in buildings['public']:
            x, y = building['xy']
            ax.plot(x, y, 's', color='#22A6B3', markersize=12, 
                   label='Public Building' if building == buildings['public'][0] else "")
            ax.text(x, y+8, 'Pub', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 住宅建筑
        for building in buildings['residential']:
            x, y = building['xy']
            ax.plot(x, y, 's', color='#F6C344', markersize=10, 
                   label='Residential Building' if building == buildings['residential'][0] else "")
            ax.text(x, y+6, 'Res', ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        # 商业建筑
        for building in buildings['commercial']:
            x, y = building['xy']
            ax.plot(x, y, 'o', color='#FD7E14', markersize=12, 
                   label='Commercial Building' if building == buildings['commercial'][0] else "")
            ax.text(x, y+6, 'Com', ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        # 绘制居民（随机选择一些显示）
        residents = self.city_state['residents']
        if residents:
            # 随机选择50个居民显示，避免过于密集
            sample_size = min(50, len(residents))
            sample_residents = np.random.choice(residents, sample_size, replace=False)
            
            for resident in sample_residents:
                x, y = resident['pos']
                ax.plot(x, y, 'o', color='#FFFFFF', markersize=2, alpha=0.6,
                       label='Residents' if resident == sample_residents[0] else "")
        
        # 设置标签和标题
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('Final City Layout - Enhanced Simulation', fontsize=16, fontweight='bold')
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # 添加统计信息
        stats_text = f"""
Simulation Statistics:
• Duration: {len(self.daily_stats)} days
• Final Population: {len(residents)} people
• Total Buildings: {len(buildings['public']) + len(buildings['residential']) + len(buildings['commercial'])} units
• Public: {len(buildings['public'])} | Residential: {len(buildings['residential'])} | Commercial: {len(buildings['commercial'])}
        """
        ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 调整布局
        plt.subplots_adjust(right=0.8)
        
        # 保存图片
        plt.savefig('enhanced_simulation_output/final_city_layout.png', 
                   dpi=300, bbox_inches='tight')
        print("✅ 最终城市布局已保存: enhanced_simulation_output/final_city_layout.png")
        
        # 显示图片（不阻塞）
        plt.show(block=False)
        plt.pause(3)  # 显示3秒
        plt.close()
    
    def create_simple_animation(self):
        """创建简单的动画（避免matplotlib动画崩溃）"""
        print("🎬 创建简单动画...")
        
        # 检查图片文件
        image_dir = Path('enhanced_simulation_output/images')
        if not image_dir.exists():
            print("❌ 图片目录不存在")
            return
        
        image_files = sorted(image_dir.glob('day_*.png'))
        if not image_files:
            print("❌ 没有找到图片文件")
            return
        
        print(f"📁 找到 {len(image_files)} 个图片文件")
        
        # 创建GIF动画
        try:
            images = []
            for img_file in image_files:
                img = Image.open(img_file)
                images.append(img)
            
            # 保存GIF
            gif_path = 'enhanced_simulation_output/city_evolution_simple.gif'
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=1000,  # 1秒每帧
                loop=0
            )
            print(f"✅ 简单动画已保存: {gif_path}")
            
        except Exception as e:
            print(f"❌ 创建GIF失败: {e}")
    
    def show_frame_by_frame(self):
        """逐帧显示（更稳定的方式）"""
        print("🎬 逐帧显示模式...")
        
        image_dir = Path('enhanced_simulation_output/images')
        image_files = sorted(image_dir.glob('day_*.png'))
        
        if not image_files:
            print("❌ 没有找到图片文件")
            return
        
        print(f"📁 找到 {len(image_files)} 个图片文件")
        print("💡 提示：按任意键继续下一帧，按 'q' 退出")
        
        for i, img_file in enumerate(image_files):
            print(f"📸 显示第 {i+1}/{len(image_files)} 帧: {img_file.name}")
            
            # 显示图片
            img = plt.imread(img_file)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(f'Day {img_file.stem.split("_")[1]}', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # 显示图片（不阻塞）
            plt.show(block=False)
            plt.pause(2)  # 显示2秒
            plt.close()
            
            # 每10帧询问是否继续
            if (i + 1) % 10 == 0:
                user_input = input(f"已显示 {i+1}/{len(image_files)} 帧，继续？(y/n): ")
                if user_input.lower() != 'y':
                    break
    
    def run_visualization(self):
        """运行可视化"""
        print("🎨 Stable City Visualizer")
        print("=" * 50)
        
        # 显示统计信息
        self.show_statistics()
        
        # 显示最终城市布局
        self.plot_final_city_layout()
        
        # 询问用户选择
        print("\n🎬 选择可视化模式:")
        print("1. 显示最终城市布局")
        print("2. 创建简单GIF动画")
        print("3. 逐帧显示")
        print("4. 全部执行")
        
        choice = input("请选择 (1-4): ").strip()
        
        if choice == '1':
            pass  # 已经显示了
        elif choice == '2':
            self.create_simple_animation()
        elif choice == '3':
            self.show_frame_by_frame()
        elif choice == '4':
            self.create_simple_animation()
            self.show_frame_by_frame()
        else:
            print("❌ 无效选择")
    
    def show_statistics(self):
        """显示统计信息"""
        print("📊 模拟统计信息:")
        print("=" * 30)
        
        # 基本信息
        simulation_info = self.city_state['simulation_info']
        print(f"🏙️ 模拟时长: {simulation_info['day']} 天")
        print(f"👥 最终人口: {simulation_info['total_residents']} 人")
        print(f"🏗️ 建筑总数: {simulation_info['total_buildings']} 个")
        print(f"💰 平均地价: {simulation_info['average_land_price']:.1f}")
        
        # 建筑分布
        buildings = self.city_state['buildings']
        print(f"\n🏗️ 建筑分布:")
        print(f"  公共建筑: {len(buildings['public'])} 个")
        print(f"  住宅建筑: {len(buildings['residential'])} 个")
        print(f"  商业建筑: {len(buildings['commercial'])} 个")
        
        # 地价统计
        land_prices = self.city_state['land_prices']
        print(f"\n💰 地价统计:")
        print(f"  最高地价: {land_prices['max_price']:.1f}")
        print(f"  最低地价: {land_prices['min_price']:.1f}")
        print(f"  平均地价: {land_prices['avg_price']:.1f}")

def main():
    """主函数"""
    visualizer = StableCityVisualizer()
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
