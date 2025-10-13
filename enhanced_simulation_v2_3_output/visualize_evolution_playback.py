#!/usr/bin/env python3
"""
高斯核地价场演化可视化播放器
逐帧显示地价场变化和建筑分布
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os
from typing import Dict, List, Tuple
import glob

class EvolutionPlayback:
    """演化播放器"""
    
    def __init__(self, output_dir: str = "enhanced_simulation_v2_3_output"):
        self.output_dir = output_dir
        self.land_price_frames = []
        self.building_frames = []
        self.months = []
        
        # 加载所有帧数据
        self._load_frames()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def _load_frames(self):
        """加载所有帧数据"""
        print("🔄 加载演化帧数据...")
        
        # 加载地价场帧
        land_price_files = sorted(glob.glob(os.path.join(self.output_dir, "land_price_frame_month_*.json")))
        for file_path in land_price_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)
                    self.land_price_frames.append(frame_data)
                    
                    # 提取月份
                    month = frame_data['month']
                    self.months.append(month)
                    
                    print(f"  ✅ 加载地价场帧: 月份 {month}")
            except Exception as e:
                print(f"  ❌ 加载失败 {file_path}: {e}")
        
        # 加载建筑位置帧
        building_files = sorted(glob.glob(os.path.join(self.output_dir, "building_positions_month_*.json")))
        for file_path in building_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)
                    self.building_frames.append(frame_data)
                    print(f"  ✅ 加载建筑帧: {frame_data['timestamp']}")
            except Exception as e:
                print(f"  ❌ 加载失败 {file_path}: {e}")
        
        print(f"📊 总共加载 {len(self.land_price_frames)} 个地价场帧，{len(self.building_frames)} 个建筑帧")
    
    def _get_buildings_for_month(self, month: int) -> Dict:
        """获取指定月份的建筑数据"""
        for building_frame in self.building_frames:
            if building_frame['timestamp'] == f'month_{month:02d}':
                return building_frame
        return {'buildings': []}
    
    def _create_frame(self, frame_idx: int):
        """创建单帧可视化"""
        if frame_idx >= len(self.land_price_frames):
            return
        
        # 获取当前帧数据
        land_price_frame = self.land_price_frames[frame_idx]
        month = land_price_frame['month']
        land_price_field = np.array(land_price_frame['land_price_field'])
        evolution_stage = land_price_frame['evolution_stage']
        
        # 获取建筑数据
        buildings = self._get_buildings_for_month(month)['buildings']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'高斯核地价场演化 - 第 {month} 个月 ({evolution_stage["name"]})', fontsize=16)
        
        # 1. 地价场热力图
        im1 = axes[0, 0].imshow(land_price_field, cmap='hot', interpolation='nearest')
        axes[0, 0].set_title(f'地价场 (范围: {np.min(land_price_field):.3f} - {np.max(land_price_field):.3f})')
        axes[0, 0].set_xlabel('X (像素)')
        axes[0, 0].set_ylabel('Y (像素)')
        plt.colorbar(im1, ax=axes[0, 0], label='地价值')
        
        # 2. 地价场等高线
        contour = axes[0, 1].contour(land_price_field, levels=10, colors='black', alpha=0.7)
        axes[0, 1].set_title('地价场等高线')
        axes[0, 1].set_xlabel('X (像素)')
        axes[0, 1].set_ylabel('Y (像素)')
        axes[0, 1].clabel(contour, inline=True, fontsize=8)
        
        # 3. 建筑分布
        axes[1, 0].set_xlim(0, land_price_field.shape[1])
        axes[1, 0].set_ylim(0, land_price_field.shape[0])
        axes[1, 0].set_title(f'建筑分布 ({len(buildings)} 个建筑)')
        axes[1, 0].set_xlabel('X (像素)')
        axes[1, 0].set_ylabel('Y (像素)')
        
        # 绘制建筑
        commercial_buildings = [b for b in buildings if b['type'] == 'commercial']
        residential_buildings = [b for b in buildings if b['type'] == 'residential']
        public_buildings = [b for b in buildings if b['type'] == 'public']
        
        if commercial_buildings:
            x_coords = [b['position'][0] for b in commercial_buildings]
            y_coords = [b['position'][1] for b in commercial_buildings]
            axes[1, 0].scatter(x_coords, y_coords, c='red', s=50, alpha=0.8, label=f'商业 ({len(commercial_buildings)})')
        
        if residential_buildings:
            x_coords = [b['position'][0] for b in residential_buildings]
            y_coords = [b['position'][1] for b in residential_buildings]
            axes[1, 0].scatter(x_coords, y_coords, c='blue', s=30, alpha=0.8, label=f'住宅 ({len(residential_buildings)})')
        
        if public_buildings:
            x_coords = [b['position'][0] for b in public_buildings]
            y_coords = [b['position'][1] for b in public_buildings]
            axes[1, 0].scatter(x_coords, y_coords, c='green', s=40, alpha=0.8, label=f'公共 ({len(public_buildings)})')
        
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 演化统计
        stats = land_price_frame['land_price_stats']
        axes[1, 1].text(0.1, 0.8, f'演化阶段: {evolution_stage["name"]}', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f'Hub σ: {evolution_stage["hub_sigma"]:.1f} px', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'道路 σ: {evolution_stage["road_sigma"]:.1f} px', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.5, f'地价范围: {stats["min"]:.3f} - {stats["max"]:.3f}', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'平均地价: {stats["mean"]:.3f}', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.3, f'标准差: {stats["std"]:.3f}', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.2, f'建筑总数: {len(buildings)}', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.1, f'月份: {month}', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('演化统计')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def play_animation(self, interval: int = 1000, save_gif: bool = False):
        """播放动画"""
        if not self.land_price_frames:
            print("❌ 没有可用的帧数据")
            return
        
        print(f"🎬 开始播放动画，共 {len(self.land_price_frames)} 帧，间隔 {interval}ms")
        
        # 创建动画
        fig = plt.figure(figsize=(16, 12))
        
        def animate(frame_idx):
            plt.clf()
            return self._create_frame(frame_idx)
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.land_price_frames),
            interval=interval, repeat=True, blit=False
        )
        
        if save_gif:
            gif_path = os.path.join(self.output_dir, "evolution_animation.gif")
            print(f"💾 保存动画到: {gif_path}")
            anim.save(gif_path, writer='pillow', fps=1)
        
        plt.show()
        return anim
    
    def show_frame(self, month: int):
        """显示指定月份的单帧"""
        frame_idx = None
        for i, m in enumerate(self.months):
            if m == month:
                frame_idx = i
                break
        
        if frame_idx is None:
            print(f"❌ 未找到月份 {month} 的帧数据")
            return
        
        print(f"🖼️ 显示第 {month} 个月的帧")
        fig = self._create_frame(frame_idx)
        plt.show()
    
    def show_all_frames(self):
        """显示所有帧（静态）"""
        if not self.land_price_frames:
            print("❌ 没有可用的帧数据")
            return
        
        print(f"🖼️ 显示所有 {len(self.land_price_frames)} 帧")
        
        for i, frame in enumerate(self.land_price_frames):
            month = frame['month']
            print(f"  显示第 {month} 个月...")
            
            fig = self._create_frame(i)
            plt.show()
            
            # 询问是否继续
            if i < len(self.land_price_frames) - 1:
                response = input("按回车继续下一帧，输入 'q' 退出: ")
                if response.lower() == 'q':
                    break

def main():
    """主函数"""
    print("🎬 高斯核地价场演化可视化播放器")
    print("=" * 50)
    
    # 创建播放器
    player = EvolutionPlayback()
    
    if not player.land_price_frames:
        print("❌ 未找到地价场帧数据")
        return
    
    print("\n📋 可用操作:")
    print("1. 播放动画 (自动播放)")
    print("2. 显示单帧 (指定月份)")
    print("3. 显示所有帧 (手动控制)")
    print("4. 保存动画GIF")
    
    while True:
        choice = input("\n请选择操作 (1-4, q退出): ").strip()
        
        if choice == 'q':
            break
        elif choice == '1':
            interval = input("请输入帧间隔(毫秒，默认1000): ").strip()
            interval = int(interval) if interval.isdigit() else 1000
            player.play_animation(interval=interval)
        elif choice == '2':
            month = input("请输入月份 (0-24): ").strip()
            if month.isdigit():
                player.show_frame(int(month))
        elif choice == '3':
            player.show_all_frames()
        elif choice == '4':
            interval = input("请输入帧间隔(毫秒，默认1000): ").strip()
            interval = int(interval) if interval.isdigit() else 1000
            player.play_animation(interval=interval, save_gif=True)
        else:
            print("❌ 无效选择，请重试")

if __name__ == "__main__":
    main()


