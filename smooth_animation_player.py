#!/usr/bin/env python3
"""
流畅动画播放器
使用matplotlib的FuncAnimation实现连续播放
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import numpy as np
import json

class SmoothAnimationPlayer:
    """流畅动画播放器"""
    
    def __init__(self):
        """初始化播放器"""
        # 设置matplotlib参数
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.max_open_warning'] = 0
        
        # 加载图片文件
        self.image_files = self._load_image_files()
        self.current_frame = 0
        
        # 加载统计数据
        self.monthly_stats = self._load_monthly_stats()
        
        # 创建图形和轴
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.axis('off')
        
        # 初始化图像对象
        self.img_plot = None
        
        print(f"🎬 流畅动画播放器初始化完成")
        print(f"📁 找到 {len(self.image_files)} 个渲染帧")
    
    def _load_image_files(self):
        """加载图片文件"""
        image_dir = Path('enhanced_simulation_output/images')
        if not image_dir.exists():
            print("❌ 图片目录不存在")
            return []
        
        # 加载月级渲染帧
        image_files = sorted(image_dir.glob('month_*.png'))
        if not image_files:
            print("❌ 没有找到月级渲染帧")
            return []
        
        return image_files
    
    def _load_monthly_stats(self):
        """加载每月统计数据"""
        try:
            with open('enhanced_simulation_output/daily_stats.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ 统计数据文件未找到")
            return []
    
    def update_frame(self, frame_num):
        """更新帧（用于动画）"""
        if frame_num >= len(self.image_files):
            return self.img_plot,
        
        # 获取图片文件
        img_file = self.image_files[frame_num]
        month = int(img_file.stem.split('_')[1])
        
        # 读取图片
        img = plt.imread(img_file)
        
        # 更新图像
        if self.img_plot is None:
            self.img_plot = self.ax.imshow(img)
        else:
            self.img_plot.set_array(img)
        
        # 更新标题
        title = f'Month {month:02d} - City Evolution'
        self.ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 清除之前的文本
        for text in self.ax.texts:
            text.remove()
        
        # 添加统计信息
        if self.monthly_stats and month < len(self.monthly_stats):
            stats = self.monthly_stats[month]
            stats_text = f"""
Population: {stats['population']} people
Buildings: {stats['public_buildings'] + stats['residential_buildings'] + stats['commercial_buildings']} total
• Public: {stats['public_buildings']}
• Residential: {stats['residential_buildings']}
• Commercial: {stats['commercial_buildings']}
            """
            self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 添加进度信息
        progress_text = f"Frame {frame_num + 1}/{len(self.image_files)}"
        self.ax.text(0.98, 0.02, progress_text, transform=self.ax.transAxes,
                   fontsize=10, horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return self.img_plot,
    
    def play_animation(self, interval=800, repeat=True):
        """播放动画"""
        if not self.image_files:
            print("❌ 没有可播放的图片")
            return
        
        print(f"🎬 开始播放动画: {len(self.image_files)} 帧, 间隔: {interval}ms")
        print("💡 动画将在新窗口中播放")
        
        # 创建动画
        anim = animation.FuncAnimation(
            self.fig, 
            self.update_frame, 
            frames=len(self.image_files),
            interval=interval,  # 毫秒
            repeat=repeat,
            blit=True
        )
        
        # 显示动画
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def show_single_frame(self, frame_index):
        """显示单帧"""
        if frame_index < 0 or frame_index >= len(self.image_files):
            print(f"❌ 帧索引超出范围: {frame_index}")
            return
        
        # 更新到指定帧
        self.update_frame(frame_index)
        
        # 显示
        plt.tight_layout()
        plt.show()

def create_gif_animation():
    """创建GIF动画"""
    print("🎬 创建GIF动画...")
    
    # 设置matplotlib参数
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 加载图片文件
    image_dir = Path('enhanced_simulation_output/images')
    image_files = sorted(image_dir.glob('month_*.png'))
    
    if not image_files:
        print("❌ 没有找到渲染帧")
        return
    
    # 加载统计数据
    try:
        with open('enhanced_simulation_output/daily_stats.json', 'r', encoding='utf-8') as f:
            monthly_stats = json.load(f)
    except FileNotFoundError:
        monthly_stats = []
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    def update_frame(frame_num):
        """更新帧"""
        if frame_num >= len(image_files):
            return ax.get_children(),
        
        # 获取图片文件
        img_file = image_files[frame_num]
        month = int(img_file.stem.split('_')[1])
        
        # 清除之前的图像
        ax.clear()
        ax.axis('off')
        
        # 读取并显示图片
        img = plt.imread(img_file)
        ax.imshow(img)
        
        # 添加标题
        title = f'Month {month:02d} - City Evolution'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 添加统计信息
        if monthly_stats and month < len(monthly_stats):
            stats = monthly_stats[month]
            stats_text = f"""
Population: {stats['population']} people
Buildings: {stats['public_buildings'] + stats['residential_buildings'] + stats['commercial_buildings']} total
• Public: {stats['public_buildings']}
• Residential: {stats['residential_buildings']}
• Commercial: {stats['commercial_buildings']}
            """
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 添加进度信息
        progress_text = f"Frame {frame_num + 1}/{len(image_files)}"
        ax.text(0.98, 0.02, progress_text, transform=ax.transAxes,
               fontsize=10, horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax.get_children(),
    
    # 创建动画
    anim = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=len(image_files),
        interval=800,  # 800ms per frame
        repeat=True,
        blit=False
    )
    
    # 保存GIF
    gif_path = 'enhanced_simulation_output/city_evolution_animation.gif'
    print(f"💾 保存GIF到: {gif_path}")
    
    try:
        anim.save(gif_path, writer='pillow', fps=1.25)  # 1.25 FPS
        print("✅ GIF动画保存成功！")
    except Exception as e:
        print(f"❌ 保存GIF失败: {e}")

def main():
    """主函数"""
    print("🎬 流畅动画播放器")
    print("=" * 50)
    
    # 检查文件是否存在
    image_dir = Path('enhanced_simulation_output/images')
    if not image_dir.exists():
        print("❌ 图片目录不存在，请先运行模拟")
        return
    
    image_files = sorted(image_dir.glob('month_*.png'))
    if not image_files:
        print("❌ 没有找到渲染帧，请先运行模拟")
        return
    
    print(f"📁 找到 {len(image_files)} 个渲染帧")
    print("\n选择播放模式:")
    print("1. 流畅动画播放")
    print("2. 显示单帧")
    print("3. 创建GIF动画")
    print("4. 退出")
    
    choice = input("请选择 (1-4): ").strip()
    
    if choice == '1':
        # 创建播放器并播放
        player = SmoothAnimationPlayer()
        if player.image_files:
            interval = input("播放间隔毫秒 (默认800): ").strip()
            interval = int(interval) if interval.isdigit() else 800
            player.play_animation(interval=interval)
    
    elif choice == '2':
        frame = input("帧索引 (0-23): ").strip()
        if frame.isdigit():
            player = SmoothAnimationPlayer()
            player.show_single_frame(int(frame))
        else:
            print("❌ 请输入有效的数字")
    
    elif choice == '3':
        create_gif_animation()
    
    elif choice == '4':
        print("👋 再见！")
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
