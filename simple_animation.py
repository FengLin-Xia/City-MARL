#!/usr/bin/env python3
"""
简单动画播放器
直接播放城市演化动画，无需用户交互
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import json

def play_city_evolution():
    """播放城市演化动画"""
    print("🎬 播放城市演化动画...")
    
    # 设置matplotlib参数
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 加载图片文件
    image_dir = Path('enhanced_simulation_output/images')
    image_files = sorted(image_dir.glob('month_*.png'))
    
    if not image_files:
        print("❌ 没有找到渲染帧")
        return
    
    print(f"📁 找到 {len(image_files)} 个渲染帧")
    
    # 加载统计数据
    try:
        with open('enhanced_simulation_output/daily_stats.json', 'r', encoding='utf-8') as f:
            monthly_stats = json.load(f)
    except FileNotFoundError:
        monthly_stats = []
        print("⚠️ 统计数据文件未找到")
    
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
        interval=1000,  # 1秒每帧
        repeat=True,
        blit=False
    )
    
    print("🎬 动画开始播放...")
    print("💡 关闭窗口停止播放")
    
    # 显示动画
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    play_city_evolution()
