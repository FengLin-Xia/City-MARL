#!/usr/bin/env python3
"""
快速播放脚本
直接播放24个月的城市演化动画
"""

import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

def quick_playback():
    """快速播放所有帧"""
    print("🎬 快速播放城市演化动画")
    print("=" * 40)
    
    # 设置matplotlib参数
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.max_open_warning'] = 0
    
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
    
    # 播放设置
    playback_speed = 0.8  # 秒/帧
    print(f"⏱️ 播放速度: {playback_speed}秒/帧")
    print("💡 按 Ctrl+C 停止播放")
    
    try:
        for i, img_file in enumerate(image_files):
            month = int(img_file.stem.split('_')[1])
            
            # 显示图片
            img = plt.imread(img_file)
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img)
            ax.axis('off')
            
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
            progress_text = f"Frame {i + 1}/{len(image_files)}"
            ax.text(0.98, 0.02, progress_text, transform=ax.transAxes,
                   fontsize=10, horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            
            # 等待指定时间
            time.sleep(playback_speed)
            
            # 关闭当前帧
            plt.close(fig)
            
            # 显示进度
            if (i + 1) % 6 == 0:
                print(f"📊 已播放 {i + 1}/{len(image_files)} 帧")
        
        print("✅ 播放完成！")
        
    except KeyboardInterrupt:
        print("\n⏹️ 播放已停止")
        plt.close('all')

def show_single_frame(frame_index=0):
    """显示单帧"""
    print(f"📸 显示第 {frame_index + 1} 帧")
    
    # 设置matplotlib参数
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 加载图片文件
    image_dir = Path('enhanced_simulation_output/images')
    image_files = sorted(image_dir.glob('month_*.png'))
    
    if frame_index >= len(image_files):
        print(f"❌ 帧索引超出范围: {frame_index}")
        return
    
    img_file = image_files[frame_index]
    month = int(img_file.stem.split('_')[1])
    
    # 显示图片
    img = plt.imread(img_file)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')
    
    # 添加标题
    title = f'Month {month:02d} - City Evolution'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 加载并显示统计信息
    try:
        with open('enhanced_simulation_output/daily_stats.json', 'r', encoding='utf-8') as f:
            monthly_stats = json.load(f)
        
        if month < len(monthly_stats):
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
    except FileNotFoundError:
        pass
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("🎬 城市演化可视化播放器")
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
    print("1. 快速播放所有帧")
    print("2. 显示单帧")
    print("3. 退出")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == '1':
        quick_playback()
    elif choice == '2':
        frame = input("帧索引 (0-23): ").strip()
        if frame.isdigit():
            show_single_frame(int(frame))
        else:
            print("❌ 请输入有效的数字")
    elif choice == '3':
        print("👋 再见！")
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
