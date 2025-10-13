#!/usr/bin/env python3
"""
简单的长期训练结果逐帧播放工具
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from pathlib import Path
import numpy as np

def simple_viewer():
    """简单的逐帧播放器"""
    print("🎬 简单逐帧播放器")
    print("="*30)
    
    # 检查输出目录
    output_dir = Path('test_long_term_output')
    if not output_dir.exists():
        print("❌ 没有找到 test_long_term_output 目录")
        return
    
    # 获取所有图片文件
    image_files = []
    for file in output_dir.glob("test_day_*.png"):
        image_files.append(str(file))
    
    if not image_files:
        print("❌ 没有找到图片文件")
        return
    
    # 按文件名排序
    image_files.sort()
    
    print(f"📊 找到 {len(image_files)} 张图片")
    print(f"📅 时间范围: {Path(image_files[0]).stem} → {Path(image_files[-1]).stem}")
    
    # 询问播放速度
    print("\n请选择播放速度：")
    print("1. 慢速 (1 FPS)")
    print("2. 中速 (2 FPS)")
    print("3. 快速 (5 FPS)")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        fps = 1
    elif choice == "2":
        fps = 2
    elif choice == "3":
        fps = 5
    else:
        fps = 2
        print("使用默认中速播放")
    
    # 开始播放
    play_animation(image_files, fps)

def play_animation(image_files, fps=2):
    """播放动画"""
    print(f"🎬 开始播放动画 (FPS: {fps})")
    print(f"📊 总帧数: {len(image_files)}")
    print(f"⏱️ 预计时长: {len(image_files)/fps:.1f} 秒")
    
    try:
        # 加载所有图片
        print("📸 正在加载图片...")
        images = []
        for i, file_path in enumerate(image_files):
            try:
                img = mpimg.imread(file_path)
                images.append(img)
                if (i + 1) % 20 == 0:
                    print(f"   已加载 {i + 1}/{len(image_files)} 张图片")
            except Exception as e:
                print(f"⚠️ 无法加载图片 {file_path}: {e}")
                continue
        
        if not images:
            print("❌ 没有可用的图片")
            return
        
        print(f"✅ 成功加载 {len(images)} 张图片")
        
        # 创建动画
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        def animate(frame):
            ax.clear()
            ax.imshow(images[frame])
            
            # 获取文件名作为标题
            filename = Path(image_files[frame]).stem
            day = filename.split('_')[1] if '_' in filename else filename
            ax.set_title(f'城市演化 - 第{day}天', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(images), 
            interval=1000//fps, repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ 播放动画时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_viewer()
