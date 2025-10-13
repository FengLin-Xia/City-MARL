#!/usr/bin/env python3
"""
查看仿真结果的脚本 - 支持静态对比和动画播放
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import os
from pathlib import Path
import time

def view_simulation_results():
    """查看仿真结果"""
    output_dir = Path('output_frames')
    
    if not output_dir.exists():
        print("输出目录不存在，请先运行仿真")
        return
    
    # 获取所有图片文件
    image_files = sorted([f for f in output_dir.glob('*.png')])
    
    if not image_files:
        print("没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 显示选项
    print("\n请选择查看方式：")
    print("1. 静态对比（开始 vs 结束）")
    print("2. 动画播放（所有帧）")
    print("3. 按天播放（每天一张）")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        show_static_comparison(image_files)
    elif choice == "2":
        play_animation(image_files)
    elif choice == "3":
        play_daily_animation(image_files)
    else:
        print("无效选择，显示静态对比")
        show_static_comparison(image_files)
    
    # 显示统计信息
    show_statistics(image_files)

def show_static_comparison(image_files):
    """显示静态对比"""
    try:
        # 显示第一张和最后一张图片
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 第一张图片
        print(f"读取第一张图片: {image_files[0]}")
        img1 = mpimg.imread(str(image_files[0]))
        ax1.imshow(img1)
        ax1.set_title(f'开始: {image_files[0].name}', fontsize=12)
        ax1.axis('off')
        
        # 最后一张图片
        print(f"读取最后一张图片: {image_files[-1]}")
        img2 = mpimg.imread(str(image_files[-1]))
        ax2.imshow(img2)
        ax2.set_title(f'结束: {image_files[-1].name}', fontsize=12)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"显示图片时出错: {e}")
        print("尝试显示图片信息...")
        
        # 显示文件信息
        for i, f in enumerate([image_files[0], image_files[-1]]):
            print(f"图片 {i+1}: {f.name}, 大小: {f.stat().st_size} bytes")

def play_animation(image_files, fps=2):
    """播放所有帧的动画"""
    print(f"开始播放动画，共 {len(image_files)} 帧，FPS: {fps}")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 加载所有图片
    images = []
    for i, img_file in enumerate(image_files):
        try:
            img = mpimg.imread(str(img_file))
            images.append(img)
            print(f"加载第 {i+1}/{len(image_files)} 帧: {img_file.name}")
        except Exception as e:
            print(f"加载图片失败 {img_file.name}: {e}")
            continue
    
    if not images:
        print("没有成功加载任何图片")
        return
    
    # 创建动画
    def animate(frame):
        ax.clear()
        ax.axis('off')
        ax.imshow(images[frame])
        
        # 解析文件名获取信息
        try:
            filename = image_files[frame].name
            parts = filename.split('_')
            day = parts[1]
            step = parts[3].split('.')[0]
            ax.set_title(f'第{day}天 第{step}步', fontsize=14, fontweight='bold')
        except:
            ax.set_title(f'帧 {frame+1}/{len(images)}', fontsize=14, fontweight='bold')
    
    # 创建动画对象
    anim = animation.FuncAnimation(
        fig, animate, frames=len(images), 
        interval=1000//fps, repeat=True, blit=False
    )
    
    plt.tight_layout()
    plt.show()

def play_daily_animation(image_files, fps=1):
    """按天播放动画（每天一张）"""
    print("开始按天播放动画...")
    
    # 按天分组
    daily_images = {}
    for img_file in image_files:
        try:
            day = img_file.name.split('_')[1]
            if day not in daily_images:
                daily_images[day] = []
            daily_images[day].append(img_file)
        except:
            continue
    
    if not daily_images:
        print("无法解析文件名")
        return
    
    # 每天选择最后一张图片（代表当天结束状态）
    daily_final_images = []
    daily_names = []
    
    for day in sorted(daily_images.keys()):
        day_files = daily_images[day]
        if day_files:
            daily_final_images.append(day_files[-1])  # 每天的最后一张
            daily_names.append(f"第{day}天")
    
    print(f"找到 {len(daily_final_images)} 天的数据")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 加载图片
    images = []
    for i, img_file in enumerate(daily_final_images):
        try:
            img = mpimg.imread(str(img_file))
            images.append(img)
            print(f"加载第 {i+1}/{len(daily_final_images)} 天: {img_file.name}")
        except Exception as e:
            print(f"加载图片失败 {img_file.name}: {e}")
            continue
    
    if not images:
        print("没有成功加载任何图片")
        return
    
    # 创建动画
    def animate(frame):
        ax.clear()
        ax.axis('off')
        ax.imshow(images[frame])
        ax.set_title(f'{daily_names[frame]}', fontsize=16, fontweight='bold')
    
    # 创建动画对象
    anim = animation.FuncAnimation(
        fig, animate, frames=len(images), 
        interval=1000//fps, repeat=True, blit=False
    )
    
    plt.tight_layout()
    plt.show()

def show_statistics(image_files):
    """显示统计信息"""
    print("\n=== 仿真统计 ===")
    print(f"总图片数: {len(image_files)}")
    
    # 按天统计
    days = {}
    for f in image_files:
        try:
            day = f.name.split('_')[1]
            if day not in days:
                days[day] = 0
            days[day] += 1
        except IndexError:
            print(f"文件名格式异常: {f.name}")
    
    print(f"仿真天数: {len(days)}")
    print(f"每天图片数: {list(days.values())}")
    
    # 显示每天的详细信息
    for day in sorted(days.keys()):
        day_files = [f for f in image_files if f.name.split('_')[1] == day]
        print(f"第{day}天: {len(day_files)}张图片")

def create_gif(image_files, output_path="simulation.gif", fps=2):
    """创建GIF动画文件"""
    print(f"开始创建GIF动画: {output_path}")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 加载所有图片
    images = []
    for i, img_file in enumerate(image_files):
        try:
            img = mpimg.imread(str(img_file))
            images.append(img)
            print(f"加载第 {i+1}/{len(image_files)} 帧: {img_file.name}")
        except Exception as e:
            print(f"加载图片失败 {img_file.name}: {e}")
            continue
    
    if not images:
        print("没有成功加载任何图片")
        return
    
    # 创建动画
    def animate(frame):
        ax.clear()
        ax.axis('off')
        ax.imshow(images[frame])
        
        # 解析文件名获取信息
        try:
            filename = image_files[frame].name
            parts = filename.split('_')
            day = parts[1]
            step = parts[3].split('.')[0]
            ax.set_title(f'第{day}天 第{step}步', fontsize=14, fontweight='bold')
        except:
            ax.set_title(f'帧 {frame+1}/{len(images)}', fontsize=14, fontweight='bold')
    
    # 创建动画对象
    anim = animation.FuncAnimation(
        fig, animate, frames=len(images), 
        interval=1000//fps, repeat=True, blit=False
    )
    
    # 保存GIF
    print(f"正在保存GIF文件...")
    anim.save(output_path, writer='pillow', fps=fps)
    print(f"GIF文件已保存: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    view_simulation_results()
