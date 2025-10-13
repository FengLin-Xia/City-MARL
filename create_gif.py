#!/usr/bin/env python3
"""
创建仿真动画GIF文件
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from pathlib import Path
import sys

def create_simulation_gif():
    """创建仿真动画GIF"""
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
    
    # 询问用户选择
    print("\n请选择GIF类型：")
    print("1. 完整动画（所有帧）")
    print("2. 按天动画（每天一张）")
    print("3. 快速预览（每5帧一张）")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        create_full_gif(image_files)
    elif choice == "2":
        create_daily_gif(image_files)
    elif choice == "3":
        create_preview_gif(image_files)
    else:
        print("无效选择，创建完整动画")
        create_full_gif(image_files)

def create_full_gif(image_files, fps=2):
    """创建完整动画GIF"""
    output_path = "simulation_full.gif"
    print(f"开始创建完整动画GIF: {output_path}")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 加载所有图片
    images = []
    for i, img_file in enumerate(image_files):
        try:
            img = mpimg.imread(str(img_file))
            images.append(img)
            if i % 10 == 0:  # 每10帧显示一次进度
                print(f"加载第 {i+1}/{len(image_files)} 帧: {img_file.name}")
        except Exception as e:
            print(f"加载图片失败 {img_file.name}: {e}")
            continue
    
    if not images:
        print("没有成功加载任何图片")
        return
    
    print(f"成功加载 {len(images)} 帧")
    
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
    try:
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"GIF文件已保存: {output_path}")
    except Exception as e:
        print(f"保存GIF失败: {e}")
        print("可能需要安装pillow: pip install pillow")
    
    plt.close()

def create_daily_gif(image_files, fps=1):
    """创建按天动画GIF"""
    output_path = "simulation_daily.gif"
    print(f"开始创建按天动画GIF: {output_path}")
    
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
    
    # 保存GIF
    print(f"正在保存GIF文件...")
    try:
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"GIF文件已保存: {output_path}")
    except Exception as e:
        print(f"保存GIF失败: {e}")
        print("可能需要安装pillow: pip install pillow")
    
    plt.close()

def create_preview_gif(image_files, fps=2, step=5):
    """创建快速预览GIF（每step帧一张）"""
    output_path = "simulation_preview.gif"
    print(f"开始创建预览动画GIF: {output_path}")
    
    # 选择每step帧一张
    selected_files = image_files[::step]
    print(f"从 {len(image_files)} 帧中选择 {len(selected_files)} 帧")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 加载图片
    images = []
    for i, img_file in enumerate(selected_files):
        try:
            img = mpimg.imread(str(img_file))
            images.append(img)
            print(f"加载第 {i+1}/{len(selected_files)} 帧: {img_file.name}")
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
            filename = selected_files[frame].name
            parts = filename.split('_')
            day = parts[1]
            step_num = parts[3].split('.')[0]
            ax.set_title(f'第{day}天 第{step_num}步 (预览)', fontsize=14, fontweight='bold')
        except:
            ax.set_title(f'预览帧 {frame+1}/{len(images)}', fontsize=14, fontweight='bold')
    
    # 创建动画对象
    anim = animation.FuncAnimation(
        fig, animate, frames=len(images), 
        interval=1000//fps, repeat=True, blit=False
    )
    
    # 保存GIF
    print(f"正在保存GIF文件...")
    try:
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"GIF文件已保存: {output_path}")
    except Exception as e:
        print(f"保存GIF失败: {e}")
        print("可能需要安装pillow: pip install pillow")
    
    plt.close()

if __name__ == "__main__":
    create_simulation_gif()



