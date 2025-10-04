#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化repulsion数据的向量方向结果
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

def load_vector_directions(filename: str):
    """加载向量方向数据"""
    points = []
    angles = []
    parcel_ids = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 4:
                    x = float(parts[0])
                    y = float(parts[1])
                    angle = float(parts[2])
                    parcel_id = int(parts[3])
                    points.append((x, y))
                    angles.append(angle)
                    parcel_ids.append(parcel_id)
    
    return points, angles, parcel_ids

def visualize_vector_field(points, angles, parcel_ids, hub2_x=112, hub2_y=121, output_file="repulsion_vector_directions_visualization.png"):
    """可视化向量场"""
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 转换为numpy数组
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])
    angles_array = np.array(angles)
    parcel_ids_array = np.array(parcel_ids)
    
    # 左上图：散点图显示角度分布（按地块着色）
    unique_parcels = np.unique(parcel_ids_array)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_parcels)))
    
    for i, parcel_id in enumerate(unique_parcels):
        mask = parcel_ids_array == parcel_id
        ax1.scatter(x_coords[mask], y_coords[mask], c=[colors[i]], s=20, alpha=0.7, label=f'Parcel {parcel_id}')
    
    ax1.scatter(hub2_x, hub2_y, c='red', s=200, marker='*', label='Hub2 (112, 121)')
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.set_title('向量方向分布图 (按地块着色)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 右上图：散点图显示角度分布（按角度着色）
    scatter = ax2.scatter(x_coords, y_coords, c=angles_array, cmap='hsv', s=20, alpha=0.7)
    ax2.scatter(hub2_x, hub2_y, c='red', s=200, marker='*', label='Hub2 (112, 121)')
    ax2.set_xlabel('X坐标')
    ax2.set_ylabel('Y坐标')
    ax2.set_title('向量方向分布图 (按角度着色)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar2 = plt.colorbar(scatter, ax=ax2)
    cbar2.set_label('角度 (度)')
    
    # 左下图：向量场箭头图（所有点）
    # 为了清晰显示，只显示部分向量
    step = max(1, len(points) // 300)  # 最多显示300个箭头
    
    x_arrows = x_coords[::step]
    y_arrows = y_coords[::step]
    angle_arrows = angles_array[::step]
    
    # 计算箭头方向
    u = [math.cos(math.radians(angle)) for angle in angle_arrows]
    v = [math.sin(math.radians(angle)) for angle in angle_arrows]
    
    ax3.quiver(x_arrows, y_arrows, u, v, angles=angle_arrows, scale=50, alpha=0.7)
    ax3.scatter(hub2_x, hub2_y, c='red', s=200, marker='*', label='Hub2 (112, 121)')
    ax3.set_xlabel('X坐标')
    ax3.set_ylabel('Y坐标')
    ax3.set_title('向量场箭头图 (所有点)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 右下图：角度分布直方图
    ax4.hist(angles_array, bins=50, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('角度 (度)')
    ax4.set_ylabel('频次')
    ax4.set_title('角度分布直方图')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"可视化结果已保存到: {output_file}")

def analyze_angle_distribution(angles, parcel_ids):
    """分析角度分布"""
    print("=== 角度分布分析 ===")
    print(f"总点数: {len(angles)}")
    print(f"角度范围: {min(angles):.2f}° - {max(angles):.2f}°")
    print(f"平均角度: {np.mean(angles):.2f}°")
    print(f"角度标准差: {np.std(angles):.2f}°")
    
    # 统计各象限的点数
    quadrants = [0, 0, 0, 0]  # 右上、左上、左下、右下
    for angle in angles:
        if 0 <= angle < 90:
            quadrants[0] += 1
        elif 90 <= angle < 180:
            quadrants[1] += 1
        elif 180 <= angle < 270:
            quadrants[2] += 1
        else:  # 270 <= angle < 360
            quadrants[3] += 1
    
    print(f"\n象限分布:")
    print(f"右上 (0°-90°): {quadrants[0]} 点 ({quadrants[0]/len(angles)*100:.1f}%)")
    print(f"左上 (90°-180°): {quadrants[1]} 点 ({quadrants[1]/len(angles)*100:.1f}%)")
    print(f"左下 (180°-270°): {quadrants[2]} 点 ({quadrants[2]/len(angles)*100:.1f}%)")
    print(f"右下 (270°-360°): {quadrants[3]} 点 ({quadrants[3]/len(angles)*100:.1f}%)")
    
    # 按地块统计
    print(f"\n各地块统计:")
    unique_parcels = np.unique(parcel_ids)
    for parcel_id in unique_parcels:
        mask = np.array(parcel_ids) == parcel_id
        parcel_angles = np.array(angles)[mask]
        print(f"地块 {parcel_id}: {len(parcel_angles)} 点, 角度范围: {min(parcel_angles):.2f}° - {max(parcel_angles):.2f}°, 平均: {np.mean(parcel_angles):.2f}°")

def main():
    """主函数"""
    print("=== Repulsion向量方向可视化 ===")
    
    # 加载数据
    points, angles, parcel_ids = load_vector_directions("vector_directions_repulsion.txt")
    print(f"成功加载 {len(points)} 个点的向量方向数据")
    
    # 分析角度分布
    analyze_angle_distribution(angles, parcel_ids)
    
    # 可视化
    visualize_vector_field(points, angles, parcel_ids)
    
    print("=== 可视化完成 ===")

if __name__ == "__main__":
    main()

