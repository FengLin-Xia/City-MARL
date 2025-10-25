#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化向量方向结果
"""

import matplotlib.pyplot as plt
import numpy as np
import math

def load_vector_directions(filename: str):
    """加载向量方向数据"""
    points = []
    angles = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 3:
                    x = float(parts[0])
                    y = float(parts[1])
                    angle = float(parts[2])
                    points.append((x, y))
                    angles.append(angle)
    
    return points, angles

def visualize_vector_field(points, angles, hub2_x=112, hub2_y=121, output_file="vector_directions_visualization.png"):
    """可视化向量场"""
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：散点图显示角度分布
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    scatter = ax1.scatter(x_coords, y_coords, c=angles, cmap='hsv', s=20, alpha=0.7)
    ax1.scatter(hub2_x, hub2_y, c='red', s=100, marker='*', label='Hub2 (112, 121)')
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.set_title('向量方向分布图\n(颜色表示角度，向右为0度)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('角度 (度)')
    
    # 右图：向量场箭头图
    # 为了清晰显示，只显示部分向量
    step = max(1, len(points) // 200)  # 最多显示200个箭头
    
    x_arrows = x_coords[::step]
    y_arrows = y_coords[::step]
    angle_arrows = angles[::step]
    
    # 计算箭头方向
    u = [math.cos(math.radians(angle)) for angle in angle_arrows]
    v = [math.sin(math.radians(angle)) for angle in angle_arrows]
    
    ax2.quiver(x_arrows, y_arrows, u, v, angles=angle_arrows, scale=50, alpha=0.7)
    ax2.scatter(hub2_x, hub2_y, c='red', s=100, marker='*', label='Hub2 (112, 121)')
    ax2.set_xlabel('X坐标')
    ax2.set_ylabel('Y坐标')
    ax2.set_title('向量场箭头图\n(箭头指向向量方向)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"可视化结果已保存到: {output_file}")

def analyze_angle_distribution(angles):
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

def main():
    """主函数"""
    print("=== 向量方向可视化 ===")
    
    # 加载数据
    points, angles = load_vector_directions("vector_directions.txt")
    print(f"成功加载 {len(points)} 个点的向量方向数据")
    
    # 分析角度分布
    analyze_angle_distribution(angles)
    
    # 可视化
    visualize_vector_field(points, angles)
    
    print("=== 可视化完成 ===")

if __name__ == "__main__":
    main()





















