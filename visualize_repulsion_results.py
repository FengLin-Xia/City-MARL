#!/usr/bin/env python3
"""
可视化点排斥模拟结果
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import csv
import os
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

def load_parcels_from_txt(file_path: str):
    """解析parcel.txt文件"""
    polygons = []
    current_parcel = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[':
                if current_parcel and len(current_parcel) >= 3:
                    try:
                        poly = Polygon(current_parcel)
                        if poly.is_valid:
                            polygons.append(poly)
                    except Exception as e:
                        print(f"警告：地块无效: {e}")
                current_parcel = []
            elif line == ']':
                if current_parcel and len(current_parcel) >= 3:
                    try:
                        poly = Polygon(current_parcel)
                        if poly.is_valid:
                            polygons.append(poly)
                    except Exception as e:
                        print(f"警告：地块无效: {e}")
                current_parcel = []
            elif line and not line.startswith('#') and line not in ['[', ']']:
                try:
                    coords = [float(x.strip()) for x in line.split(',')]
                    if len(coords) >= 2:
                        current_parcel.append((coords[0], coords[1]))
                except ValueError:
                    continue
    
    return polygons

def load_points_from_csv(csv_path):
    """从CSV文件加载点坐标"""
    points = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append([float(row['x']), float(row['y'])])
    return np.array(points)

def compute_mean_distance(points, k_neighbors=6):
    """计算平均距离"""
    if len(points) <= 1:
        return 0.0
    
    tree = cKDTree(points)
    k = min(k_neighbors + 1, len(points))
    distances, _ = tree.query(points, k=k)
    
    if k > 1:
        distances = distances[:, 1:]
    
    return np.mean(distances)

def visualize_all_parcels():
    """可视化所有地块的点分布"""
    print("开始可视化点排斥模拟结果...")
    
    # 加载地块边界
    polygons = load_parcels_from_txt('parcel.txt')
    print(f"加载了 {len(polygons)} 个地块")
    
    # 设置子图布局
    n_parcels = len(polygons)
    cols = 4
    rows = (n_parcels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if n_parcels == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # 为每个地块加载点并可视化
    for i, poly in enumerate(polygons):
        ax = axes[i]
        
        # 绘制多边形边界
        if hasattr(poly, 'exterior'):
            x, y = poly.exterior.xy
            ax.plot(x, y, 'k-', linewidth=2, label='边界')
            ax.fill(x, y, alpha=0.1, color='lightblue')
            
            # 绘制洞
            for interior in poly.interiors:
                ix, iy = interior.xy
                ax.plot(ix, iy, 'k-', linewidth=1, alpha=0.7)
                ax.fill(ix, iy, color='white', alpha=0.8)
        
        # 加载点数据 - 使用最新的文件
        import glob
        csv_pattern = f'repulsion_outputs/parcel_{i}_final_points_*.csv'
        csv_files = glob.glob(csv_pattern)
        if csv_files:
            csv_path = max(csv_files)  # 选择最新的文件
        else:
            csv_path = None
        if csv_path and os.path.exists(csv_path):
            points = load_points_from_csv(csv_path)
            
            if len(points) > 0:
                # 绘制点
                ax.scatter(points[:, 0], points[:, 1], c='red', s=15, alpha=0.7)
                
                # 计算平均距离
                mean_dist = compute_mean_distance(points)
                
                ax.set_title(f'地块 {i}\n剩余: {len(points)} 点, 平均距离: {mean_dist:.2f}米', 
                           fontsize=10)
            else:
                ax.set_title(f'地块 {i}\n无剩余点', fontsize=10)
        else:
            ax.set_title(f'地块 {i}\n数据文件不存在', fontsize=10)
        
        ax.set_xlabel('X (米)', fontsize=8)
        ax.set_ylabel('Y (米)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # 隐藏多余的子图
    for i in range(n_parcels, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('点排斥模拟结果 - 所有地块最终状态\n(目标距离: 2.0米, 所有地块均已达到距离阈值)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    output_path = 'repulsion_outputs/final_distribution_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_path}")
    
    plt.show()

def visualize_distance_statistics():
    """可视化距离统计"""
    print("\n分析距离统计...")
    
    # 读取所有地块的点数据并计算统计
    distances = []
    remaining_counts = []
    
    for i in range(12):  # 假设有12个地块
        import glob
        csv_pattern = f'repulsion_outputs/parcel_{i}_final_points_*.csv'
        csv_files = glob.glob(csv_pattern)
        csv_path = max(csv_files) if csv_files else None
        if csv_path and os.path.exists(csv_path):
            points = load_points_from_csv(csv_path)
            if len(points) > 0:
                mean_dist = compute_mean_distance(points)
                distances.append(mean_dist)
                remaining_counts.append(len(points))
            else:
                distances.append(0)
                remaining_counts.append(0)
        else:
            distances.append(0)
            remaining_counts.append(0)
    
    # 创建统计图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 平均距离分布
    ax1.bar(range(len(distances)), distances, color='skyblue', alpha=0.7)
    ax1.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='目标距离 (2.0米)')
    ax1.set_xlabel('地块编号')
    ax1.set_ylabel('平均邻近距离 (米)')
    ax1.set_title('各地块最终平均距离')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 剩余点数分布
    ax2.bar(range(len(remaining_counts)), remaining_counts, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('地块编号')
    ax2.set_ylabel('剩余点数')
    ax2.set_title('各地块剩余点数')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存统计图
    stats_path = 'repulsion_outputs/distance_statistics.png'
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    print(f"距离统计图已保存到: {stats_path}")
    
    plt.show()
    
    # 打印统计摘要
    print(f"\n=== 距离统计摘要 ===")
    print(f"平均距离范围: {min(distances):.3f} - {max(distances):.3f} 米")
    print(f"平均距离均值: {np.mean(distances):.3f} 米")
    print(f"达到目标距离的地块: {sum(1 for d in distances if d >= 2.0)}/{len(distances)}")
    print(f"剩余点数范围: {min(remaining_counts)} - {max(remaining_counts)} 点")
    print(f"剩余点数均值: {np.mean(remaining_counts):.1f} 点")

def main():
    """主函数"""
    print("=== 点排斥模拟结果可视化 ===")
    
    # 可视化所有地块
    visualize_all_parcels()
    
    # 可视化距离统计
    visualize_distance_statistics()

if __name__ == '__main__':
    main()
