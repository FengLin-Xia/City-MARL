#!/usr/bin/env python3
"""
多地块排斥力布点系统
在多个地块内以排斥力迭代布点，目标最近邻间距≈100m
当"非边界点数=50"时首次达到即停止
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Config:
    """配置参数"""
    target_spacing_m: float = 10.0  # 5像素 = 10米 (根据1像素=2米的换算)
    num_points_per_parcel: int = 100
    boundary_tol_m: float = 1.0
    max_iters: int = 5000
    dt: float = 0.6
    repulsion_strength: float = 1.0
    repulsion_cap: float = 2.0
    seed: int = 42
    init_spread_sigma: float = 2.0  # 减小初始化散布半径，适应更小的目标间距

def load_parcels_from_txt(file_path: str) -> Dict[str, Polygon]:
    """
    读取parcel.txt文件，解析多个地块边界
    返回 {parcel_id: shapely.Polygon}
    """
    parcels = {}
    current_parcel = []
    parcel_id = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[':
                # 开始新地块
                if current_parcel:
                    # 保存前一个地块
                    if len(current_parcel) >= 3:
                        try:
                            poly = Polygon(current_parcel)
                            if poly.is_valid:
                                parcels[f'parcel_{parcel_id}'] = poly
                                parcel_id += 1
                        except Exception as e:
                            print(f"警告：地块 {parcel_id} 无效: {e}")
                    current_parcel = []
            elif line == ']':
                # 地块结束
                if current_parcel and len(current_parcel) >= 3:
                    try:
                        poly = Polygon(current_parcel)
                        if poly.is_valid:
                            parcels[f'parcel_{parcel_id}'] = poly
                            parcel_id += 1
                    except Exception as e:
                        print(f"警告：地块 {parcel_id} 无效: {e}")
                current_parcel = []
            elif line and not line.startswith('#'):
                # 解析坐标行
                try:
                    coords = [float(x.strip()) for x in line.split(',')]
                    if len(coords) >= 2:
                        current_parcel.append((coords[0], coords[1]))
                except ValueError:
                    continue
    
    print(f"成功加载 {len(parcels)} 个地块")
    return parcels

def init_points_in_parcel(poly: Polygon, n: int, seed: int, sigma: float) -> np.ndarray:
    """
    在多边形质心附近高斯抖动初始化n个点
    保证点先落在多边形内
    """
    np.random.seed(seed)
    
    # 获取多边形质心
    centroid = poly.centroid
    cx, cy = centroid.x, centroid.y
    
    points = []
    attempts = 0
    max_attempts = n * 10  # 最多尝试次数
    
    while len(points) < n and attempts < max_attempts:
        # 在质心附近高斯分布采样
        x = np.random.normal(cx, sigma)
        y = np.random.normal(cy, sigma)
        point = Point(x, y)
        
        # 检查点是否在多边形内
        if poly.contains(point) or poly.touches(point):
            points.append([x, y])
        
        attempts += 1
    
    # 如果采样不足，在边界框内均匀采样
    if len(points) < n:
        bounds = poly.bounds
        minx, miny, maxx, maxy = bounds
        
        while len(points) < n and attempts < max_attempts * 2:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)
            
            if poly.contains(point) or poly.touches(point):
                points.append([x, y])
            
            attempts += 1
    
    return np.array(points)

def repel_step(points: np.ndarray, poly: Polygon, config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行一次排斥力更新（RK2方法）
    返回: (new_points, forces)
    """
    n = len(points)
    if n == 0:
        return points, np.zeros_like(points)
    
    # 构建KDTree用于近邻搜索
    tree = cKDTree(points)
    r_cut = 2 * config.target_spacing_m
    
    forces = np.zeros_like(points)
    
    # 计算排斥力
    for i in range(n):
        point_i = points[i]
        
        # 查找半径内的邻居
        neighbor_indices = tree.query_ball_point(point_i, r_cut)
        
        for j in neighbor_indices:
            if i == j:
                continue
                
            point_j = points[j]
            diff = point_i - point_j
            dist = np.linalg.norm(diff)
            
            if dist < 1e-10:  # 避免除零
                continue
                
            # Hooke-like软弹簧模型
            if dist < config.target_spacing_m:
                # 排斥力：朝向目标间距
                direction = diff / dist
                force_magnitude = config.repulsion_strength * (config.target_spacing_m - dist)
                force = force_magnitude * direction
                forces[i] += force
    
    # 限制力的大小
    force_magnitudes = np.linalg.norm(forces, axis=1)
    max_force = config.repulsion_cap * config.target_spacing_m
    scale = np.minimum(1.0, max_force / (force_magnitudes + 1e-10))
    forces *= scale[:, np.newaxis]
    
    # RK2时间步进
    k1 = forces
    mid_points = points + 0.5 * config.dt * k1
    
    # 重新计算中点处的力（简化版，使用原力）
    k2 = forces  # 简化：假设力变化不大
    
    new_points = points + config.dt * k2
    
    # 边界约束：投影到多边形内
    for i in range(n):
        point = Point(new_points[i])
        if not poly.contains(point):
            # 投影到最近边界点
            boundary = poly.boundary
            nearest_point = boundary.interpolate(boundary.project(point))
            new_points[i] = [nearest_point.x, nearest_point.y]
    
    return new_points, forces

def interior_count(points: np.ndarray, poly: Polygon, boundary_tol: float) -> int:
    """
    统计与边界距离 > boundary_tol 的点数
    """
    if len(points) == 0:
        return 0
    
    count = 0
    for point_coords in points:
        point = Point(point_coords)
        dist_to_boundary = poly.boundary.distance(point)
        if dist_to_boundary > boundary_tol:
            count += 1
    
    return count

def run_for_parcel(poly: Polygon, config: Config) -> Dict:
    """
    对单个地块运行排斥力迭代
    返回结果字典
    """
    print(f"开始处理地块，目标间距: {config.target_spacing_m}m")
    
    # 初始化点
    points = init_points_in_parcel(poly, config.num_points_per_parcel, config.seed, config.init_spread_sigma)
    print(f"初始化 {len(points)} 个点")
    
    if len(points) == 0:
        return {
            'points': np.array([]),
            'steps': 0,
            'interior': 0,
            'hit_threshold': False,
            'nn_stats': {'p5': 0, 'median': 0, 'p95': 0}
        }
    
    # 迭代
    prev_interior = -1
    hit_threshold = False
    
    for step in range(config.max_iters):
        # 计算当前非边界点数量
        current_interior = interior_count(points, poly, config.boundary_tol_m)
        
        # 检查是否首次达到50个非边界点
        if prev_interior != 50 and current_interior == 50:
            print(f"步骤 {step}: 首次达到50个非边界点，停止迭代")
            hit_threshold = True
            break
        
        # 执行排斥力更新
        points, forces = repel_step(points, poly, config)
        
        prev_interior = current_interior
        
        # 每100步输出一次进度
        if step % 100 == 0:
            print(f"步骤 {step}: 非边界点数 = {current_interior}")
    
    # 计算最终统计
    final_interior = interior_count(points, poly, config.boundary_tol_m)
    
    # 计算最近邻距离统计
    if len(points) > 1:
        tree = cKDTree(points)
        distances = []
        for i, point in enumerate(points):
            # 找到最近邻（排除自己）
            dists, indices = tree.query(point, k=2)
            if len(dists) > 1:
                distances.append(dists[1])  # 第二近的点（第一近是自己）
        
        if distances:
            nn_stats = {
                'p5': np.percentile(distances, 5),
                'median': np.percentile(distances, 50),
                'p95': np.percentile(distances, 95)
            }
        else:
            nn_stats = {'p5': 0, 'median': 0, 'p95': 0}
    else:
        nn_stats = {'p5': 0, 'median': 0, 'p95': 0}
    
    print(f"完成: 总步数={step+1}, 最终非边界点数={final_interior}, 达到阈值={hit_threshold}")
    print(f"最近邻距离统计: P5={nn_stats['p5']:.1f}m, 中位数={nn_stats['median']:.1f}m, P95={nn_stats['p95']:.1f}m")
    
    return {
        'points': points,
        'steps': step + 1,
        'interior': final_interior,
        'hit_threshold': hit_threshold,
        'nn_stats': nn_stats
    }

def save_outputs(parcel_id: str, result: Dict, poly: Polygon, out_dir: str):
    """保存CSV、JSON和可视化结果"""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/plots", exist_ok=True)
    
    # 保存CSV
    csv_path = f"{out_dir}/{parcel_id}_points.csv"
    points = result['points']
    with open(csv_path, 'w') as f:
        f.write("parcel_id,x,y\n")
        for point in points:
            f.write(f"{parcel_id},{point[0]:.6f},{point[1]:.6f}\n")
    
    # 可视化
    plt.figure(figsize=(12, 10))
    
    # 绘制多边形边界
    if hasattr(poly, 'exterior'):
        x, y = poly.exterior.xy
        plt.plot(x, y, 'k-', linewidth=2, label='边界')
        
        # 绘制洞
        for interior in poly.interiors:
            ix, iy = interior.xy
            plt.plot(ix, iy, 'k-', linewidth=1, alpha=0.7)
    
    # 绘制点
    if len(points) > 0:
        # 计算边界距离，区分边界点和内部点
        boundary_points = []
        interior_points = []
        
        for point_coords in points:
            point = Point(point_coords)
            dist_to_boundary = poly.boundary.distance(point)
            if dist_to_boundary <= 1.0:  # 边界容差
                boundary_points.append(point_coords)
            else:
                interior_points.append(point_coords)
        
        if interior_points:
            interior_points = np.array(interior_points)
            plt.scatter(interior_points[:, 0], interior_points[:, 1], 
                      c='blue', s=20, alpha=0.7, label=f'内部点 ({len(interior_points)})')
        
        if boundary_points:
            boundary_points = np.array(boundary_points)
            plt.scatter(boundary_points[:, 0], boundary_points[:, 1], 
                      c='red', s=20, alpha=0.7, label=f'边界点 ({len(boundary_points)})')
    
    plt.xlabel('X (米)')
    plt.ylabel('Y (米)')
    plt.title(f'{parcel_id} - 最终布点结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plot_path = f"{out_dir}/plots/{parcel_id}_final.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存: {csv_path}, {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='多地块排斥力布点系统')
    parser.add_argument('--parcels', required=True, help='地块文件路径')
    parser.add_argument('--out', default='out', help='输出目录')
    parser.add_argument('--target-spacing', type=float, default=100.0, help='目标间距(米)')
    parser.add_argument('--n-per-parcel', type=int, default=100, help='每地块点数')
    parser.add_argument('--boundary-tol', type=float, default=1.0, help='边界容差(米)')
    parser.add_argument('--max-iters', type=int, default=5000, help='最大迭代次数')
    parser.add_argument('--dt', type=float, default=0.6, help='时间步长')
    parser.add_argument('--repulsion-strength', type=float, default=1.0, help='排斥力强度')
    parser.add_argument('--repulsion-cap', type=float, default=2.0, help='排斥力上限倍数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--init-spread', type=float, default=5.0, help='初始化散布半径(米)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config(
        target_spacing_m=args.target_spacing,
        num_points_per_parcel=args.n_per_parcel,
        boundary_tol_m=args.boundary_tol,
        max_iters=args.max_iters,
        dt=args.dt,
        repulsion_strength=args.repulsion_strength,
        repulsion_cap=args.repulsion_cap,
        seed=args.seed,
        init_spread_sigma=args.init_spread
    )
    
    # 加载地块
    parcels = load_parcels_from_txt(args.parcels)
    if not parcels:
        print("错误：未找到有效地块")
        return
    
    # 处理每个地块
    summary = {}
    
    for parcel_id, poly in parcels.items():
        print(f"\n处理 {parcel_id}...")
        result = run_for_parcel(poly, config)
        
        # 保存结果
        save_outputs(parcel_id, result, poly, args.out)
        
        # 记录摘要
        summary[parcel_id] = {
            'steps': result['steps'],
            'interior_count': result['interior'],
            'hit_threshold': result['hit_threshold'],
            'nn_stats': result['nn_stats']
        }
    
    # 保存总摘要
    summary_path = f"{args.out}/summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n完成！结果保存在 {args.out}/")
    print(f"摘要: {summary_path}")
    
    # 打印摘要
    print("\n=== 处理摘要 ===")
    for parcel_id, stats in summary.items():
        print(f"{parcel_id}:")
        print(f"  迭代步数: {stats['steps']}")
        print(f"  最终非边界点数: {stats['interior_count']}")
        print(f"  达到阈值: {stats['hit_threshold']}")
        print(f"  最近邻距离 - P5: {stats['nn_stats']['p5']:.1f}m, "
              f"中位数: {stats['nn_stats']['median']:.1f}m, "
              f"P95: {stats['nn_stats']['p95']:.1f}m")

if __name__ == '__main__':
    main()
