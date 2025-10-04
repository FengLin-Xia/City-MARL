#!/usr/bin/env python3
"""
可视化第20步的parcel_1点分布
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

def load_parcels_from_txt(file_path: str):
    """解析parcel.txt文件"""
    parcels = {}
    current_parcel = []
    parcel_id = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[':
                if current_parcel and len(current_parcel) >= 3:
                    try:
                        poly = Polygon(current_parcel)
                        if poly.is_valid:
                            parcels[f'parcel_{parcel_id}'] = poly
                            parcel_id += 1
                    except Exception as e:
                        print(f"警告：地块 {parcel_id} 无效: {e}")
                current_parcel = []
            elif line == ']':
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
                try:
                    coords = [float(x.strip()) for x in line.split(',')]
                    if len(coords) >= 2:
                        current_parcel.append((coords[0], coords[1]))
                except ValueError:
                    continue
    
    return parcels

def init_points_in_parcel(poly: Polygon, n: int, seed: int, sigma: float):
    """在多边形内初始化点"""
    np.random.seed(seed)
    centroid = poly.centroid
    cx, cy = centroid.x, centroid.y
    
    points = []
    attempts = 0
    max_attempts = n * 20
    
    while len(points) < n and attempts < max_attempts:
        x = np.random.normal(cx, sigma)
        y = np.random.normal(cy, sigma)
        point = Point(x, y)
        
        if poly.contains(point) or poly.touches(point):
            points.append([x, y])
        attempts += 1
    
    return np.array(points)

def repel_step(points, poly, target_spacing, dt, repulsion_strength, repulsion_cap):
    """执行一次排斥力更新"""
    n = len(points)
    if n == 0:
        return points
    
    tree = cKDTree(points)
    r_cut = 2 * target_spacing
    forces = np.zeros_like(points)
    
    # 计算排斥力
    for i in range(n):
        point_i = points[i]
        neighbor_indices = tree.query_ball_point(point_i, r_cut)
        
        for j in neighbor_indices:
            if i == j:
                continue
                
            point_j = points[j]
            diff = point_i - point_j
            dist = np.linalg.norm(diff)
            
            if dist < 1e-10:
                continue
                
            if dist < target_spacing:
                direction = diff / dist
                force_magnitude = repulsion_strength * (target_spacing - dist)
                force = force_magnitude * direction
                forces[i] += force
    
    # 限制力的大小
    force_magnitudes = np.linalg.norm(forces, axis=1)
    max_force = repulsion_cap * target_spacing
    scale = np.minimum(1.0, max_force / (force_magnitudes + 1e-10))
    forces *= scale[:, np.newaxis]
    
    # 更新位置
    new_points = points + dt * forces
    
    # 边界约束
    for i in range(n):
        point = Point(new_points[i])
        if not poly.contains(point):
            boundary = poly.boundary
            nearest_point = boundary.interpolate(boundary.project(point))
            new_points[i] = [nearest_point.x, nearest_point.y]
    
    return new_points

def interior_count(points, poly, boundary_tol):
    """统计非边界点数量"""
    if len(points) == 0:
        return 0
    
    count = 0
    for point_coords in points:
        point = Point(point_coords)
        dist_to_boundary = poly.boundary.distance(point)
        if dist_to_boundary > boundary_tol:
            count += 1
    return count

def visualize_step20():
    """可视化第20步的parcel_1点分布"""
    print("开始可视化第20步的parcel_1点分布...")
    
    # 加载地块
    parcels = load_parcels_from_txt('parcel.txt')
    if 'parcel_1' not in parcels:
        print("错误：未找到parcel_1")
        return
    
    poly = parcels['parcel_1']
    print(f"parcel_1 面积: {poly.area:.2f} 平方米")
    print(f"parcel_1 质心: ({poly.centroid.x:.2f}, {poly.centroid.y:.2f})")
    
    # 参数设置
    target_spacing = 10.0  # 5像素 = 10米
    n_per_parcel = 50
    boundary_tol = 1.0
    dt = 0.6
    repulsion_strength = 1.0
    repulsion_cap = 2.0
    seed = 42
    init_spread_sigma = 2.0
    
    # 初始化点
    points = init_points_in_parcel(poly, n_per_parcel, seed, init_spread_sigma)
    print(f"初始化 {len(points)} 个点")
    
    # 运行20步迭代
    for step in range(20):
        current_interior = interior_count(points, poly, boundary_tol)
        print(f"步骤 {step}: 非边界点数 = {current_interior}")
        
        points = repel_step(points, poly, target_spacing, dt, repulsion_strength, repulsion_cap)
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：初始状态
    ax1.set_title('初始状态 (步骤 0)')
    
    # 重新初始化点用于初始状态显示
    init_points = init_points_in_parcel(poly, n_per_parcel, seed, init_spread_sigma)
    
    # 绘制多边形
    if hasattr(poly, 'exterior'):
        x, y = poly.exterior.xy
        ax1.plot(x, y, 'k-', linewidth=2, label='边界')
        ax1.fill(x, y, alpha=0.1, color='lightblue')
        
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax1.plot(ix, iy, 'k-', linewidth=1, alpha=0.7)
            ax1.fill(ix, iy, color='white', alpha=0.8)
    
    # 绘制初始点
    if len(init_points) > 0:
        ax1.scatter(init_points[:, 0], init_points[:, 1], 
                   c='red', s=20, alpha=0.7, label=f'初始点 ({len(init_points)})')
    
    ax1.set_xlabel('X (米)')
    ax1.set_ylabel('Y (米)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 右图：第20步状态
    ax2.set_title('第20步状态')
    
    # 绘制多边形
    if hasattr(poly, 'exterior'):
        x, y = poly.exterior.xy
        ax2.plot(x, y, 'k-', linewidth=2, label='边界')
        ax2.fill(x, y, alpha=0.1, color='lightblue')
        
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax2.plot(ix, iy, 'k-', linewidth=1, alpha=0.7)
            ax2.fill(ix, iy, color='white', alpha=0.8)
    
    # 绘制第20步的点，区分边界点和内部点
    if len(points) > 0:
        boundary_points = []
        interior_points = []
        
        for point_coords in points:
            point = Point(point_coords)
            dist_to_boundary = poly.boundary.distance(point)
            if dist_to_boundary <= 1.0:
                boundary_points.append(point_coords)
            else:
                interior_points.append(point_coords)
        
        if interior_points:
            interior_points = np.array(interior_points)
            ax2.scatter(interior_points[:, 0], interior_points[:, 1], 
                       c='blue', s=20, alpha=0.7, label=f'内部点 ({len(interior_points)})')
        
        if boundary_points:
            boundary_points = np.array(boundary_points)
            ax2.scatter(boundary_points[:, 0], boundary_points[:, 1], 
                       c='red', s=20, alpha=0.7, label=f'边界点 ({len(boundary_points)})')
        
        # 计算最近邻距离统计
        if len(points) > 1:
            tree = cKDTree(points)
            distances = []
            for i, point in enumerate(points):
                dists, indices = tree.query(point, k=2)
                if len(dists) > 1:
                    distances.append(dists[1])
            
            if distances:
                print(f"第20步最近邻距离统计:")
                print(f"  最小: {min(distances):.2f}米")
                print(f"  最大: {max(distances):.2f}米")
                print(f"  平均: {np.mean(distances):.2f}米")
                print(f"  中位数: {np.median(distances):.2f}米")
    
    ax2.set_xlabel('X (米)')
    ax2.set_ylabel('Y (米)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('parcel_1_step20_visualization.png', dpi=150, bbox_inches='tight')
    print("可视化结果已保存到 parcel_1_step20_visualization.png")
    plt.show()

if __name__ == '__main__':
    visualize_step20()

