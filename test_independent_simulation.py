#!/usr/bin/env python3
"""
测试独立地块模拟
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
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

def init_points(polygon, num_points, seed):
    """初始化点"""
    np.random.seed(seed)
    centroid = polygon.centroid
    cx, cy = centroid.x, centroid.y
    
    points = []
    attempts = 0
    max_attempts = num_points * 20
    
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    sigma = min(width, height) * 0.1
    
    while len(points) < num_points and attempts < max_attempts:
        x = np.random.normal(cx, sigma)
        y = np.random.normal(cy, sigma)
        point = Point(x, y)
        
        if polygon.contains(point):
            points.append([x, y])
        attempts += 1
    
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

def compute_repulsion_forces(points, force_cutoff=5.0, eps=1e-6):
    """计算排斥力"""
    n = len(points)
    if n <= 1:
        return np.zeros_like(points)
    
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, eps)
    
    mask = dist < force_cutoff
    force_magnitude = np.where(mask, 1.0 / (dist**2), 0.0)
    direction = diff / dist[:, :, None]
    forces = np.sum(force_magnitude[:, :, None] * direction, axis=1)
    
    return forces

def step(points, polygon, config):
    """执行一步"""
    if len(points) == 0:
        return points, {}, True
    
    # 计算力
    forces = compute_repulsion_forces(points, config['force_cutoff'], config['eps'])
    
    # 计算平均距离
    mean_dist = compute_mean_distance(points, config['k_neighbors'])
    
    # 自适应步长
    if mean_dist > 0:
        scale = np.clip(mean_dist / config['target_dist'], config['min_scale'], config['max_scale'])
        step_size = config['learning_rate'] * scale
    else:
        step_size = config['learning_rate']
    
    # 归一化力
    force_norms = np.linalg.norm(forces, axis=1)
    normalized_forces = np.where(
        force_norms[:, None] > config['eps'],
        forces / force_norms[:, None],
        forces
    )
    
    # 计算位移
    displacement = step_size * normalized_forces
    
    # 裁剪位移
    displacement = np.clip(displacement, -config['max_step'], config['max_step'])
    
    # 更新位置
    new_points = points + displacement
    
    # 边界检测
    valid_points = []
    for point_coords in new_points:
        point = Point(point_coords)
        if polygon.contains(point):
            valid_points.append(point_coords)
    
    valid_points = np.array(valid_points) if valid_points else np.array([]).reshape(0, 2)
    
    # 检查停止条件
    stopped = mean_dist >= config['target_distance_threshold']
    
    stats = {
        'mean_distance': mean_dist,
        'step_size': step_size,
        'remaining_count': len(valid_points)
    }
    
    return valid_points, stats, stopped

def main():
    """主函数"""
    print("=== 独立地块模拟测试 ===")
    
    # 配置
    config = {
        'target_dist': 2.0,
        'target_distance_threshold': 2.0,
        'learning_rate': 0.5,
        'min_scale': 0.5,
        'max_scale': 2.0,
        'max_step': 1.0,
        'force_cutoff': 5.0,
        'eps': 1e-6,
        'k_neighbors': 6,
        'num_points': 50,  # 减少点数以加快测试
        'max_iter': 1000
    }
    
    # 加载地块
    polygons = load_parcels_from_txt('parcel.txt')
    print(f"加载了 {len(polygons)} 个地块")
    
    # 只测试前3个地块
    test_polygons = polygons[:3]
    
    # 初始化
    all_points = []
    parcel_stopped = []
    
    for i, poly in enumerate(test_polygons):
        points = init_points(poly, config['num_points'], 42 + i)
        all_points.append(points)
        parcel_stopped.append(False)
        print(f"地块 {i}: 初始化 {len(points)} 个点")
    
    # 迭代
    iteration = 0
    total_stopped = 0
    
    while iteration < config['max_iter'] and total_stopped < len(test_polygons):
        iteration += 1
        
        for i, (poly, points, stopped) in enumerate(zip(test_polygons, all_points, parcel_stopped)):
            if stopped or len(points) == 0:
                continue
            
            new_points, stats, stopped_now = step(points, poly, config)
            all_points[i] = new_points
            
            if stopped_now:
                parcel_stopped[i] = True
                total_stopped += 1
                print(f"步骤 {iteration}: 地块 {i} 达到距离阈值 {config['target_distance_threshold']}, "
                      f"当前平均距离: {stats['mean_distance']:.3f}, 剩余点数: {len(new_points)}")
        
        # 打印进度
        if iteration % 20 == 0:
            remaining_counts = [len(points) for points in all_points]
            stopped_info = [f"已停止" if stopped else f"{remaining_counts[i]}" 
                           for i, stopped in enumerate(parcel_stopped)]
            print(f"步骤 {iteration}: 各地块剩余点数 {stopped_info}, 已停止: {total_stopped}/{len(test_polygons)}")
    
    # 最终结果
    print(f"\n=== 模拟完成 ===")
    print(f"总迭代步数: {iteration}")
    print(f"已停止地块数: {total_stopped}/{len(test_polygons)}")
    
    for i, (remaining, stopped) in enumerate(zip([len(points) for points in all_points], parcel_stopped)):
        status = "已停止" if stopped else "运行中"
        mean_dist = compute_mean_distance(all_points[i], config['k_neighbors']) if len(all_points[i]) > 0 else 0
        print(f"  地块 {i}: 剩余 {remaining} 点, 状态: {status}, 平均距离: {mean_dist:.3f}")

if __name__ == '__main__':
    main()









