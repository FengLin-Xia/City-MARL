#!/usr/bin/env python3
"""
地块分析 - 不依赖matplotlib的版本
"""

import json
import numpy as np
from shapely.geometry import Polygon, Point

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

def analyze_parcel(poly: Polygon, parcel_id: str):
    """分析单个地块"""
    print(f"\n=== {parcel_id} ===")
    print(f"面积: {poly.area:.2f} 平方米")
    print(f"周长: {poly.length:.2f} 米")
    print(f"质心: ({poly.centroid.x:.2f}, {poly.centroid.y:.2f})")
    print(f"边界框: {poly.bounds}")
    print(f"是否有效: {poly.is_valid}")
    print(f"是否有洞: {len(poly.interiors) > 0}")
    
    return {
        'area': poly.area,
        'perimeter': poly.length,
        'centroid': [poly.centroid.x, poly.centroid.y],
        'bounds': poly.bounds,
        'is_valid': poly.is_valid,
        'has_holes': len(poly.interiors) > 0,
        'num_holes': len(poly.interiors)
    }

def generate_points_in_parcel(poly: Polygon, n: int, seed: int = 42):
    """在地块内生成点"""
    np.random.seed(seed)
    centroid = poly.centroid
    cx, cy = centroid.x, centroid.y
    
    points = []
    attempts = 0
    max_attempts = n * 20
    
    while len(points) < n and attempts < max_attempts:
        # 在质心附近高斯分布采样
        x = np.random.normal(cx, 10)  # 标准差10米
        y = np.random.normal(cy, 10)
        point = Point(x, y)
        
        if poly.contains(point) or poly.touches(point):
            points.append([x, y])
        
        attempts += 1
    
    print(f"成功生成 {len(points)} 个点 (尝试 {attempts} 次)")
    return np.array(points)

def calculate_point_distances(points):
    """计算点之间的距离统计"""
    if len(points) < 2:
        return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
    
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    
    distances = []
    for i, point in enumerate(points):
        # 找到最近邻（排除自己）
        dists, indices = tree.query(point, k=2)
        if len(dists) > 1:
            distances.append(dists[1])  # 第二近的点
    
    if distances:
        return {
            'min': min(distances),
            'max': max(distances),
            'mean': np.mean(distances),
            'std': np.std(distances),
            'median': np.median(distances)
        }
    else:
        return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'median': 0}

def main():
    print("开始地块分析...")
    
    # 加载地块
    parcels = load_parcels_from_txt('parcel.txt')
    print(f"成功加载 {len(parcels)} 个地块")
    
    # 分析每个地块
    results = {}
    
    for parcel_id, poly in parcels.items():
        # 分析地块
        analysis = analyze_parcel(poly, parcel_id)
        
        # 生成测试点
        test_points = generate_points_in_parcel(poly, 50, seed=42)
        
        # 计算点距离统计
        if len(test_points) > 1:
            distances = calculate_point_distances(test_points)
            print(f"点距离统计: 最小={distances['min']:.2f}m, 最大={distances['max']:.2f}m, 平均={distances['mean']:.2f}m")
        else:
            distances = {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'median': 0}
        
        # 保存结果
        results[parcel_id] = {
            'analysis': analysis,
            'test_points': test_points.tolist(),
            'distances': distances
        }
    
    # 保存结果到JSON
    with open('parcel_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n分析完成！结果保存在 parcel_analysis_results.json")
    
    # 打印总结
    print("\n=== 总结 ===")
    total_area = sum(r['analysis']['area'] for r in results.values())
    total_perimeter = sum(r['analysis']['perimeter'] for r in results.values())
    print(f"总地块数: {len(results)}")
    print(f"总面积: {total_area:.2f} 平方米")
    print(f"总周长: {total_perimeter:.2f} 米")
    
    # 找出最大和最小的地块
    areas = [(pid, r['analysis']['area']) for pid, r in results.items()]
    areas.sort(key=lambda x: x[1])
    
    print(f"最小地块: {areas[0][0]} ({areas[0][1]:.2f} 平方米)")
    print(f"最大地块: {areas[-1][0]} ({areas[-1][1]:.2f} 平方米)")

if __name__ == '__main__':
    main()









