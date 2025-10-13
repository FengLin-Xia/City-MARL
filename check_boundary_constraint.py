#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查边界约束是否起作用
"""

import json
import math

def point_in_polygon_winding(x, y, polygon):
    """winding number算法检查点是否在多边形内"""
    n = len(polygon)
    if n < 3:
        return False
    
    wn = 0
    for i in range(n - 1):
        x0, y0 = polygon[i]
        x1, y1 = polygon[i + 1]
        if y0 <= y:
            if y1 > y:
                if (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) > 0:
                    wn += 1
        else:
            if y1 <= y:
                if (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) < 0:
                    wn -= 1
    return wn != 0

def main():
    # 读取边界数据
    boundary_points = []
    with open('north_bound.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                boundary_points.append((x, y))
    
    print(f"Loaded {len(boundary_points)} boundary points")
    
    # 读取变形结果
    with open('enhanced_simulation_v4_0_output/deform_demo_north_boundary/vf_slots.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['original'])} deformation results")
    
    # 检查边界约束
    points_outside = []
    points_near_boundary = []
    
    for i, (orig, new) in enumerate(zip(data['original'], data['new'])):
        orig_x, orig_y = orig['x'], orig['y']
        new_x, new_y = new['x'], new['y']
        
        # 检查新点是否在边界内
        if not point_in_polygon_winding(new_x, new_y, boundary_points):
            points_outside.append({
                'index': i,
                'orig': (orig_x, orig_y),
                'new': (new_x, new_y),
                'displacement': math.sqrt((new_x - orig_x)**2 + (new_y - orig_y)**2)
            })
        
        # 计算到边界的最小距离
        min_dist_to_boundary = float('inf')
        for bx, by in boundary_points:
            dist = math.sqrt((orig_x - bx)**2 + (orig_y - by)**2)
            min_dist_to_boundary = min(min_dist_to_boundary, dist)
        
        if min_dist_to_boundary < 5.0:  # 距离边界5像素内的点
            points_near_boundary.append({
                'index': i,
                'orig': (orig_x, orig_y),
                'new': (new_x, new_y),
                'dist_to_boundary': min_dist_to_boundary,
                'displacement': math.sqrt((new_x - orig_x)**2 + (new_y - orig_y)**2)
            })
    
    print(f"\n=== 边界约束分析 ===")
    print(f"边界外的点数量: {len(points_outside)}")
    print(f"边界附近的点数量: {len(points_near_boundary)}")
    
    if points_outside:
        print(f"\n边界外的点示例 (前5个):")
        for pt in points_outside[:5]:
            print(f"  点 {pt['index']}: {pt['orig']} -> {pt['new']} (位移: {pt['displacement']:.2f})")
    
    if points_near_boundary:
        print(f"\n边界附近的点示例 (前5个):")
        for pt in points_near_boundary[:5]:
            print(f"  点 {pt['index']}: {pt['orig']} -> {pt['new']} (距边界: {pt['dist_to_boundary']:.2f}, 位移: {pt['displacement']:.2f})")
    
    # 检查边界约束是否有效
    if len(points_outside) == 0:
        print(f"\n[SUCCESS] 边界约束有效：所有点都被约束在边界内")
    else:
        print(f"\n[FAILED] 边界约束无效：有 {len(points_outside)} 个点越出了边界")
        print("可能原因：")
        print("1. 边界模式设置为 'clip'，但点的位移太大")
        print("2. 边界检测算法有问题")
        print("3. 边界数据格式不正确")

if __name__ == '__main__':
    main()
