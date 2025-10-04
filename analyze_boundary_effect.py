#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析边界约束效果
"""

import json
import math

def main():
    # 读取边界数据
    boundary_points = []
    with open('north_bound.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                boundary_points.append((x, y))

    # 读取变形结果
    with open('enhanced_simulation_v4_0_output/deform_demo_north_boundary/vf_slots.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 分析边界附近的位移模式
    boundary_effects = []
    for i, (orig, new) in enumerate(zip(data['original'], data['new'])):
        orig_x, orig_y = orig['x'], orig['y']
        new_x, new_y = new['x'], new['y']
        
        # 计算到边界的最小距离
        min_dist_to_boundary = float('inf')
        for bx, by in boundary_points:
            dist = math.sqrt((orig_x - bx)**2 + (orig_y - by)**2)
            min_dist_to_boundary = min(min_dist_to_boundary, dist)
        
        displacement = math.sqrt((new_x - orig_x)**2 + (new_y - orig_y)**2)
        
        if min_dist_to_boundary < 15.0:  # 距离边界15像素内的点
            boundary_effects.append({
                'dist_to_boundary': min_dist_to_boundary,
                'displacement': displacement
            })

    # 按距离边界远近排序
    boundary_effects.sort(key=lambda x: x['dist_to_boundary'])

    print('=== 边界约束效果分析 ===')
    print(f'边界附近的总点数 (<15px): {len(boundary_effects)}')
    
    # 按距离分组分析
    very_close = [e for e in boundary_effects if e['dist_to_boundary'] < 3.0]
    close = [e for e in boundary_effects if 3.0 <= e['dist_to_boundary'] < 6.0]
    medium = [e for e in boundary_effects if 6.0 <= e['dist_to_boundary'] < 10.0]
    far = [e for e in boundary_effects if e['dist_to_boundary'] >= 10.0]
    
    print('\n按距离分组的位移统计:')
    groups = [
        ('非常近 (<3px)', very_close),
        ('较近 (3-6px)', close),
        ('中等 (6-10px)', medium),
        ('较远 (>=10px)', far)
    ]
    
    for name, group in groups:
        if group:
            avg_displacement = sum(e['displacement'] for e in group) / len(group)
            max_displacement = max(e['displacement'] for e in group)
            print(f'  {name}: {len(group)}个点, 平均位移: {avg_displacement:.2f}px, 最大位移: {max_displacement:.2f}px')
    
    # 检查边界约束是否明显
    if very_close and far:
        avg_very_close = sum(e['displacement'] for e in very_close) / len(very_close)
        avg_far = sum(e['displacement'] for e in far) / len(far)
        constraint_ratio = avg_far / avg_very_close if avg_very_close > 0 else 0
        print(f'\n约束效果: 远点位移是近点的 {constraint_ratio:.2f} 倍')
        
        if constraint_ratio > 1.5:
            print('  -> 边界约束效果明显：靠近边界的点位移被显著限制')
        elif constraint_ratio > 1.2:
            print('  -> 边界约束效果中等：靠近边界的点位移有所限制')
        else:
            print('  -> 边界约束效果较弱：靠近边界的点位移限制不明显')

if __name__ == '__main__':
    main()
