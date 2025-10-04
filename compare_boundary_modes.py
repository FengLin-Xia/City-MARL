#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比边界作为约束 vs 边界作为道路的效果差异
"""

import json
import math

def analyze_deformation_result(filepath, mode_name):
    """分析变形结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 计算位移统计
    displacements = []
    for orig, new in zip(data['original'], data['new']):
        dx = new['x'] - orig['x']
        dy = new['y'] - orig['y']
        dist = math.sqrt(dx*dx + dy*dy)
        displacements.append(dist)
    
    avg_displacement = sum(displacements) / len(displacements)
    max_displacement = max(displacements)
    min_displacement = min(displacements)
    
    return {
        'mode': mode_name,
        'avg_displacement': avg_displacement,
        'max_displacement': max_displacement,
        'min_displacement': min_displacement,
        'total_points': len(displacements)
    }

def main():
    print("=== 边界模式对比分析 ===\n")
    
    # 分析两种模式的结果
    mode1 = analyze_deformation_result(
        'enhanced_simulation_v4_0_output/deform_demo_north_boundary/vf_slots.json',
        '边界仅作约束 (boundary constraint only)'
    )
    
    mode2 = analyze_deformation_result(
        'enhanced_simulation_v4_0_output/deform_demo_north_boundary_as_road/vf_slots.json',
        '边界作为道路+约束 (boundary as road + constraint)'
    )
    
    print(f"模式1: {mode1['mode']}")
    print(f"  平均位移: {mode1['avg_displacement']:.3f}px")
    print(f"  最大位移: {mode1['max_displacement']:.3f}px")
    print(f"  最小位移: {mode1['min_displacement']:.3f}px")
    print(f"  总点数: {mode1['total_points']}")
    
    print(f"\n模式2: {mode2['mode']}")
    print(f"  平均位移: {mode2['avg_displacement']:.3f}px")
    print(f"  最大位移: {mode2['max_displacement']:.3f}px")
    print(f"  最小位移: {mode2['min_displacement']:.3f}px")
    print(f"  总点数: {mode2['total_points']}")
    
    # 计算差异
    avg_diff = mode2['avg_displacement'] - mode1['avg_displacement']
    max_diff = mode2['max_displacement'] - mode1['max_displacement']
    avg_ratio = mode2['avg_displacement'] / mode1['avg_displacement'] if mode1['avg_displacement'] > 0 else 1
    
    print(f"\n=== 差异分析 ===")
    print(f"平均位移差异: {avg_diff:+.3f}px ({avg_ratio:.3f}x)")
    print(f"最大位移差异: {max_diff:+.3f}px")
    
    if avg_diff > 0.5:
        print("\n结论: 边界作为道路时，点位移明显增加")
        print("  -> 边界确实起到了道路吸引作用")
    elif avg_diff < -0.5:
        print("\n结论: 边界作为道路时，点位移明显减少")
        print("  -> 边界可能起到了额外的约束作用")
    else:
        print("\n结论: 两种模式下位移差异较小")
        print("  -> 边界作为道路的影响相对较小")
    
    # 分析边界附近点的具体影响
    print(f"\n=== 详细分析 ===")
    print("当边界作为道路时:")
    print("1. 边界多边形会像道路一样产生吸引力和切向梳理力")
    print("2. 边界附近的点会受到双重影响:")
    print("   - 道路的吸引/梳理力")
    print("   - 边界约束力")
    print("3. 这可能导致边界附近的点位移模式发生变化")

if __name__ == '__main__':
    main()
