#!/usr/bin/env python3
"""
分析建筑位置分布问题
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_building_positions(json_file):
    """分析建筑位置分布"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    residential_buildings = []
    commercial_buildings = []
    public_buildings = []
    
    for building in data['buildings']:
        if building['type'] == 'residential':
            residential_buildings.append(building)
        elif building['type'] == 'commercial':
            commercial_buildings.append(building)
        elif building['type'] == 'public':
            public_buildings.append(building)
    
    print(f"=== 建筑位置分析 ===")
    print(f"住宅建筑数量: {len(residential_buildings)}")
    print(f"商业建筑数量: {len(commercial_buildings)}")
    print(f"公共建筑数量: {len(public_buildings)}")
    
    # 分析住宅建筑
    print(f"\n=== 住宅建筑分析 ===")
    for i, building in enumerate(residential_buildings):
        pos = building['position']
        sdf = building['sdf_value']
        print(f"res_{i+1}: 位置[{pos[0]:3d}, {pos[1]:3d}], SDF值: {sdf:.3f}")
    
    # 分析商业建筑
    print(f"\n=== 商业建筑分析 ===")
    for i, building in enumerate(commercial_buildings):
        pos = building['position']
        sdf = building['sdf_value']
        print(f"com_{i+1}: 位置[{pos[0]:3d}, {pos[1]:3d}], SDF值: {sdf:.3f}")
    
    # 检查位置重复
    print(f"\n=== 位置重复检查 ===")
    all_positions = [building['position'] for building in data['buildings']]
    unique_positions = set()
    duplicates = []
    
    for pos in all_positions:
        pos_tuple = tuple(pos)
        if pos_tuple in unique_positions:
            duplicates.append(pos)
        else:
            unique_positions.add(pos_tuple)
    
    if duplicates:
        print(f"发现 {len(duplicates)} 个重复位置:")
        for pos in duplicates:
            print(f"  重复位置: [{pos[0]}, {pos[1]}]")
    else:
        print("没有发现重复位置")
    
    # 检查边界问题
    print(f"\n=== 边界检查 ===")
    edge_buildings = []
    for building in data['buildings']:
        pos = building['position']
        if pos[0] <= 5 or pos[0] >= 251 or pos[1] <= 5 or pos[1] >= 251:
            edge_buildings.append(building)
    
    if edge_buildings:
        print(f"发现 {len(edge_buildings)} 个边界建筑:")
        for building in edge_buildings:
            pos = building['position']
            print(f"  {building['id']}: 位置[{pos[0]}, {pos[1]}]")
    else:
        print("没有发现边界建筑")
    
    # 可视化
    visualize_positions(residential_buildings, commercial_buildings, public_buildings)

def visualize_positions(residential, commercial, public):
    """可视化建筑位置"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制住宅建筑
    if residential:
        res_x = [b['position'][0] for b in residential]
        res_y = [b['position'][1] for b in residential]
        ax.scatter(res_x, res_y, c='yellow', s=100, label='Residential', alpha=0.7, edgecolors='black')
    
    # 绘制商业建筑
    if commercial:
        com_x = [b['position'][0] for b in commercial]
        com_y = [b['position'][1] for b in commercial]
        ax.scatter(com_x, com_y, c='orange', s=80, label='Commercial', alpha=0.7, edgecolors='black')
    
    # 绘制公共建筑
    if public:
        pub_x = [b['position'][0] for b in public]
        pub_y = [b['position'][1] for b in public]
        ax.scatter(pub_x, pub_y, c='cyan', s=120, label='Public', alpha=0.7, edgecolors='black')
    
    # 绘制交通枢纽
    ax.scatter([40, 216], [128, 128], c='blue', s=200, marker='s', label='Transport Hubs', alpha=0.8)
    
    # 绘制主干道
    ax.plot([40, 216], [128, 128], 'gray', linewidth=3, alpha=0.5, label='Main Road')
    
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Building Position Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('building_positions_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 分析最新的建筑位置文件
    json_file = "enhanced_simulation_v2_3_output/building_positions_month_21.json"
    analyze_building_positions(json_file)
