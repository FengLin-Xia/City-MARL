#!/usr/bin/env python3
"""
分析建筑分布模式 - 理解为什么建筑集中在左侧枢纽
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_building_distribution():
    """分析建筑分布模式"""
    
    # 加载建筑位置数据
    with open('enhanced_simulation_v2_3_output/building_positions_month_21.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    buildings = data['buildings']
    
    # 枢纽位置
    hubs = [[40, 128], [216, 128]]
    
    print("🏗️ 建筑分布分析")
    print("=" * 50)
    
    # 按类型统计建筑
    building_types = defaultdict(list)
    for building in buildings:
        building_type = building['type']
        position = building['position']
        building_types[building_type].append(position)
    
    # 分析每种建筑类型的分布
    for building_type, positions in building_types.items():
        print(f"\n📊 {building_type.upper()} 建筑分布:")
        print(f"   总数: {len(positions)}")
        
        if not positions:
            continue
        
        # 计算到两个枢纽的距离
        distances_to_hub1 = []
        distances_to_hub2 = []
        
        for pos in positions:
            dist1 = np.sqrt((pos[0] - hubs[0][0])**2 + (pos[1] - hubs[0][1])**2)
            dist2 = np.sqrt((pos[0] - hubs[1][0])**2 + (pos[1] - hubs[1][1])**2)
            distances_to_hub1.append(dist1)
            distances_to_hub2.append(dist2)
        
        # 统计距离分布
        avg_dist1 = np.mean(distances_to_hub1)
        avg_dist2 = np.mean(distances_to_hub2)
        min_dist1 = np.min(distances_to_hub1)
        min_dist2 = np.min(distances_to_hub2)
        max_dist1 = np.max(distances_to_hub1)
        max_dist2 = np.max(distances_to_hub2)
        
        print(f"   到左侧枢纽 (40,128): 平均={avg_dist1:.1f}, 最小={min_dist1:.1f}, 最大={max_dist1:.1f}")
        print(f"   到右侧枢纽 (216,128): 平均={avg_dist2:.1f}, 最小={min_dist2:.1f}, 最大={max_dist2:.1f}")
        
        # 判断是否集中在某个枢纽
        if avg_dist1 < avg_dist2:
            print(f"   ➡️ 建筑集中在左侧枢纽附近")
        else:
            print(f"   ➡️ 建筑集中在右侧枢纽附近")
        
        # 统计在枢纽附近的建筑数量（距离<50像素）
        near_hub1 = sum(1 for d in distances_to_hub1 if d < 50)
        near_hub2 = sum(1 for d in distances_to_hub2 if d < 50)
        print(f"   距离左侧枢纽<50px: {near_hub1} 个")
        print(f"   距离右侧枢纽<50px: {near_hub2} 个")
    
    # 可视化建筑分布
    visualize_distribution(building_types, hubs)

def visualize_distribution(building_types, hubs):
    """可视化建筑分布"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制枢纽
    for i, hub in enumerate(hubs):
        ax.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                  edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
    
    # 绘制主干道
    ax.plot([hubs[0][0], hubs[1][0]], [hubs[0][1], hubs[1][1]], 
            color='gray', linewidth=3, alpha=0.7, label='Trunk Road')
    
    # 绘制建筑
    colors = {'residential': '#F6C344', 'commercial': '#FD7E14', 'public': '#0B5ED7'}
    markers = {'residential': 's', 'commercial': 'o', 'public': '^'}
    
    for building_type, positions in building_types.items():
        if not positions:
            continue
        
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        ax.scatter(x_coords, y_coords, c=colors[building_type], 
                  marker=markers[building_type], s=50, alpha=0.7,
                  label=f'{building_type.title()} ({len(positions)})')
    
    # 绘制50像素半径的圆圈
    for i, hub in enumerate(hubs):
        circle = plt.Circle((hub[0], hub[1]), 50, fill=False, 
                           color='red', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        ax.text(hub[0], hub[1] + 60, f'Hub {i+1} (50px radius)', 
                ha='center', va='bottom', fontsize=10, color='red')
    
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Building Distribution Analysis - Month 21')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def analyze_sdf_distribution():
    """分析SDF场分布"""
    print("\n🔍 SDF场分布分析")
    print("=" * 30)
    
    # 加载SDF场数据（只读取部分来分析）
    try:
        with open('enhanced_simulation_v2_3_output/sdf_field_month_21.json', 'r', encoding='utf-8') as f:
            # 只读取前几行来了解结构
            first_lines = []
            for i, line in enumerate(f):
                if i < 10:
                    first_lines.append(line)
                else:
                    break
        
        print("SDF场文件结构:")
        for line in first_lines:
            print(f"  {line.strip()}")
            
        print("\n⚠️ SDF场文件过大，无法完整分析")
        print("   建议检查SDF生成逻辑，特别是:")
        print("   1. 点SDF和线SDF的融合是否平衡")
        print("   2. 衰减参数λ是否设置合理")
        print("   3. 主干道SDF是否覆盖整个地图")
        
    except Exception as e:
        print(f"❌ 无法读取SDF场文件: {e}")

if __name__ == "__main__":
    analyze_building_distribution()
    analyze_sdf_distribution()


