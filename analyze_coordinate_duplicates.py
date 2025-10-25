#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析导出文件中的坐标重复情况
"""

import os
import re
from collections import defaultdict

def parse_coordinates_from_file(file_path):
    """从文件中解析坐标信息"""
    coordinates = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return coordinates
            
            # 解析格式: agent_id(x,y,z)value
            pattern = r'(\d+)\(([^,]+),([^,]+),([^)]+)\)([^,\s]*)'
            matches = re.findall(pattern, content)
            
            for match in matches:
                agent_id, x, y, z, value = match
                coord = (float(x), float(y), float(z))
                coordinates.append({
                    'agent_id': agent_id,
                    'coordinate': coord,
                    'value': value,
                    'file': os.path.basename(file_path)
                })
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")
    
    return coordinates

def analyze_coordinate_duplicates(directory="outputs"):
    """分析坐标重复情况"""
    all_coordinates = []
    coordinate_to_agents = defaultdict(list)
    coordinate_to_files = defaultdict(list)
    
    # 读取所有导出文件
    for i in range(1, 31):
        file_path = os.path.join(directory, f"export_month_{i:02d}.txt")
        if os.path.exists(file_path):
            coords = parse_coordinates_from_file(file_path)
            all_coordinates.extend(coords)
            
            for coord_info in coords:
                coord = coord_info['coordinate']
                coordinate_to_agents[coord].append(coord_info['agent_id'])
                coordinate_to_files[coord].append(coord_info['file'])
    
    print(f"=== 坐标重复分析报告 ===")
    print(f"总坐标数: {len(all_coordinates)}")
    print(f"唯一坐标数: {len(coordinate_to_agents)}")
    
    # 找出重复的坐标
    duplicate_coords = []
    for coord, agents in coordinate_to_agents.items():
        if len(agents) > 1:
            duplicate_coords.append({
                'coordinate': coord,
                'agents': agents,
                'files': coordinate_to_files[coord],
                'count': len(agents)
            })
    
    if duplicate_coords:
        print(f"\n发现 {len(duplicate_coords)} 个重复坐标:")
        for dup in sorted(duplicate_coords, key=lambda x: x['count'], reverse=True):
            print(f"坐标 {dup['coordinate']}: 被 {dup['count']} 次选择")
            print(f"  - Agent IDs: {dup['agents']}")
            print(f"  - 文件: {dup['files']}")
            print()
    else:
        print("\n✅ 没有发现重复坐标！")
    
    # 按agent统计
    agent_stats = defaultdict(int)
    for coord_info in all_coordinates:
        agent_stats[coord_info['agent_id']] += 1
    
    print(f"\n=== Agent统计 ===")
    for agent_id, count in sorted(agent_stats.items()):
        print(f"Agent {agent_id}: {count} 个坐标")
    
    return duplicate_coords

if __name__ == "__main__":
    duplicates = analyze_coordinate_duplicates()
    
    if duplicates:
        print(f"\n[ERROR] 发现 {len(duplicates)} 个重复坐标")
    else:
        print(f"\n[SUCCESS] 所有坐标都是唯一的")
