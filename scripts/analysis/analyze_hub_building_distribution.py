#!/usr/bin/env python3
"""
分析三个Hub附近的建筑分布情况
比较Hub1、Hub2、Hub3的建筑数量是否合理
"""

import json
import os
import glob
import math

def analyze_hub_building_distribution():
    """分析Hub建筑分布"""
    
    # Hub位置
    hub1_pos = [20, 55]  # Hub1
    hub2_pos = [90, 55]  # Hub2  
    hub3_pos = [67, 94]  # Hub3
    
    print("=== Hub建筑分布分析 ===")
    print(f"Hub1位置: {hub1_pos}")
    print(f"Hub2位置: {hub2_pos}")
    print(f"Hub3位置: {hub3_pos}")
    print()
    
    # 分析范围（30格半径）
    analysis_radius = 30
    
    # 统计每个Hub附近的建筑
    hub_stats = {
        'hub1': {'residential': 0, 'commercial': 0, 'industrial': 0, 'total': 0},
        'hub2': {'residential': 0, 'commercial': 0, 'industrial': 0, 'total': 0},
        'hub3': {'residential': 0, 'commercial': 0, 'industrial': 0, 'total': 0}
    }
    
    # 读取所有simplified文件
    simplified_files = glob.glob("enhanced_simulation_v3_1_output/simplified/simplified_buildings_*.txt")
    simplified_files = sorted(simplified_files)
    
    print(f"分析 {len(simplified_files)} 个文件...")
    
    for file_path in simplified_files:
        # 提取月份
        filename = os.path.basename(file_path)
        month_str = filename.replace("simplified_buildings_", "").replace(".txt", "")
        
        try:
            month = int(month_str)
        except ValueError:
            continue
            
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            continue
            
        # 解析建筑数据
        buildings = parse_building_data(content)
        
        # 分析每个Hub附近的建筑
        for building in buildings:
            building_type, x, y = building
            
            # 检查是否在Hub1附近
            distance_to_hub1 = math.sqrt((x - hub1_pos[0])**2 + (y - hub1_pos[1])**2)
            if distance_to_hub1 <= analysis_radius:
                if building_type == 0:  # residential
                    hub_stats['hub1']['residential'] += 1
                elif building_type == 1:  # commercial
                    hub_stats['hub1']['commercial'] += 1
                elif building_type == 2:  # industrial
                    hub_stats['hub1']['industrial'] += 1
                hub_stats['hub1']['total'] += 1
            
            # 检查是否在Hub2附近
            distance_to_hub2 = math.sqrt((x - hub2_pos[0])**2 + (y - hub2_pos[1])**2)
            if distance_to_hub2 <= analysis_radius:
                if building_type == 0:  # residential
                    hub_stats['hub2']['residential'] += 1
                elif building_type == 1:  # commercial
                    hub_stats['hub2']['commercial'] += 1
                elif building_type == 2:  # industrial
                    hub_stats['hub2']['industrial'] += 1
                hub_stats['hub2']['total'] += 1
            
            # 检查是否在Hub3附近
            distance_to_hub3 = math.sqrt((x - hub3_pos[0])**2 + (y - hub3_pos[1])**2)
            if distance_to_hub3 <= analysis_radius:
                if building_type == 0:  # residential
                    hub_stats['hub3']['residential'] += 1
                elif building_type == 1:  # commercial
                    hub_stats['hub3']['commercial'] += 1
                elif building_type == 2:  # industrial
                    hub_stats['hub3']['industrial'] += 1
                hub_stats['hub3']['total'] += 1
    
    # 打印统计结果
    print("=== 建筑分布统计 (30格半径内) ===")
    print()
    
    for hub_name, stats in hub_stats.items():
        hub_pos = {'hub1': hub1_pos, 'hub2': hub2_pos, 'hub3': hub3_pos}[hub_name]
        print(f"{hub_name.upper()} ({hub_pos[0]}, {hub_pos[1]}):")
        print(f"  住宅建筑: {stats['residential']} 个")
        print(f"  商业建筑: {stats['commercial']} 个") 
        print(f"  工业建筑: {stats['industrial']} 个")
        print(f"  总计: {stats['total']} 个")
        print()
    
    # 比较分析
    print("=== 比较分析 ===")
    
    # 计算比例
    total_buildings = sum(stats['total'] for stats in hub_stats.values())
    
    for hub_name, stats in hub_stats.items():
        percentage = (stats['total'] / total_buildings * 100) if total_buildings > 0 else 0
        print(f"{hub_name.upper()}: {stats['total']} 个建筑 ({percentage:.1f}%)")
    
    print()
    
    # 分析合理性
    print("=== 合理性分析 ===")
    
    # Hub1和Hub2是主要Hub，应该有更多建筑
    hub1_total = hub_stats['hub1']['total']
    hub2_total = hub_stats['hub2']['total']
    hub3_total = hub_stats['hub3']['total']
    
    print(f"Hub1 (主要Hub): {hub1_total} 个建筑")
    print(f"Hub2 (主要Hub): {hub2_total} 个建筑")
    print(f"Hub3 (新增Hub): {hub3_total} 个建筑")
    
    # 计算比例
    if hub1_total > 0:
        hub3_vs_hub1_ratio = hub3_total / hub1_total
        print(f"Hub3/Hub1 比例: {hub3_vs_hub1_ratio:.2f}")
    
    if hub2_total > 0:
        hub3_vs_hub2_ratio = hub3_total / hub2_total
        print(f"Hub3/Hub2 比例: {hub3_vs_hub2_ratio:.2f}")
    
    # 判断合理性
    if hub3_total > 0:
        if hub3_vs_hub1_ratio >= 0.3 and hub3_vs_hub2_ratio >= 0.3:
            print("✅ Hub3建筑数量合理：与主要Hub相比有足够的建筑生成")
        elif hub3_vs_hub1_ratio >= 0.1 and hub3_vs_hub2_ratio >= 0.1:
            print("⚠️ Hub3建筑数量偏少：与主要Hub相比建筑数量较少")
        else:
            print("❌ Hub3建筑数量过少：与主要Hub相比建筑数量明显不足")
    else:
        print("❌ Hub3没有建筑生成")
    
    return hub_stats

def parse_building_data(content):
    """解析建筑数据"""
    buildings = []
    
    if not content:
        return buildings
    
    # 分割建筑数据
    parts = content.split(', ')
    
    for part in parts:
        if '(' in part and ')' in part:
            try:
                # 解析格式: 类型(坐标, 坐标, 0)
                type_part = part.split('(')[0]
                coords_part = part.split('(')[1].split(')')[0]
                coords = coords_part.split(', ')
                
                if len(coords) >= 2:
                    building_type = int(type_part)
                    x = float(coords[0])
                    y = float(coords[1])
                    buildings.append((building_type, x, y))
            except (ValueError, IndexError):
                continue
    
    return buildings

if __name__ == "__main__":
    analyze_hub_building_distribution()
