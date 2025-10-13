#!/usr/bin/env python3
"""
调试Hub3智能体决策
检查智能体是否优先选择Hub3附近的槽位
"""

import json
import numpy as np
import os

def analyze_hub3_agent_decision():
    """分析Hub3周围的智能体决策情况"""
    
    # Hub3位置
    hub3_x, hub3_y = 67, 94
    
    # 检查几个关键月份的建筑数据
    months_to_check = [2, 10, 20, 30]
    
    print("=== Hub3 智能体决策分析 ===")
    print(f"Hub3 位置: ({hub3_x}, {hub3_y})")
    print()
    
    for month in months_to_check:
        try:
            # 读取建筑数据
            if month == 1:
                filename = f"enhanced_simulation_v3_1_output/building_positions_month_{month:02d}.json"
            else:
                filename = f"enhanced_simulation_v3_1_output/building_delta_month_{month:02d}.json"
            
            if os.path.exists(filename):
                print(f"--- Month {month} ---")
                print(f"✅ 找到建筑文件: {filename}")
                
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 分析建筑数据
                if 'buildings' in data:
                    buildings = data['buildings']
                elif isinstance(data, list):
                    buildings = data
                else:
                    buildings = []
                
                print(f"总建筑数量: {len(buildings)}")
                
                # 按类型分组
                building_types = {}
                for building in buildings:
                    building_type = building.get('type', 'unknown')
                    if building_type not in building_types:
                        building_types[building_type] = []
                    building_types[building_type].append(building)
                
                print("建筑类型分布:")
                for building_type, buildings_list in building_types.items():
                    print(f"  {building_type}: {len(buildings_list)} 个")
                
                # 检查Hub3周围的建筑
                hub3_buildings = []
                for building in buildings:
                    if 'position' in building:
                        x, y = building['position']
                        distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
                        if distance <= 30:  # 30像素范围内
                            hub3_buildings.append((building, distance))
                
                print(f"Hub3 周围建筑数量 (30像素内): {len(hub3_buildings)}")
                
                if hub3_buildings:
                    print("Hub3 周围建筑详情:")
                    for building, distance in sorted(hub3_buildings, key=lambda x: x[1]):
                        building_type = building.get('type', 'unknown')
                        x, y = building['position']
                        building_id = building.get('id', 'unknown')
                        print(f"  ID: {building_id}, 类型: {building_type}, 位置: ({x}, {y}), 距离: {distance:.1f}")
                else:
                    print("❌ Hub3 周围没有建筑")
                    
                    # 检查最近的建筑
                    if buildings:
                        min_distance = float('inf')
                        nearest_building = None
                        for building in buildings:
                            if 'position' in building:
                                x, y = building['position']
                                distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_building = building
                        
                        if nearest_building:
                            x, y = nearest_building['position']
                            building_type = nearest_building.get('type', 'unknown')
                            building_id = nearest_building.get('id', 'unknown')
                            print(f"最近建筑: ID {building_id}, 类型 {building_type}, 位置 ({x}, {y}), 距离: {min_distance:.1f}")
            
            else:
                print(f"--- Month {month} ---")
                print(f"❌ 建筑文件不存在: {filename}")
            
            print()
            
        except Exception as e:
            print(f"❌ 分析出错: {e}")
    
    # 检查智能体决策逻辑
    print("=== 智能体决策逻辑分析 ===")
    print("检查智能体如何选择建筑位置...")
    
    try:
        with open('enhanced_city_simulation_v3_1.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 查找智能体决策相关代码
        if 'GovernmentAgent' in code:
            print("✅ 有政府智能体")
        else:
            print("❌ 没有政府智能体")
            
        if 'BusinessAgent' in code:
            print("✅ 有商业智能体")
        else:
            print("❌ 没有商业智能体")
            
        if 'ResidentAgent' in code:
            print("✅ 有居民智能体")
        else:
            print("❌ 没有居民智能体")
        
        # 查找建筑放置逻辑
        if 'place_building' in code:
            print("✅ 有建筑放置逻辑")
        else:
            print("❌ 没有建筑放置逻辑")
            
        if 'building_placement' in code:
            print("✅ 有建筑放置相关代码")
        else:
            print("❌ 没有建筑放置相关代码")
            
    except Exception as e:
        print(f"❌ 读取代码文件出错: {e}")
    
    # 检查智能体模块
    print("\n=== 智能体模块分析 ===")
    try:
        with open('logic/enhanced_agents.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 查找关键方法
        key_methods = [
            'place_building',
            'select_location',
            'evaluate_location',
            'make_decision'
        ]
        
        for method in key_methods:
            if method in code:
                print(f"✅ 找到 {method} 方法")
            else:
                print(f"❌ 未找到 {method} 方法")
                
    except Exception as e:
        print(f"❌ 读取智能体模块出错: {e}")

if __name__ == "__main__":
    analyze_hub3_agent_decision()
