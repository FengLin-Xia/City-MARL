#!/usr/bin/env python3
"""
检查槽位使用情况
分析为什么第1个月后有建筑但后续没有新建筑
"""

import json
import numpy as np

def check_slot_usage():
    """检查槽位使用情况"""
    
    print("=== 检查槽位使用情况 ===")
    
    # 读取第1个月的建筑数据
    with open('enhanced_simulation_v3_1_output/building_positions_month_01.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    buildings = data.get('buildings', [])
    print(f"第1个月总建筑数量: {len(buildings)}")
    
    # 按类型分组
    building_types = {}
    for building in buildings:
        building_type = building.get('type', 'unknown')
        if building_type not in building_types:
            building_types[building_type] = []
        building_types[building_type].append(building)
    
    print("第1个月建筑类型分布:")
    for building_type, buildings_list in building_types.items():
        print(f"  {building_type}: {len(buildings_list)} 个")
    
    # 分析建筑位置分布
    print("\n第1个月建筑位置分析:")
    for building_type, buildings_list in building_types.items():
        if buildings_list:
            positions = [building['position'] for building in buildings_list]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            print(f"  {building_type}:")
            print(f"    X范围: {min(x_coords)} - {max(x_coords)}")
            print(f"    Y范围: {min(y_coords)} - {max(y_coords)}")
            print(f"    中心: ({np.mean(x_coords):.1f}, {np.mean(y_coords):.1f})")
    
    # 检查Hub3附近的建筑
    hub3_x, hub3_y = 67, 94
    hub3_buildings = []
    for building in buildings:
        x, y = building['position']
        distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
        if distance <= 30:
            hub3_buildings.append((building, distance))
    
    print(f"\nHub3附近建筑 (30像素内): {len(hub3_buildings)} 个")
    if hub3_buildings:
        for building, distance in sorted(hub3_buildings, key=lambda x: x[1]):
            building_type = building.get('type', 'unknown')
            x, y = building['position']
            building_id = building.get('id', 'unknown')
            print(f"  ID: {building_id}, 类型: {building_type}, 位置: ({x}, {y}), 距离: {distance:.1f}")
    
    # 检查层状态
    print("\n=== 检查层状态 ===")
    try:
        with open('enhanced_simulation_v3_1_output/layer_state_month_02.json', 'r', encoding='utf-8') as f:
            layer_data = json.load(f)
        
        layers = layer_data.get('layers', {})
        commercial_layers = layers.get('commercial', [])
        residential_layers = layers.get('residential', [])
        
        print("商业建筑层状态:")
        for i, layer in enumerate(commercial_layers):
            status_icon = {
                'locked': '🔒',
                'active': '🟢',
                'complete': '✅'
            }.get(layer['status'], '❓')
            
            print(f"  {status_icon} 层 {i}: {layer['layer_id']} - {layer['status']}")
            print(f"    容量: {layer['placed']}/{layer['capacity_effective']} (密度: {layer['density']:.1%})")
            print(f"    激活季度: {layer['activated_quarter'] if layer['activated_quarter'] >= 0 else '未激活'}")
        
        print("\n住宅建筑层状态:")
        for i, layer in enumerate(residential_layers):
            status_icon = {
                'locked': '🔒',
                'active': '🟢',
                'complete': '✅'
            }.get(layer['status'], '❓')
            
            print(f"  {status_icon} 层 {i}: {layer['layer_id']} - {layer['status']}")
            print(f"    容量: {layer['placed']}/{layer['capacity_effective']} (密度: {layer['density']:.1%})")
            print(f"    激活季度: {layer['activated_quarter'] if layer['activated_quarter'] >= 0 else '未激活'}")
        
        # 分析问题
        print("\n=== 问题分析 ===")
        
        # 检查是否有激活的层
        active_commercial = [layer for layer in commercial_layers if layer['status'] == 'active']
        active_residential = [layer for layer in residential_layers if layer['status'] == 'active']
        
        print(f"激活的层数量:")
        print(f"  商业建筑: {len(active_commercial)} 个")
        print(f"  住宅建筑: {len(active_residential)} 个")
        
        if len(active_commercial) > 0:
            for layer in active_commercial:
                available_slots = layer['capacity_effective'] - layer['placed']
                print(f"  商业层 {layer['layer_id']}: 可用槽位 {available_slots}")
        
        if len(active_residential) > 0:
            for layer in active_residential:
                available_slots = layer['capacity_effective'] - layer['placed']
                print(f"  住宅层 {layer['layer_id']}: 可用槽位 {available_slots}")
        
        # 检查是否所有激活层都已满
        all_commercial_full = all(layer['density'] >= 0.95 for layer in active_commercial)
        all_residential_full = all(layer['density'] >= 0.95 for layer in active_residential)
        
        if all_commercial_full:
            print("⚠️ 所有激活的商业层都已满 (密度≥95%)")
        if all_residential_full:
            print("⚠️ 所有激活的住宅层都已满 (密度≥95%)")
        
        if all_commercial_full and all_residential_full:
            print("❌ 问题确认：所有激活层都已满，无法生成新建筑")
            print("   需要激活下一层或增加新的等值线层")
    
    except Exception as e:
        print(f"❌ 读取层状态文件出错: {e}")

if __name__ == "__main__":
    check_slot_usage()
