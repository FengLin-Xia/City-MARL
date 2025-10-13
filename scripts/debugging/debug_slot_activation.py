#!/usr/bin/env python3
"""
调试槽位激活逻辑
检查Hub3的槽位是否被正确激活
"""

import json
import numpy as np
import os

def debug_slot_activation():
    """调试槽位激活逻辑"""
    
    # Hub3位置
    hub3_x, hub3_y = 67, 94
    
    print("=== 调试槽位激活逻辑 ===")
    print(f"Hub3 位置: ({hub3_x}, {hub3_y})")
    print()
    
    # 检查第1个月的建筑分布
    print("--- 第1个月建筑分布分析 ---")
    try:
        with open('enhanced_simulation_v3_1_output/building_positions_month_01.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        buildings = data.get('buildings', [])
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
        
        # 分析建筑位置分布
        print("\n建筑位置分析:")
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
                hub3_buildings = []
                for building in buildings_list:
                    x, y = building['position']
                    distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
                    if distance <= 30:
                        hub3_buildings.append((building, distance))
                
                print(f"    Hub3附近建筑: {len(hub3_buildings)} 个")
                if hub3_buildings:
                    for building, distance in sorted(hub3_buildings, key=lambda x: x[1]):
                        x, y = building['position']
                        print(f"      ID: {building['id']}, 位置: ({x}, {y}), 距离: {distance:.1f}")
        
        # 检查层状态
        print("\n--- 层状态分析 ---")
        try:
            with open('enhanced_simulation_v3_1_output/layer_state_month_02.json', 'r', encoding='utf-8') as f:
                layer_data = json.load(f)
            
            layers = layer_data.get('layers', {})
            
            print("商业建筑层状态:")
            commercial_layers = layers.get('commercial', [])
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
            residential_layers = layers.get('residential', [])
            for i, layer in enumerate(residential_layers):
                status_icon = {
                    'locked': '🔒',
                    'active': '🟢',
                    'complete': '✅'
                }.get(layer['status'], '❓')
                
                print(f"  {status_icon} 层 {i}: {layer['layer_id']} - {layer['status']}")
                print(f"    容量: {layer['placed']}/{layer['capacity_effective']} (密度: {layer['density']:.1%})")
                print(f"    激活季度: {layer['activated_quarter'] if layer['activated_quarter'] >= 0 else '未激活'}")
        
        except Exception as e:
            print(f"❌ 读取层状态文件出错: {e}")
        
        # 分析问题
        print("\n--- 问题分析 ---")
        
        # 检查是否有Hub3附近的建筑
        hub3_nearby = False
        for building in buildings:
            x, y = building['position']
            distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
            if distance <= 30:
                hub3_nearby = True
                break
        
        if not hub3_nearby:
            print("❌ Hub3附近没有建筑")
            print("可能的原因:")
            print("1. 槽位生成问题：Hub3的槽位没有被正确生成")
            print("2. 槽位激活问题：Hub3的槽位没有被激活")
            print("3. 建筑放置问题：智能体没有选择Hub3附近的槽位")
            print("4. 等值线问题：Hub3的等值线没有被正确处理")
        else:
            print("✅ Hub3附近有建筑")
        
        # 检查层激活状态
        active_commercial_layers = [layer for layer in commercial_layers if layer['status'] == 'active']
        active_residential_layers = [layer for layer in residential_layers if layer['status'] == 'active']
        
        print(f"\n激活的层数量:")
        print(f"  商业建筑: {len(active_commercial_layers)} 个")
        print(f"  住宅建筑: {len(active_residential_layers)} 个")
        
        if len(active_commercial_layers) == 0:
            print("⚠️ 没有激活的商业建筑层")
        if len(active_residential_layers) == 0:
            print("⚠️ 没有激活的住宅建筑层")
    
    except Exception as e:
        print(f"❌ 分析出错: {e}")

if __name__ == "__main__":
    debug_slot_activation()
