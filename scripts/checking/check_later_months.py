#!/usr/bin/env python3
"""
检查后面月份Hub3的建筑生成情况
"""

import json
import numpy as np
import os

def check_later_months():
    """检查后面月份Hub3的建筑生成情况"""
    
    # Hub3位置
    hub3_x, hub3_y = 67, 94
    
    print("=== 检查后面月份Hub3建筑生成情况 ===")
    print(f"Hub3 位置: ({hub3_x}, {hub3_y})")
    print()
    
    # 检查多个月份
    months_to_check = [1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
    
    for month in months_to_check:
        try:
            print(f"--- Month {month} ---")
            
            # 读取建筑数据
            if month == 1:
                filename = f"enhanced_simulation_v3_1_output/building_positions_month_{month:02d}.json"
            else:
                filename = f"enhanced_simulation_v3_1_output/building_delta_month_{month:02d}.json"
            
            if os.path.exists(filename):
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
            
            else:
                print(f"❌ 建筑文件不存在: {filename}")
            
            # 检查层状态
            layer_filename = f"enhanced_simulation_v3_1_output/layer_state_month_{month:02d}.json"
            if os.path.exists(layer_filename):
                with open(layer_filename, 'r', encoding='utf-8') as f:
                    layer_data = json.load(f)
                
                layers = layer_data.get('layers', {})
                commercial_layers = layers.get('commercial', [])
                residential_layers = layers.get('residential', [])
                
                # 统计激活的层
                active_commercial = [layer for layer in commercial_layers if layer['status'] == 'active']
                active_residential = [layer for layer in residential_layers if layer['status'] == 'active']
                complete_commercial = [layer for layer in commercial_layers if layer['status'] == 'complete']
                complete_residential = [layer for layer in residential_layers if layer['status'] == 'complete']
                
                print(f"层状态: 商业激活{len(active_commercial)}/完成{len(complete_commercial)}, 住宅激活{len(active_residential)}/完成{len(complete_residential)}")
                
                # 显示激活层的详细信息
                if active_commercial:
                    for layer in active_commercial:
                        print(f"  商业激活层: {layer['layer_id']} - 密度: {layer['density']:.1%}")
                if active_residential:
                    for layer in active_residential:
                        print(f"  住宅激活层: {layer['layer_id']} - 密度: {layer['density']:.1%}")
            else:
                print("❌ 层状态文件不存在")
            
            print()
            
        except Exception as e:
            print(f"❌ 分析出错: {e}")
            print()
    
    # 总结分析
    print("=== 总结分析 ===")
    print("如果Hub3在早期月份有建筑，但后期没有新建筑，可能的原因：")
    print("1. 槽位激活逻辑问题：后续层没有被激活")
    print("2. 建筑生成配额问题：每月生成数量有限")
    print("3. 层完成逻辑问题：当前层完成后没有激活下一层")
    print("4. 等值线更新问题：年度更新后等值线发生变化")

if __name__ == "__main__":
    check_later_months()
