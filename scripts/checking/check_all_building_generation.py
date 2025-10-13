#!/usr/bin/env python3
"""
检查所有影响建筑生长的地方
特别关注Hub3相关的逻辑
"""

import json
import numpy as np
import os

def check_all_building_generation():
    """检查所有影响建筑生长的地方"""
    
    print("=== 检查所有影响建筑生长的地方 ===")
    
    # 1. 检查建筑生成调用链
    print("\n1. 检查建筑生成调用链")
    print("   _quarterly_update() -> _generate_buildings_with_slots() -> _generate_residential_with_slots() / _generate_commercial_with_slots()")
    
    # 2. 检查季度更新逻辑
    print("\n2. 检查季度更新逻辑")
    print("   季度更新应该调用建筑生成，但可能有问题")
    
    # 3. 检查建筑生成条件
    print("\n3. 检查建筑生成条件")
    
    # 读取第1个月的建筑数据
    with open('enhanced_simulation_v3_1_output/building_positions_month_01.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    buildings = data.get('buildings', [])
    print(f"   第1个月建筑数量: {len(buildings)}")
    
    # 4. 检查槽位生成和激活
    print("\n4. 检查槽位生成和激活")
    
    # 检查层状态
    with open('enhanced_simulation_v3_1_output/layer_state_month_02.json', 'r', encoding='utf-8') as f:
        layer_data = json.load(f)
    
    layers = layer_data.get('layers', {})
    commercial_layers = layers.get('commercial', [])
    residential_layers = layers.get('residential', [])
    
    print("   商业建筑层:")
    for i, layer in enumerate(commercial_layers):
        if layer['status'] in ['active', 'complete']:
            print(f"     层 {i}: {layer['layer_id']} - {layer['status']} - 密度: {layer['density']:.1%}")
    
    print("   住宅建筑层:")
    for i, layer in enumerate(residential_layers):
        if layer['status'] in ['active', 'complete']:
            print(f"     层 {i}: {layer['layer_id']} - {layer['status']} - 密度: {layer['density']:.1%}")
    
    # 5. 检查Hub3相关的特殊逻辑
    print("\n5. 检查Hub3相关的特殊逻辑")
    
    # 检查Hub3位置
    hub3_x, hub3_y = 67, 94
    print(f"   Hub3位置: ({hub3_x}, {hub3_y})")
    
    # 检查第1个月Hub3附近的建筑
    hub3_buildings = []
    for building in buildings:
        x, y = building['position']
        distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
        if distance <= 30:
            hub3_buildings.append((building, distance))
    
    print(f"   第1个月Hub3附近建筑: {len(hub3_buildings)} 个")
    if hub3_buildings:
        for building, distance in sorted(hub3_buildings, key=lambda x: x[1]):
            building_type = building.get('type', 'unknown')
            x, y = building['position']
            print(f"     {building_type}: ({x}, {y}), 距离: {distance:.1f}")
    
    # 6. 检查建筑生成的具体逻辑
    print("\n6. 检查建筑生成的具体逻辑")
    
    # 模拟建筑生成过程
    print("   模拟建筑生成过程:")
    
    # 检查可用槽位
    available_commercial_slots = 0
    available_residential_slots = 0
    
    for layer in commercial_layers:
        if layer['status'] == 'active':
            available_commercial_slots += layer['capacity_effective'] - layer['placed']
    
    for layer in residential_layers:
        if layer['status'] == 'active':
            available_residential_slots += layer['capacity_effective'] - layer['placed']
    
    print(f"   可用商业槽位: {available_commercial_slots}")
    print(f"   可用住宅槽位: {available_residential_slots}")
    
    # 7. 检查建筑生成目标
    print("\n7. 检查建筑生成目标")
    
    # 根据代码逻辑，建筑生成目标应该是：
    residential_target = min(12, available_residential_slots)  # 基础目标
    commercial_target = min(5, available_commercial_slots)     # 基础目标
    
    print(f"   住宅建筑目标: {residential_target}")
    print(f"   商业建筑目标: {commercial_target}")
    
    # 8. 检查可能的问题
    print("\n8. 检查可能的问题")
    
    if available_commercial_slots > 0 and commercial_target == 0:
        print("   ❌ 问题1: 有可用商业槽位，但目标为0")
    
    if available_residential_slots > 0 and residential_target == 0:
        print("   ❌ 问题2: 有可用住宅槽位，但目标为0")
    
    if available_commercial_slots == 0:
        print("   ⚠️ 问题3: 没有可用的商业槽位")
    
    if available_residential_slots == 0:
        print("   ⚠️ 问题4: 没有可用的住宅槽位")
    
    # 9. 检查Hub3槽位分布
    print("\n9. 检查Hub3槽位分布")
    
    # 检查Hub3附近的槽位是否在激活层中
    hub3_nearby_slots = 0
    
    for layer in commercial_layers:
        if layer['status'] == 'active':
            # 这里需要检查槽位位置，但层状态文件中没有槽位位置信息
            # 我们需要从等值线系统获取
            pass
    
    print(f"   Hub3附近的激活槽位: {hub3_nearby_slots}")
    
    # 10. 检查建筑生成的实际执行
    print("\n10. 检查建筑生成的实际执行")
    
    # 检查第2个月是否有新建筑
    month_2_file = "enhanced_simulation_v3_1_output/building_delta_month_02.json"
    if os.path.exists(month_2_file):
        with open(month_2_file, 'r', encoding='utf-8') as f:
            month_2_data = json.load(f)
        
        new_buildings = month_2_data.get('new_buildings', [])
        print(f"   第2个月新建筑数量: {len(new_buildings)}")
        
        if len(new_buildings) == 0:
            print("   ❌ 问题5: 第2个月没有生成新建筑")
        else:
            print("   ✅ 第2个月有生成新建筑")
    else:
        print("   ❌ 问题6: 第2个月增量文件不存在")
    
    # 11. 总结可能的问题
    print("\n=== 可能的问题总结 ===")
    print("1. 建筑生成目标计算错误")
    print("2. 槽位获取逻辑有问题")
    print("3. 建筑生成方法没有被正确调用")
    print("4. Hub3的槽位没有被正确识别为可用")
    print("5. 建筑生成条件不满足")
    print("6. 季度更新逻辑有问题")

if __name__ == "__main__":
    check_all_building_generation()
