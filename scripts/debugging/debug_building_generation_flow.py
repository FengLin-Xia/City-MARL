#!/usr/bin/env python3
"""
调试建筑生成的完整流程
找出问题所在
"""

import json
import numpy as np
import os

def debug_building_generation_flow():
    """调试建筑生成的完整流程"""
    
    print("=== 调试建筑生成的完整流程 ===")
    
    # 1. 检查建筑生成的调用时机
    print("\n1. 检查建筑生成的调用时机")
    print("   建筑生成应该在 _quarterly_update() 中调用")
    print("   让我们检查季度更新的逻辑")
    
    # 2. 检查季度更新逻辑
    print("\n2. 检查季度更新逻辑")
    
    # 读取层状态文件，看看季度更新是否正常
    months_to_check = [1, 2, 3, 6, 9, 12]
    
    for month in months_to_check:
        layer_file = f"enhanced_simulation_v3_1_output/layer_state_month_{month:02d}.json"
        if os.path.exists(layer_file):
            with open(layer_file, 'r', encoding='utf-8') as f:
                layer_data = json.load(f)
            
            quarter = layer_data.get('quarter', -1)
            layers = layer_data.get('layers', {})
            
            commercial_layers = layers.get('commercial', [])
            residential_layers = layers.get('residential', [])
            
            # 统计激活层
            active_commercial = [layer for layer in commercial_layers if layer['status'] == 'active']
            active_residential = [layer for layer in residential_layers if layer['status'] == 'active']
            
            print(f"   Month {month} (Quarter {quarter}):")
            print(f"     激活层: 商业 {len(active_commercial)}, 住宅 {len(active_residential)}")
            
            # 检查可用槽位
            available_commercial = 0
            available_residential = 0
            
            for layer in active_commercial:
                available_commercial += layer['capacity_effective'] - layer['placed']
            
            for layer in active_residential:
                available_residential += layer['capacity_effective'] - layer['placed']
            
            print(f"     可用槽位: 商业 {available_commercial}, 住宅 {available_residential}")
    
    # 3. 检查建筑生成目标计算
    print("\n3. 检查建筑生成目标计算")
    
    # 根据代码逻辑，建筑生成目标应该是：
    # residential_target = min(random.randint(12, 20), available_residential_slots)
    # commercial_target = min(random.randint(5, 12), available_commercial_slots)
    
    print("   建筑生成目标计算逻辑:")
    print("   residential_target = min(random.randint(12, 20), available_residential_slots)")
    print("   commercial_target = min(random.randint(5, 12), available_commercial_slots)")
    
    # 4. 检查建筑生成的实际执行
    print("\n4. 检查建筑生成的实际执行")
    
    # 检查第2个月是否有新建筑
    month_2_file = "enhanced_simulation_v3_1_output/building_delta_month_02.json"
    if os.path.exists(month_2_file):
        with open(month_2_file, 'r', encoding='utf-8') as f:
            month_2_data = json.load(f)
        
        new_buildings = month_2_data.get('new_buildings', [])
        print(f"   第2个月新建筑数量: {len(new_buildings)}")
        
        if len(new_buildings) == 0:
            print("   ❌ 第2个月没有生成新建筑")
        else:
            print("   ✅ 第2个月有生成新建筑")
    else:
        print("   ❌ 第2个月增量文件不存在")
    
    # 5. 检查建筑生成的条件
    print("\n5. 检查建筑生成的条件")
    
    # 检查是否有建筑生成的条件
    print("   建筑生成的条件:")
    print("   1. 有激活的层")
    print("   2. 激活层中有可用槽位")
    print("   3. 建筑生成目标 > 0")
    print("   4. _generate_buildings_with_slots() 被调用")
    
    # 6. 检查可能的问题点
    print("\n6. 检查可能的问题点")
    
    # 问题1: 建筑生成目标为0
    print("   问题1: 建筑生成目标为0")
    print("   可能原因: available_slots 为0 或 random.randint() 返回0")
    
    # 问题2: 建筑生成方法没有被调用
    print("   问题2: 建筑生成方法没有被调用")
    print("   可能原因: _quarterly_update() 中的条件不满足")
    
    # 问题3: 建筑生成后没有保存
    print("   问题3: 建筑生成后没有保存")
    print("   可能原因: 建筑生成成功但没有保存到文件")
    
    # 问题4: 建筑生成逻辑有bug
    print("   问题4: 建筑生成逻辑有bug")
    print("   可能原因: _generate_residential_with_slots() 或 _generate_commercial_with_slots() 有bug")
    
    # 7. 检查建筑生成的具体实现
    print("\n7. 检查建筑生成的具体实现")
    
    # 让我们模拟建筑生成过程
    print("   模拟建筑生成过程:")
    
    # 读取第1个月的建筑数据
    with open('enhanced_simulation_v3_1_output/building_positions_month_01.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    buildings = data.get('buildings', [])
    print(f"   第1个月建筑数量: {len(buildings)}")
    
    # 按类型分组
    building_types = {}
    for building in buildings:
        building_type = building.get('type', 'unknown')
        if building_type not in building_types:
            building_types[building_type] = []
        building_types[building_type].append(building)
    
    print("   第1个月建筑类型分布:")
    for building_type, buildings_list in building_types.items():
        print(f"     {building_type}: {len(buildings_list)} 个")
    
    # 8. 检查建筑生成的目标计算
    print("\n8. 检查建筑生成的目标计算")
    
    # 根据代码逻辑，如果第1个月有建筑，那么第2个月应该继续生成
    # 但实际情况是第2个月没有新建筑
    
    print("   根据代码逻辑:")
    print("   - 第1个月有建筑，说明槽位系统工作正常")
    print("   - 第2个月没有新建筑，说明建筑生成逻辑有问题")
    
    # 9. 检查建筑生成的具体问题
    print("\n9. 检查建筑生成的具体问题")
    
    # 让我们检查建筑生成的关键代码
    print("   关键代码检查:")
    print("   _generate_buildings_with_slots() 方法:")
    print("   1. 获取可用槽位数量")
    print("   2. 计算建筑生成目标")
    print("   3. 调用 _generate_residential_with_slots() 和 _generate_commercial_with_slots()")
    print("   4. 将新建筑添加到 city_state")
    
    # 10. 检查可能的问题
    print("\n10. 检查可能的问题")
    
    print("   可能的问题:")
    print("   1. get_available_slots() 返回空列表")
    print("   2. 建筑生成目标计算错误")
    print("   3. 建筑生成方法有bug")
    print("   4. 建筑生成后没有正确保存")
    print("   5. 季度更新逻辑有问题")
    
    # 11. 建议的调试方法
    print("\n11. 建议的调试方法")
    print("   1. 在 _generate_buildings_with_slots() 中添加调试输出")
    print("   2. 检查 get_available_slots() 的返回值")
    print("   3. 检查建筑生成目标的计算")
    print("   4. 检查建筑生成方法的执行")
    print("   5. 检查建筑保存逻辑")

if __name__ == "__main__":
    debug_building_generation_flow()
