#!/usr/bin/env python3
"""
检查第2个月的情况
直接测试建筑生成逻辑
"""

import json
import numpy as np
from logic.isocontour_building_system import IsocontourBuildingSystem
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from enhanced_city_simulation_v3_1 import ProgressiveGrowthSystem

def check_month_2_debug():
    """检查第2个月的情况"""
    
    print("=== 检查第2个月的情况 ===")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建系统
    land_price_system = GaussianLandPriceSystem(config)
    isocontour_system = IsocontourBuildingSystem(config)
    progressive_system = ProgressiveGrowthSystem(config)
    
    # 读取地价场数据
    with open('enhanced_simulation_v3_1_output/land_price_frame_month_02.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    land_price_field = np.array(data['land_price_field'])
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    # 初始化系统
    land_price_system.initialize_system(transport_hubs, map_size)
    isocontour_system.initialize_system(land_price_field, transport_hubs, map_size)
    progressive_system.initialize_layers(isocontour_system, land_price_field)
    
    # 激活第一层
    progressive_system._activate_layer('commercial', 0, 0)
    progressive_system._activate_layer('residential', 0, 0)
    
    print("系统初始化完成")
    
    # 检查可用槽位
    available_residential_slots = len(progressive_system.get_available_slots('residential', 100))
    available_commercial_slots = len(progressive_system.get_available_slots('commercial', 100))
    
    print(f"可用槽位数量 - 住宅: {available_residential_slots}, 商业: {available_commercial_slots}")
    
    # 计算建筑生成目标
    import random
    residential_target = min(random.randint(12, 20), available_residential_slots)
    commercial_target = min(random.randint(5, 12), available_commercial_slots)
    
    print(f"建筑生成目标 - 住宅: {residential_target}, 商业: {commercial_target}")
    
    # 测试建筑生成
    if residential_target > 0:
        print("测试住宅建筑生成...")
        available_slots = progressive_system.get_available_slots('residential', residential_target)
        print(f"获取到的住宅槽位: {len(available_slots)}")
        
        if available_slots:
            print("住宅槽位位置:")
            for i, slot in enumerate(available_slots[:5]):
                print(f"  槽位 {i+1}: ({slot.pos[0]}, {slot.pos[1]})")
    
    if commercial_target > 0:
        print("测试商业建筑生成...")
        available_slots = progressive_system.get_available_slots('commercial', commercial_target)
        print(f"获取到的商业槽位: {len(available_slots)}")
        
        if available_slots:
            print("商业槽位位置:")
            for i, slot in enumerate(available_slots[:5]):
                print(f"  槽位 {i+1}: ({slot.pos[0]}, {slot.pos[1]})")
    
    # 检查Hub3附近的槽位
    hub3_x, hub3_y = 67, 94
    print(f"\n检查Hub3附近的槽位 (Hub3位置: {hub3_x}, {hub3_y})")
    
    # 检查住宅槽位
    residential_slots = progressive_system.get_available_slots('residential', 100)
    hub3_residential = []
    for slot in residential_slots:
        x, y = slot.pos
        distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
        if distance <= 30:
            hub3_residential.append((slot, distance))
    
    print(f"Hub3附近的可用住宅槽位: {len(hub3_residential)}")
    if hub3_residential:
        for slot, distance in sorted(hub3_residential, key=lambda x: x[1]):
            print(f"  住宅槽位: ({slot.pos[0]}, {slot.pos[1]}), 距离: {distance:.1f}")
    
    # 检查商业槽位
    commercial_slots = progressive_system.get_available_slots('commercial', 100)
    hub3_commercial = []
    for slot in commercial_slots:
        x, y = slot.pos
        distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
        if distance <= 30:
            hub3_commercial.append((slot, distance))
    
    print(f"Hub3附近的可用商业槽位: {len(hub3_commercial)}")
    if hub3_commercial:
        for slot, distance in sorted(hub3_commercial, key=lambda x: x[1]):
            print(f"  商业槽位: ({slot.pos[0]}, {slot.pos[1]}), 距离: {distance:.1f}")
    
    # 总结
    print(f"\n=== 总结 ===")
    if available_residential_slots > 0 and residential_target > 0:
        print("✅ 住宅建筑应该能生成")
    else:
        print("❌ 住宅建筑无法生成")
    
    if available_commercial_slots > 0 and commercial_target > 0:
        print("✅ 商业建筑应该能生成")
    else:
        print("❌ 商业建筑无法生成")
    
    if len(hub3_residential) > 0 or len(hub3_commercial) > 0:
        print("✅ Hub3附近有可用槽位")
    else:
        print("❌ Hub3附近没有可用槽位")

if __name__ == "__main__":
    check_month_2_debug()
