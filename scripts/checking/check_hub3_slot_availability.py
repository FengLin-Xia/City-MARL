#!/usr/bin/env python3
"""
检查Hub3槽位是否被正确识别为可用
"""

import json
import numpy as np
from logic.isocontour_building_system import IsocontourBuildingSystem
from logic.enhanced_sdf_system import GaussianLandPriceSystem

def check_hub3_slot_availability():
    """检查Hub3槽位是否被正确识别为可用"""
    
    print("=== 检查Hub3槽位是否被正确识别为可用 ===")
    
    # Hub3位置
    hub3_x, hub3_y = 67, 94
    print(f"Hub3位置: ({hub3_x}, {hub3_y})")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建系统
    land_price_system = GaussianLandPriceSystem(config)
    isocontour_system = IsocontourBuildingSystem(config)
    
    # 读取地价场数据
    with open('enhanced_simulation_v3_1_output/land_price_frame_month_02.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    land_price_field = np.array(data['land_price_field'])
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    # 初始化系统
    land_price_system.initialize_system(transport_hubs, map_size)
    isocontour_system.initialize_system(land_price_field, transport_hubs, map_size)
    
    # 获取等值线数据
    contour_data = isocontour_system.get_contour_data_for_visualization()
    
    print(f"\n等值线数据:")
    commercial_contours = contour_data.get('commercial_contours', [])
    residential_contours = contour_data.get('residential_contours', [])
    
    print(f"商业等值线数量: {len(commercial_contours)}")
    print(f"住宅等值线数量: {len(residential_contours)}")
    
    # 模拟槽位生成过程
    print("\n=== 模拟槽位生成过程 ===")
    
    from enhanced_city_simulation_v3_1 import ProgressiveGrowthSystem
    
    progressive_system = ProgressiveGrowthSystem(config)
    progressive_system.initialize_layers(isocontour_system, land_price_field)
    
    # 激活第一层
    progressive_system._activate_layer('commercial', 0, 0)
    progressive_system._activate_layer('residential', 0, 0)
    
    print("激活层后:")
    print(f"商业激活层: {progressive_system.active_layers['commercial']}")
    print(f"住宅激活层: {progressive_system.active_layers['residential']}")
    
    # 检查激活层的槽位
    print("\n=== 检查激活层的槽位 ===")
    
    # 商业建筑层
    commercial_layers = progressive_system.layers['commercial']
    active_commercial_layer = commercial_layers[progressive_system.active_layers['commercial']]
    
    print(f"商业激活层: {active_commercial_layer.layer_id}")
    print(f"层状态: {active_commercial_layer.status}")
    print(f"总槽位: {len(active_commercial_layer.slots)}")
    
    # 检查槽位状态
    used_slots = [slot for slot in active_commercial_layer.slots if slot.used]
    dead_slots = [slot for slot in active_commercial_layer.slots if slot.dead]
    available_slots = [slot for slot in active_commercial_layer.slots if not slot.used and not slot.dead]
    
    print(f"已使用槽位: {len(used_slots)}")
    print(f"死槽: {len(dead_slots)}")
    print(f"可用槽位: {len(available_slots)}")
    
    # 检查Hub3附近的槽位
    print(f"\n=== 检查Hub3附近的商业槽位 ===")
    hub3_commercial_slots = []
    for slot in available_slots:
        x, y = slot.pos
        distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
        if distance <= 30:
            hub3_commercial_slots.append((slot, distance))
    
    print(f"Hub3附近的可用商业槽位: {len(hub3_commercial_slots)}")
    if hub3_commercial_slots:
        for slot, distance in sorted(hub3_commercial_slots, key=lambda x: x[1]):
            print(f"  槽位: ({slot.pos[0]}, {slot.pos[1]}), 距离: {distance:.1f}")
    else:
        print("  ❌ Hub3附近没有可用的商业槽位")
    
    # 住宅建筑层
    residential_layers = progressive_system.layers['residential']
    active_residential_layer = residential_layers[progressive_system.active_layers['residential']]
    
    print(f"\n住宅激活层: {active_residential_layer.layer_id}")
    print(f"层状态: {active_residential_layer.status}")
    print(f"总槽位: {len(active_residential_layer.slots)}")
    
    # 检查槽位状态
    used_slots = [slot for slot in active_residential_layer.slots if slot.used]
    dead_slots = [slot for slot in active_residential_layer.slots if slot.dead]
    available_slots = [slot for slot in active_residential_layer.slots if not slot.used and not slot.dead]
    
    print(f"已使用槽位: {len(used_slots)}")
    print(f"死槽: {len(dead_slots)}")
    print(f"可用槽位: {len(available_slots)}")
    
    # 检查Hub3附近的槽位
    print(f"\n=== 检查Hub3附近的住宅槽位 ===")
    hub3_residential_slots = []
    for slot in available_slots:
        x, y = slot.pos
        distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
        if distance <= 30:
            hub3_residential_slots.append((slot, distance))
    
    print(f"Hub3附近的可用住宅槽位: {len(hub3_residential_slots)}")
    if hub3_residential_slots:
        for slot, distance in sorted(hub3_residential_slots, key=lambda x: x[1]):
            print(f"  槽位: ({slot.pos[0]}, {slot.pos[1]}), 距离: {distance:.1f}")
    else:
        print("  ❌ Hub3附近没有可用的住宅槽位")
    
    # 测试get_available_slots方法
    print(f"\n=== 测试get_available_slots方法 ===")
    
    available_commercial = progressive_system.get_available_slots('commercial', 100)
    available_residential = progressive_system.get_available_slots('residential', 100)
    
    print(f"get_available_slots返回的商业槽位: {len(available_commercial)}")
    print(f"get_available_slots返回的住宅槽位: {len(available_residential)}")
    
    # 检查返回的槽位中是否有Hub3附近的
    hub3_in_available_commercial = []
    hub3_in_available_residential = []
    
    for slot in available_commercial:
        x, y = slot.pos
        distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
        if distance <= 30:
            hub3_in_available_commercial.append((slot, distance))
    
    for slot in available_residential:
        x, y = slot.pos
        distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
        if distance <= 30:
            hub3_in_available_residential.append((slot, distance))
    
    print(f"get_available_slots返回的Hub3附近商业槽位: {len(hub3_in_available_commercial)}")
    print(f"get_available_slots返回的Hub3附近住宅槽位: {len(hub3_in_available_residential)}")
    
    if hub3_in_available_commercial:
        print("Hub3附近的可用商业槽位:")
        for slot, distance in sorted(hub3_in_available_commercial, key=lambda x: x[1]):
            print(f"  ({slot.pos[0]}, {slot.pos[1]}), 距离: {distance:.1f}")
    
    if hub3_in_available_residential:
        print("Hub3附近的可用住宅槽位:")
        for slot, distance in sorted(hub3_in_available_residential, key=lambda x: x[1]):
            print(f"  ({slot.pos[0]}, {slot.pos[1]}), 距离: {distance:.1f}")
    
    # 总结
    print(f"\n=== 总结 ===")
    if len(hub3_in_available_commercial) > 0 or len(hub3_in_available_residential) > 0:
        print("✅ Hub3的槽位被正确识别为可用")
        print("   问题可能在建筑生成的其他环节")
    else:
        print("❌ Hub3的槽位没有被识别为可用")
        print("   问题在槽位识别逻辑")

if __name__ == "__main__":
    check_hub3_slot_availability()
