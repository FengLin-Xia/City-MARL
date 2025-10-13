#!/usr/bin/env python3
"""
调试可用槽位问题
检查为什么有激活层但没有可用槽位
"""

import json
import numpy as np
from logic.isocontour_building_system import IsocontourBuildingSystem
from logic.enhanced_sdf_system import GaussianLandPriceSystem

def debug_available_slots():
    """调试可用槽位问题"""
    
    print("=== 调试可用槽位问题 ===")
    
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
    
    print("等值线数据:")
    commercial_contours = contour_data.get('commercial_contours', [])
    residential_contours = contour_data.get('residential_contours', [])
    
    print(f"商业等值线数量: {len(commercial_contours)}")
    print(f"住宅等值线数量: {len(residential_contours)}")
    
    # 模拟槽位生成过程
    print("\n=== 模拟槽位生成过程 ===")
    
    # 模拟ProgressiveGrowthSystem的槽位生成
    from enhanced_city_simulation_v3_1 import ProgressiveGrowthSystem
    
    progressive_system = ProgressiveGrowthSystem(config)
    progressive_system.initialize_layers(isocontour_system, land_price_field)
    
    print("商业建筑层:")
    for i, layer in enumerate(progressive_system.layers['commercial']):
        print(f"  层 {i}: {layer.layer_id} - {layer.status}")
        print(f"    容量: {layer.capacity}, 已放置: {layer.placed}")
        print(f"    槽位详情:")
        
        # 检查槽位状态
        used_slots = [slot for slot in layer.slots if slot.used]
        dead_slots = [slot for slot in layer.slots if slot.dead]
        available_slots = [slot for slot in layer.slots if not slot.used and not slot.dead]
        
        print(f"      已使用: {len(used_slots)}")
        print(f"      死槽: {len(dead_slots)}")
        print(f"      可用: {len(available_slots)}")
        
        # 显示可用槽位的位置
        if available_slots:
            print(f"      可用槽位位置:")
            for j, slot in enumerate(available_slots[:5]):  # 只显示前5个
                print(f"        槽位 {j+1}: ({slot.pos[0]}, {slot.pos[1]})")
            if len(available_slots) > 5:
                print(f"        ... 还有 {len(available_slots) - 5} 个")
    
    print("\n住宅建筑层:")
    for i, layer in enumerate(progressive_system.layers['residential']):
        print(f"  层 {i}: {layer.layer_id} - {layer.status}")
        print(f"    容量: {layer.capacity}, 已放置: {layer.placed}")
        print(f"    槽位详情:")
        
        # 检查槽位状态
        used_slots = [slot for slot in layer.slots if slot.used]
        dead_slots = [slot for slot in layer.slots if slot.dead]
        available_slots = [slot for slot in layer.slots if not slot.used and not slot.dead]
        
        print(f"      已使用: {len(used_slots)}")
        print(f"      死槽: {len(dead_slots)}")
        print(f"      可用: {len(available_slots)}")
        
        # 显示可用槽位的位置
        if available_slots:
            print(f"      可用槽位位置:")
            for j, slot in enumerate(available_slots[:5]):  # 只显示前5个
                print(f"        槽位 {j+1}: ({slot.pos[0]}, {slot.pos[1]})")
            if len(available_slots) > 5:
                print(f"        ... 还有 {len(available_slots) - 5} 个")
    
    # 测试get_available_slots方法
    print("\n=== 测试get_available_slots方法 ===")
    
    # 激活第一层
    progressive_system._activate_layer('commercial', 0, 0)
    progressive_system._activate_layer('residential', 0, 0)
    
    # 获取可用槽位
    available_commercial = progressive_system.get_available_slots('commercial', 100)
    available_residential = progressive_system.get_available_slots('residential', 100)
    
    print(f"可用商业槽位: {len(available_commercial)}")
    print(f"可用住宅槽位: {len(available_residential)}")
    
    if available_commercial:
        print("商业槽位位置:")
        for i, slot in enumerate(available_commercial[:5]):
            print(f"  槽位 {i+1}: ({slot.pos[0]}, {slot.pos[1]})")
    
    if available_residential:
        print("住宅槽位位置:")
        for i, slot in enumerate(available_residential[:5]):
            print(f"  槽位 {i+1}: ({slot.pos[0]}, {slot.pos[1]})")

if __name__ == "__main__":
    debug_available_slots()
