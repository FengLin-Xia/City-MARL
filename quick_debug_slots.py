#!/usr/bin/env python3
"""
快速调试槽位系统初始化
"""

import json
import numpy as np
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem
from enhanced_city_simulation_v3_1 import ProgressiveGrowthSystem

def quick_debug_slots():
    """快速调试槽位系统"""
    print("🔍 快速调试槽位系统...")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 初始化系统
    land_price_system = GaussianLandPriceSystem(config)
    isocontour_system = IsocontourBuildingSystem(config)
    progressive_growth_system = ProgressiveGrowthSystem(config)
    
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 测试Month 0
    month = 0
    print(f"\n📊 测试第 {month} 个月:")
    
    # 更新地价场
    land_price_system.update_land_price_field(month, {})
    land_price_field = land_price_system.get_land_price_field()
    
    # 初始化等值线系统
    isocontour_system.initialize_system(land_price_field, transport_hubs, map_size, month, land_price_system)
    
    # 获取等值线数据
    contour_data = isocontour_system.get_contour_data_for_visualization()
    
    commercial_contours = contour_data.get('commercial_contours', [])
    residential_contours = contour_data.get('residential_contours', [])
    
    print(f"等值线: 商业 {len(commercial_contours)}, 住宅 {len(residential_contours)}")
    
    # 初始化槽位系统
    print("\n🏗️ 初始化槽位系统...")
    progressive_growth_system.initialize_layers(isocontour_system, land_price_field)
    
    # 检查槽位系统状态
    commercial_layers = progressive_growth_system.layers['commercial']
    residential_layers = progressive_growth_system.layers['residential']
    
    print(f"商业层数量: {len(commercial_layers)}")
    print(f"住宅层数量: {len(residential_layers)}")
    
    # 统计槽位数量
    total_commercial_slots = sum(len(layer.slots) for layer in commercial_layers)
    total_residential_slots = sum(len(layer.slots) for layer in residential_layers)
    
    print(f"商业槽位总数: {total_commercial_slots}")
    print(f"住宅槽位总数: {total_residential_slots}")
    
    # 检查前几层的详细信息
    print("\n📋 商业层详情:")
    for i, layer in enumerate(commercial_layers[:3]):
        print(f"  层{i}: {layer.status}, 槽位{len(layer.slots)}个, 容量{layer.capacity}")
        if layer.slots:
            print(f"    前3个槽位位置: {[slot.pos for slot in layer.slots[:3]]}")
    
    print("\n📋 住宅层详情:")
    for i, layer in enumerate(residential_layers[:3]):
        print(f"  层{i}: {layer.status}, 槽位{len(layer.slots)}个, 容量{layer.capacity}")
        if layer.slots:
            print(f"    前3个槽位位置: {[slot.pos for slot in layer.slots[:3]]}")

if __name__ == "__main__":
    quick_debug_slots()


