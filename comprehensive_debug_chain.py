#!/usr/bin/env python3
"""
全面调试调用链 - 追踪从地价场演化到建筑生成的完整流程
"""

import json
import os
import numpy as np
from pathlib import Path

# 导入系统模块
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem
from enhanced_city_simulation_v3_1 import ProgressiveGrowthSystem, EnhancedCitySimulationV3_1

def debug_complete_chain():
    """调试完整的调用链"""
    print("🔍 全面调试调用链")
    print("=" * 80)
    
    # 1. 检查配置文件
    print("\n1️⃣ 检查配置文件...")
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    land_price_evolution = config.get('land_price_evolution', {})
    print(f"   地价演化启用: {land_price_evolution.get('enabled', False)}")
    print(f"   道路激活月份: {land_price_evolution.get('road_activation_month', 0)}")
    print(f"   Hub激活月份: {land_price_evolution.get('hub_activation_month', 7)}")
    print(f"   Hub初始峰值: {land_price_evolution.get('hub_initial_peak', 0.7)}")
    print(f"   Hub最终峰值: {land_price_evolution.get('hub_final_peak', 1.0)}")
    
    # 2. 检查地价场演化
    print("\n2️⃣ 检查地价场演化...")
    land_price_system = GaussianLandPriceSystem(config)
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 测试关键月份
    test_months = [0, 6, 7, 12, 24]
    for month in test_months:
        print(f"\n   第 {month} 个月:")
        
        # 更新地价场
        land_price_system.update_land_price_field(month, {})
        land_price_field = land_price_system.get_land_price_field()
        
        # 获取演化阶段
        evolution_stage = land_price_system._get_evolution_stage(month)
        component_strengths = evolution_stage.get('component_strengths', {})
        
        print(f"     阶段: {evolution_stage['name']}")
        print(f"     组件强度: 道路={component_strengths.get('road', 0):.2f}, "
              f"Hub1={component_strengths.get('hub1', 0):.2f}, "
              f"Hub2={component_strengths.get('hub2', 0):.2f}, "
              f"Hub3={component_strengths.get('hub3', 0):.2f}")
        
        # 检查Hub位置的地价值
        hub1_value = land_price_field[55, 20]  # Hub1
        hub2_value = land_price_field[55, 90]  # Hub2
        hub3_value = land_price_field[94, 67]  # Hub3
        
        print(f"     Hub地价值: Hub1={hub1_value:.3f}, Hub2={hub2_value:.3f}, Hub3={hub3_value:.3f}")
    
    # 3. 检查等值线提取
    print("\n3️⃣ 检查等值线提取...")
    isocontour_system = IsocontourBuildingSystem(config)
    
    for month in test_months:
        print(f"\n   第 {month} 个月:")
        
        # 更新地价场
        land_price_system.update_land_price_field(month, {})
        land_price_field = land_price_system.get_land_price_field()
        
        # 初始化等值线系统
        isocontour_system.initialize_system(land_price_field, transport_hubs, map_size, month, land_price_system)
        
        # 获取等值线数据
        contour_data = isocontour_system.get_contour_data_for_visualization()
        
        commercial_contours = contour_data.get('commercial_contours', [])
        residential_contours = contour_data.get('residential_contours', [])
        
        print(f"     商业等值线: {len(commercial_contours)} 条")
        print(f"     住宅等值线: {len(residential_contours)} 条")
        
        # 检查等值线位置
        if commercial_contours:
            first_contour = commercial_contours[0]
            if len(first_contour) > 0:
                center_x = np.mean([p[0] for p in first_contour])
                center_y = np.mean([p[1] for p in first_contour])
                print(f"     商业等值线中心: ({center_x:.1f}, {center_y:.1f})")
        
        if residential_contours:
            first_contour = residential_contours[0]
            if len(first_contour) > 0:
                center_x = np.mean([p[0] for p in first_contour])
                center_y = np.mean([p[1] for p in first_contour])
                print(f"     住宅等值线中心: ({center_x:.1f}, {center_y:.1f})")
    
    # 4. 检查槽位系统初始化
    print("\n4️⃣ 检查槽位系统初始化...")
    progressive_growth_system = ProgressiveGrowthSystem(config)
    
    for month in test_months:
        print(f"\n   第 {month} 个月:")
        
        # 更新地价场
        land_price_system.update_land_price_field(month, {})
        land_price_field = land_price_system.get_land_price_field()
        
        # 初始化等值线系统
        isocontour_system.initialize_system(land_price_field, transport_hubs, map_size, month, land_price_system)
        
        # 初始化槽位系统
        progressive_growth_system.initialize_layers(isocontour_system, land_price_field)
        
        # 检查槽位系统状态
        commercial_layers = progressive_growth_system.layers['commercial']
        residential_layers = progressive_growth_system.layers['residential']
        
        total_commercial_slots = sum(len(layer.slots) for layer in commercial_layers)
        total_residential_slots = sum(len(layer.slots) for layer in residential_layers)
        
        print(f"     商业层: {len(commercial_layers)} 个, 槽位: {total_commercial_slots} 个")
        print(f"     住宅层: {len(residential_layers)} 个, 槽位: {total_residential_slots} 个")
        
        # 检查槽位位置分布
        if commercial_layers and commercial_layers[0].slots:
            first_slot = commercial_layers[0].slots[0]
            print(f"     商业槽位示例: ({first_slot.pos[0]}, {first_slot.pos[1]})")
        
        if residential_layers and residential_layers[0].slots:
            first_slot = residential_layers[0].slots[0]
            print(f"     住宅槽位示例: ({first_slot.pos[0]}, {first_slot.pos[1]})")
    
    # 5. 检查实际模拟运行
    print("\n5️⃣ 检查实际模拟运行...")
    
    # 检查输出文件
    output_dir = "enhanced_simulation_v3_1_output"
    if os.path.exists(output_dir):
        print("   输出目录存在")
        
        # 检查建筑数据
        building_files = [f for f in os.listdir(output_dir) if f.startswith("building_positions_month_") and f.endswith(".json")]
        building_files.sort()
        
        print(f"   建筑文件数量: {len(building_files)}")
        
        # 检查关键月份的建筑数据
        for month in [0, 6, 12, 24]:
            file_path = os.path.join(output_dir, f"building_positions_month_{month:02d}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                buildings = data.get('buildings', [])
                print(f"   第 {month} 个月: {len(buildings)} 个建筑")
                
                # 分析建筑位置分布
                if buildings:
                    positions = [b['position'] for b in buildings]
                    x_coords = [p[0] for p in positions]
                    y_coords = [p[1] for p in positions]
                    
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    
                    print(f"     建筑中心: ({center_x:.1f}, {center_y:.1f})")
                    print(f"     X范围: {min(x_coords):.1f} - {max(x_coords):.1f}")
                    print(f"     Y范围: {min(y_coords):.1f} - {max(y_coords):.1f}")
                    
                    # 检查建筑类型分布
                    type_counts = {}
                    for building in buildings:
                        building_type = building.get('type', 'unknown')
                        type_counts[building_type] = type_counts.get(building_type, 0) + 1
                    
                    print(f"     建筑类型: {type_counts}")
    else:
        print("   ❌ 输出目录不存在")
    
    # 6. 检查可能的问题点
    print("\n6️⃣ 检查可能的问题点...")
    
    # 检查是否有缓存问题
    print("   检查缓存问题:")
    if hasattr(land_price_system, '_cached_land_price_field'):
        print("     ❌ 地价系统有缓存字段")
    else:
        print("     ✅ 地价系统无缓存字段")
    
    # 检查等值线系统状态
    print("   检查等值线系统状态:")
    print(f"     当前月份: {getattr(isocontour_system, 'current_month', 'None')}")
    print(f"     地价系统引用: {getattr(isocontour_system, 'land_price_system', 'None')}")
    
    # 检查槽位系统状态
    print("   检查槽位系统状态:")
    print(f"     商业层数量: {len(progressive_growth_system.layers['commercial'])}")
    print(f"     住宅层数量: {len(progressive_growth_system.layers['residential'])}")
    print(f"     活跃层: {progressive_growth_system.active_layers}")
    
    print("\n" + "=" * 80)
    print("🔍 调试完成！请检查上述输出找出问题所在。")

def debug_specific_issue():
    """调试特定问题"""
    print("\n🎯 调试特定问题...")
    
    # 检查Month 0和Month 24的差异
    print("\n📊 对比Month 0和Month 24...")
    
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    land_price_system = GaussianLandPriceSystem(config)
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # Month 0
    land_price_system.update_land_price_field(0, {})
    month0_field = land_price_system.get_land_price_field()
    
    # Month 24
    land_price_system.update_land_price_field(24, {})
    month24_field = land_price_system.get_land_price_field()
    
    # 比较地价场
    print(f"Month 0 地价场范围: {np.min(month0_field):.3f} - {np.max(month0_field):.3f}")
    print(f"Month 24 地价场范围: {np.min(month24_field):.3f} - {np.max(month24_field):.3f}")
    
    # 比较Hub位置
    for i, hub in enumerate(transport_hubs):
        x, y = hub[0], hub[1]
        month0_value = month0_field[y, x]
        month24_value = month24_field[y, x]
        print(f"Hub{i+1} ({x}, {y}): Month 0 = {month0_value:.3f}, Month 24 = {month24_value:.3f}")
    
    # 检查差异
    diff_field = month24_field - month0_field
    max_diff = np.max(np.abs(diff_field))
    print(f"最大差异: {max_diff:.3f}")
    
    if max_diff < 0.001:
        print("❌ 地价场几乎没有变化！这是问题所在！")
    else:
        print("✅ 地价场有显著变化")

if __name__ == "__main__":
    debug_complete_chain()
    debug_specific_issue()


