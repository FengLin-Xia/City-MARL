#!/usr/bin/env python3
"""
调试槽位系统、地价场、高斯核联动问题
检查为什么高斯核变化没有影响建筑生成
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入系统模块
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem
from enhanced_city_simulation_v3_1 import ProgressiveGrowthSystem

def debug_landprice_evolution():
    """调试地价场演化"""
    print("🔍 调试地价场演化...")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 初始化地价系统
    land_price_system = GaussianLandPriceSystem(config)
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 检查不同月份的地价场演化
    test_months = [0, 6, 12, 18, 24, 30, 36]
    
    print("\n📊 地价场演化阶段分析:")
    for month in test_months:
        # 获取演化阶段
        evolution_stage = land_price_system._get_evolution_stage(month)
        component_strengths = evolution_stage.get('component_strengths', {})
        
        print(f"\n第 {month} 个月:")
        print(f"  阶段: {evolution_stage['name']} - {evolution_stage['description']}")
        print(f"  组件强度: 道路={component_strengths.get('road', 0):.2f}, "
              f"Hub1={component_strengths.get('hub1', 0):.2f}, "
              f"Hub2={component_strengths.get('hub2', 0):.2f}, "
              f"Hub3={component_strengths.get('hub3', 0):.2f}")
        
        # 更新地价场
        land_price_system.update_land_price_field(month, {})
        land_price_field = land_price_system.get_land_price_field()
        
        # 计算地价场统计
        max_value = np.max(land_price_field)
        mean_value = np.mean(land_price_field)
        hub1_value = land_price_field[55, 20]  # Hub1位置
        hub2_value = land_price_field[55, 90]  # Hub2位置
        hub3_value = land_price_field[94, 67]  # Hub3位置
        
        print(f"  地价场统计: 最大值={max_value:.3f}, 平均值={mean_value:.3f}")
        print(f"  Hub地价: Hub1={hub1_value:.3f}, Hub2={hub2_value:.3f}, Hub3={hub3_value:.3f}")

def debug_isocontour_extraction():
    """调试等值线提取"""
    print("\n🔍 调试等值线提取...")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 初始化系统
    land_price_system = GaussianLandPriceSystem(config)
    isocontour_system = IsocontourBuildingSystem(config)
    
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 检查不同月份的等值线提取
    test_months = [0, 6, 12, 18, 24, 30, 36]
    
    print("\n📊 等值线提取分析:")
    for month in test_months:
        print(f"\n第 {month} 个月:")
        
        # 更新地价场
        land_price_system.update_land_price_field(month, {})
        land_price_field = land_price_system.get_land_price_field()
        
        # 初始化等值线系统
        isocontour_system.initialize_system(land_price_field, transport_hubs, map_size, month, land_price_system)
        
        # 获取等值线数据
        contour_data = isocontour_system.get_contour_data_for_visualization()
        
        commercial_contours = contour_data.get('commercial_contours', [])
        residential_contours = contour_data.get('residential_contours', [])
        
        print(f"  商业等值线数量: {len(commercial_contours)}")
        print(f"  住宅等值线数量: {len(residential_contours)}")
        
        # 检查活跃Hub
        isocontour_system.current_month = month  # 设置当前月份
        active_hubs = isocontour_system._get_active_hubs()
        print(f"  活跃Hub: {[f'Hub{i+1}' for i, hub in enumerate(transport_hubs) if hub in active_hubs]}")
        
        # 分析等值线长度
        if commercial_contours:
            commercial_lengths = [len(contour) for contour in commercial_contours]
            print(f"  商业等值线长度: {commercial_lengths}")
        
        if residential_contours:
            residential_lengths = [len(contour) for contour in residential_contours]
            print(f"  住宅等值线长度: {residential_lengths}")

def debug_slot_system_initialization():
    """调试槽位系统初始化"""
    print("\n🔍 调试槽位系统初始化...")
    
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
    
    # 检查不同月份的槽位系统初始化
    test_months = [0, 6, 12, 18, 24, 30, 36]
    
    print("\n📊 槽位系统初始化分析:")
    for month in test_months:
        print(f"\n第 {month} 个月:")
        
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
        
        print(f"  商业层数量: {len(commercial_layers)}")
        print(f"  住宅层数量: {len(residential_layers)}")
        
        # 统计槽位数量
        total_commercial_slots = sum(len(layer.slots) for layer in commercial_layers)
        total_residential_slots = sum(len(layer.slots) for layer in residential_layers)
        
        print(f"  商业槽位总数: {total_commercial_slots}")
        print(f"  住宅槽位总数: {total_residential_slots}")
        
        # 检查层状态
        for i, layer in enumerate(commercial_layers[:3]):  # 只显示前3层
            print(f"  商业层{i}: {layer.status}, 槽位{len(layer.slots)}个")
        
        for i, layer in enumerate(residential_layers[:3]):  # 只显示前3层
            print(f"  住宅层{i}: {layer.status}, 槽位{len(layer.slots)}个")

def debug_building_generation_logic():
    """调试建筑生成逻辑"""
    print("\n🔍 调试建筑生成逻辑...")
    
    # 检查实际生成的建筑数据
    output_dir = "enhanced_simulation_v3_1_output"
    
    if not os.path.exists(output_dir):
        print("❌ 输出目录不存在，请先运行模拟")
        return
    
    # 分析建筑生成模式
    print("\n📊 建筑生成模式分析:")
    
    months_with_buildings = []
    total_buildings_by_month = {}
    
    for month in range(37):  # 0-36个月
        json_file = os.path.join(output_dir, f"building_positions_month_{month:02d}.json")
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                buildings = data.get('buildings', [])
                if buildings:
                    months_with_buildings.append(month)
                    total_buildings_by_month[month] = len(buildings)
                    
                    # 分析建筑类型分布
                    type_counts = {}
                    for building in buildings:
                        building_type = building.get('type', 'unknown')
                        type_counts[building_type] = type_counts.get(building_type, 0) + 1
                    
                    if month % 6 == 0:  # 每6个月打印一次
                        print(f"第 {month} 个月: {len(buildings)} 个建筑")
                        for building_type, count in type_counts.items():
                            print(f"  {building_type}: {count}")
            
            except Exception as e:
                print(f"⚠️ 读取第 {month} 个月数据时出错: {e}")
    
    print(f"\n📈 建筑生成总结:")
    print(f"有建筑的月份: {months_with_buildings}")
    print(f"建筑数量变化: {total_buildings_by_month}")
    
    # 分析生成模式
    if len(months_with_buildings) > 1:
        first_month = min(months_with_buildings)
        last_month = max(months_with_buildings)
        print(f"建筑生成时间范围: 第 {first_month} 个月 到 第 {last_month} 个月")
        
        # 检查是否有明显的生成模式
        building_growth = []
        for month in sorted(total_buildings_by_month.keys()):
            building_growth.append(total_buildings_by_month[month])
        
        if len(building_growth) > 1:
            growth_rate = [building_growth[i] - building_growth[i-1] for i in range(1, len(building_growth))]
            print(f"建筑增长模式: {growth_rate}")

def debug_gaussian_kernel_impact():
    """调试高斯核对建筑生成的影响"""
    print("\n🔍 调试高斯核对建筑生成的影响...")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 检查地价演化配置
    land_price_evolution = config.get('land_price_evolution', {})
    print(f"\n📊 地价演化配置:")
    print(f"  启用: {land_price_evolution.get('enabled', False)}")
    print(f"  道路激活月份: {land_price_evolution.get('road_activation_month', 0)}")
    print(f"  道路峰值: {land_price_evolution.get('road_peak_value', 0.7)}")
    print(f"  Hub激活月份: {land_price_evolution.get('hub_activation_month', 7)}")
    print(f"  Hub初始峰值: {land_price_evolution.get('hub_initial_peak', 0.7)}")
    print(f"  Hub最终峰值: {land_price_evolution.get('hub_final_peak', 1.0)}")
    print(f"  Hub3保持现有: {land_price_evolution.get('hub3_keep_existing', True)}")
    
    # 检查等值线配置
    isocontour_config = config.get('isocontour_layout', {})
    commercial_config = isocontour_config.get('commercial', {})
    residential_config = isocontour_config.get('residential', {})
    
    print(f"\n📊 等值线配置:")
    print(f"  商业百分位数: {commercial_config.get('percentiles', [])}")
    print(f"  住宅百分位数: {residential_config.get('percentiles', [])}")
    
    # 检查渐进式增长配置
    progressive_config = config.get('progressive_growth', {})
    print(f"\n📊 渐进式增长配置:")
    print(f"  启用: {progressive_config.get('enabled', True)}")
    print(f"  严格满格要求: {progressive_config.get('strict_fill_required', True)}")
    print(f"  死槽容忍率: {progressive_config.get('allow_dead_slots_ratio', 0.05)}")

def main():
    """主函数"""
    print("🔍 槽位系统、地价场、高斯核联动调试")
    print("=" * 60)
    
    try:
        # 1. 调试地价场演化
        debug_landprice_evolution()
        
        # 2. 调试等值线提取
        debug_isocontour_extraction()
        
        # 3. 调试槽位系统初始化
        debug_slot_system_initialization()
        
        # 4. 调试建筑生成逻辑
        debug_building_generation_logic()
        
        # 5. 调试高斯核影响
        debug_gaussian_kernel_impact()
        
        print("\n✅ 调试完成！")
        print("\n🔍 关键问题检查:")
        print("1. 地价场是否按预期演化？")
        print("2. 等值线提取是否响应地价场变化？")
        print("3. 槽位系统是否在正确时机初始化？")
        print("4. 建筑生成是否基于槽位系统？")
        print("5. 高斯核变化是否影响建筑放置？")
        
    except Exception as e:
        print(f"❌ 调试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
