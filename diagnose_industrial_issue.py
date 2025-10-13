#!/usr/bin/env python3
"""
诊断工业建筑生成问题
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from enhanced_city_simulation_v3_3 import GaussianLandPriceSystemV3_3, ContourExtractionSystemV3_3

def diagnose_industrial_issue():
    """诊断工业建筑生成问题"""
    print("🔍 诊断工业建筑生成问题...")
    
    # 加载极端配置
    with open('configs/city_config_v3_3_extreme.json', 'r') as f:
        config = json.load(f)
    
    # 创建系统
    land_price_system = GaussianLandPriceSystemV3_3(config)
    contour_system = ContourExtractionSystemV3_3(config)
    
    # 初始化
    map_size = [110, 110]
    transport_hubs = config['city']['transport_hubs']
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 获取地价场
    land_price_field = land_price_system.get_land_price_field()
    
    print(f"地价场统计:")
    print(f"  最小值: {land_price_field.min():.3f}")
    print(f"  最大值: {land_price_field.max():.3f}")
    print(f"  平均值: {land_price_field.mean():.3f}")
    print(f"  标准差: {land_price_field.std():.3f}")
    
    # 检查工业等值线阈值
    industrial_levels = config['isocontour_layout']['industrial']['levels']
    print(f"\n工业等值线阈值: {industrial_levels}")
    
    # 检查每个阈值在地价场中的覆盖情况
    for level in industrial_levels:
        coverage = np.sum(land_price_field >= level) / land_price_field.size
        print(f"  阈值 {level}: 覆盖率 {coverage:.1%}")
    
    # 尝试提取工业等值线
    print(f"\n尝试提取工业等值线...")
    contours = contour_system.extract_contours_from_land_price(
        land_price_field, 'industrial', map_size
    )
    
    print(f"提取到的工业等值线数量: {len(contours)}")
    
    if contours:
        for i, contour in enumerate(contours):
            print(f"  等值线 {i+1}: {len(contour)} 个点")
    else:
        print("  ❌ 没有提取到工业等值线！")
    
    # 可视化地价场和等值线
    visualize_land_price_and_contours(land_price_field, industrial_levels, contours, map_size)
    
    # 建议修复方案
    suggest_fix(land_price_field, industrial_levels)

def visualize_land_price_and_contours(land_price_field, industrial_levels, contours, map_size):
    """可视化地价场和等值线"""
    print("\n📊 可视化地价场和等值线...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：地价场
    im1 = ax1.imshow(land_price_field, cmap='YlOrRd', alpha=0.8)
    ax1.set_title('Land Price Field')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 绘制工业等值线阈值
    for level in industrial_levels:
        contours_found = ax1.contour(land_price_field, levels=[level], colors=['blue'], 
                                   linewidths=2, alpha=0.8)
        ax1.clabel(contours_found, inline=True, fontsize=10, fmt=f'{level:.2f}')
    
    # 右图：提取的等值线
    ax2.imshow(land_price_field, cmap='YlOrRd', alpha=0.3)
    ax2.set_title('Extracted Industrial Contours')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    if contours:
        for i, contour in enumerate(contours):
            contour_array = np.array(contour)
            ax2.plot(contour_array[:, 0], contour_array[:, 1], 'b-', linewidth=2, 
                    label=f'Contour {i+1}' if i < 3 else '')
    else:
        ax2.text(map_size[0]//2, map_size[1]//2, 'No Contours Found', 
                ha='center', va='center', fontsize=16, color='red')
    
    if contours:
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('industrial_contour_diagnosis.png', dpi=150, bbox_inches='tight')
    plt.show()

def suggest_fix(land_price_field, current_levels):
    """建议修复方案"""
    print("\n💡 修复建议:")
    
    # 分析地价场分布
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("地价场百分位数:")
    for p in percentiles:
        value = np.percentile(land_price_field, p)
        print(f"  {p}%: {value:.3f}")
    
    # 建议新的工业等值线阈值
    p25 = np.percentile(land_price_field, 25)
    p50 = np.percentile(land_price_field, 50)
    p75 = np.percentile(land_price_field, 75)
    
    suggested_levels = [p25, p50, p75]
    print(f"\n建议的工业等值线阈值: {[f'{x:.3f}' for x in suggested_levels]}")
    
    # 检查建议阈值的覆盖率
    print("建议阈值的覆盖率:")
    for level in suggested_levels:
        coverage = np.sum(land_price_field >= level) / land_price_field.size
        print(f"  {level:.3f}: {coverage:.1%}")
    
    # 创建修复后的配置
    create_fixed_config(suggested_levels)

def create_fixed_config(suggested_levels):
    """创建修复后的配置"""
    print("\n🔧 创建修复后的配置...")
    
    # 加载极端配置
    with open('configs/city_config_v3_3_extreme.json', 'r') as f:
        config = json.load(f)
    
    # 更新工业等值线阈值
    config['isocontour_layout']['industrial']['levels'] = suggested_levels
    
    # 保存修复后的配置
    with open('configs/city_config_v3_3_extreme_fixed.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ 已创建修复后的配置文件: configs/city_config_v3_3_extreme_fixed.json")
    print(f"   新的工业等值线阈值: {[f'{x:.3f}' for x in suggested_levels]}")

def test_fixed_config():
    """测试修复后的配置"""
    print("\n🧪 测试修复后的配置...")
    
    # 加载修复后的配置
    with open('configs/city_config_v3_3_extreme_fixed.json', 'r') as f:
        config = json.load(f)
    
    # 创建模拟系统
    from enhanced_city_simulation_v3_3 import EnhancedCitySimulationV3_3
    simulation = EnhancedCitySimulationV3_3(config)
    simulation.initialize_simulation()
    
    # 运行短期模拟
    simulation.run_simulation(total_months=6)
    
    # 分析结果
    import os
    output_dir = 'enhanced_simulation_v3_3_output'
    
    def rebuild_building_state(output_dir, target_month):
        with open(os.path.join(output_dir, 'building_positions_month_00.json'), 'r') as f:
            base_data = json.load(f)
        
        buildings = base_data['buildings'].copy()
        
        for month in range(1, target_month + 1):
            delta_file = os.path.join(output_dir, f'building_delta_month_{month:02d}.json')
            if os.path.exists(delta_file):
                with open(delta_file, 'r') as f:
                    delta_data = json.load(f)
                
                for building in delta_data.get('new_buildings', []):
                    building_type = building['building_type']
                    buildings[building_type].append(building)
        
        return buildings
    
    buildings = rebuild_building_state(output_dir, 6)
    
    # 统计建筑
    total_buildings = 0
    building_counts = {}
    for building_type, building_list in buildings.items():
        if building_type != 'public':
            count = len(building_list)
            building_counts[building_type] = count
            total_buildings += count
    
    print(f"修复后的建筑分布: {building_counts}")
    print(f"总建筑数: {total_buildings}")
    
    if building_counts.get('industrial', 0) > 0:
        print("✅ 工业建筑生成成功！")
    else:
        print("❌ 工业建筑仍然没有生成")

def main():
    """主函数"""
    print("🔍 工业建筑生成问题诊断")
    
    # 诊断问题
    diagnose_industrial_issue()
    
    # 测试修复后的配置
    test_fixed_config()
    
    print("\n✅ 诊断完成！")

if __name__ == "__main__":
    main()
