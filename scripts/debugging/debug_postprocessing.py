#!/usr/bin/env python3
"""
调试后处理问题
"""

import os
import json
import sys

# 添加当前目录到路径，以便导入模块
sys.path.append('.')

def debug_postprocessing():
    """调试后处理问题"""
    print("=== 调试后处理问题 ===")
    
    # 检查文件是否存在
    data_file = "enhanced_simulation_v3_1_output/building_positions_month_23.json"
    if not os.path.exists(data_file):
        print(f"文件不存在: {data_file}")
        return
    
    # 加载数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    buildings = data.get('buildings', [])
    print(f"加载了 {len(buildings)} 个建筑")
    
    # 统计商业建筑
    commercial_buildings = [b for b in buildings if b['type'] == 'commercial']
    print(f"商业建筑数: {len(commercial_buildings)}")
    
    # 检查 Hub2 附近的商业建筑
    hub2_position = [90, 55]
    hub2_radius = 30
    
    hub2_commercial = []
    for building in commercial_buildings:
        x, y = building['position']
        distance = ((x - hub2_position[0])**2 + (y - hub2_position[1])**2)**0.5
        if distance <= hub2_radius:
            hub2_commercial.append((building, distance))
    
    print(f"Hub2 附近的商业建筑数: {len(hub2_commercial)}")
    
    if hub2_commercial:
        print("Hub2 附近的商业建筑:")
        for building, distance in hub2_commercial[:5]:  # 显示前5个
            print(f"  {building['id']}: {building['position']} (距离: {distance:.1f})")
    
    # 检查是否有工业建筑
    industrial_buildings = [b for b in buildings if b['type'] == 'industrial']
    print(f"工业建筑数: {len(industrial_buildings)}")
    
    # 检查是否有转换信息
    converted_buildings = [b for b in buildings if 'original_type' in b]
    print(f"有转换信息的建筑数: {len(converted_buildings)}")
    
    if converted_buildings:
        print("转换的建筑:")
        for building in converted_buildings[:3]:
            print(f"  {building['id']}: {building['original_type']} -> {building['type']}")

def test_postprocessing_function():
    """测试后处理函数"""
    print("\n=== 测试后处理函数 ===")
    
    # 导入后处理函数
    try:
        from enhanced_city_simulation_v3_1 import EnhancedCitySimulationV3_1
        
        # 创建模拟实例
        sim = EnhancedCitySimulationV3_1()
        
        # 创建测试数据
        test_buildings = [
            {
                'id': 'com_1',
                'type': 'commercial',
                'position': [95, 55],  # 在 Hub2 附近
                'land_price_value': 0.8
            },
            {
                'id': 'com_2',
                'type': 'commercial', 
                'position': [20, 55],  # 远离 Hub2
                'land_price_value': 0.7
            }
        ]
        
        print("测试数据:")
        for building in test_buildings:
            print(f"  {building['id']}: {building['type']} at {building['position']}")
        
        # 调用后处理函数
        processed = sim._post_process_building_types(test_buildings, 1)
        
        print("\n后处理结果:")
        for building in processed:
            print(f"  {building['id']}: {building['type']} at {building['position']}")
            if 'original_type' in building:
                print(f"    原始类型: {building['original_type']}")
                print(f"    转换原因: {building['conversion_reason']}")
        
    except Exception as e:
        print(f"导入或测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_postprocessing()
    test_postprocessing_function()
