#!/usr/bin/env python3
"""
测试后处理逻辑
验证 _post_process_building_types 函数是否正确工作
"""

import json
import numpy as np

def _post_process_building_types(buildings, month):
    """后处理建筑类型，实现 Hub2 工业中心效果"""
    # Hub2 工业中心配置
    hub2_position = [90, 55]  # Hub2 位置
    hub2_radius = 30  # 影响半径
    
    processed_buildings = []
    for building in buildings:
        # 创建建筑副本
        processed_building = building.copy()
        
        # 检查是否在 Hub2 工业中心附近
        if building['type'] == 'commercial':
            x, y = building['position']
            distance = ((x - hub2_position[0])**2 + (y - hub2_position[1])**2)**0.5
            
            if distance <= hub2_radius:
                # 转换为工业建筑类型
                processed_building['type'] = 'industrial'
                processed_building['original_type'] = 'commercial'
                processed_building['hub_influence'] = 'hub2_industrial_zone'
                processed_building['conversion_reason'] = f'Hub2工业中心影响 (距离: {distance:.1f})'
        
        processed_buildings.append(processed_building)
    
    return processed_buildings

def test_postprocessing():
    """测试后处理逻辑"""
    print("=== 测试后处理逻辑 ===")
    
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
        },
        {
            'id': 'res_1',
            'type': 'residential',
            'position': [90, 50],  # 在 Hub2 附近
            'land_price_value': 0.6
        }
    ]
    
    print("原始建筑数据:")
    for building in test_buildings:
        print(f"  {building['id']}: {building['type']} at {building['position']}")
    
    # 应用后处理
    processed_buildings = _post_process_building_types(test_buildings, 1)
    
    print("\n后处理后的建筑数据:")
    for building in processed_buildings:
        print(f"  {building['id']}: {building['type']} at {building['position']}")
        if 'original_type' in building:
            print(f"    原始类型: {building['original_type']}")
            print(f"    转换原因: {building['conversion_reason']}")

def test_real_data():
    """测试真实数据"""
    print("\n=== 测试真实数据 ===")
    
    # 加载真实数据
    data_file = "enhanced_simulation_v3_1_output/building_positions_month_23.json"
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        buildings = data.get('buildings', [])
        print(f"加载了 {len(buildings)} 个建筑")
        
        # 统计原始类型
        original_types = {}
        for building in buildings:
            building_type = building['type']
            if building_type not in original_types:
                original_types[building_type] = 0
            original_types[building_type] += 1
        
        print("原始建筑类型分布:")
        for building_type, count in original_types.items():
            print(f"  {building_type}: {count}个")
        
        # 应用后处理
        processed_buildings = _post_process_building_types(buildings, 23)
        
        # 统计处理后类型
        processed_types = {}
        converted_count = 0
        for building in processed_buildings:
            building_type = building['type']
            if building_type not in processed_types:
                processed_types[building_type] = 0
            processed_types[building_type] += 1
            
            if 'original_type' in building:
                converted_count += 1
        
        print("\n后处理后的建筑类型分布:")
        for building_type, count in processed_types.items():
            print(f"  {building_type}: {count}个")
        
        print(f"\n转换的建筑数: {converted_count}")
        
        # 显示转换示例
        converted_examples = [b for b in processed_buildings if 'original_type' in b][:3]
        if converted_examples:
            print("\n转换示例:")
            for building in converted_examples:
                print(f"  {building['id']}: {building['original_type']} -> {building['type']}")
                print(f"    位置: {building['position']}")
                print(f"    原因: {building['conversion_reason']}")
        
    except FileNotFoundError:
        print(f"文件不存在: {data_file}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    test_postprocessing()
    test_real_data()
