#!/usr/bin/env python3
"""
修复第0个月的数据，应用 Hub2 工业中心后处理
"""

import json
import os

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

def fix_month_00_data():
    """修复第0个月的数据"""
    print("=== 修复第0个月数据 ===")
    
    # 加载原始数据
    data_file = "enhanced_simulation_v3_1_output/building_positions_month_00.json"
    if not os.path.exists(data_file):
        print(f"文件不存在: {data_file}")
        return
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    buildings = data.get('buildings', [])
    print(f"原始建筑数: {len(buildings)}")
    
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
    processed_buildings = _post_process_building_types(buildings, 0)
    
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
    
    # 更新数据
    data['buildings'] = processed_buildings
    
    # 备份原文件
    backup_file = data_file + ".backup"
    if not os.path.exists(backup_file):
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(json.load(open(data_file, 'r', encoding='utf-8')), f, indent=2, ensure_ascii=False)
        print(f"原文件已备份到: {backup_file}")
    
    # 保存修复后的数据
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"修复后的数据已保存到: {data_file}")
    
    # 显示转换示例
    converted_examples = [b for b in processed_buildings if 'original_type' in b][:3]
    if converted_examples:
        print("\n转换示例:")
        for building in converted_examples:
            print(f"  {building['id']}: {building['original_type']} -> {building['type']}")
            print(f"    位置: {building['position']}")
            print(f"    原因: {building['conversion_reason']}")

if __name__ == "__main__":
    fix_month_00_data()
