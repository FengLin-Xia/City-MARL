#!/usr/bin/env python3
"""
测试 Hub2 工业中心数据后处理效果
验证导出数据和可视化显示的一致性
"""

import os
import json
import numpy as np
from typing import Dict, List

def test_hub2_data_postprocessing():
    """测试 Hub2 工业中心数据后处理效果"""
    print("=== 测试 Hub2 工业中心数据后处理效果 ===")
    
    # 检查数据文件
    data_dir = "enhanced_simulation_v3_1_output"
    if not os.path.exists(data_dir):
        print("错误: 数据目录不存在")
        return
    
    # 查找可用的月份数据
    months = []
    for file in os.listdir(data_dir):
        if file.startswith("building_positions_month_") and file.endswith(".json"):
            month_str = file.replace("building_positions_month_", "").replace(".json", "")
            try:
                month = int(month_str)
                months.append(month)
            except ValueError:
                continue
    
    months = sorted(months)
    if not months:
        print("错误: 没有找到可用的数据文件")
        return
    
    print(f"找到 {len(months)} 个月份的数据: {months}")
    print()
    
    # Hub2 配置
    hub2_position = [90, 55]
    hub2_radius = 30
    
    # 分析每个月份的数据
    for month in months[-3:]:  # 分析最后3个月
        print(f"--- 第 {month} 月数据分析 ---")
        
        # 加载数据
        data_file = os.path.join(data_dir, f"building_positions_month_{month:02d}.json")
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        buildings = data.get('buildings', [])
        print(f"总建筑数: {len(buildings)}")
        
        # 统计建筑类型
        building_types = {}
        hub2_buildings = []
        industrial_buildings = []
        
        for building in buildings:
            building_type = building['type']
            if building_type not in building_types:
                building_types[building_type] = 0
            building_types[building_type] += 1
            
            # 检查是否在 Hub2 附近
            x, y = building['position']
            distance = np.sqrt((x - hub2_position[0])**2 + (y - hub2_position[1])**2)
            if distance <= hub2_radius:
                hub2_buildings.append(building)
            
            # 统计工业建筑
            if building_type == 'industrial':
                industrial_buildings.append(building)
        
        print(f"建筑类型分布:")
        for building_type, count in building_types.items():
            print(f"  {building_type}: {count}个")
        
        print(f"Hub2 附近建筑数: {len(hub2_buildings)}")
        print(f"工业建筑总数: {len(industrial_buildings)}")
        
        # 分析 Hub2 附近的建筑类型
        hub2_types = {}
        for building in hub2_buildings:
            building_type = building['type']
            if building_type not in hub2_types:
                hub2_types[building_type] = 0
            hub2_types[building_type] += 1
        
        print(f"Hub2 附近建筑类型分布:")
        for building_type, count in hub2_types.items():
            print(f"  {building_type}: {count}个")
        
        # 检查是否有转换信息
        converted_buildings = []
        for building in buildings:
            if 'original_type' in building:
                converted_buildings.append(building)
        
        if converted_buildings:
            print(f"转换的建筑数: {len(converted_buildings)}")
            for building in converted_buildings[:3]:  # 显示前3个
                print(f"  {building['id']}: {building['original_type']} -> {building['type']}")
                if 'conversion_reason' in building:
                    print(f"    原因: {building['conversion_reason']}")
        else:
            print("没有找到转换的建筑")
        
        print()

def test_data_consistency():
    """测试数据一致性"""
    print("=== 测试数据一致性 ===")
    
    # 检查增量文件
    data_dir = "enhanced_simulation_v3_1_output"
    delta_files = []
    
    for file in os.listdir(data_dir):
        if file.startswith("building_delta_month_") and file.endswith(".json"):
            delta_files.append(file)
    
    if delta_files:
        print(f"找到 {len(delta_files)} 个增量文件")
        
        # 检查最新的增量文件
        latest_delta = sorted(delta_files)[-1]
        delta_file = os.path.join(data_dir, latest_delta)
        
        with open(delta_file, 'r', encoding='utf-8') as f:
            delta_data = json.load(f)
        
        new_buildings = delta_data.get('new_buildings', [])
        print(f"最新增量文件: {latest_delta}")
        print(f"新增建筑数: {len(new_buildings)}")
        
        # 检查新增建筑的类型
        new_types = {}
        for building in new_buildings:
            building_type = building['type']
            if building_type not in new_types:
                new_types[building_type] = 0
            new_types[building_type] += 1
        
        print("新增建筑类型分布:")
        for building_type, count in new_types.items():
            print(f"  {building_type}: {count}个")
    else:
        print("没有找到增量文件")

def main():
    """主函数"""
    print("Hub2 工业中心数据后处理测试")
    print("=" * 50)
    
    # 测试数据后处理效果
    test_hub2_data_postprocessing()
    print()
    
    # 测试数据一致性
    test_data_consistency()

if __name__ == "__main__":
    main()
