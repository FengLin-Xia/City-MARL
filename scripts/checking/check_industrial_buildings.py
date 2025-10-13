#!/usr/bin/env python3
"""
检查工业建筑是否存在
"""

import json
import os

def rebuild_building_state(output_dir, target_month):
    """重建完整的建筑状态"""
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

def main():
    """主函数"""
    print("🔍 检查工业建筑...")
    
    buildings = rebuild_building_state('enhanced_simulation_v3_3_output', 23)
    
    print("建筑统计:")
    for building_type, building_list in buildings.items():
        if building_type != 'public':
            print(f"  {building_type}: {len(building_list)}个")
    
    # 检查工业建筑的具体位置
    if 'industrial' in buildings and buildings['industrial']:
        print("\n工业建筑位置:")
        for i, building in enumerate(buildings['industrial'][:5]):  # 只显示前5个
            pos = building['xy']
            print(f"  {i+1}: [{pos[0]}, {pos[1]}]")
    else:
        print("\n❌ 没有找到工业建筑！")
    
    # 检查工业建筑在哪些月份生成
    print("\n检查工业建筑生成月份:")
    industrial_months = []
    
    for month in range(1, 24):
        delta_file = os.path.join('enhanced_simulation_v3_3_output', f'building_delta_month_{month:02d}.json')
        if os.path.exists(delta_file):
            with open(delta_file, 'r') as f:
                delta_data = json.load(f)
            
            industrial_count = 0
            for building in delta_data.get('new_buildings', []):
                if building['building_type'] == 'industrial':
                    industrial_count += 1
            
            if industrial_count > 0:
                industrial_months.append((month, industrial_count))
                print(f"  第{month}个月: {industrial_count}个工业建筑")
    
    if not industrial_months:
        print("  ❌ 没有在任何月份找到工业建筑！")
    
    # 检查层状态中的工业层
    print("\n检查工业层状态:")
    layer_file = os.path.join('enhanced_simulation_v3_3_output', 'layer_state_month_23.json')
    if os.path.exists(layer_file):
        with open(layer_file, 'r') as f:
            layer_data = json.load(f)
        
        industrial_layers = []
        for layer in layer_data.get('layers', {}).get('layers', []):
            if 'industrial' in layer.get('layer_id', ''):
                industrial_layers.append(layer)
        
        print(f"  找到 {len(industrial_layers)} 个工业层:")
        for layer in industrial_layers:
            print(f"    {layer['layer_id']}: {layer['status']}, 容量={layer['capacity']}, 已放置={layer['placed']}")

if __name__ == "__main__":
    main()
