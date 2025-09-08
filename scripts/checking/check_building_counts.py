#!/usr/bin/env python3
"""
检查建筑数量变化
"""

import json
import glob
import os

def rebuild_building_state(output_dir, target_month):
    """重建指定月份的建筑状态"""
    # 加载第0个月
    month_0_file = os.path.join(output_dir, 'building_positions_month_00.json')
    with open(month_0_file, 'r') as f:
        data = json.load(f)
        buildings = data['buildings'].copy()
    
    # 加载增量数据
    delta_files = sorted(glob.glob(os.path.join(output_dir, 'building_delta_month_*.json')))
    
    for delta_file in delta_files:
        with open(delta_file, 'r') as f:
            delta_data = json.load(f)
            month = delta_data['month']
            new_buildings = delta_data['new_buildings']
            
            if month <= target_month:
                for building in new_buildings:
                    building_type = building['building_type']
                    if building_type in buildings:
                        buildings[building_type].append(building)
    
    return buildings

def main():
    """主函数"""
    print("检查建筑数量变化...")
    
    # 检查前12个月的建筑数量
    for month in range(13):
        buildings = rebuild_building_state('enhanced_simulation_v3_3_output', month)
        residential = len(buildings['residential'])
        commercial = len(buildings['commercial'])
        industrial = len(buildings['industrial'])
        total = residential + commercial + industrial
        
        print(f'第{month}个月: 住宅{residential}个, 商业{commercial}个, 工业{industrial}个, 总计{total}个')

if __name__ == "__main__":
    main()
