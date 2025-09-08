#!/usr/bin/env python3
"""
数据覆盖检查脚本
检查建筑数据和SDF数据的完整性和覆盖情况
"""

import json
import glob
import os

def check_data_coverage():
    """检查数据覆盖情况"""
    output_dir = 'enhanced_simulation_v2_3_output'
    
    print("🔍 数据覆盖检查")
    print("=" * 50)
    
    # 检查建筑数据
    print("\n📊 建筑数据文件:")
    building_files = glob.glob(f'{output_dir}/building_positions_month_*.json')
    building_files.sort()
    
    building_months = []
    for file_path in building_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                month = data['timestamp']
                building_count = len(data['buildings'])
                building_months.append(month)
                print(f"  {file_path}: month_{month}, {building_count} buildings")
        except Exception as e:
            print(f"  ❌ {file_path}: 加载失败 - {e}")
    
    # 检查SDF数据
    print("\n🗺️ SDF数据文件:")
    sdf_files = glob.glob(f'{output_dir}/sdf_field_month_*.json')
    sdf_files.sort()
    
    sdf_months = []
    for file_path in sdf_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                month = data['month']
                sdf_months.append(month)
                sdf_shape = f"{len(data['sdf_field'])}x{len(data['sdf_field'][0])}"
                print(f"  {file_path}: month_{month}, shape: {sdf_shape}")
        except Exception as e:
            print(f"  ❌ {file_path}: 加载失败 - {e}")
    
    # 分析覆盖情况
    print("\n📈 覆盖分析:")
    print(f"  建筑数据月份: {building_months}")
    print(f"  SDF数据月份: {sdf_months}")
    
    # 检查缺失的月份
    all_months = set(building_months + sdf_months)
    if all_months:
        min_month = min(all_months)
        max_month = max(all_months)
        expected_months = set(range(min_month, max_month + 1))
        missing_months = expected_months - all_months
        
        print(f"\n  数据范围: {min_month} - {max_month}")
        print(f"  期望月份: {sorted(expected_months)}")
        print(f"  缺失月份: {sorted(missing_months)}")
        
        # 检查建筑数据缺失
        building_set = set(building_months)
        missing_building = expected_months - building_set
        if missing_building:
            print(f"  ❌ 建筑数据缺失月份: {sorted(missing_building)}")
        
        # 检查SDF数据缺失
        sdf_set = set(sdf_months)
        missing_sdf = expected_months - sdf_set
        if missing_sdf:
            print(f"  ❌ SDF数据缺失月份: {sorted(missing_sdf)}")
    
    # 检查文件大小
    print("\n📏 文件大小检查:")
    for file_path in building_files + sdf_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            size_kb = size / 1024
            print(f"  {file_path}: {size_kb:.1f} KB")
        else:
            print(f"  ❌ {file_path}: 文件不存在")

if __name__ == "__main__":
    check_data_coverage()


