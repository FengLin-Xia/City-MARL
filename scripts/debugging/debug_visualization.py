#!/usr/bin/env python3
"""
调试可视化播放器的数据加载问题
"""

import json
import glob
import os

def debug_visualization_loader():
    """调试可视化播放器的数据加载"""
    print("🔍 调试可视化播放器数据加载...")
    
    output_dir = "enhanced_simulation_v3_1_output"
    
    # 检查文件是否存在
    print(f"\n📁 检查输出目录: {output_dir}")
    if not os.path.exists(output_dir):
        print(f"❌ 输出目录不存在: {output_dir}")
        return
    
    # 查找所有文件
    print("\n📋 查找所有数据文件:")
    
    # 地价场文件
    land_price_files = sorted(glob.glob(f"{output_dir}/land_price_frame_month_*.json"))
    print(f"   地价场文件: {len(land_price_files)} 个")
    for i, file_path in enumerate(land_price_files[:5]):  # 只显示前5个
        print(f"     {i+1}. {os.path.basename(file_path)}")
    if len(land_price_files) > 5:
        print(f"     ... 还有 {len(land_price_files) - 5} 个文件")
    
    # 建筑文件
    building_files = sorted(glob.glob(f"{output_dir}/building_positions_month_*.json"))
    print(f"   建筑文件: {len(building_files)} 个")
    for i, file_path in enumerate(building_files[:5]):
        print(f"     {i+1}. {os.path.basename(file_path)}")
    if len(building_files) > 5:
        print(f"     ... 还有 {len(building_files) - 5} 个文件")
    
    # 层状态文件
    layer_files = sorted(glob.glob(f"{output_dir}/layer_state_month_*.json"))
    print(f"   层状态文件: {len(layer_files)} 个")
    for i, file_path in enumerate(layer_files[:5]):
        print(f"     {i+1}. {os.path.basename(file_path)}")
    if len(layer_files) > 5:
        print(f"     ... 还有 {len(layer_files) - 5} 个文件")
    
    # 测试加载第一个文件
    print("\n🧪 测试加载第一个文件:")
    
    if land_price_files:
        test_file = land_price_files[0]
        print(f"   测试文件: {os.path.basename(test_file)}")
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"   ✅ 加载成功")
            print(f"   月份: {data.get('month')}")
            print(f"   键: {list(data.keys())}")
            
            if 'land_price_field' in data:
                field = data['land_price_field']
                print(f"   地价场形状: {len(field)} x {len(field[0]) if field else 0}")
                print(f"   地价场类型: {type(field)}")
            
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
    
    # 检查月份范围
    print("\n📅 检查月份范围:")
    
    months = set()
    
    # 从地价场文件提取月份
    for file_path in land_price_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                month = data.get('month')
                if month is not None:
                    months.add(month)
        except:
            pass
    
    # 从建筑文件提取月份
    for file_path in building_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 建筑文件可能没有month字段，从文件名提取
                filename = os.path.basename(file_path)
                if 'month_' in filename:
                    month_str = filename.split('month_')[1].split('.')[0]
                    try:
                        month = int(month_str)
                        months.add(month)
                    except:
                        pass
        except:
            pass
    
    # 从层状态文件提取月份
    for file_path in layer_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                month = data.get('month')
                if month is not None:
                    months.add(month)
        except:
            pass
    
    months_list = sorted(list(months))
    print(f"   发现月份: {months_list}")
    print(f"   月份范围: {min(months_list)} - {max(months_list)}")
    print(f"   总月份数: {len(months_list)}")
    
    # 检查数据完整性
    print("\n🔍 检查数据完整性:")
    
    for month in [0, 12, 23]:  # 检查几个关键月份
        print(f"\n   月份 {month}:")
        
        # 检查地价场
        land_price_file = f"{output_dir}/land_price_frame_month_{month:02d}.json"
        if os.path.exists(land_price_file):
            print(f"     ✅ 地价场文件存在")
        else:
            print(f"     ❌ 地价场文件不存在")
        
        # 检查建筑文件
        building_file = f"{output_dir}/building_positions_month_{month:02d}.json"
        if os.path.exists(building_file):
            print(f"     ✅ 建筑文件存在")
        else:
            print(f"     ❌ 建筑文件不存在")
        
        # 检查层状态文件
        layer_file = f"{output_dir}/layer_state_month_{month:02d}.json"
        if os.path.exists(layer_file):
            print(f"     ✅ 层状态文件存在")
        else:
            print(f"     ❌ 层状态文件不存在")

if __name__ == "__main__":
    debug_visualization_loader()


