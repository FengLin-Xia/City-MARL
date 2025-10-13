#!/usr/bin/env python3
"""
调试Hub3槽位生成
检查Hub3周围是否有有效的建筑槽位
"""

import json
import numpy as np
import os

def analyze_hub3_slots():
    """分析Hub3周围的建筑槽位生成情况"""
    
    # Hub3位置
    hub3_x, hub3_y = 67, 94
    
    # 检查几个关键月份
    months_to_check = [2, 10, 20, 30]
    
    print("=== Hub3 槽位生成分析 ===")
    print(f"Hub3 位置: ({hub3_x}, {hub3_y})")
    print()
    
    for month in months_to_check:
        try:
            # 检查是否有槽位文件
            slot_filename = f"enhanced_simulation_v3_1_output/isocontour_building_slots_month_{month:02d}.json"
            
            if os.path.exists(slot_filename):
                print(f"--- Month {month} ---")
                print(f"✅ 找到槽位文件: {slot_filename}")
                
                # 读取槽位数据
                with open(slot_filename, 'r', encoding='utf-8') as f:
                    slot_data = json.load(f)
                
                # 分析槽位数据
                if 'building_slots' in slot_data:
                    all_slots = slot_data['building_slots']
                    print(f"总槽位数量: {len(all_slots)}")
                    
                    # 按类型分组
                    slot_types = {}
                    for slot in all_slots:
                        slot_type = slot.get('type', 'unknown')
                        if slot_type not in slot_types:
                            slot_types[slot_type] = []
                        slot_types[slot_type].append(slot)
                    
                    print("槽位类型分布:")
                    for slot_type, slots in slot_types.items():
                        print(f"  {slot_type}: {len(slots)} 个")
                    
                    # 检查Hub3周围的槽位
                    hub3_slots = []
                    for slot in all_slots:
                        if 'position' in slot:
                            x, y = slot['position']
                            distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
                            if distance <= 30:  # 30像素范围内
                                hub3_slots.append((slot, distance))
                    
                    print(f"Hub3 周围槽位数量 (30像素内): {len(hub3_slots)}")
                    
                    if hub3_slots:
                        print("Hub3 周围槽位详情:")
                        for slot, distance in sorted(hub3_slots, key=lambda x: x[1]):
                            slot_type = slot.get('type', 'unknown')
                            x, y = slot['position']
                            print(f"  类型: {slot_type}, 位置: ({x}, {y}), 距离: {distance:.1f}")
                    else:
                        print("❌ Hub3 周围没有槽位")
                        
                        # 检查最近的槽位
                        if all_slots:
                            min_distance = float('inf')
                            nearest_slot = None
                            for slot in all_slots:
                                if 'position' in slot:
                                    x, y = slot['position']
                                    distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
                                    if distance < min_distance:
                                        min_distance = distance
                                        nearest_slot = slot
                            
                            if nearest_slot:
                                x, y = nearest_slot['position']
                                slot_type = nearest_slot.get('type', 'unknown')
                                print(f"最近槽位: 类型 {slot_type}, 位置 ({x}, {y}), 距离: {min_distance:.1f}")
                
                else:
                    print("❌ 槽位文件中没有 'building_slots' 字段")
                    print("文件内容:", list(slot_data.keys()))
            
            else:
                print(f"--- Month {month} ---")
                print(f"❌ 槽位文件不存在: {slot_filename}")
                
                # 检查是否有其他相关文件
                related_files = []
                for file in os.listdir("enhanced_simulation_v3_1_output"):
                    if f"month_{month:02d}" in file:
                        related_files.append(file)
                
                if related_files:
                    print("相关文件:")
                    for file in related_files:
                        print(f"  {file}")
                else:
                    print("没有找到相关文件")
            
            print()
            
        except Exception as e:
            print(f"❌ 分析出错: {e}")
    
    # 检查槽位生成逻辑
    print("=== 槽位生成逻辑分析 ===")
    print("检查 IsocontourBuildingSystem 的槽位生成逻辑...")
    
    try:
        with open('logic/isocontour_building_system.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 查找关键方法
        key_methods = [
            'generate_building_slots',
            '_generate_slots_from_contour',
            '_create_slot_positions',
            '_validate_slot'
        ]
        
        for method in key_methods:
            if method in code:
                print(f"✅ 找到 {method} 方法")
            else:
                print(f"❌ 未找到 {method} 方法")
        
        # 检查槽位保存逻辑
        if 'save_building_slots' in code or 'building_slots' in code:
            print("✅ 有槽位保存逻辑")
        else:
            print("⚠️  可能缺少槽位保存逻辑")
            
    except Exception as e:
        print(f"❌ 读取代码文件出错: {e}")
    
    # 检查主模拟文件中的槽位生成调用
    print("\n=== 主模拟文件分析 ===")
    try:
        with open('enhanced_city_simulation_v3_1.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 查找槽位生成相关代码
        if 'generate_building_slots' in code:
            print("✅ 主模拟文件中有槽位生成调用")
        else:
            print("❌ 主模拟文件中没有槽位生成调用")
            
        if 'isocontour_system' in code:
            print("✅ 主模拟文件中有等值线系统引用")
        else:
            print("❌ 主模拟文件中没有等值线系统引用")
            
    except Exception as e:
        print(f"❌ 读取主模拟文件出错: {e}")

if __name__ == "__main__":
    analyze_hub3_slots()
