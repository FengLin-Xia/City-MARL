#!/usr/bin/env python3
"""
实施方案A：直接修正现有的 txt 文件
从增量文件中提取工业建筑数据并添加到对应的 txt 文件中
"""

import json
import os
import glob

def implement_plan_a():
    """实施方案A：修正 txt 文件 - 识别并改写工业建筑"""
    print("=== 实施方案A：修正 simplified txt 文件 ===")
    
    # 类型映射
    type_map = {'residential': 0, 'commercial': 1, 'office': 2, 'public': 3, 'industrial': 2}
    
    # Hub2 工业中心配置
    hub2_position = [90, 55]  # Hub2 位置
    hub2_radius = 30  # 影响半径
    
    # 查找所有增量文件
    delta_files = glob.glob("enhanced_simulation_v3_1_output/building_delta_month_*.json")
    delta_files = sorted(delta_files)
    
    print(f"找到 {len(delta_files)} 个增量文件")
    
    for delta_file in delta_files:
        # 提取月份
        filename = os.path.basename(delta_file)
        month_str = filename.replace("building_delta_month_", "").replace(".json", "")
        month = int(month_str)
        
        print(f"\n--- 处理第 {month} 月 ---")
        
        # 读取对应的 simplified txt 文件
        txt_file = f"enhanced_simulation_v3_1_output/simplified/simplified_buildings_{month:02d}.txt"
        if not os.path.exists(txt_file):
            print(f"  txt 文件不存在: {txt_file}")
            continue
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            txt_content = f.read().strip()
        
        if not txt_content:
            print(f"  txt 文件为空，跳过")
            continue
        
        # 解析现有内容
        existing_buildings = []
        if txt_content:
            parts = txt_content.split(', ')
            for part in parts:
                if '(' in part and ')' in part:
                    try:
                        # 解析格式: 类型(坐标, 坐标, 0)
                        type_part = part.split('(')[0]
                        coords_part = part.split('(')[1].split(')')[0]
                        coords = coords_part.split(', ')
                        if len(coords) >= 2:
                            building_type = int(type_part)
                            x = float(coords[0])
                            y = float(coords[1])
                            existing_buildings.append((building_type, x, y))
                    except (ValueError, IndexError) as e:
                        print(f"    解析错误: {part} - {e}")
                        continue
        
        print(f"  现有建筑数: {len(existing_buildings)}")
        
        # 识别并转换工业建筑
        converted_count = 0
        processed_buildings = []
        
        for building_type, x, y in existing_buildings:
            # 检查是否在 Hub2 工业中心附近且是商业建筑
            if building_type == 1:  # 商业建筑
                distance = ((x - hub2_position[0])**2 + (y - hub2_position[1])**2)**0.5
                if distance <= hub2_radius:
                    # 转换为工业建筑
                    processed_buildings.append((2, x, y))  # 2 = 工业建筑
                    converted_count += 1
                    print(f"    转换工业建筑: 商业建筑 at ({x}, {y}) -> 工业建筑 (距离Hub2: {distance:.1f})")
                else:
                    # 保持商业建筑
                    processed_buildings.append((building_type, x, y))
            else:
                # 其他类型建筑保持不变
                processed_buildings.append((building_type, x, y))
        
        print(f"  转换了 {converted_count} 个工业建筑")
        
        # 重新生成 txt 内容
        formatted_buildings = []
        for building_type, x, y in processed_buildings:
            formatted_buildings.append(f"{building_type}({x:.3f}, {y:.3f}, 0)")
        
        new_txt_content = ", ".join(formatted_buildings)
        
        # 备份原文件
        backup_file = txt_file + ".backup"
        if not os.path.exists(backup_file):
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            print(f"  原文件已备份到: {backup_file}")
        
        # 保存修正后的文件
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(new_txt_content)
        
        print(f"  已修正: {txt_file}")
        print(f"  总建筑数: {len(processed_buildings)}")
        
        # 统计类型
        type_counts = {}
        for building_type, x, y in processed_buildings:
            type_counts[building_type] = type_counts.get(building_type, 0) + 1
        
        print(f"  建筑类型分布:")
        for building_type, count in sorted(type_counts.items()):
            type_name = {0: '住宅', 1: '商业', 2: '工业', 3: '公共'}.get(building_type, f'类型{building_type}')
            print(f"    {type_name}: {count}个")

def verify_results():
    """验证修正结果"""
    print("\n=== 验证修正结果 ===")
    
    # Hub2 工业中心配置
    hub2_position = [90, 55]  # Hub2 位置
    hub2_radius = 30  # 影响半径
    
    # 检查几个关键月份的 txt 文件
    test_months = [21, 24, 27, 30, 33]  # 这些月份应该有工业建筑
    
    for month in test_months:
        txt_file = f"enhanced_simulation_v3_1_output/simplified/simplified_buildings_{month:02d}.txt"
        if not os.path.exists(txt_file):
            print(f"第 {month} 月文件不存在")
            continue
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"第 {month} 月文件为空")
            continue
        
        # 解析建筑数据
        buildings = []
        parts = content.split(', ')
        for part in parts:
            if '(' in part and ')' in part:
                try:
                    type_part = part.split('(')[0]
                    coords_part = part.split('(')[1].split(')')[0]
                    coords = coords_part.split(', ')
                    if len(coords) >= 2:
                        building_type = int(type_part)
                        x = float(coords[0])
                        y = float(coords[1])
                        buildings.append((building_type, x, y))
                except (ValueError, IndexError):
                    continue
        
        # 统计建筑类型
        type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for building_type, x, y in buildings:
            type_counts[building_type] += 1
        
        # 统计Hub2附近的工业建筑
        hub2_industrial_count = 0
        for building_type, x, y in buildings:
            if building_type == 2:  # 工业建筑
                distance = ((x - hub2_position[0])**2 + (y - hub2_position[1])**2)**0.5
                if distance <= hub2_radius:
                    hub2_industrial_count += 1
        
        print(f"第 {month} 月建筑统计:")
        print(f"  住宅: {type_counts[0]} 个")
        print(f"  商业: {type_counts[1]} 个")
        print(f"  工业: {type_counts[2]} 个")
        print(f"  公共: {type_counts[3]} 个")
        print(f"  Hub2附近工业建筑: {hub2_industrial_count} 个")
        
        if type_counts[2] > 0:
            print(f"  ✅ 成功转换工业建筑")
        else:
            print(f"  ❌ 没有工业建筑")

def main():
    """主函数"""
    implement_plan_a()
    verify_results()

if __name__ == "__main__":
    main()
