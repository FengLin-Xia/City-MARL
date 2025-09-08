#!/usr/bin/env python3
"""
修正 simplified txt 文件，添加工业建筑类型
"""

import json
import os
import glob

def fix_simplified_txt_files():
    """修正 simplified txt 文件"""
    print("=== 修正 simplified txt 文件 ===")
    
    # 类型映射
    type_map = {'residential': 0, 'commercial': 1, 'office': 2, 'public': 3, 'industrial': 2}
    
    # 查找所有增量文件
    delta_files = glob.glob("enhanced_simulation_v3_1_output/building_delta_month_*.json")
    delta_files = sorted(delta_files)
    
    for delta_file in delta_files:
        # 提取月份
        filename = os.path.basename(delta_file)
        month_str = filename.replace("building_delta_month_", "").replace(".json", "")
        month = int(month_str)
        
        print(f"处理第 {month} 月...")
        
        # 加载增量数据
        with open(delta_file, 'r', encoding='utf-8') as f:
            delta_data = json.load(f)
        
        new_buildings = delta_data.get('new_buildings', [])
        if not new_buildings:
            print(f"  第 {month} 月无新建筑")
            continue
        
        # 检查是否有工业建筑
        industrial_buildings = [b for b in new_buildings if b['type'] == 'industrial']
        if not industrial_buildings:
            print(f"  第 {month} 月无工业建筑")
            continue
        
        print(f"  找到 {len(industrial_buildings)} 个工业建筑")
        
        # 读取对应的 simplified txt 文件
        txt_file = f"enhanced_simulation_v3_1_output/simplified/simplified_buildings_{month:02d}.txt"
        if not os.path.exists(txt_file):
            print(f"  txt 文件不存在: {txt_file}")
            continue
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            txt_content = f.read().strip()
        
        if not txt_content:
            print(f"  txt 文件为空")
            continue
        
        # 解析现有内容
        existing_buildings = []
        if txt_content:
            parts = txt_content.split(', ')
            for part in parts:
                if '(' in part and ')' in part:
                    # 解析格式: 类型(坐标, 坐标, 0)
                    type_part = part.split('(')[0]
                    coords_part = part.split('(')[1].split(')')[0]
                    coords = coords_part.split(', ')
                    if len(coords) >= 2:
                        building_type = int(type_part)
                        x = float(coords[0])
                        y = float(coords[1])
                        existing_buildings.append((building_type, x, y))
        
        # 添加工业建筑
        for building in industrial_buildings:
            pos = building['position']
            x, y = pos[0], pos[1]
            building_type = type_map['industrial']  # 2
            existing_buildings.append((building_type, x, y))
        
        # 重新生成 txt 内容
        formatted_buildings = []
        for building_type, x, y in existing_buildings:
            formatted_buildings.append(f"{building_type}({x:.3f}, {y:.3f}, 0)")
        
        new_txt_content = ", ".join(formatted_buildings)
        
        # 保存修正后的文件
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(new_txt_content)
        
        print(f"  已修正 {txt_file}")
        print(f"  总建筑数: {len(existing_buildings)}")
        
        # 统计类型
        type_counts = {}
        for building_type, x, y in existing_buildings:
            type_counts[building_type] = type_counts.get(building_type, 0) + 1
        
        print(f"  建筑类型分布:")
        for building_type, count in sorted(type_counts.items()):
            type_name = {0: '住宅', 1: '商业', 2: '工业', 3: '公共'}.get(building_type, f'类型{building_type}')
            print(f"    {type_name}: {count}个")

def main():
    """主函数"""
    fix_simplified_txt_files()

if __name__ == "__main__":
    main()
