#!/usr/bin/env python3
"""
分析v5.0导出文件中的槽位重复选择问题
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

def parse_export_file(file_path: str) -> List[Tuple[str, float, float, float]]:
    """
    解析导出文件，提取槽位坐标信息
    
    Returns:
        List of (agent, x, y, z) tuples
    """
    coordinates = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            return coordinates
            
        # 解析格式: agent_id(x,y,z)value, agent_id(x,y,z)value, ...
        # 例如: 3(124.3,83.0,0)9.8, 3(120.9,75.7,0)18.4
        
        # 使用正则表达式直接匹配所有坐标
        # 匹配格式: agent_id(x,y,z)value
        pattern = r'(\d+)\(([^,]+),([^,]+),([^)]+)\)([^,]*?)(?=,\s*\d+\(|$)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            agent_id = match[0]
            x = float(match[1])
            y = float(match[2])
            z = float(match[3])
            
            coordinates.append((agent_id, x, y, z))
                
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")
        
    return coordinates

def analyze_slot_duplicates(output_dir: str) -> Dict[str, any]:
    """
    分析槽位重复选择问题
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        分析结果字典
    """
    print("=== 分析v5.0槽位重复选择问题 ===")
    
    all_coordinates = []
    month_coordinates = defaultdict(list)
    agent_coordinates = defaultdict(list)
    
    # 遍历所有导出文件
    for filename in sorted(os.listdir(output_dir)):
        if filename.startswith('export_month_') and filename.endswith('.txt'):
            file_path = os.path.join(output_dir, filename)
            
            # 提取月份
            month_match = re.search(r'export_month_(\d+)\.txt', filename)
            if not month_match:
                continue
            month = int(month_match.group(1))
            
            # 解析文件
            coordinates = parse_export_file(file_path)
            
            if coordinates:
                print(f"月份 {month:2d}: {len(coordinates)} 个槽位")
                for agent_id, x, y, z in coordinates:
                    print(f"  Agent {agent_id}: ({x:.1f}, {y:.1f}, {z:.1f})")
                
                all_coordinates.extend(coordinates)
                month_coordinates[month] = coordinates
                
                # 按智能体分组
                for agent_id, x, y, z in coordinates:
                    agent_coordinates[agent_id].append((month, x, y, z))
    
    print(f"\n总计: {len(all_coordinates)} 个槽位选择")
    
    # 检查重复坐标
    print("\n--- 检查重复坐标 ---")
    
    # 按坐标分组
    coord_groups = defaultdict(list)
    for agent_id, x, y, z in all_coordinates:
        coord_key = (x, y, z)
        coord_groups[coord_key].append(agent_id)
    
    # 查找重复
    duplicates = {}
    for coord_key, agents in coord_groups.items():
        if len(agents) > 1:
            x, y, z = coord_key
            duplicates[coord_key] = agents
            print(f"[ERROR] 重复坐标 ({x:.1f}, {y:.1f}, {z:.1f}): 被智能体 {agents} 选择")
    
    if not duplicates:
        print("[OK] 未发现重复坐标")
    else:
        print(f"[ERROR] 发现 {len(duplicates)} 个重复坐标")
    
    # 检查同一智能体的重复选择
    print("\n--- 检查智能体内重复选择 ---")
    
    agent_duplicates = {}
    for agent_id, coords in agent_coordinates.items():
        agent_coord_groups = defaultdict(list)
        for month, x, y, z in coords:
            coord_key = (x, y, z)
            agent_coord_groups[coord_key].append(month)
        
        agent_dups = {}
        for coord_key, months in agent_coord_groups.items():
            if len(months) > 1:
                x, y, z = coord_key
                agent_dups[coord_key] = months
                print(f"[ERROR] 智能体 {agent_id} 重复选择坐标 ({x:.1f}, {y:.1f}, {z:.1f}): 月份 {months}")
        
        if agent_dups:
            agent_duplicates[agent_id] = agent_dups
    
    if not agent_duplicates:
        print("[OK] 智能体内无重复选择")
    else:
        print(f"[ERROR] {len(agent_duplicates)} 个智能体存在重复选择")
    
    # 统计信息
    print("\n--- 统计信息 ---")
    print(f"总槽位选择数: {len(all_coordinates)}")
    print(f"涉及月份数: {len(month_coordinates)}")
    print(f"涉及智能体数: {len(agent_coordinates)}")
    
    for agent_id, coords in agent_coordinates.items():
        print(f"智能体 {agent_id}: {len(coords)} 次选择")
    
    # 返回分析结果
    result = {
        "total_selections": len(all_coordinates),
        "months_count": len(month_coordinates),
        "agents_count": len(agent_coordinates),
        "duplicate_coordinates": len(duplicates),
        "agent_duplicates": len(agent_duplicates),
        "duplicates": duplicates,
        "agent_duplicates": agent_duplicates,
        "month_coordinates": dict(month_coordinates),
        "agent_coordinates": dict(agent_coordinates)
    }
    
    return result

def main():
    """主函数"""
    output_dir = "test_output"
    
    if not os.path.exists(output_dir):
        print(f"输出目录不存在: {output_dir}")
        return
    
    # 分析槽位重复选择
    result = analyze_slot_duplicates(output_dir)
    
    # 总结
    print("\n=== 分析总结 ===")
    if result["duplicate_coordinates"] == 0 and len(result["agent_duplicates"]) == 0:
        print("[SUCCESS] v5.0架构槽位选择机制正常，未发现重复选择问题")
    else:
        print("[FAILED] 发现槽位重复选择问题:")
        if result["duplicate_coordinates"] > 0:
            print(f"  - 跨智能体重复坐标: {result['duplicate_coordinates']} 个")
        if len(result["agent_duplicates"]) > 0:
            print(f"  - 智能体内重复选择: {len(result['agent_duplicates'])} 个智能体")

if __name__ == "__main__":
    main()
