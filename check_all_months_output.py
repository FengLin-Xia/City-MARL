#!/usr/bin/env python3
"""
检查所有月份的输出

检查所有月份的输出是否都只有两个固定槽位
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence
from exporters.v5_0.export_system import V5ExportSystem


def check_all_months_output():
    """检查所有月份的输出"""
    print("=" * 80)
    print("检查所有月份的输出")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 模拟完整的训练过程
        print(f"\n   模拟完整训练过程:")
        
        step_logs = []
        env_states = []
        
        # 运行30个月
        for step in range(30):
            print(f"\n   步骤 {step} (月份 {env.current_month}):")
            
            # 获取当前智能体
            current_agent = env.current_agent
            print(f"   - 当前智能体: {current_agent}")
            
            # 获取候选动作
            candidates = env.get_action_candidates(current_agent)
            if candidates:
                # 选择第一个候选
                selected_candidate = candidates[0]
                selected_slots = selected_candidate.meta.get("slots", [])
                
                if selected_slots:
                    slot_id = selected_slots[0]
                    print(f"   - 选择槽位: {slot_id}")
                    
                    # 执行动作
                    sequence = Sequence(
                        agent=current_agent,
                        actions=[selected_candidate.id]
                    )
                    
                    next_state, reward, done, info = env.step(current_agent, sequence)
                    
                    # 记录步骤日志和环境状态
                    if env.step_logs:
                        step_logs.append(env.step_logs[-1])
                        env_states.append(next_state)
                        
                        # 检查StepLog中的槽位位置信息
                        log = env.step_logs[-1]
                        if hasattr(log, 'slot_positions') and log.slot_positions:
                            slot_pos = log.slot_positions[0]
                            print(f"   - 槽位位置: ({slot_pos['x']:.1f}, {slot_pos['y']:.1f}, {slot_pos['angle']:.1f})")
                        else:
                            print(f"   - 无槽位位置信息")
            else:
                print(f"   - 无候选动作")
        
        # 检查收集的数据
        print(f"\n   收集的数据检查:")
        print(f"   - StepLog数量: {len(step_logs)}")
        print(f"   - 环境状态数量: {len(env_states)}")
        
        # 分析槽位使用情况
        print(f"\n   槽位使用情况分析:")
        
        slot_usage = {}
        position_usage = {}
        
        for i, log in enumerate(step_logs):
            if hasattr(log, 'slot_positions') and log.slot_positions:
                slot_pos = log.slot_positions[0]
                slot_id = slot_pos['slot_id']
                position = (slot_pos['x'], slot_pos['y'])
                
                # 统计槽位使用
                if slot_id in slot_usage:
                    slot_usage[slot_id] += 1
                else:
                    slot_usage[slot_id] = 1
                
                # 统计位置使用
                if position in position_usage:
                    position_usage[position] += 1
                else:
                    position_usage[position] = 1
                
                print(f"   - 步骤 {i}: {slot_id} 在 ({position[0]:.1f}, {position[1]:.1f})")
        
        # 分析结果
        print(f"\n   分析结果:")
        print(f"   - 使用的槽位数量: {len(slot_usage)}")
        print(f"   - 使用的位置数量: {len(position_usage)}")
        
        # 检查重复槽位
        duplicate_slots = {slot_id: count for slot_id, count in slot_usage.items() if count > 1}
        if duplicate_slots:
            print(f"   - [WARNING] 发现重复槽位: {len(duplicate_slots)} 个")
            for slot_id, count in duplicate_slots.items():
                print(f"     - 槽位 {slot_id}: {count} 次使用")
        else:
            print(f"   - [PASS] 无重复槽位")
        
        # 检查重复位置
        duplicate_positions = {pos: count for pos, count in position_usage.items() if count > 1}
        if duplicate_positions:
            print(f"   - [WARNING] 发现重复位置: {len(duplicate_positions)} 个")
            for pos, count in duplicate_positions.items():
                print(f"     - 位置 {pos}: {count} 次使用")
        else:
            print(f"   - [PASS] 无重复位置")
        
        # 检查是否只有两个固定槽位
        if len(slot_usage) <= 2:
            print(f"   - [WARNING] 只使用了 {len(slot_usage)} 个槽位")
            print(f"     - 槽位列表: {list(slot_usage.keys())}")
        else:
            print(f"   - [PASS] 使用了 {len(slot_usage)} 个槽位")
        
        if len(position_usage) <= 2:
            print(f"   - [WARNING] 只使用了 {len(position_usage)} 个位置")
            print(f"     - 位置列表: {list(position_usage.keys())}")
        else:
            print(f"   - [PASS] 使用了 {len(position_usage)} 个位置")
        
        # 测试导出系统
        print(f"\n   测试导出系统:")
        
        # 创建导出系统
        export_system = V5ExportSystem('configs/city_config_v5_0.json')
        print(f"   - 导出系统创建成功")
        
        # 测试导出
        output_dir = "check_output"
        try:
            results = export_system.export_all(step_logs, env_states, output_dir)
            print(f"   - 导出成功: {results}")
            
            # 检查导出文件
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                print(f"   - 导出文件: {files}")
                
                # 检查TXT文件内容
                txt_files = [f for f in files if f.startswith('v4_compatible_month_')]
                if txt_files:
                    print(f"   - TXT文件数量: {len(txt_files)}")
                    
                    # 分析所有TXT文件的内容
                    all_coordinates = []
                    for file in sorted(txt_files):
                        file_path = os.path.join(output_dir, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    lines = content.split('\n')
                                    for line in lines:
                                        # 解析坐标 (x,y,angle)
                                        if '(' in line and ')' in line:
                                            coord_part = line[line.find('(')+1:line.find(')')]
                                            parts = coord_part.split(',')
                                            if len(parts) >= 2:
                                                x = float(parts[0])
                                                y = float(parts[1])
                                                all_coordinates.append((x, y))
                                                print(f"   - {file}: ({x:.1f}, {y:.1f})")
                        except Exception as e:
                            print(f"   - {file}: 读取失败 - {e}")
                    
                    # 分析坐标分布
                    unique_coordinates = set(all_coordinates)
                    print(f"   - 总坐标数量: {len(all_coordinates)}")
                    print(f"   - 唯一坐标数量: {len(unique_coordinates)}")
                    
                    if len(unique_coordinates) <= 2:
                        print(f"   - [WARNING] 只有 {len(unique_coordinates)} 个唯一坐标")
                        print(f"     - 坐标列表: {list(unique_coordinates)}")
                    else:
                        print(f"   - [PASS] 有 {len(unique_coordinates)} 个唯一坐标")
                    
                    # 检查重复坐标
                    coord_counts = {}
                    for coord in all_coordinates:
                        if coord in coord_counts:
                            coord_counts[coord] += 1
                        else:
                            coord_counts[coord] = 1
                    
                    duplicate_coords = {coord: count for coord, count in coord_counts.items() if count > 1}
                    if duplicate_coords:
                        print(f"   - [WARNING] 发现重复坐标: {len(duplicate_coords)} 个")
                        for coord, count in duplicate_coords.items():
                            print(f"     - 坐标 {coord}: {count} 次使用")
                    else:
                        print(f"   - [PASS] 无重复坐标")
        except Exception as e:
            print(f"   - 导出失败: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"   [FAIL] 检查失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("检查完成!")
    print("=" * 80)


if __name__ == "__main__":
    check_all_months_output()
