#!/usr/bin/env python3
"""
最终问题分析

分析为什么重复动作和重复建筑问题都没有解决
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def debug_final_analysis():
    """最终问题分析"""
    print("=" * 80)
    print("最终问题分析")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 问题1: 重复动作分析
        print(f"\n   问题1: 重复动作分析")
        
        # 检查候选动作的多样性
        print(f"\n   候选动作多样性检查:")
        for agent in ["IND", "EDU", "COUNCIL"]:
            candidates = env.get_action_candidates(agent)
            print(f"   - {agent}: {len(candidates)} 个候选")
            
            if candidates:
                # 分析动作ID分布
                action_ids = [c.id for c in candidates]
                unique_action_ids = set(action_ids)
                print(f"     - 唯一动作ID: {sorted(unique_action_ids)}")
                
                # 分析槽位分布
                slot_ids = []
                for candidate in candidates:
                    slots = candidate.meta.get("slots", [])
                    if slots:
                        slot_ids.extend(slots)
                
                unique_slots = set(slot_ids)
                print(f"     - 唯一槽位: {len(unique_slots)} 个")
                print(f"     - 槽位示例: {list(unique_slots)[:5]}")
                
                # 检查是否有足够的多样性
                if len(unique_action_ids) == 1:
                    print(f"     - [WARNING] 只有一个动作类型")
                else:
                    print(f"     - [PASS] 有多个动作类型")
                
                if len(unique_slots) < 5:
                    print(f"     - [WARNING] 槽位选择有限")
                else:
                    print(f"     - [PASS] 槽位选择丰富")
        
        # 问题2: 重复建筑分析
        print(f"\n   问题2: 重复建筑分析")
        
        # 模拟多个步骤执行
        print(f"\n   模拟多步骤执行:")
        
        building_log = []
        
        for step in range(5):
            print(f"\n   步骤 {step}:")
            
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
                    
                    # 检查槽位是否已被占用
                    if slot_id in env.occupied_slots:
                        print(f"   - [WARNING] 选择已占用的槽位: {slot_id}")
                    else:
                        print(f"   - [PASS] 选择未占用的槽位: {slot_id}")
                    
                    # 记录建筑信息
                    if slot_id in env.slots:
                        slot = env.slots[slot_id]
                        building_info = {
                            'step': step,
                            'agent': current_agent,
                            'slot_id': slot_id,
                            'position': (slot['x'], slot['y']),
                            'action_id': selected_candidate.id
                        }
                        building_log.append(building_info)
                        print(f"   - 建筑位置: ({slot['x']:.1f}, {slot['y']:.1f})")
                
                # 执行动作
                sequence = Sequence(
                    agent=current_agent,
                    actions=[selected_candidate.id]
                )
                
                print(f"   - 执行前已占用槽位: {len(env.occupied_slots)}")
                next_state, reward, done, info = env.step(current_agent, sequence)
                print(f"   - 执行后已占用槽位: {len(env.occupied_slots)}")
                
                # 检查新候选是否包含已占用槽位
                new_candidates = env.get_action_candidates(current_agent)
                if new_candidates:
                    new_slots = set()
                    for candidate in new_candidates:
                        slots = candidate.meta.get("slots", [])
                        if slots:
                            new_slots.update(slots)
                    
                    occupied_in_new = env.occupied_slots.intersection(new_slots)
                    if occupied_in_new:
                        print(f"   - [WARNING] 新候选中包含已占用槽位: {list(occupied_in_new)}")
                    else:
                        print(f"   - [PASS] 新候选中不包含已占用槽位")
            else:
                print(f"   - [FAIL] 无候选动作")
        
        # 分析建筑日志
        print(f"\n   建筑日志分析:")
        print(f"   - 总建筑数: {len(building_log)}")
        
        # 检查重复槽位
        slot_usage = {}
        for building in building_log:
            slot_id = building['slot_id']
            if slot_id in slot_usage:
                slot_usage[slot_id].append(building)
            else:
                slot_usage[slot_id] = [building]
        
        duplicate_slots = {slot_id: buildings for slot_id, buildings in slot_usage.items() if len(buildings) > 1}
        
        if duplicate_slots:
            print(f"   - [WARNING] 发现重复槽位: {len(duplicate_slots)} 个")
            for slot_id, buildings in duplicate_slots.items():
                print(f"     - 槽位 {slot_id}: {len(buildings)} 次使用")
                for building in buildings:
                    print(f"       - 步骤 {building['step']}: {building['agent']} 在 ({building['position'][0]:.1f}, {building['position'][1]:.1f})")
        else:
            print(f"   - [PASS] 无重复槽位")
        
        # 检查导出系统
        print(f"\n   导出系统检查:")
        
        # 检查步骤日志
        if env.step_logs:
            print(f"   - 步骤日志数量: {len(env.step_logs)}")
            for i, log in enumerate(env.step_logs):
                print(f"   - 日志 {i+1}: {log}")
                
                # 检查槽位位置信息
                if hasattr(log, 'slot_positions') and log.slot_positions:
                    print(f"     - 槽位位置: {log.slot_positions}")
                else:
                    print(f"     - 无槽位位置信息")
        
        # 检查导出文件
        print(f"\n   导出文件检查:")
        
        # 检查输出目录
        output_dir = "outputs"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"   - 输出目录: {output_dir}")
            print(f"   - 文件数量: {len(files)}")
            
            # 检查v4兼容文件
            v4_files = [f for f in files if f.startswith("v4_compatible_month_")]
            if v4_files:
                print(f"   - v4兼容文件: {len(v4_files)} 个")
                for file in sorted(v4_files)[:3]:  # 只显示前3个
                    file_path = os.path.join(output_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            print(f"     - {file}: {len(content)} 字符")
                            if content:
                                lines = content.strip().split('\n')
                                print(f"       - 行数: {len(lines)}")
                                if lines:
                                    print(f"       - 第一行: {lines[0]}")
                                    if len(lines) > 1:
                                        print(f"       - 第二行: {lines[1]}")
                    except Exception as e:
                        print(f"     - {file}: 读取失败 - {e}")
            else:
                print(f"   - 无v4兼容文件")
        else:
            print(f"   - 输出目录不存在")
        
        # 根本原因分析
        print(f"\n   根本原因分析:")
        
        # 检查槽位占用机制
        print(f"\n   槽位占用机制检查:")
        print(f"   - 已占用槽位: {len(env.occupied_slots)}")
        print(f"   - 已占用槽位列表: {list(env.occupied_slots)}")
        
        # 检查槽位状态
        occupied_slot_count = 0
        for slot_id, slot in env.slots.items():
            if isinstance(slot, dict) and slot.get('occupied', False):
                occupied_slot_count += 1
                print(f"   - 槽位 {slot_id}: 已占用, 位置 ({slot.get('x', 0):.1f}, {slot.get('y', 0):.1f})")
        
        print(f"   - 总占用槽位数: {occupied_slot_count}")
        
        # 检查候选动作生成
        print(f"\n   候选动作生成检查:")
        for agent in ["IND", "EDU", "COUNCIL"]:
            candidates = env.get_action_candidates(agent)
            if candidates:
                # 检查候选动作是否包含已占用槽位
                candidate_slots = set()
                for candidate in candidates:
                    slots = candidate.meta.get("slots", [])
                    if slots:
                        candidate_slots.update(slots)
                
                occupied_in_candidates = env.occupied_slots.intersection(candidate_slots)
                if occupied_in_candidates:
                    print(f"   - {agent}: [WARNING] 候选动作包含已占用槽位: {list(occupied_in_candidates)}")
                else:
                    print(f"   - {agent}: [PASS] 候选动作不包含已占用槽位")
        
    except Exception as e:
        print(f"   [FAIL] 分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    debug_final_analysis()
