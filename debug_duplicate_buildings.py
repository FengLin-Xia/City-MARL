#!/usr/bin/env python3
"""
调试重复建筑问题

分析为什么导出的内容中建筑仍然重复在同一槽位上出现
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def debug_duplicate_buildings():
    """调试重复建筑问题"""
    print("=" * 80)
    print("调试重复建筑问题")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 检查槽位占用机制
        print(f"\n   槽位占用机制检查:")
        print(f"   - 已占用槽位: {len(env.occupied_slots)}")
        print(f"   - 已占用槽位列表: {list(env.occupied_slots)}")
        
        # 模拟多个智能体执行
        print(f"\n   模拟多智能体执行:")
        
        building_log = []
        
        for step in range(10):
            print(f"\n   步骤 {step}:")
            
            # 获取当前智能体
            current_agent = env.current_agent
            print(f"   - 当前智能体: {current_agent}")
            
            # 获取候选动作
            candidates = env.get_action_candidates(current_agent)
            print(f"   - 候选数量: {len(candidates)}")
            
            if candidates:
                # 分析候选动作的槽位
                candidate_slots = []
                for candidate in candidates:
                    slots = candidate.meta.get("slots", [])
                    if slots:
                        candidate_slots.extend(slots)
                
                unique_slots = set(candidate_slots)
                print(f"   - 候选槽位数量: {len(unique_slots)}")
                print(f"   - 候选槽位示例: {list(unique_slots)[:5]}")
                
                # 检查已占用槽位是否在候选集中
                occupied_in_candidates = env.occupied_slots.intersection(unique_slots)
                if occupied_in_candidates:
                    print(f"   - [WARNING] 已占用槽位仍在候选集中: {list(occupied_in_candidates)}")
                else:
                    print(f"   - [PASS] 已占用槽位已从候选集中移除")
                
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
                print(f"   - 执行后已占用槽位列表: {list(env.occupied_slots)}")
                
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
        
        # 检查槽位占用更新机制
        print(f"\n   槽位占用更新机制检查:")
        
        # 检查_update_occupied_slots方法是否被调用
        print(f"   - 检查_update_occupied_slots方法调用")
        
        # 手动测试槽位占用更新
        print(f"\n   手动测试槽位占用更新:")
        
        # 重置环境
        env.reset()
        
        # 获取初始候选
        candidates = env.get_action_candidates("IND")
        if candidates:
            selected_candidate = candidates[0]
            selected_slots = selected_candidate.meta.get("slots", [])
            
            if selected_slots:
                slot_id = selected_slots[0]
                print(f"   - 选择槽位: {slot_id}")
                print(f"   - 执行前已占用槽位: {len(env.occupied_slots)}")
                
                # 执行动作
                sequence = Sequence(
                    agent="IND",
                    actions=[selected_candidate.id]
                )
                
                next_state, reward, done, info = env.step("IND", sequence)
                print(f"   - 执行后已占用槽位: {len(env.occupied_slots)}")
                print(f"   - 执行后已占用槽位列表: {list(env.occupied_slots)}")
                
                # 检查槽位是否被正确标记
                if slot_id in env.occupied_slots:
                    print(f"   - [PASS] 槽位 {slot_id} 被正确标记为已占用")
                else:
                    print(f"   - [FAIL] 槽位 {slot_id} 未被标记为已占用")
                
                # 获取新候选
                new_candidates = env.get_action_candidates("IND")
                if new_candidates:
                    new_slots = set()
                    for candidate in new_candidates:
                        slots = candidate.meta.get("slots", [])
                        if slots:
                            new_slots.update(slots)
                    
                    if slot_id in new_slots:
                        print(f"   - [WARNING] 已占用槽位 {slot_id} 仍在候选集中")
                    else:
                        print(f"   - [PASS] 已占用槽位 {slot_id} 已从候选集中移除")
        
        # 检查导出系统
        print(f"\n   导出系统检查:")
        
        # 检查建筑记录
        if env.buildings:
            print(f"   - 建筑记录数量: {len(env.buildings)}")
            for i, building in enumerate(env.buildings):
                print(f"   - 建筑 {i+1}: {building}")
        else:
            print(f"   - 无建筑记录")
        
        # 检查步骤日志
        if env.step_logs:
            print(f"   - 步骤日志数量: {len(env.step_logs)}")
            for i, log in enumerate(env.step_logs):
                print(f"   - 日志 {i+1}: {log}")
        else:
            print(f"   - 无步骤日志")
        
    except Exception as e:
        print(f"   [FAIL] 调试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("调试完成!")
    print("=" * 80)


if __name__ == "__main__":
    debug_duplicate_buildings()

