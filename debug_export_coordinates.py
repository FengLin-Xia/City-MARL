#!/usr/bin/env python3
"""
调试导出坐标问题

分析导出系统中槽位坐标获取问题
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def debug_export_coordinates():
    """调试导出坐标问题"""
    print("=" * 80)
    print("调试导出坐标问题")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 检查槽位数据结构
        print(f"\n   槽位数据结构检查:")
        print(f"   - 环境槽位数量: {len(env.slots)}")
        print(f"   - 环境状态槽位数量: {len(state.slots)}")
        
        # 检查环境槽位
        print(f"\n   环境槽位检查:")
        for i, (slot_id, slot) in enumerate(env.slots.items()):
            if i < 3:  # 只显示前3个
                print(f"   - 槽位 {slot_id}: {type(slot)}")
                if hasattr(slot, 'x'):
                    print(f"     - 坐标: ({slot.x:.1f}, {slot.y:.1f}, {slot.angle:.1f})")
                elif isinstance(slot, dict):
                    print(f"     - 坐标: ({slot.get('x', 0):.1f}, {slot.get('y', 0):.1f}, {slot.get('angle', 0):.1f})")
        
        # 检查环境状态槽位
        print(f"\n   环境状态槽位检查:")
        for i, slot in enumerate(state.slots):
            if i < 3:  # 只显示前3个
                print(f"   - 槽位 {i}: {type(slot)}")
                if hasattr(slot, 'x'):
                    print(f"     - 坐标: ({slot.x:.1f}, {slot.y:.1f}, {slot.angle:.1f})")
                elif isinstance(slot, dict):
                    print(f"     - 坐标: ({slot.get('x', 0):.1f}, {slot.get('y', 0):.1f}, {slot.get('angle', 0):.1f})")
        
        # 模拟执行动作
        print(f"\n   模拟执行动作:")
        
        # 获取候选动作
        candidates = env.get_action_candidates("IND")
        if candidates:
            selected_candidate = candidates[0]
            selected_slots = selected_candidate.meta.get("slots", [])
            
            if selected_slots:
                slot_id = selected_slots[0]
                print(f"   - 选择槽位: {slot_id}")
                
                # 获取槽位坐标
                if slot_id in env.slots:
                    slot = env.slots[slot_id]
                    if hasattr(slot, 'x'):
                        print(f"   - 槽位坐标: ({slot.x:.1f}, {slot.y:.1f}, {slot.angle:.1f})")
                    elif isinstance(slot, dict):
                        print(f"   - 槽位坐标: ({slot.get('x', 0):.1f}, {slot.get('y', 0):.1f}, {slot.get('angle', 0):.1f})")
                
                # 执行动作
                sequence = Sequence(
                    agent="IND",
                    actions=[selected_candidate.id]
                )
                
                next_state, reward, done, info = env.step("IND", sequence)
                print(f"   - 执行后已占用槽位: {len(env.occupied_slots)}")
                
                # 检查执行后的环境状态
                print(f"\n   执行后环境状态检查:")
                print(f"   - 环境状态槽位数量: {len(next_state.slots)}")
                
                # 检查已占用槽位的坐标
                print(f"\n   已占用槽位坐标检查:")
                for slot_id in env.occupied_slots:
                    if slot_id in env.slots:
                        slot = env.slots[slot_id]
                        if hasattr(slot, 'x'):
                            print(f"   - 槽位 {slot_id}: ({slot.x:.1f}, {slot.y:.1f}, {slot.angle:.1f})")
                        elif isinstance(slot, dict):
                            print(f"   - 槽位 {slot_id}: ({slot.get('x', 0):.1f}, {slot.get('y', 0):.1f}, {slot.get('angle', 0):.1f})")
                
                # 检查步骤日志
                print(f"\n   步骤日志检查:")
                if env.step_logs:
                    log = env.step_logs[-1]
                    print(f"   - 最后日志: {log}")
                    print(f"   - 选择的动作: {log.chosen}")
                    
                    # 检查动作ID和槽位ID的对应关系
                    for action_id in log.chosen:
                        print(f"   - 动作ID {action_id}:")
                        
                        # 检查动作参数
                        action_params = env.config.get("action_params", {}).get(str(action_id), {})
                        print(f"     - 动作参数: {action_params}")
                        
                        # 检查槽位索引
                        if action_id < len(next_state.slots):
                            slot = next_state.slots[action_id]
                            print(f"     - 槽位索引 {action_id}: {type(slot)}")
                            if isinstance(slot, dict):
                                print(f"       - 坐标: ({slot.get('x', 0):.1f}, {slot.get('y', 0):.1f}, {slot.get('angle', 0):.1f})")
        
        # 检查导出系统
        print(f"\n   导出系统检查:")
        
        # 检查动作参数配置
        action_params = env.config.get("action_params", {})
        print(f"   - 动作参数数量: {len(action_params)}")
        
        for action_id, params in action_params.items():
            if int(action_id) < 3:  # 只显示前3个
                print(f"   - 动作 {action_id}: {params}")
        
        # 检查导出系统的坐标获取
        print(f"\n   导出系统坐标获取测试:")
        
        # 模拟导出系统的坐标获取
        if env.step_logs:
            log = env.step_logs[-1]
            for action_id in log.chosen:
                print(f"   - 动作ID {action_id}:")
                
                # 方法1：通过动作参数
                action_params = env.config.get("action_params", {}).get(str(action_id), {})
                print(f"     - 动作参数: {action_params}")
                
                # 方法2：通过槽位索引
                if action_id < len(next_state.slots):
                    slot = next_state.slots[action_id]
                    print(f"     - 槽位索引 {action_id}: {type(slot)}")
                    if isinstance(slot, dict):
                        x = slot.get('x', 0.0)
                        y = slot.get('y', 0.0)
                        angle = slot.get('angle', 0.0)
                        print(f"       - 坐标: ({x:.1f}, {y:.1f}, {angle:.1f})")
                
                # 方法3：通过已占用槽位
                for slot_id in env.occupied_slots:
                    if slot_id in env.slots:
                        slot = env.slots[slot_id]
                        if hasattr(slot, 'x'):
                            print(f"     - 已占用槽位 {slot_id}: ({slot.x:.1f}, {slot.y:.1f}, {slot.angle:.1f})")
                        elif isinstance(slot, dict):
                            print(f"     - 已占用槽位 {slot_id}: ({slot.get('x', 0):.1f}, {slot.get('y', 0):.1f}, {slot.get('angle', 0):.1f})")
        
    except Exception as e:
        print(f"   [FAIL] 调试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("调试完成!")
    print("=" * 80)


if __name__ == "__main__":
    debug_export_coordinates()
