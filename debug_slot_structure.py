#!/usr/bin/env python3
"""
调试槽位数据结构

检查槽位数据的实际结构
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def debug_slot_structure():
    """调试槽位数据结构"""
    print("=" * 80)
    print("调试槽位数据结构")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 检查槽位数据结构
        print(f"\n   槽位数据结构:")
        print(f"   - 槽位数量: {len(state.slots)}")
        print(f"   - 槽位类型: {type(state.slots)}")
        
        if state.slots:
            print(f"   - 第一个槽位: {state.slots[0]}")
            print(f"   - 第一个槽位类型: {type(state.slots[0])}")
            
            if isinstance(state.slots[0], dict):
                print(f"   - 第一个槽位键: {list(state.slots[0].keys())}")
                print(f"   - 第一个槽位值: {state.slots[0]}")
            else:
                print(f"   - 第一个槽位属性: {dir(state.slots[0])}")
        
        # 检查环境中的槽位数据
        print(f"\n   环境槽位数据:")
        print(f"   - 环境槽位数量: {len(env.slots)}")
        print(f"   - 环境槽位类型: {type(env.slots)}")
        
        if env.slots:
            slot_keys = list(env.slots.keys())[:5]  # 显示前5个槽位
            print(f"   - 槽位键示例: {slot_keys}")
            
            for key in slot_keys[:3]:  # 显示前3个槽位的详细信息
                slot = env.slots[key]
                print(f"   - 槽位 {key}: {slot}")
        
        # 检查动作候选的槽位信息
        print(f"\n   动作候选槽位信息:")
        candidates = env.get_action_candidates("IND")
        if candidates:
            candidate = candidates[0]
            print(f"   - 候选动作ID: {candidate.id}")
            print(f"   - 候选动作元数据: {candidate.meta}")
            print(f"   - 槽位信息: {candidate.meta.get('slots', [])}")
        
    except Exception as e:
        print(f"   [FAIL] 调试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("调试完成!")
    print("=" * 80)


if __name__ == "__main__":
    debug_slot_structure()
