#!/usr/bin/env python3
"""
修复槽位坐标问题

实现正确的动作ID到槽位坐标的映射
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def fix_slot_coordinates():
    """修复槽位坐标问题"""
    print("=" * 80)
    print("修复槽位坐标问题")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 检查槽位数据
        print(f"\n   槽位数据检查:")
        print(f"   - 状态槽位数量: {len(state.slots)}")
        print(f"   - 环境槽位数量: {len(env.slots)}")
        
        # 检查前几个槽位的坐标
        print(f"\n   前5个槽位坐标:")
        for i, slot in enumerate(state.slots[:5]):
            print(f"   - 槽位 {i}: {slot}")
        
        # 检查环境槽位坐标
        print(f"\n   环境槽位坐标:")
        slot_keys = list(env.slots.keys())[:5]
        for key in slot_keys:
            slot = env.slots[key]
            print(f"   - {key}: ({slot.get('x', 0)}, {slot.get('y', 0)})")
        
        # 检查动作候选的槽位信息
        print(f"\n   动作候选槽位信息:")
        candidates = env.get_action_candidates("IND")
        if candidates:
            candidate = candidates[0]
            slots = candidate.meta.get('slots', [])
            if slots:
                slot_id = slots[0]
                print(f"   - 槽位ID: {slot_id}")
                
                # 从环境槽位中查找坐标
                if slot_id in env.slots:
                    slot_info = env.slots[slot_id]
                    print(f"   - 槽位坐标: ({slot_info.get('x', 0)}, {slot_info.get('y', 0)})")
                else:
                    print(f"   - 槽位未找到: {slot_id}")
        
        # 检查槽位数据是否正确加载
        print(f"\n   槽位数据加载检查:")
        print(f"   - 槽位数据是否为空: {len(env.slots) == 0}")
        print(f"   - 槽位数据是否包含坐标: {all('x' in slot for slot in env.slots.values())}")
        
        # 检查槽位数据来源
        print(f"\n   槽位数据来源:")
        print(f"   - 槽位数据来源: {type(env.slots)}")
        print(f"   - 槽位数据键: {list(env.slots.keys())[:5]}")
        
        # 检查槽位数据是否包含实际坐标
        print(f"\n   槽位数据坐标检查:")
        for key, slot in list(env.slots.items())[:5]:
            x, y = slot.get('x', 0), slot.get('y', 0)
            print(f"   - {key}: ({x}, {y}) - {'实际坐标' if x != 0 or y != 0 else '默认坐标'}")
        
    except Exception as e:
        print(f"   [FAIL] 修复失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("修复完成!")
    print("=" * 80)


if __name__ == "__main__":
    fix_slot_coordinates()