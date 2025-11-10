#!/usr/bin/env python3
"""
调试动作槽位映射

检查动作ID到槽位的映射关系
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def debug_action_slot_mapping():
    """调试动作槽位映射"""
    print("=" * 80)
    print("调试动作槽位映射")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 检查不同智能体的动作候选
        agents = ["IND", "EDU", "COUNCIL"]
        
        for agent in agents:
            print(f"\n   {agent} 智能体动作候选:")
            candidates = env.get_action_candidates(agent)
            
            if candidates:
                print(f"   - 候选数量: {len(candidates)}")
                
                # 显示前3个候选的详细信息
                for i, candidate in enumerate(candidates[:3]):
                    print(f"   - 候选 {i+1}:")
                    print(f"     * 动作ID: {candidate.id}")
                    print(f"     * 槽位列表: {candidate.meta.get('slots', [])}")
                    print(f"     * 槽位数量: {len(candidate.meta.get('slots', []))}")
                    
                    # 获取槽位坐标
                    slots = candidate.meta.get('slots', [])
                    if slots:
                        slot_id = slots[0]  # 第一个槽位
                        print(f"     * 第一个槽位ID: {slot_id}")
                        
                        # 从环境槽位中查找坐标
                        if slot_id in env.slots:
                            slot_info = env.slots[slot_id]
                            print(f"     * 槽位坐标: ({slot_info.get('x', 0)}, {slot_info.get('y', 0)})")
                        else:
                            print(f"     * 槽位未找到: {slot_id}")
            else:
                print(f"   - 无候选动作")
        
        # 检查动作参数配置
        print(f"\n   动作参数配置:")
        action_params = env.config.get("action_params", {})
        print(f"   - 动作参数数量: {len(action_params)}")
        
        # 显示前几个动作参数
        for action_id, params in list(action_params.items())[:5]:
            print(f"   - 动作 {action_id}: {params}")
        
    except Exception as e:
        print(f"   [FAIL] 调试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("调试完成!")
    print("=" * 80)


if __name__ == "__main__":
    debug_action_slot_mapping()

