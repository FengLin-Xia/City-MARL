#!/usr/bin/env python3
"""
测试已占用槽位过滤

检查已经被选择的槽位是否还会再次进入候选集
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def test_occupied_slots_filtering():
    """测试已占用槽位过滤"""
    print("=" * 80)
    print("测试已占用槽位过滤")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 检查初始状态
        print(f"\n   初始状态检查:")
        print(f"   - 已占用槽位: {len(env.occupied_slots)}")
        print(f"   - 已占用槽位列表: {list(env.occupied_slots)}")
        
        # 获取初始候选动作
        print(f"\n   初始候选动作:")
        candidates = env.get_action_candidates("IND")
        print(f"   - IND候选数量: {len(candidates)}")
        
        if candidates:
            # 分析候选动作的槽位
            candidate_slots = []
            for candidate in candidates:
                slots = candidate.meta.get("slots", [])
                if slots:
                    candidate_slots.extend(slots)
            
            unique_slots = set(candidate_slots)
            print(f"   - 候选槽位数量: {len(unique_slots)}")
            print(f"   - 候选槽位示例: {list(unique_slots)[:10]}")
        
        # 模拟执行一个动作
        print(f"\n   模拟执行动作:")
        if candidates:
            # 选择第一个候选动作
            selected_candidate = candidates[0]
            selected_slots = selected_candidate.meta.get("slots", [])
            print(f"   - 选择的动作ID: {selected_candidate.id}")
            print(f"   - 选择的槽位: {selected_slots}")
            
            # 创建序列并执行
            sequence = Sequence(
                agent="IND",
                actions=[selected_candidate.id]
            )
            
            # 执行动作
            next_state, reward, done, info = env.step("IND", sequence)
            print(f"   - 执行后奖励: {reward}")
            print(f"   - 执行后已占用槽位: {len(env.occupied_slots)}")
            print(f"   - 执行后已占用槽位列表: {list(env.occupied_slots)}")
            
            # 检查执行后的候选动作
            print(f"\n   执行后候选动作:")
            new_candidates = env.get_action_candidates("IND")
            print(f"   - 新候选数量: {len(new_candidates)}")
            
            if new_candidates:
                # 分析新候选动作的槽位
                new_candidate_slots = []
                for candidate in new_candidates:
                    slots = candidate.meta.get("slots", [])
                    if slots:
                        new_candidate_slots.extend(slots)
                
                new_unique_slots = set(new_candidate_slots)
                print(f"   - 新候选槽位数量: {len(new_unique_slots)}")
                print(f"   - 新候选槽位示例: {list(new_unique_slots)[:10]}")
                
                # 检查是否有重复槽位
                overlapping_slots = unique_slots.intersection(new_unique_slots)
                print(f"   - 重复槽位数量: {len(overlapping_slots)}")
                print(f"   - 重复槽位示例: {list(overlapping_slots)[:5]}")
                
                # 检查已占用槽位是否还在候选集中
                occupied_in_candidates = env.occupied_slots.intersection(new_unique_slots)
                print(f"   - 已占用槽位仍在候选集中: {len(occupied_in_candidates)}")
                if occupied_in_candidates:
                    print(f"   - 仍在候选集的已占用槽位: {list(occupied_in_candidates)}")
                    print(f"   - [WARNING] 已占用槽位仍在候选集中！")
                else:
                    print(f"   - [PASS] 已占用槽位已从候选集中移除")
            else:
                print(f"   - [FAIL] 执行后无候选动作")
        
        # 测试多次执行
        print(f"\n   测试多次执行:")
        for i in range(3):
            print(f"\n   第 {i+1} 次执行:")
            
            # 获取当前活跃智能体
            current_agent = env.current_agent
            print(f"   - 当前智能体: {current_agent}")
            
            # 获取当前候选
            current_candidates = env.get_action_candidates(current_agent)
            if not current_candidates:
                print(f"   - 无候选动作，停止测试")
                break
            
            # 选择第一个候选
            selected_candidate = current_candidates[0]
            selected_slots = selected_candidate.meta.get("slots", [])
            
            print(f"   - 选择槽位: {selected_slots}")
            print(f"   - 执行前已占用槽位: {len(env.occupied_slots)}")
            
            # 执行动作
            sequence = Sequence(
                agent=current_agent,
                actions=[selected_candidate.id]
            )
            
            next_state, reward, done, info = env.step(current_agent, sequence)
            
            print(f"   - 执行后已占用槽位: {len(env.occupied_slots)}")
            print(f"   - 执行后已占用槽位列表: {list(env.occupied_slots)}")
            
            # 检查新候选是否包含已占用槽位
            new_candidates = env.get_action_candidates("IND")
            if new_candidates:
                new_slots = set()
                for candidate in new_candidates:
                    slots = candidate.meta.get("slots", [])
                    if slots:
                        new_slots.update(slots)
                
                occupied_in_new = env.occupied_slots.intersection(new_slots)
                print(f"   - 新候选中包含已占用槽位: {len(occupied_in_new)}")
                if occupied_in_new:
                    print(f"   - [WARNING] 已占用槽位仍在候选集中: {list(occupied_in_new)}")
                else:
                    print(f"   - [PASS] 已占用槽位已从候选集中移除")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_occupied_slots_filtering()
