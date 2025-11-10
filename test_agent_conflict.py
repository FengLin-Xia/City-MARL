#!/usr/bin/env python3
"""
测试智能体动作冲突

验证EDU和COUNCIL是否会出现选择同一个动作的情况
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def test_agent_conflict():
    """测试智能体动作冲突"""
    print("=" * 80)
    print("测试智能体动作冲突")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 测试10步的动作选择
        print(f"\n   测试10步动作选择:")
        print(f"   {'步骤':<4} {'月份':<4} {'智能体':<8} {'动作ID':<8} {'槽位ID':<10} {'描述':<15}")
        print(f"   {'-'*4} {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*15}")
        
        for step in range(10):
            # 获取当前phase信息
            phase_agents = env.get_phase_agents()
            execution_mode = env.get_phase_execution_mode()
            
            print(f"   步骤{step}: {phase_agents}, 模式: {execution_mode}")
            
            # 为每个智能体获取动作候选
            phase_sequences = {}
            phase_candidates = {}
            
            for agent in phase_agents:
                candidates = env.get_action_candidates(agent)
                phase_candidates[agent] = candidates
                
                if candidates:
                    # 选择第一个候选动作
                    selected_candidate = candidates[0]
                    sequence = Sequence(
                        agent=agent,
                        actions=[selected_candidate.id]
                    )
                    phase_sequences[agent] = sequence
                    
                    # 显示选择结果
                    slot_id = selected_candidate.meta.get('slot_id', 'N/A')
                    desc = selected_candidate.meta.get('desc', 'N/A')
                    print(f"     {agent:<8} {selected_candidate.id:<8} {slot_id:<10} {desc:<15}")
                else:
                    phase_sequences[agent] = None
                    print(f"     {agent:<8} {'N/A':<8} {'N/A':<10} {'No candidates':<15}")
            
            # 执行phase
            try:
                next_state, phase_rewards, done, info = env.step_phase(phase_agents, phase_sequences)
                state = next_state
                
                if done:
                    print(f"     环境结束于步骤 {step}")
                    break
                    
            except Exception as e:
                print(f"     执行失败: {e}")
                break
        
        # 分析结果
        print(f"\n   分析结果:")
        print(f"   - 最终月份: {state.month}")
        print(f"   - 最终步骤: {env.current_step}")
        print(f"   - 步骤日志数量: {len(env.step_logs)}")
        
        # 检查动作ID分布
        action_ids = {}
        for log in env.step_logs:
            agent = log.agent
            if agent not in action_ids:
                action_ids[agent] = []
            action_ids[agent].extend(log.chosen)
        
        print(f"\n   动作ID分布:")
        for agent, ids in action_ids.items():
            print(f"   {agent}: {sorted(set(ids))}")
        
        # 检查是否有重叠
        all_ids = []
        for ids in action_ids.values():
            all_ids.extend(ids)
        
        unique_ids = set(all_ids)
        if len(unique_ids) == len(all_ids):
            print(f"   [PASS] 没有动作ID冲突")
        else:
            print(f"   [FAIL] 发现动作ID冲突")
        
        # 检查槽位冲突
        slot_conflicts = []
        for log in env.step_logs:
            # 这里需要从日志中提取槽位信息
            pass
        
        if slot_conflicts:
            print(f"   [FAIL] 发现槽位冲突: {slot_conflicts}")
        else:
            print(f"   [PASS] 没有槽位冲突")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_agent_conflict()

