#!/usr/bin/env python3
"""
调试重复动作问题

分析为什么智能体一直在做重复的同样动作
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def debug_repetitive_actions():
    """调试重复动作问题"""
    print("=" * 80)
    print("调试重复动作问题")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 检查智能体配置
        print(f"\n   智能体配置:")
        agents_config = env.config.get("agents", {})
        agent_order = agents_config.get("order", [])
        print(f"   - 智能体顺序: {agent_order}")
        
        agent_defs = agents_config.get("defs", {})
        for agent, config in agent_defs.items():
            action_ids = config.get("action_ids", [])
            print(f"   - {agent}: 动作ID {action_ids}")
        
        # 检查动作参数
        print(f"\n   动作参数:")
        action_params = env.config.get("action_params", {})
        for action_id, params in action_params.items():
            desc = params.get("desc", "")
            cost = params.get("cost", 0)
            reward = params.get("reward", 0)
            print(f"   - 动作 {action_id}: {desc}, 成本={cost}, 奖励={reward}")
        
        # 检查候选动作的多样性
        print(f"\n   候选动作多样性分析:")
        
        for agent in agent_order:
            print(f"\n   {agent} 智能体:")
            candidates = env.get_action_candidates(agent)
            
            if candidates:
                print(f"   - 候选数量: {len(candidates)}")
                
                # 分析动作ID分布
                action_ids = [c.id for c in candidates]
                unique_action_ids = set(action_ids)
                print(f"   - 唯一动作ID: {sorted(unique_action_ids)}")
                print(f"   - 动作ID数量: {len(unique_action_ids)}")
                
                # 分析槽位分布
                slot_ids = []
                for candidate in candidates:
                    slots = candidate.meta.get("slots", [])
                    if slots:
                        slot_ids.extend(slots)
                
                unique_slots = set(slot_ids)
                print(f"   - 唯一槽位数量: {len(unique_slots)}")
                print(f"   - 槽位分布: {len(slot_ids)} 个候选使用 {len(unique_slots)} 个不同槽位")
                
                # 检查是否有足够的多样性
                if len(unique_action_ids) == 1:
                    print(f"   - [WARNING] 只有一个动作类型: {unique_action_ids}")
                else:
                    print(f"   - [PASS] 有多个动作类型")
                
                if len(unique_slots) < 10:
                    print(f"   - [WARNING] 槽位选择有限: 只有 {len(unique_slots)} 个不同槽位")
                else:
                    print(f"   - [PASS] 槽位选择丰富")
            else:
                print(f"   - [FAIL] 无候选动作")
        
        # 检查调度器行为
        print(f"\n   调度器行为分析:")
        for step in range(10):  # 检查前10步
            active_agents = env.scheduler.get_active_agents(step)
            execution_mode = env.scheduler.get_execution_mode(step)
            print(f"   - 步骤 {step}: {active_agents} ({execution_mode})")
        
        # 检查预算和成本
        print(f"\n   预算和成本分析:")
        for agent in agent_order:
            budget = env.budgets.get(agent, 0)
            print(f"   - {agent} 预算: {budget}")
            
            candidates = env.get_action_candidates(agent)
            if candidates:
                costs = [c.meta.get("cost", 0) for c in candidates]
                unique_costs = set(costs)
                print(f"   - {agent} 成本分布: {sorted(unique_costs)}")
                
                # 检查是否有可负担的动作
                affordable_actions = [c for c in candidates if c.meta.get("cost", 0) <= budget]
                print(f"   - {agent} 可负担动作: {len(affordable_actions)}/{len(candidates)}")
        
        # 检查槽位占用情况
        print(f"\n   槽位占用分析:")
        print(f"   - 已占用槽位: {len(env.occupied_slots)}")
        print(f"   - 总槽位数: {len(env.slots)}")
        print(f"   - 占用率: {len(env.occupied_slots) / len(env.slots) * 100:.1f}%")
        
        # 检查候选范围限制
        print(f"\n   候选范围限制分析:")
        hubs_config = env.config.get("hubs", {})
        if hubs_config.get("mode") == "explicit":
            print(f"   - 候选范围限制: 已启用")
            print(f"   - 模式: {hubs_config.get('candidate_mode')}")
            
            # 检查不同月份的候选范围
            for month in [0, 5, 10, 15, 20, 25, 29]:
                env.current_month = month
                candidates = env.get_action_candidates("IND")
                print(f"   - 月份 {month}: {len(candidates)} 个候选")
        else:
            print(f"   - 候选范围限制: 未启用")
        
        # 检查奖励机制
        print(f"\n   奖励机制分析:")
        reward_mechanisms = env.config.get("reward_mechanisms", {})
        enabled_rewards = [name for name, config in reward_mechanisms.items() 
                          if config.get("enabled", False)]
        print(f"   - 启用的奖励机制: {enabled_rewards}")
        
        # 检查探索策略
        print(f"\n   探索策略分析:")
        exploration_config = env.config.get("mappo", {}).get("exploration", {})
        temperature = exploration_config.get("temperature", 1.0)
        print(f"   - 探索温度: {temperature}")
        
        if temperature < 0.5:
            print(f"   - [WARNING] 探索温度过低，可能导致重复动作")
        else:
            print(f"   - [PASS] 探索温度正常")
        
    except Exception as e:
        print(f"   [FAIL] 调试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("调试完成!")
    print("=" * 80)


if __name__ == "__main__":
    debug_repetitive_actions()

