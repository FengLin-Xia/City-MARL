#!/usr/bin/env python3
"""
测试完整的奖励系统
"""

import sys
import json
import numpy as np

# 加载配置
with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

print("=== 测试完整奖励系统 ===")

from solvers.v4_1.rl_selector import RLPolicySelector
from envs.v4_1.city_env import CityEnvironment

# 初始化
selector = RLPolicySelector(cfg)
env = CityEnvironment(cfg)
state = env.reset(seed=42)

print(f"[INFO] 当前智能体: {state['current_agent']}")

# 获取动作池
actions, action_feats, mask = env.get_action_pool('EDU')
print(f"[INFO] EDU动作池大小: {len(actions)}")

if actions:
    # 使用RL选择器选择动作（这会计算得分）
    _, selected_action = selector.choose_action_sequence(
        slots=env.slots,
        candidates=set(actions[i].footprint_slots[0] for i in range(len(actions)) if actions[i].footprint_slots),
        occupied=env._get_occupied_slots(),
        lp_provider=env._create_lp_provider(),
        agent_types=['EDU'],
        sizes={'EDU': ['S', 'M', 'L']}
    )
    
    if selected_action:
        print(f"[INFO] 选择的动作: score={selected_action.score:.3f}, reward={selected_action.reward:.1f}, cost={selected_action.cost:.1f}")
        
        # 计算奖励
        reward = env._calculate_reward('EDU', selected_action)
        print(f"[INFO] 计算的奖励: {reward:.3f}")
        
        # 执行动作
        next_state, step_reward, done, info = env.step('EDU', selected_action)
        print(f"[INFO] 步骤奖励: {step_reward:.3f}")
        print(f"[INFO] 是否结束: {done}")
        print(f"[INFO] 信息: {info}")
        
        # 检查建筑状态
        print(f"[INFO] 公共建筑数量: {len(env.buildings['public'])}")
        print(f"[INFO] 工业建筑数量: {len(env.buildings['industrial'])}")

# 测试IND智能体
print(f"\n[INFO] 切换到IND智能体")
actions_ind, action_feats_ind, mask_ind = env.get_action_pool('IND')
print(f"[INFO] IND动作池大小: {len(actions_ind)}")

if actions_ind:
    _, selected_action_ind = selector.choose_action_sequence(
        slots=env.slots,
        candidates=set(actions_ind[i].footprint_slots[0] for i in range(len(actions_ind)) if actions_ind[i].footprint_slots),
        occupied=env._get_occupied_slots(),
        lp_provider=env._create_lp_provider(),
        agent_types=['IND'],
        sizes={'IND': ['S', 'M', 'L']}
    )
    
    if selected_action_ind:
        print(f"[INFO] IND选择的动作: score={selected_action_ind.score:.3f}, reward={selected_action_ind.reward:.1f}")
        
        # 计算奖励
        reward_ind = env._calculate_reward('IND', selected_action_ind)
        print(f"[INFO] IND计算的奖励: {reward_ind:.3f}")
        
        # 执行动作
        next_state, step_reward_ind, done, info = env.step('IND', selected_action_ind)
        print(f"[INFO] IND步骤奖励: {step_reward_ind:.3f}")

# 检查月度奖励
print(f"\n[INFO] 月度奖励统计:")
for agent, rewards in env.monthly_rewards.items():
    if rewards:
        print(f"  {agent}: 总奖励={sum(rewards):.3f}, 平均={np.mean(rewards):.3f}, 数量={len(rewards)}")

print("\n=== 完整奖励系统测试完成 ===")

