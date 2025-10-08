#!/usr/bin/env python3
"""
测试奖励系统
"""

import sys
import json
import numpy as np

# 加载配置
with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

print("=== 测试奖励系统 ===")

# 测试RL选择器
from solvers.v4_1.rl_selector import RLPolicySelector
selector = RLPolicySelector(cfg)
print("[OK] RL选择器初始化成功")

# 测试环境
from envs.v4_1.city_env import CityEnvironment
env = CityEnvironment(cfg)
print("[OK] 环境初始化成功")

# 重置环境
state = env.reset(seed=42)
print(f"[INFO] 环境重置成功，当前智能体: {state['current_agent']}")

# 获取动作池
actions, action_feats, mask = env.get_action_pool('EDU')
print(f"[INFO] EDU动作池大小: {len(actions)}")

if actions:
    action = actions[0]
    print(f"[INFO] 第一个动作: agent={action.agent}, size={action.size}")
    print(f"[INFO] 动作得分: score={action.score}, reward={action.reward}, cost={action.cost}, prestige={action.prestige}")
    
    # 测试奖励计算
    reward = env._calculate_reward('EDU', action)
    print(f"[INFO] 计算的奖励: {reward}")
    
    # 测试协作奖励
    coop_reward = env._calculate_cooperation_reward('EDU', action)
    print(f"[INFO] 协作奖励: {coop_reward}")
    
else:
    print("[WARNING] 没有可用动作")

# 测试IND智能体
actions_ind, action_feats_ind, mask_ind = env.get_action_pool('IND')
print(f"[INFO] IND动作池大小: {len(actions_ind)}")

if actions_ind:
    action_ind = actions_ind[0]
    print(f"[INFO] IND第一个动作: score={action_ind.score}, reward={action_ind.reward}")
    
    # 测试奖励计算
    reward_ind = env._calculate_reward('IND', action_ind)
    print(f"[INFO] IND计算的奖励: {reward_ind}")

print("\n=== 奖励系统测试完成 ===")

