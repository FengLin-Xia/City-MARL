#!/usr/bin/env python3
"""
调试归一化过程
"""

import sys
import json
import numpy as np

# 加载配置
with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

print("=== 调试归一化过程 ===")

from logic.v4_enumeration import ActionEnumerator, ActionScorer
from envs.v4_1.city_env import CityEnvironment

env = CityEnvironment(cfg)
state = env.reset(seed=42)

slots = env.slots
candidates = env._get_candidate_slots()
occupied = env._get_occupied_slots()
lp_provider = env._create_lp_provider()

# 枚举动作
enumerator = ActionEnumerator(slots)
actions = enumerator.enumerate_actions(
    candidates=candidates,
    occupied=occupied,
    agent_types=['EDU'],
    sizes={'EDU': ['S', 'M', 'L']},
    lp_provider=lp_provider,
    adjacency='4-neighbor',
    caps=cfg.get('growth_v4_1', {}).get('enumeration', {}).get('caps', {}),
)

print(f"[INFO] 动作数量: {len(actions)}")

if actions:
    # 创建ActionScorer
    enum_cfg = cfg.get('growth_v4_1', {}).get('enumeration', {})
    obj = enum_cfg.get('objective', {})
    objective = {
        'EDU': obj.get('EDU', {'w_r': 0.3, 'w_p': 0.6, 'w_c': 0.1}),
        'IND': obj.get('IND', {'w_r': 0.6, 'w_p': 0.2, 'w_c': 0.2}),
    }
    normalize = str(obj.get('normalize', 'per-month-pool-minmax'))
    eval_params = cfg.get('growth_v4_1', {}).get('evaluation', {})
    
    scorer = ActionScorer(objective, normalize, eval_params=eval_params)
    
    # 手动执行score_actions的步骤
    print("\n=== 步骤1: 计算原始cost/reward/prestige ===")
    for i, a in enumerate(actions):
        scorer._calc_crp(a)
        print(f"动作{i}: reward={a.reward}, cost={a.cost}, prestige={a.prestige}")
    
    print("\n=== 步骤2: 归一化 ===")
    costs = [a.cost for a in actions]
    rewards = [a.reward for a in actions]
    prestiges = [a.prestige for a in actions]
    
    c_min, c_max = (min(costs) if costs else 0.0), (max(costs) if costs else 1.0)
    r_min, r_max = (min(rewards) if rewards else 0.0), (max(rewards) if rewards else 1.0)
    p_min, p_max = (min(prestiges) if prestiges else 0.0), (max(prestiges) if prestiges else 1.0)
    
    print(f"cost范围: [{c_min}, {c_max}]")
    print(f"reward范围: [{r_min}, {r_max}]")
    print(f"prestige范围: [{p_min}, {p_max}]")
    
    def norm(v, lo, hi):
        if hi - lo <= 1e-9:
            return 0.0
        return (v - lo) / (hi - lo)
    
    print("\n=== 步骤3: 计算得分 ===")
    for i, a in enumerate(actions):
        w = objective.get(a.agent, {"w_r": 0.5, "w_p": 0.3, "w_c": 0.2})
        nr = norm(a.reward, r_min, r_max)
        np_ = norm(a.prestige, p_min, p_max)
        nc = norm(a.cost, c_min, c_max)
        
        print(f"动作{i}权重: {w}")
        print(f"动作{i}归一化: nr={nr:.3f}, np={np_:.3f}, nc={nc:.3f}")
        
        score = float(w.get('w_r', 0.5)) * nr + float(w.get('w_p', 0.3)) * np_ - float(w.get('w_c', 0.2)) * nc
        print(f"动作{i}最终得分: {score:.3f}")
        a.score = score

print("\n=== 归一化调试完成 ===")

