#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试IND动作score为0的问题
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List

# 确保导入路径正确
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from logic.v4_enumeration import ActionScorer, Action

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def debug_ind_score():
    """调试IND动作score为0的问题"""
    print("=== 调试IND动作Score为0的问题 ===")
    
    # 1. 创建ActionScorer
    config_path = 'configs/city_config_v4_1.json'
    cfg = load_config(config_path)
    
    enum_cfg = cfg.get('growth_v4_1', {}).get('enumeration', {})
    obj = enum_cfg.get('objective', {})
    objective = {
        'EDU': obj.get('EDU', {'w_r': 0.3, 'w_p': 0.6, 'w_c': 0.1}),
        'IND': obj.get('IND', {'w_r': 0.6, 'w_p': 0.2, 'w_c': 0.2}),
    }
    normalize = str(obj.get('normalize', 'per-month-pool-minmax'))
    eval_params = cfg.get('growth_v4_1', {}).get('evaluation', {})
    
    scorer = ActionScorer(objective, normalize, eval_params=eval_params)
    print(f"ActionScorer目标权重: {objective}")
    
    # 2. 创建测试动作（模拟实际数据）
    actions = [
        Action(agent='IND', size='S', footprint_slots=['s1'], zone='mid', LP_norm=0.5, adjacency={}, cost=0, reward=0, prestige=0),
        Action(agent='IND', size='S', footprint_slots=['s2'], zone='mid', LP_norm=0.5, adjacency={}, cost=0, reward=0, prestige=0),
        Action(agent='IND', size='S', footprint_slots=['s3'], zone='mid', LP_norm=0.5, adjacency={}, cost=0, reward=0, prestige=0),
    ]
    
    # 3. 模拟river_distance_provider
    def mock_river_distance_provider(slot_id: str) -> float:
        return 50.0
    
    # 4. 手动调用_calc_crp计算原始cost/reward/prestige
    print("\n=== 原始cost/reward/prestige计算 ===")
    for i, action in enumerate(actions):
        scorer._calc_crp(action, mock_river_distance_provider)
        print(f"Action {i} ({action.agent} {action.size}):")
        print(f"  cost: {action.cost}")
        print(f"  reward: {action.reward}")
        print(f"  prestige: {action.prestige}")
    
    # 5. 调用score_actions进行归一化和score计算
    print("\n=== 归一化和Score计算 ===")
    scored_actions = scorer.score_actions(actions, mock_river_distance_provider)
    
    # 6. 分析归一化过程
    costs = [a.cost for a in actions]
    rewards = [a.reward for a in actions]
    prestiges = [a.prestige for a in actions]
    
    c_min, c_max = min(costs), max(costs)
    r_min, r_max = min(rewards), max(rewards)
    p_min, p_max = min(prestiges), max(prestiges)
    
    print(f"归一化范围:")
    print(f"  cost: [{c_min:.2f}, {c_max:.2f}]")
    print(f"  reward: [{r_min:.2f}, {r_max:.2f}]")
    print(f"  prestige: [{p_min:.2f}, {p_max:.2f}]")
    
    def norm(v, lo, hi):
        if hi - lo <= 1e-9:
            return 0.0
        return (v - lo) / (hi - lo)
    
    # 7. 手动计算score并对比
    print(f"\n=== 手动计算Score对比 ===")
    for i, action in enumerate(scored_actions):
        w = objective.get(action.agent, {"w_r": 0.5, "w_p": 0.3, "w_c": 0.2})
        
        nr = norm(action.reward, r_min, r_max)
        np_ = norm(action.prestige, p_min, p_max)
        nc = norm(action.cost, c_min, c_max)
        
        manual_score = float(w.get('w_r', 0.5)) * nr + float(w.get('w_p', 0.3)) * np_ - float(w.get('w_c', 0.2)) * nc
        
        print(f"Action {i} ({action.agent} {action.size}):")
        print(f"  归一化值: nr={nr:.3f}, np_={np_:.3f}, nc={nc:.3f}")
        print(f"  权重: w_r={w.get('w_r', 0.5)}, w_p={w.get('w_p', 0.3)}, w_c={w.get('w_c', 0.2)}")
        print(f"  手动计算score: {manual_score:.3f}")
        print(f"  ActionScorer.score: {action.score:.3f}")
        print(f"  差异: {abs(manual_score - action.score):.6f}")
        
        # 分析为什么score很小
        if abs(action.score) < 0.01:
            print(f"  *** Score很小的原因分析 ***")
            print(f"    正向贡献: {w.get('w_r', 0.5) * nr + w.get('w_p', 0.3) * np_:.3f}")
            print(f"    负向贡献: -{w.get('w_c', 0.2) * nc:.3f}")
            print(f"    净贡献: {manual_score:.3f}")

if __name__ == "__main__":
    debug_ind_score()





