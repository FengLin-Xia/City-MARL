#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solvers.v4_1.rl_selector import RLPolicySelector
def test_modifications():
    """测试我们的修改是否正常工作"""
    
    # 使用简单的配置
    cfg = {
        'solver': {
            'rl': {
                'temperature': 2.0,
                'epsilon': 0.4
            }
        },
        'growth_v4_1': {
            'enumeration': {
                'length_max': 5,
                'beam_width': 16,
                'max_expansions': 2000
            }
        }
    }
    
    # 创建selector
    selector = RLPolicySelector(cfg)
    
    print("=== 测试修改效果 ===")
    print(f"初始探索率: {selector.epsilon}")
    
    # 检查温度参数
    for agent, actor in selector.actors.items():
        print(f"{agent} Actor 温度: {actor.temperature}")
    
    # 测试探索衰减
    print("\n测试探索衰减:")
    for i in range(5):
        selector.update_exploration(i)
        print(f"Episode {i}: 探索率={selector.epsilon:.3f}")
        for agent, actor in selector.actors.items():
            print(f"  {agent} 温度: {actor.temperature:.3f}")
    
    # 测试S型建筑限制
    print("\n测试S型建筑限制:")
    from logic.v4_enumeration import Action
    test_actions = [
        Action(agent="IND", size="S", footprint_slots=["s_1"], cost=100, reward=50, prestige=0.1, zone="IND", LP_norm=1.0, adjacency=0),
        Action(agent="IND", size="S", footprint_slots=["s_2"], cost=100, reward=50, prestige=0.1, zone="IND", LP_norm=1.0, adjacency=0),
        Action(agent="IND", size="S", footprint_slots=["s_3"], cost=100, reward=50, prestige=0.1, zone="IND", LP_norm=1.0, adjacency=0),
        Action(agent="IND", size="M", footprint_slots=["s_4"], cost=200, reward=120, prestige=0.2, zone="IND", LP_norm=1.0, adjacency=0),
        Action(agent="IND", size="L", footprint_slots=["s_5"], cost=300, reward=200, prestige=0.3, zone="IND", LP_norm=1.0, adjacency=0),
    ]
    
    limited_actions = selector._limit_s_size_actions(test_actions, max_s_ratio=0.4)
    print(f"原始动作数: {len(test_actions)}")
    print(f"限制后动作数: {len(limited_actions)}")
    
    size_counts = {'S': 0, 'M': 0, 'L': 0}
    for action in limited_actions:
        size_counts[action.size] += 1
    
    for size, count in size_counts.items():
        print(f"{size}: {count}")
    
    # 测试探索奖励
    print("\n测试探索奖励:")
    print(f"当前探索率: {selector.epsilon:.3f}")
    
    # 先记录原始score
    original_scores = {}
    for action in limited_actions:
        original_scores[action.size] = action.score
    
    bonus_actions = selector._add_exploration_bonus(limited_actions)
    for action in bonus_actions:
        original_score = original_scores.get(action.size, 0)
        print(f"{action.size}型建筑: 原始score={original_score:.3f}, 奖励后score={action.score:.3f}")

if __name__ == "__main__":
    test_modifications()
