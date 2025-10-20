#!/usr/bin/env python3
"""
测试不同类型的动作奖励差异
"""

import torch
import numpy as np
from envs.v4_1.city_env import CityEnvironment
from solvers.v4_1.rl_selector import RLPolicySelector
import json

def test_different_action_types():
    """测试不同类型动作的奖励差异"""
    print("开始测试不同动作类型的奖励差异...")
    
    # 加载配置
    with open("configs/city_config_v4_1.json", 'r') as f:
        cfg = json.load(f)
    
    # 创建环境和选择器
    env = CityEnvironment(cfg)
    selector = RLPolicySelector(cfg)
    
    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 重置环境
    state = env.reset()
    current_agent = state['current_agent']
    
    # 获取可用动作
    from logic.v4_enumeration import ActionEnumerator
    enumerator = ActionEnumerator(env.slots)
    
    def lp_provider(slot_id):
        try:
            x, y = map(int, slot_id.split('_'))
            return env.land_price_system.get_land_price([x, y])
        except:
            return 1.0
    
    actions = enumerator.enumerate_actions(
        candidates=state['candidate_slots'],
        occupied=state['occupied_slots'],
        agent_types=env.rl_cfg['agents'],
        sizes={'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L']},
        lp_provider=lp_provider
    )
    
    # 使用ActionScorer计算cost和reward
    from logic.v4_enumeration import ActionScorer
    enum_cfg = cfg.get('growth_v4_1', {}).get('enumeration', {})
    obj = enum_cfg.get('objective', {})
    objective = {
        'EDU': obj.get('EDU', {'w_r': 0.3, 'w_p': 0.6, 'w_c': 0.1}),
        'IND': obj.get('IND', {'w_r': 0.6, 'w_p': 0.2, 'w_c': 0.2}),
    }
    normalize = str(obj.get('normalize', 'per-month-pool-minmax'))
    eval_params = cfg.get('growth_v4_1', {}).get('evaluation', {})
    
    scorer = ActionScorer(objective, normalize, eval_params=eval_params, slots=env.slots)
    buildings = state['buildings']
    if isinstance(buildings, dict):
        buildings_list = []
        for category, b_list in buildings.items():
            if isinstance(b_list, list):
                buildings_list.extend(b_list)
        buildings = buildings_list
    
    actions = scorer.score_actions(actions, river_distance_provider=None, buildings=buildings)
    
    # 按类型分组测试
    action_types = {}
    for action in actions:
        key = f"{action.agent}_{action.size}"
        if key not in action_types:
            action_types[key] = []
        action_types[key].append(action)
    
    print(f"发现的动作类型: {list(action_types.keys())}")
    
    rewards_by_type = {}
    
    for action_type, action_list in action_types.items():
        if len(action_list) > 0:
            # 测试第一个动作
            action = action_list[0]
            print(f"\n=== 测试动作类型: {action_type} ===")
            print(f"Action: cost={action.cost:.1f}, reward={action.reward:.1f}, NPV={action.reward * 12 - action.cost:.1f}")
            
            # 重置环境
            env.reset()
            
            # 创建序列
            from logic.v4_enumeration import Sequence
            sequence = Sequence(
                actions=[action], 
                sum_cost=action.cost, 
                sum_reward=action.reward, 
                sum_prestige=action.prestige, 
                score=action.score
            )
            
            # 执行动作
            actual_agent = env.current_agent
            next_state, reward, done, info = env.step(actual_agent, sequence)
            
            rewards_by_type[action_type] = reward
            print(f"环境返回奖励: {reward:.6f}")
    
    # 计算奖励差异
    if len(rewards_by_type) > 1:
        reward_values = list(rewards_by_type.values())
        min_reward = min(reward_values)
        max_reward = max(reward_values)
        delta_reward = max_reward - min_reward
        
        print(f"\n=== 奖励差异分析 ===")
        print(f"奖励范围: [{min_reward:.6f}, {max_reward:.6f}]")
        print(f"奖励差异: {delta_reward:.6f}")
        
        if delta_reward > 0.01:
            print("✅ 不同类型动作有显著奖励差异")
        else:
            print("❌ 不同类型动作奖励差异太小")

if __name__ == "__main__":
    test_different_action_types()

