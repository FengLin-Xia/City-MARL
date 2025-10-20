#!/usr/bin/env python3
"""
测试PPO训练过程中的奖励处理
"""

import torch
import numpy as np
from envs.v4_1.city_env import CityEnvironment
from solvers.v4_1.rl_selector import RLPolicySelector
import json

def test_ppo_reward_flow():
    """测试PPO训练过程中的奖励流"""
    print("开始PPO奖励流测试...")
    
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
    
    print(f"当前智能体: {current_agent}")
    
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
    
    print(f"可用动作数量: {len(actions)}")
    
    # 选择前3个动作进行测试
    test_actions = actions[:3]
    
    for i, action in enumerate(test_actions):
        print(f"\n=== 测试动作 {i+1} ===")
        print(f"Action: {action}")
        
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
        
        # 执行动作并获取奖励
        actual_agent = env.current_agent
        next_state, reward, done, info = env.step(actual_agent, sequence)
        
        print(f"环境返回的即时奖励: {reward:.6f}")
        print(f"动作的cost: {action.cost:.1f}")
        print(f"动作的reward: {action.reward:.1f}")
        print(f"NPV计算: {action.reward * 12 - action.cost:.1f}")
        
        # 检查episode_history中的奖励记录
        if hasattr(env, 'episode_history') and env.episode_history:
            last_record = env.episode_history[-1]
            print(f"Episode历史中的奖励: {last_record.get('reward', 'N/A'):.6f}")

if __name__ == "__main__":
    test_ppo_reward_flow()

