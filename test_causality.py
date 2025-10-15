#!/usr/bin/env python3
"""
因果性测试 - 检查动作是否真的影响回报
"""

import torch
import numpy as np
from envs.v4_1.city_env import CityEnvironment
from solvers.v4_1.rl_selector import RLPolicySelector
import json

def test_action_causality():
    """测试同一状态下不同动作的回报差异"""
    print("开始因果性测试...")
    
    # 加载配置
    with open("configs/city_config_v4_1.json", 'r') as f:
        cfg = json.load(f)
    
    # 创建环境和选择器
    env = CityEnvironment(cfg)
    selector = RLPolicySelector(cfg)
    
    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 重置环境获取初始状态
    state = env.reset()
    current_agent = state['current_agent']
    
    # 获取可用动作（使用selector的内部方法）
    from logic.v4_enumeration import ActionEnumerator
    
    # 创建动作枚举器
    enumerator = ActionEnumerator(env.slots)
    
    # 创建地价提供者包装函数
    def lp_provider(slot_id):
        # 从slot_id提取坐标 (假设格式为 "x_y")
        try:
            x, y = map(int, slot_id.split('_'))
            return env.land_price_system.get_land_price([x, y])
        except:
            return 1.0  # 默认地价
    
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
    # 检查buildings结构并修复
    buildings = state['buildings']
    print(f"Buildings type: {type(buildings)}")
    print(f"Buildings content: {buildings}")
    
    # 如果buildings是字符串列表，转换为正确的格式
    if isinstance(buildings, dict):
        buildings_list = []
        for category, b_list in buildings.items():
            if isinstance(b_list, list):
                buildings_list.extend(b_list)
        buildings = buildings_list
    elif isinstance(buildings, list):
        # 检查列表中的元素类型
        if buildings and isinstance(buildings[0], str):
            # 如果buildings是字符串列表，创建一个简单的建筑列表
            buildings = []
    
    actions = scorer.score_actions(actions, river_distance_provider=None, buildings=buildings)
    
    if len(actions) < 5:
        print(f"可用动作太少: {len(actions)}")
        return
    
    # 选择前5个动作进行测试
    test_actions = actions[:5]
    returns = []
    
    for i, action in enumerate(test_actions):
        print(f"测试动作 {i+1}: {action}")
        
        # 重置环境到相同状态
        env.reset()
        
        # 创建序列（单个动作）
        from logic.v4_enumeration import Sequence
        sequence = Sequence(
            actions=[action], 
            sum_cost=action.cost, 
            sum_reward=action.reward, 
            sum_prestige=action.prestige, 
            score=action.score
        )
        
        # 执行单个动作序列
        # 确保使用正确的当前智能体
        actual_agent = env.current_agent
        next_state, reward, done, info = env.step(actual_agent, sequence)
        total_reward = reward
        
        # 运行几步看回报（使用随机策略）
        for _ in range(5):  # 运行5步
            if done:
                break
            
            # 获取当前智能体的随机动作
            current_agent = next_state['current_agent']
            random_actions = enumerator.enumerate_actions(
                candidates=next_state['candidate_slots'],
                occupied=next_state['occupied_slots'],
                agent_types=env.rl_cfg['agents'],
                sizes={'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L']},
                lp_provider=lp_provider
            )
            
            if random_actions:
                random_action = random_actions[np.random.randint(len(random_actions))]
                random_sequence = Sequence(
                    actions=[random_action], 
                    sum_cost=random_action.cost, 
                    sum_reward=random_action.reward, 
                    sum_prestige=random_action.prestige, 
                    score=random_action.score
                )
                # 确保使用正确的当前智能体
                actual_agent = env.current_agent
                next_state, step_reward, done, info = env.step(actual_agent, random_sequence)
                total_reward += step_reward
            else:
                break
        
        returns.append(total_reward)
        print(f"  总回报: {total_reward:.3f}")
    
    # 计算回报差异
    min_return = min(returns)
    max_return = max(returns)
    delta_return = max_return - min_return
    
    print(f"\n因果性测试结果:")
    print(f"  回报范围: [{min_return:.3f}, {max_return:.3f}]")
    print(f"  差异 Δreturn: {delta_return:.3f}")
    
    if delta_return < 0.01:
        print("动作对回报影响很小，可能需要shaping奖励")
        print("建议: 添加即时shaping奖励 α*(score - λ*cost + β*rent_gain)")
    else:
        print("动作对回报有明显影响，问题可能在梯度传播")
    
    return delta_return

if __name__ == "__main__":
    test_action_causality()
