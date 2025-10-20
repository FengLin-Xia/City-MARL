#!/usr/bin/env python3
"""
调试奖励计算问题
分析奖励方向、重复计入、缩放问题
"""

import json
import csv
import sys
import os
from typing import Dict, List
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v4_1.city_env import CityEnvironment
from solvers.v4_1.rl_selector import RLPolicySelector


def debug_reward_calculation():
    """调试奖励计算"""
    print("=== 调试奖励计算 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    selector = RLPolicySelector(cfg)
    
    # 重置环境
    state = env.reset(seed=42)
    
    # 创建CSV文件记录奖励详情
    csv_file = 'debug_reward_analysis.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'step', 'month', 'agent', 'action_type', 'action_size',
            'action_cost', 'action_reward', 'action_prestige', 'action_score',
            'base_reward', 'quality_reward', 'progress_reward', 'cooperation_bonus',
            'total_reward', 'slot_id', 'slot_x', 'slot_y'
        ])
        
        # 运行几步进行调试
        for step in range(5):
            current_agent = env.current_agent
            current_month = env.current_month
            
            print(f"\n--- Step {step}: Month {current_month}, Agent {current_agent} ---")
            
            # 获取动作池
            actions, action_feats, mask = env.get_action_pool(current_agent)
            
            if not actions:
                print("   No available actions")
                break
            
            print(f"   Available actions: {len(actions)}")
            
            # 选择第一个动作进行调试
            if actions:
                selected_action = actions[0]
                
                print(f"   Selected action:")
                print(f"     Agent: {selected_action.agent}")
                print(f"     Size: {selected_action.size}")
                print(f"     Cost: {selected_action.cost}")
                print(f"     Reward: {selected_action.reward}")
                print(f"     Prestige: {selected_action.prestige}")
                print(f"     Score: {selected_action.score}")
                print(f"     Footprint slots: {selected_action.footprint_slots}")
                
                # 计算奖励
                total_reward = env._calculate_reward(current_agent, selected_action)
                
                print(f"   Calculated reward: {total_reward}")
                
                # 获取槽位位置信息
                slot_id = selected_action.footprint_slots[0] if selected_action.footprint_slots else ''
                slot_x, slot_y = 0.0, 0.0
                if slot_id in env.slots:
                    slot = env.slots[slot_id]
                    slot_x = float(getattr(slot, 'fx', getattr(slot, 'x', 0.0)))
                    slot_y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
                
                # 记录到CSV
                writer.writerow([
                    step, current_month, current_agent,
                    selected_action.agent, selected_action.size,
                    selected_action.cost, selected_action.reward, 
                    selected_action.prestige, selected_action.score,
                    selected_action.score,  # base_reward
                    0.0,  # quality_reward (需要单独计算)
                    0.0,  # progress_reward (需要单独计算)
                    0.0,  # cooperation_bonus (需要单独计算)
                    total_reward,
                    slot_id, slot_x, slot_y
                ])
                
                # 执行动作
                from logic.v4_enumeration import Sequence
                sequence = Sequence(
                    actions=[selected_action],
                    sum_cost=selected_action.cost,
                    sum_reward=selected_action.reward,
                    sum_prestige=selected_action.prestige,
                    score=selected_action.score
                )
                
                next_state, reward, done, info = env.step(current_agent, sequence)
                state = next_state
                
                if done:
                    break
    
    print(f"\n奖励分析已保存到: {csv_file}")
    
    # 分析ActionScorer的score计算
    print("\n=== 分析ActionScorer的Score计算 ===")
    if actions:
        sample_action = actions[0]
        print(f"Sample action score calculation:")
        print(f"  Cost: {sample_action.cost}")
        print(f"  Reward: {sample_action.reward}")
        print(f"  Prestige: {sample_action.prestige}")
        print(f"  Score: {sample_action.score}")
        print(f"  Expected score (reward - cost): {sample_action.reward - sample_action.cost}")
        
        # 检查是否有重复计入问题
        if abs(sample_action.score - (sample_action.reward - sample_action.cost)) > 0.1:
            print("  WARNING: Score calculation may be incorrect!")


def analyze_existing_results():
    """分析现有训练结果"""
    print("\n=== 分析现有训练结果 ===")
    
    history_path = 'models/v4_1_rl/slot_selection_history.json'
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return
    
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    episodes = history.get('episodes', [])
    if not episodes:
        print("No episodes found in history")
        return
    
    print(f"Found {len(episodes)} episodes")
    
    # 分析第一个episode的奖励分布
    if episodes:
        first_episode = episodes[0]
        steps = first_episode.get('steps', [])
        
        print(f"\nFirst episode analysis (ID: {first_episode.get('episode_id', 0)}):")
        print(f"  Total return: {first_episode.get('episode_return', 0):.2f}")
        print(f"  Number of steps: {len(steps)}")
        
        # 分析每步的奖励
        step_rewards = []
        action_scores = []
        
        for step in steps:
            reward = step.get('reward', 0)
            sequence_score = step.get('sequence_score', 0)
            step_rewards.append(reward)
            action_scores.append(sequence_score)
        
        if step_rewards:
            print(f"  Step rewards: min={min(step_rewards):.2f}, max={max(step_rewards):.2f}, avg={np.mean(step_rewards):.2f}")
            print(f"  Action scores: min={min(action_scores):.2f}, max={max(action_scores):.2f}, avg={np.mean(action_scores):.2f}")
            
            # 检查是否有异常值
            if min(step_rewards) < -1000:
                print("  WARNING: Extremely negative rewards detected!")
            if min(action_scores) < -1000:
                print("  WARNING: Extremely negative action scores detected!")


if __name__ == "__main__":
    debug_reward_calculation()
    analyze_existing_results()





