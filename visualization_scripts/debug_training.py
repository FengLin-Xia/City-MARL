#!/usr/bin/env python3
"""
调试版训练脚本 - 诊断成功率问题
"""

import numpy as np
import json
import time
import os
import torch
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from envs.terrain_grid_nav_env import TerrainGridNavEnv
from agents.ppo_terrain_agent import TerrainPPOAgent


def debug_episode(env, agent, episode_num=0):
    """调试单个episode"""
    print(f"\n=== 调试Episode {episode_num} ===")
    
    obs, _ = env.reset()
    print(f"起点: {env.pos}, 终点: {env.goal}")
    print(f"起点高度: {env.terrain[env.pos[0], env.pos[1]]:.2f}")
    print(f"终点高度: {env.terrain[env.goal[0], env.goal[1]]:.2f}")
    
    step_count = 0
    total_reward = 0
    path = [env.pos]
    
    while step_count < env.max_steps:
        # 获取动作掩膜
        action_mask = obs['action_mask']
        print(f"步骤 {step_count}: 位置 {env.pos}, 动作掩膜: {action_mask}")
        
        # 检查是否有合法动作
        if sum(action_mask) == 0:
            print("警告：没有合法动作！")
            break
        
        # 获取智能体动作
        state_features = agent.get_state_features(obs)
        terrain_features = agent.get_terrain_features(obs)
        
        with torch.no_grad():
            action_logits, value = agent.network(state_features, terrain_features)
            masked_logits = action_logits - (1 - torch.FloatTensor(action_mask).to(agent.device)) * 1e8
            action_probs = torch.softmax(masked_logits, dim=-1)
            action = torch.argmax(action_probs).item()
        
        print(f"选择的动作: {action} (上右下左)")
        
        # 执行动作
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        path.append(env.pos)
        
        print(f"奖励: {reward:.3f}, 总奖励: {total_reward:.3f}")
        if 'height_change' in info:
            print(f"高度变化: {info['height_change']:.2f}, 坡度: {info['slope']:.2f}")
        
        if done:
            print("到达目标！")
            break
        elif truncated:
            print("达到最大步数")
            break
        
        step_count += 1
    
    print(f"Episode结束: 步数={step_count}, 总奖励={total_reward:.3f}, 成功={done}")
    print(f"路径长度: {len(path)}")
    return done, total_reward, path


def main():
    """主函数"""
    
    # 使用最新的地形数据
    terrain_file = "data/terrain/terrain_1755281528.json"
    
    # 加载地形数据
    with open(terrain_file, 'r') as f:
        terrain_data = json.load(f)
    height_map = np.array(terrain_data['height_map'], dtype=np.float32)
    
    print(f"地形尺寸: {height_map.shape}")
    print(f"高程范围: {height_map.min():.2f} ~ {height_map.max():.2f}")
    
    # 选择起始点
    start_point = (20, 20)
    goal_point = (110, 110)
    
    print(f"起点: {start_point}, 终点: {goal_point}")
    print(f"起点高度: {height_map[start_point[0], start_point[1]]:.1f}")
    print(f"终点高度: {height_map[goal_point[0], goal_point[1]]:.1f}")
    
    # 创建环境（降低惩罚权重）
    env = TerrainGridNavEnv(
        H=height_map.shape[0], W=height_map.shape[1],
        max_steps=200,
        height_range=(terrain_data['original_bounds']['z_min'], 
                     terrain_data['original_bounds']['z_max']),
        slope_penalty_weight=0.05,  # 降低坡度惩罚
        height_penalty_weight=0.02,  # 降低高度惩罚
        custom_terrain=height_map,
        fixed_start=start_point,
        fixed_goal=goal_point
    )
    
    # 创建智能体
    agent = TerrainPPOAgent(
        state_dim=13,
        action_dim=4,
        hidden_dim=256,
        lr=2e-4
    )
    
    # 测试未训练的智能体
    print("\n=== 测试未训练的智能体 ===")
    success_count = 0
    total_rewards = []
    
    for i in range(10):
        success, reward, path = debug_episode(env, agent, i)
        if success:
            success_count += 1
        total_rewards.append(reward)
    
    print(f"\n未训练智能体结果:")
    print(f"成功率: {success_count/10:.1%}")
    print(f"平均奖励: {np.mean(total_rewards):.3f}")
    
    # 简单训练几个episode
    print("\n=== 简单训练 ===")
    for episode in range(5):
        states, actions, rewards, values, log_probs, dones, path, success = \
            agent.collect_episode(env)
        
        print(f"训练Episode {episode}: 成功={success}, 总奖励={sum(rewards):.3f}")
        
        # 更新网络
        agent.update(states, actions, rewards, values, log_probs, dones)
    
    # 测试训练后的智能体
    print("\n=== 测试训练后的智能体 ===")
    success_count = 0
    total_rewards = []
    
    for i in range(10):
        success, reward, path = debug_episode(env, agent, i)
        if success:
            success_count += 1
        total_rewards.append(reward)
    
    print(f"\n训练后智能体结果:")
    print(f"成功率: {success_count/10:.1%}")
    print(f"平均奖励: {np.mean(total_rewards):.3f}")


if __name__ == "__main__":
    main()
