#!/usr/bin/env python3
"""
测试智能体成功率
"""

import numpy as np
from envs.terrain_grid_nav_env import TerrainGridNavEnv
from agents.ppo_terrain_agent import TerrainPPOAgent

def test_success_rate(num_episodes=100):
    """测试智能体成功率"""
    print(f"测试 {num_episodes} 个episodes的成功率...")
    
    # 创建环境
    env = TerrainGridNavEnv(
        H=20, W=20,
        max_steps=120,
        height_range=(0.0, 10.0),
        slope_penalty_weight=0.2,
        height_penalty_weight=0.15
    )
    
    # 创建智能体
    agent = TerrainPPOAgent(
        state_dim=13,
        action_dim=4,
        hidden_dim=256,
        lr=2e-4
    )
    
    # 测试统计
    success_count = 0
    total_rewards = []
    path_lengths = []
    
    for episode in range(num_episodes):
        # 运行一个episode
        result = agent.test_episode(env, render=False)
        
        if result['success']:
            success_count += 1
        
        total_rewards.append(result['total_reward'])
        path_lengths.append(result['path_length'])
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            current_success_rate = success_count / (episode + 1)
            print(f"Episode {episode + 1:3d}: 成功率 = {current_success_rate:.1%} ({success_count}/{episode + 1})")
    
    # 最终统计
    final_success_rate = success_count / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_path_length = np.mean(path_lengths)
    
    print("\n" + "=" * 50)
    print("测试结果:")
    print(f"总episodes: {num_episodes}")
    print(f"成功episodes: {success_count}")
    print(f"失败episodes: {num_episodes - success_count}")
    print(f"最终成功率: {final_success_rate:.1%}")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均路径长度: {avg_path_length:.1f}")
    
    # 分析失败原因
    if final_success_rate < 0.8:
        print("\n可能的问题:")
        print("1. 地形惩罚太强，智能体难以找到好路径")
        print("2. 步数限制太短，无法到达目标")
        print("3. 智能体学习不足，需要更多训练")
        print("4. 地形配置过于困难")
    
    return final_success_rate, avg_reward

if __name__ == "__main__":
    # 测试未训练的智能体（随机策略）
    print("测试未训练的智能体（随机策略）:")
    test_success_rate(50)
    
    print("\n" + "=" * 50)
    print("注意：这是未训练智能体的表现，成功率应该很低")
    print("训练后的智能体应该会有更高的成功率")

