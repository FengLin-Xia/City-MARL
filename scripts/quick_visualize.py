#!/usr/bin/env python3
"""
快速可视化脚本 - 直接显示智能体行为
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.simple_road_env import SimpleRoadEnv
from agents.simple_ppo import SimpleActorCritic


def visualize_agent_behavior():
    """可视化智能体行为"""
    print("🎨 可视化智能体行为")
    
    # 创建环境
    env = SimpleRoadEnv()
    
    # 创建智能体
    actor_critic = SimpleActorCritic()
    
    # 设置matplotlib
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('智能体行为可视化', fontsize=16, fontweight='bold')
    
    # 左图：训练后的智能体
    print("🤖 运行训练后的智能体...")
    obs, _ = env.reset()
    
    path_x = [obs['position'][0]]
    path_y = [obs['position'][1]]
    total_reward = 0
    success = False
    
    for step in range(200):
        # 获取动作
        obs_tensor = {
            'position': torch.FloatTensor(obs['position']).unsqueeze(0),
            'goal': torch.FloatTensor(obs['goal']).unsqueeze(0),
            'local_dem': torch.FloatTensor(obs['local_dem']).unsqueeze(0)
        }
        
        action, _, _ = actor_critic.get_action(obs_tensor)
        
        # 执行动作
        next_obs, reward, done, truncated, info = env.step(action.numpy().squeeze())
        
        # 更新路径
        path_x.append(next_obs['position'][0])
        path_y.append(next_obs['position'][1])
        total_reward += reward
        
        if done and info.get('reason') == 'reached_goal':
            success = True
            break
        
        if done:
            break
        
        obs = next_obs
    
    # 绘制训练后的智能体路径
    ax1.imshow(env.dem, cmap='terrain', origin='lower', alpha=0.7)
    
    # 绘制理想路径
    ideal_x = [env.start_pos[0], env.goal_pos[0]]
    ideal_y = [env.start_pos[1], env.goal_pos[1]]
    ax1.plot(ideal_x, ideal_y, 'r--', linewidth=2, alpha=0.6, label='理想路径')
    
    # 绘制智能体路径
    ax1.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8, label='智能体路径')
    ax1.plot(path_x[0], path_y[0], 'go', markersize=15, label='起点', markeredgecolor='black', markeredgewidth=2)
    ax1.plot(env.goal_pos[0], env.goal_pos[1], 'ro', markersize=15, label='终点', markeredgecolor='black', markeredgewidth=2)
    ax1.plot(path_x[-1], path_y[-1], 'bo', markersize=10, label='当前位置')
    
    ax1.set_title(f'训练后的智能体\n奖励: {total_reward:.1f}, 成功: {"✅" if success else "❌"}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X坐标', fontsize=12)
    ax1.set_ylabel('Y坐标', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 右图：随机策略
    print("🎲 运行随机策略...")
    obs, _ = env.reset()
    
    random_path_x = [obs['position'][0]]
    random_path_y = [obs['position'][1]]
    random_total_reward = 0
    random_success = False
    
    for step in range(200):
        # 随机动作
        action = env.action_space.sample()
        
        # 执行动作
        next_obs, reward, done, truncated, info = env.step(action)
        
        # 更新路径
        random_path_x.append(next_obs['position'][0])
        random_path_y.append(next_obs['position'][1])
        random_total_reward += reward
        
        if done and info.get('reason') == 'reached_goal':
            random_success = True
            break
        
        if done:
            break
        
        obs = next_obs
    
    # 绘制随机策略路径
    ax2.imshow(env.dem, cmap='terrain', origin='lower', alpha=0.7)
    
    # 绘制理想路径
    ax2.plot(ideal_x, ideal_y, 'r--', linewidth=2, alpha=0.6, label='理想路径')
    
    # 绘制随机路径
    ax2.plot(random_path_x, random_path_y, 'orange', linewidth=3, alpha=0.8, label='随机路径')
    ax2.plot(random_path_x[0], random_path_y[0], 'go', markersize=15, label='起点', markeredgecolor='black', markeredgewidth=2)
    ax2.plot(env.goal_pos[0], env.goal_pos[1], 'ro', markersize=15, label='终点', markeredgecolor='black', markeredgewidth=2)
    ax2.plot(random_path_x[-1], random_path_y[-1], 'orange', marker='o', markersize=10, label='当前位置')
    
    ax2.set_title(f'随机策略\n奖励: {random_total_reward:.1f}, 成功: {"✅" if random_success else "❌"}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X坐标', fontsize=12)
    ax2.set_ylabel('Y坐标', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 结果对比:")
    print(f"   训练后智能体: 奖励={total_reward:.1f}, 成功={success}")
    print(f"   随机策略: 奖励={random_total_reward:.1f}, 成功={random_success}")


if __name__ == "__main__":
    visualize_agent_behavior()


