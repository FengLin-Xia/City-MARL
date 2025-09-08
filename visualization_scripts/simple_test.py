#!/usr/bin/env python3
"""
简单测试脚本
"""

import numpy as np
import json
import torch
from envs.terrain_grid_nav_env import TerrainGridNavEnv
from agents.ppo_terrain_agent import TerrainPPOAgent

def main():
    print("开始简单测试...")
    
    # 检查地形文件是否存在
    terrain_file = "data/terrain/terrain_1755281528.json"
    try:
        with open(terrain_file, 'r') as f:
            terrain_data = json.load(f)
        print(f"地形文件加载成功")
    except Exception as e:
        print(f"地形文件加载失败: {e}")
        return
    
    # 加载地形数据
    height_map = np.array(terrain_data['height_map'], dtype=np.float32)
    print(f"地形尺寸: {height_map.shape}")
    print(f"高程范围: {height_map.min():.2f} ~ {height_map.max():.2f}")
    
    # 选择起始点
    start_point = (20, 20)
    goal_point = (110, 110)
    
    print(f"起点: {start_point}, 终点: {goal_point}")
    print(f"起点高度: {height_map[start_point[0], start_point[1]]:.1f}")
    print(f"终点高度: {height_map[goal_point[0], goal_point[1]]:.1f}")
    
    # 创建环境
    try:
        env = TerrainGridNavEnv(
            H=height_map.shape[0], W=height_map.shape[1],
            max_steps=200,
            height_range=(terrain_data['original_bounds']['z_min'], 
                         terrain_data['original_bounds']['z_max']),
            slope_penalty_weight=0.05,
            height_penalty_weight=0.02,
            custom_terrain=height_map,
            fixed_start=start_point,
            fixed_goal=goal_point
        )
        print("环境创建成功")
    except Exception as e:
        print(f"环境创建失败: {e}")
        return
    
    # 创建智能体
    try:
        agent = TerrainPPOAgent(
            state_dim=13,
            action_dim=4,
            hidden_dim=256,
            lr=2e-4
        )
        print("智能体创建成功")
    except Exception as e:
        print(f"智能体创建失败: {e}")
        return
    
    # 测试单个episode
    print("\n测试单个episode...")
    try:
        obs, _ = env.reset()
        print(f"环境重置成功，起点: {env.pos}, 终点: {env.goal}")
        
        # 运行几步
        for step in range(5):
            action_mask = obs['action_mask']
            print(f"步骤 {step}: 位置 {env.pos}, 动作掩膜: {action_mask}")
            
            # 随机选择动作
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            if valid_actions:
                action = np.random.choice(valid_actions)
                print(f"选择动作: {action}")
                
                obs, reward, done, truncated, info = env.step(action)
                print(f"奖励: {reward:.3f}, 完成: {done}, 截断: {truncated}")
                
                if done or truncated:
                    break
            else:
                print("没有有效动作")
                break
        
        print("单个episode测试完成")
        
    except Exception as e:
        print(f"单个episode测试失败: {e}")
        return
    
    # 测试智能体收集episode
    print("\n测试智能体收集episode...")
    try:
        states, actions, rewards, values, log_probs, dones, path, success = \
            agent.collect_episode(env)
        
        print(f"收集episode成功: 成功={success}, 路径长度={len(path)}, 总奖励={sum(rewards):.3f}")
        
    except Exception as e:
        print(f"智能体收集episode失败: {e}")
        return
    
    print("所有测试完成")

if __name__ == "__main__":
    main()

