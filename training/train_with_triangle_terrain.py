#!/usr/bin/env python3
"""
使用三角面填充地形的强化学习训练
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
from agents.ppo_terrain_agent import TerrainPPOAgent
from envs.terrain_grid_nav_env import TerrainGridNavEnv

def load_latest_terrain_data():
    """加载最新的地形数据"""
    data_dir = Path("data/terrain")
    if not data_dir.exists():
        print("❌ 地形数据目录不存在")
        return None
    
    # 查找最新的地形文件
    terrain_files = list(data_dir.glob("terrain_*.json"))
    if not terrain_files:
        print("❌ 没有找到地形数据文件")
        return None
    
    # 按修改时间排序，取最新的
    latest_file = max(terrain_files, key=lambda f: f.stat().st_mtime)
    print(f"✅ 加载最新地形数据: {latest_file}")
    
    with open(latest_file, 'r') as f:
        terrain_data = json.load(f)
    
    return terrain_data

def analyze_terrain_for_training(terrain_data):
    """分析地形数据是否适合训练"""
    print("📊 分析地形数据...")
    
    height_map = np.array(terrain_data['height_map'])
    mask = np.array(terrain_data['mask'])
    grid_size = terrain_data['grid_size']
    
    print(f"   网格大小: {grid_size}")
    print(f"   掩码覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
    
    # 检查有效区域
    valid_heights = height_map[mask]
    if len(valid_heights) == 0:
        print("❌ 没有有效的地形区域")
        return None
    
    print(f"   有效高程范围: [{np.min(valid_heights):.3f}, {np.max(valid_heights):.3f}]")
    print(f"   平均高程: {np.mean(valid_heights):.3f}")
    
    # 找到合适的起始点和终点
    start_goal = find_good_start_goal(height_map, mask)
    if start_goal is None:
        print("❌ 无法找到合适的起始点和终点")
        return None
    
    start_pos, goal_pos = start_goal
    print(f"   起始点: {start_pos}")
    print(f"   终点: {goal_pos}")
    
    return {
        'height_map': height_map,
        'mask': mask,
        'grid_size': grid_size,
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'valid_heights': valid_heights
    }

def find_good_start_goal(height_map, mask):
    """找到合适的起始点和终点"""
    # 找到所有有效点
    valid_positions = np.where(mask)
    if len(valid_positions[0]) < 2:
        return None
    
    # 随机选择起始点和终点
    indices = np.random.choice(len(valid_positions[0]), 2, replace=False)
    start_pos = (valid_positions[0][indices[0]], valid_positions[1][indices[0]])
    goal_pos = (valid_positions[0][indices[1]], valid_positions[1][indices[1]])
    
    # 确保距离足够远
    distance = np.sqrt((start_pos[0] - goal_pos[0])**2 + (start_pos[1] - goal_pos[1])**2)
    if distance < 20:  # 如果太近，重新选择
        return find_good_start_goal(height_map, mask)
    
    return start_pos, goal_pos

def train_with_triangle_terrain():
    """使用三角面填充地形进行训练"""
    print("🚀 开始三角面填充地形训练")
    print("=" * 50)
    
    # 1. 加载地形数据
    terrain_data = load_latest_terrain_data()
    if terrain_data is None:
        return
    
    # 2. 分析地形数据
    training_data = analyze_terrain_for_training(terrain_data)
    if training_data is None:
        return
    
    # 3. 创建环境
    print("\n🔄 创建训练环境...")
    env = TerrainGridNavEnv(
        H=training_data['grid_size'][0],
        W=training_data['grid_size'][1],
        custom_terrain=training_data['height_map'],
        fixed_start=training_data['start_pos'],
        fixed_goal=training_data['goal_pos']
    )
    
    # 4. 创建智能体
    print("🔄 创建PPO智能体...")
    state_dim = 13  # 根据环境观察空间调整
    agent = TerrainPPOAgent(
        state_dim=state_dim,
        action_dim=4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01
    )
    
    # 5. 训练参数
    num_episodes = 10000
    save_interval = 1000
    
    print(f"\n🎯 开始训练 ({num_episodes} episodes)...")
    print(f"   起始点: {training_data['start_pos']}")
    print(f"   终点: {training_data['goal_pos']}")
    print(f"   网格大小: {training_data['grid_size']}")
    
    # 6. 训练循环
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_rates': [],
        'episode_heights': [],
        'episode_slopes': []
    }
    
    for episode in range(num_episodes):
        # 收集一个episode的数据
        states, actions, rewards, values, log_probs, dones, path, success = agent.collect_episode(env)
        
        # 更新智能体
        agent.update(states, actions, rewards, values, log_probs, dones)
        
        # 记录统计信息
        episode_reward = sum(rewards)
        episode_length = len(rewards)
        episode_height = np.mean([training_data['height_map'][pos] for pos in path if mask[pos]])
        episode_slope = calculate_average_slope(path, training_data['height_map'])
        
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(episode_length)
        training_stats['episode_heights'].append(episode_height)
        training_stats['episode_slopes'].append(episode_slope)
        
        # 计算最近的成功率
        recent_successes = training_stats['success_rates'][-100:] if training_stats['success_rates'] else []
        recent_success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0
        training_stats['success_rates'].append(success)
        
        # 打印进度
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Success={success}, "
                  f"Recent Success Rate={recent_success_rate:.2%}")
        
        # 保存数据
        if episode % save_interval == 0 and episode > 0:
            save_training_data(training_stats, episode, training_data)
    
    # 7. 保存最终结果
    save_training_data(training_stats, num_episodes, training_data, is_final=True)
    
    print(f"\n✅ 训练完成!")
    print(f"   最终成功率: {sum(training_stats['success_rates'][-100:])/100:.2%}")
    print(f"   平均奖励: {np.mean(training_stats['episode_rewards'][-100:]):.2f}")

def calculate_average_slope(path, height_map):
    """计算路径的平均坡度"""
    if len(path) < 2:
        return 0.0
    
    slopes = []
    for i in range(len(path) - 1):
        pos1, pos2 = path[i], path[i + 1]
        height1 = height_map[pos1]
        height2 = height_map[pos2]
        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        if distance > 0:
            slope = abs(height2 - height1) / distance
            slopes.append(slope)
    
    return np.mean(slopes) if slopes else 0.0

def save_training_data(training_stats, episode, training_data, is_final=False):
    """保存训练数据"""
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # 准备保存数据
    save_data = {
        'episode': episode,
        'training_stats': convert_numpy_types(training_stats),
        'terrain_info': {
            'grid_size': training_data['grid_size'],
            'start_pos': training_data['start_pos'],
            'goal_pos': training_data['goal_pos'],
            'height_range': [float(np.min(training_data['valid_heights'])), 
                           float(np.max(training_data['valid_heights']))]
        },
        'is_final': is_final
    }
    
    # 保存文件
    filename = f"triangle_terrain_training_{episode}.json" if not is_final else "triangle_terrain_training_final.json"
    filepath = Path("data/training") / filename
    filepath.parent.mkdir(exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"💾 训练数据已保存: {filepath}")

if __name__ == "__main__":
    train_with_triangle_terrain()

