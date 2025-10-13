#!/usr/bin/env python3
"""
使用直接Mesh处理结果的训练脚本
"""

import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from envs.terrain_grid_nav_env import TerrainGridNavEnv
from agents.ppo_terrain_agent import TerrainPPOAgent
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DirectMeshTrainer:
    """直接Mesh地形训练器"""
    
    def __init__(self, terrain_file: str = "data/terrain/terrain_direct_mesh_fixed.json"):
        self.terrain_file = terrain_file
        self.terrain_data = None
        self.env = None
        self.agent = None
        self.training_stats = {
            'episode_rewards': [],
            'success_rates': [],
            'episode_lengths': [],
            'total_episodes': 0,
            'total_success': 0,
            'final_success_rate': 0.0,
            'final_avg_reward': 0.0,
            'final_avg_length': 0.0,
            'start_point': None,
            'goal_point': None,
            'terrain_file': terrain_file
        }
        
    def load_terrain_data(self):
        """加载地形数据"""
        if not os.path.exists(self.terrain_file):
            print(f"❌ 地形文件不存在: {self.terrain_file}")
            return False
        
        with open(self.terrain_file, 'r') as f:
            self.terrain_data = json.load(f)
        
        print(f"✅ 成功加载地形数据")
        print(f"   网格尺寸: {self.terrain_data['grid_size']}")
        print(f"   有效点数: {self.terrain_data['valid_points_count']}")
        print(f"   覆盖率: {self.terrain_data['coverage_percentage']:.1f}%")
        
        return True
    
    def find_land_points(self, height_threshold: float = 0.0) -> tuple:
        """在陆地上找到合适的起始点和终点"""
        height_map = np.array(self.terrain_data['height_map'])
        mask = np.array(self.terrain_data['mask'])
        
        # 找到所有有效的陆地点
        valid_indices = np.where((mask) & (height_map > height_threshold))
        
        if len(valid_indices[0]) < 2:
            print("❌ 没有足够的陆地点")
            return None, None
        
        # 随机选择两个不同的点
        indices = np.random.choice(len(valid_indices[0]), 2, replace=False)
        
        start_idx = (valid_indices[0][indices[0]], valid_indices[1][indices[0]])
        goal_idx = (valid_indices[0][indices[1]], valid_indices[1][indices[1]])
        
        start_height = height_map[start_idx]
        goal_height = height_map[goal_idx]
        
        print(f"✅ 找到起始点和终点")
        print(f"   起始点: {start_idx}, 高程: {start_height:.2f}")
        print(f"   终点: {goal_idx}, 高程: {goal_height:.2f}")
        
        return start_idx, goal_idx
    
    def create_environment(self, start_point: tuple, goal_point: tuple):
        """创建环境"""
        height_map = np.array(self.terrain_data['height_map'])
        mask = np.array(self.terrain_data['mask'])
        
        # 创建自定义地形数据 - 只传递高程图
        custom_terrain = height_map
        
        grid_size = self.terrain_data['grid_size']
        self.env = TerrainGridNavEnv(
            H=grid_size[0],
            W=grid_size[1],
            max_steps=400,
            custom_terrain=custom_terrain,
            fixed_start=start_point,
            fixed_goal=goal_point,
            slope_penalty_weight=0.0,  # 暂时移除地形惩罚
            height_penalty_weight=0.0
        )
        
        print(f"✅ 环境创建成功")
        print(f"   网格尺寸: {grid_size}")
        print(f"   最大步数: {self.env.max_steps}")
    
    def create_agent(self):
        """创建智能体"""
        # 计算状态维度（基础状态特征）
        # position(2) + goal(2) + distance_to_goal(1) + current_height(1) + 
        # goal_height(1) + height_difference(1) + current_slope(1) + action_mask(4) = 13
        state_dim = 13
        action_dim = self.env.action_space.n
        
        self.agent = TerrainPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            target_kl=0.01,
            train_pi_iters=80,
            train_v_iters=80,
            lam=0.97,
            max_grad_norm=0.5
        )
        
        print(f"✅ 智能体创建成功")
        print(f"   状态维度: {state_dim}")
        print(f"   动作维度: {action_dim}")
    
    def train(self, num_episodes: int = 1000, eval_interval: int = 50):
        """训练智能体"""
        print(f"🚀 开始训练，总episodes: {num_episodes}")
        print("=" * 50)
        
        for episode in range(1, num_episodes + 1):
            # 收集一个episode的数据
            states, actions, rewards, values, log_probs, dones, path, success = self.agent.collect_episode(self.env)
            
            # 更新智能体
            if len(states) > 0:
                self.agent.update(states, actions, rewards, values, log_probs, dones)
            
            # 记录统计信息
            episode_reward = rewards.sum().item()
            episode_length = len(rewards)
            episode_success = success
            
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            self.training_stats['total_episodes'] += 1
            
            if episode_success:
                self.training_stats['total_success'] += 1
            
            # 计算当前成功率
            current_success_rate = self.training_stats['total_success'] / self.training_stats['total_episodes']
            self.training_stats['success_rates'].append(current_success_rate)
            
            # 定期输出训练状态
            if episode % eval_interval == 0:
                recent_rewards = self.training_stats['episode_rewards'][-eval_interval:]
                recent_lengths = self.training_stats['episode_lengths'][-eval_interval:]
                recent_successes = sum(1 for r in recent_rewards if r > 0)
                
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)
                recent_success_rate = recent_successes / eval_interval
                
                print(f"Episode {episode:4d} | "
                      f"成功率: {current_success_rate:.1%} | "
                      f"最近成功率: {recent_success_rate:.1%} | "
                      f"平均奖励: {avg_reward:6.1f} | "
                      f"平均长度: {avg_length:4.1f}")
        
        # 计算最终统计
        self.training_stats['final_success_rate'] = current_success_rate
        self.training_stats['final_avg_reward'] = np.mean(self.training_stats['episode_rewards'])
        self.training_stats['final_avg_length'] = np.mean(self.training_stats['episode_lengths'])
        
        print("\n✅ 训练完成!")
        print(f"   总episodes: {self.training_stats['total_episodes']}")
        print(f"   成功次数: {self.training_stats['total_success']}")
        print(f"   最终成功率: {self.training_stats['final_success_rate']:.1%}")
        print(f"   平均奖励: {self.training_stats['final_avg_reward']:.2f}")
        print(f"   平均路径长度: {self.training_stats['final_avg_length']:.1f}")
    
    def save_training_data(self, output_file: str):
        """保存训练数据"""
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
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
            elif isinstance(obj, tuple):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # 转换所有numpy类型
        training_data = convert_numpy_types(self.training_stats)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"✅ 训练数据已保存到: {output_file}")
    
    def visualize_training(self, save_path: str = None):
        """可视化训练结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('直接Mesh地形训练结果', fontsize=16)
        
        episodes = list(range(1, len(self.training_stats['episode_rewards']) + 1))
        
        # 1. 成功率变化
        axes[0, 0].plot(episodes, self.training_stats['success_rates'], 'b-', linewidth=2)
        axes[0, 0].set_title('成功率变化')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('成功率')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. 奖励变化
        window_size = min(50, len(self.training_stats['episode_rewards']) // 10)
        if window_size > 1:
            moving_avg = np.convolve(self.training_stats['episode_rewards'], 
                                   np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = episodes[window_size-1:]
            axes[0, 1].plot(moving_avg_episodes, moving_avg, 'r-', linewidth=2, 
                           label=f'移动平均({window_size})')
        
        axes[0, 1].plot(episodes, self.training_stats['episode_rewards'], 'gray', alpha=0.3, linewidth=0.5)
        axes[0, 1].set_title('奖励变化')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('总奖励')
        axes[0, 1].grid(True, alpha=0.3)
        if window_size > 1:
            axes[0, 1].legend()
        
        # 3. 路径长度变化
        if window_size > 1:
            moving_avg_length = np.convolve(self.training_stats['episode_lengths'], 
                                          np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(moving_avg_episodes, moving_avg_length, 'g-', linewidth=2, 
                           label=f'移动平均({window_size})')
        
        axes[1, 0].plot(episodes, self.training_stats['episode_lengths'], 'gray', alpha=0.3, linewidth=0.5)
        axes[1, 0].set_title('路径长度变化')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('路径长度')
        axes[1, 0].grid(True, alpha=0.3)
        if window_size > 1:
            axes[1, 0].legend()
        
        # 4. 奖励分布
        axes[1, 1].hist(self.training_stats['episode_rewards'], bins=30, alpha=0.7, 
                       color='skyblue', edgecolor='black')
        axes[1, 1].axvline(np.mean(self.training_stats['episode_rewards']), color='red', 
                          linestyle='--', linewidth=2, 
                          label=f'平均值: {np.mean(self.training_stats["episode_rewards"]):.2f}')
        axes[1, 1].set_title('奖励分布')
        axes[1, 1].set_xlabel('总奖励')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练结果图已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    # 创建训练器
    trainer = DirectMeshTrainer()
    
    # 加载地形数据
    if not trainer.load_terrain_data():
        return
    
    # 找到起始点和终点
    start_point, goal_point = trainer.find_land_points(height_threshold=0.0)
    if start_point is None or goal_point is None:
        return
    
    # 保存起始点和终点
    trainer.training_stats['start_point'] = list(start_point)
    trainer.training_stats['goal_point'] = list(goal_point)
    
    # 创建环境
    trainer.create_environment(start_point, goal_point)
    
    # 创建智能体
    trainer.create_agent()
    
    # 开始训练
    trainer.train(num_episodes=1000, eval_interval=50)
    
    # 保存训练数据
    trainer.save_training_data("training_data/direct_mesh_training_stats.json")
    
    # 可视化训练结果
    trainer.visualize_training("visualization_output/direct_mesh_training_results.png")


if __name__ == "__main__":
    main()
