#!/usr/bin/env python3
"""
地形道路强化学习训练脚本
支持PPO、DQN等算法
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.terrain_road_env import TerrainRoadEnvironment
from agents.terrain_policy import TerrainPolicyNetwork, TerrainValueNetwork, TerrainActorCritic

class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, 
                 env: TerrainRoadEnvironment,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = 'auto'):
        
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 创建网络
        self.policy_network = TerrainPolicyNetwork(
            grid_size=env.grid_size,
            action_space=env.action_space,
            hidden_dim=256
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'total_losses': []
        }
    
    def collect_episode(self, max_steps: int = 1000) -> Tuple[List, List, List, List]:
        """收集一个episode的数据"""
        obs, _ = self.env.reset()
        observations, actions, rewards, values, log_probs = [], [], [], [], []
        
        for step in range(max_steps):
            # 转换观察为tensor
            obs_tensor = self._obs_to_tensor(obs)
            
            # 获取动作
            action, action_logits, value = self.policy_network.get_action(obs_tensor)
            
            # 计算log概率
            action_probs = torch.softmax(action_logits, dim=-1)
            log_prob = torch.log(action_probs[0, action])
            
            # 执行动作
            next_obs, reward, done, truncated, _ = self.env.step(action)
            
            # 存储数据
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            
            obs = next_obs
            
            if done or truncated:
                break
        
        return observations, actions, rewards, values, log_probs
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   next_value: float = 0.0) -> List[float]:
        """计算广义优势估计 (GAE)"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update_policy(self, observations: List, actions: List, 
                     old_log_probs: List, advantages: List, returns: List,
                     num_epochs: int = 10, batch_size: int = 64):
        """更新策略网络"""
        # 转换为tensor
        obs_tensor = self._batch_obs_to_tensor(observations)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # 标准化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # 多轮更新
        for epoch in range(num_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(observations))
            
            for start_idx in range(0, len(observations), batch_size):
                end_idx = min(start_idx + batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = {k: v[batch_indices] for k, v in obs_tensor.items()}
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # 前向传播
                action_logits, values = self.policy_network(batch_obs)
                
                # 计算新的log概率
                action_probs = torch.softmax(action_logits, dim=-1)
                log_probs = torch.log_softmax(action_logits, dim=-1)
                new_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = nn.MSELoss()(values.squeeze(-1), batch_returns)
                
                # 熵损失
                entropy = -(action_probs * log_probs).sum(dim=-1).mean()
                entropy_loss = -entropy
                
                # 总损失
                total_loss = (policy_loss + 
                            self.value_loss_coef * value_loss + 
                            self.entropy_coef * entropy_loss)
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 记录损失
                self.training_stats['policy_losses'].append(policy_loss.item())
                self.training_stats['value_losses'].append(value_loss.item())
                self.training_stats['entropy_losses'].append(entropy_loss.item())
                self.training_stats['total_losses'].append(total_loss.item())
    
    def _obs_to_tensor(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """将观察转换为tensor"""
        return {k: torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0) 
                for k, v in obs.items()}
    
    def _batch_obs_to_tensor(self, observations: List[Dict]) -> Dict[str, torch.Tensor]:
        """将批量观察转换为tensor"""
        batch_obs = {}
        for key in observations[0].keys():
            batch_obs[key] = torch.stack([
                torch.tensor(obs[key], dtype=torch.float32, device=self.device) 
                for obs in observations
            ])
        return batch_obs
    
    def train(self, num_episodes: int = 1000, save_interval: int = 100):
        """训练智能体"""
        print(f"开始训练，共{num_episodes}个episodes...")
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # 收集episode数据
            observations, actions, rewards, values, log_probs = self.collect_episode()
            
            # 计算优势
            advantages = self.compute_gae(rewards, values)
            returns = [r + self.gamma * v for r, v in zip(rewards, values)]
            
            # 更新策略
            self.update_policy(observations, actions, log_probs, advantages, returns)
            
            # 记录统计
            episode_reward = sum(rewards)
            episode_length = len(rewards)
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-10:])
                avg_length = np.mean(self.training_stats['episode_lengths'][-10:])
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Reward: {episode_reward:.2f}, Length: {episode_length}, "
                      f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")
            
            # 保存模型
            if (episode + 1) % save_interval == 0:
                self.save_model(f"ppo_terrain_episode_{episode + 1}.pth")
        
        print("训练完成!")
    
    def save_model(self, filename: str):
        """保存模型"""
        save_path = Path("models") / filename
        save_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, save_path)
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, filename: str):
        """加载模型"""
        load_path = Path("models") / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        print(f"模型已从 {load_path} 加载")

class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, 
                 env: TerrainRoadEnvironment,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 target_update_freq: int = 100,
                 device: str = 'auto'):
        
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_count = 0
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 创建网络
        self.q_network = TerrainValueNetwork(
            grid_size=env.grid_size,
            action_space=env.action_space,
            hidden_dim=256
        ).to(self.device)
        
        self.target_network = TerrainValueNetwork(
            grid_size=env.grid_size,
            action_space=env.action_space,
            hidden_dim=256
        ).to(self.device)
        
        # 复制权重
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'q_losses': [],
            'epsilons': []
        }
    
    def train(self, num_episodes: int = 1000, save_interval: int = 100):
        """训练智能体"""
        print(f"开始训练，共{num_episodes}个episodes...")
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            step_count = 0
            
            while step_count < 1000:  # 最大步数
                # 转换观察为tensor
                obs_tensor = self._obs_to_tensor(obs)
                
                # 选择动作
                action = self.q_network.get_action(obs_tensor, self.epsilon)
                
                # 执行动作
                next_obs, reward, done, truncated, _ = self.env.step(action)
                
                # 存储经验 (这里简化了，实际应该使用经验回放)
                # 直接更新网络
                self._update_q_network(obs_tensor, action, reward, next_obs, done)
                
                total_reward += reward
                step_count += 1
                obs = next_obs
                
                if done or truncated:
                    break
            
            # 更新epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # 记录统计
            self.training_stats['episode_rewards'].append(total_reward)
            self.training_stats['episode_lengths'].append(step_count)
            self.training_stats['epsilons'].append(self.epsilon)
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-10:])
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}, "
                      f"Avg Reward: {avg_reward:.2f}")
            
            # 保存模型
            if (episode + 1) % save_interval == 0:
                self.save_model(f"dqn_terrain_episode_{episode + 1}.pth")
        
        print("训练完成!")
    
    def _update_q_network(self, obs_tensor, action, reward, next_obs, done):
        """更新Q网络"""
        # 获取当前Q值
        current_q_values = self.q_network(obs_tensor)
        current_q = current_q_values[0, action]
        
        # 获取目标Q值
        with torch.no_grad():
            next_obs_tensor = self._obs_to_tensor(next_obs)
            next_q_values = self.target_network(next_obs_tensor)
            max_next_q = torch.max(next_q_values)
            target_q = reward + (self.gamma * max_next_q * (1 - done))
        
        # 计算损失
        loss = self.criterion(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 记录损失
        self.training_stats['q_losses'].append(loss.item())
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _obs_to_tensor(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """将观察转换为tensor"""
        return {k: torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0) 
                for k, v in obs.items()}
    
    def save_model(self, filename: str):
        """保存模型"""
        save_path = Path("models") / filename
        save_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'epsilon': self.epsilon
        }, save_path)
        print(f"模型已保存到: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='地形道路强化学习训练')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'dqn'],
                       help='训练算法')
    parser.add_argument('--mesh_file', type=str, default=None,
                       help='mesh文件路径')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[50, 50],
                       help='网格大小')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='训练episodes数量')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备选择')
    
    args = parser.parse_args()
    
    # 创建环境
    env = TerrainRoadEnvironment(
        mesh_file=args.mesh_file,
        grid_size=tuple(args.grid_size),
        max_steps=1000
    )
    
    print(f"环境信息: {env.get_terrain_info()}")
    
    # 创建智能体
    if args.algorithm == 'ppo':
        agent = PPOAgent(env, lr=args.lr, device=args.device)
    elif args.algorithm == 'dqn':
        agent = DQNAgent(env, lr=args.lr, device=args.device)
    else:
        raise ValueError(f"不支持的算法: {args.algorithm}")
    
    # 开始训练
    agent.train(num_episodes=args.episodes)

if __name__ == "__main__":
    main()
