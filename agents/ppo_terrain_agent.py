#!/usr/bin/env python3
"""
地形PPO智能体 - 适配TerrainGridNavEnv的PPO智能体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class TerrainPPONetwork(nn.Module):
    """地形PPO网络 - 处理高度和坡度信息"""
    
    def __init__(self, state_dim: int = 13, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        # 共享特征网络 - 处理基础状态信息
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 地形特征网络 - 处理局部地形信息
        self.terrain_net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # 2通道：地形+坡度
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # 自适应池化到2x2
            nn.Flatten(),
            nn.Linear(64, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor网络 - 策略头
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic网络 - 价值头
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state_features: torch.Tensor, terrain_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 处理基础状态特征
        state_features = self.feature_net(state_features)
        
        # 处理地形特征
        terrain_features = self.terrain_net(terrain_features)
        
        # 融合特征
        combined_features = torch.cat([state_features, terrain_features], dim=1)
        
        # 输出策略和价值
        action_logits = self.actor(combined_features)
        value = self.critic(combined_features)
        
        return action_logits, value


class TerrainPPOAgent:
    """地形PPO智能体"""
    
    def __init__(self, state_dim: int = 13, action_dim: int = 4, hidden_dim: int = 64,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, target_kl: float = 0.01,
                 train_pi_iters: int = 80, train_v_iters: int = 80,
                 lam: float = 0.97, max_grad_norm: float = 0.5):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 网络
        self.network = TerrainPPONetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        
        # 训练记录
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
    
    def get_state_features(self, obs: Dict) -> torch.Tensor:
        """提取基础状态特征"""
        # 基础状态信息：位置、目标、距离、高度、坡度
        state_features = np.concatenate([
            obs['position'].astype(np.float32),
            obs['goal'].astype(np.float32),
            obs['distance_to_goal'].astype(np.float32),
            obs['current_height'].astype(np.float32),
            obs['goal_height'].astype(np.float32),
            obs['height_difference'].astype(np.float32),
            obs['current_slope'].astype(np.float32),
            obs['action_mask'].astype(np.float32)
        ])
        
        return torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
    
    def get_terrain_features(self, obs: Dict) -> torch.Tensor:
        """提取地形特征"""
        # 地形特征：局部地形和坡度
        terrain_features = np.stack([
            obs['local_terrain'],
            obs['local_slope']
        ], axis=0)  # (2, 5, 5)
        
        return torch.FloatTensor(terrain_features).unsqueeze(0).to(self.device)
    
    def get_action(self, obs: Dict) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """获取动作"""
        state_features = self.get_state_features(obs)
        terrain_features = self.get_terrain_features(obs)
        
        with torch.no_grad():
            action_logits, value = self.network(state_features, terrain_features)
            
            # 应用动作掩膜
            action_mask = torch.FloatTensor(obs['action_mask']).to(self.device)
            masked_logits = action_logits - (1 - action_mask) * 1e8
            
            # 采样动作
            action_probs = F.softmax(masked_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        return int(action.cpu().numpy()[0]), log_prob, value
    
    def compute_returns_and_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                                     dones: torch.Tensor, gamma: float = 0.99, 
                                     gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算回报和优势函数"""
        # 确保dones是FloatTensor
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算GAE
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            advantages[i] = delta + gamma * gae_lambda * (1 - dones[i]) * last_advantage
            last_advantage = advantages[i]
        
        returns = advantages + values
        
        return returns, advantages
    
    def collect_episode(self, env) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                          torch.Tensor, torch.Tensor, torch.Tensor]:
        """收集一个episode的数据"""
        obs, _ = env.reset()
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        path = [env.pos]
        success = False
        
        while True:
            state_features = self.get_state_features(obs)
            terrain_features = self.get_terrain_features(obs)
            
            with torch.no_grad():
                action_logits, value = self.network(state_features, terrain_features)
                
                # 应用动作掩膜
                action_mask = torch.FloatTensor(obs['action_mask']).to(self.device)
                masked_logits = action_logits - (1 - action_mask) * 1e8
                
                # 采样动作
                action_probs = F.softmax(masked_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            # 存储数据
            states.append(torch.cat([state_features.squeeze(), terrain_features.squeeze().flatten()]))
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(int(action.cpu().numpy()[0]))
            rewards.append(reward)
            dones.append(done or truncated)
            path.append(env.pos)
            if done:
                success = True
            
            if done or truncated:
                break
        
        # 转换为张量
        states = torch.stack(states).detach()
        actions = torch.stack(actions).squeeze().detach()
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.stack(values).squeeze().detach()
        log_probs = torch.stack(log_probs).squeeze().detach()
        dones = torch.FloatTensor(dones).to(self.device)
        
        return states, actions, rewards, values, log_probs, dones, path, success
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
               values: torch.Tensor, log_probs: torch.Tensor, dones: torch.Tensor):
        """更新网络"""
        self._update_network(states, actions, rewards, values, log_probs, dones, entropy_bonus=0.01)
    
    def update_with_entropy_bonus(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                                 values: torch.Tensor, log_probs: torch.Tensor, dones: torch.Tensor, 
                                 entropy_bonus: float):
        """带熵奖励的网络更新"""
        self._update_network(states, actions, rewards, values, log_probs, dones, entropy_bonus)
    
    def _update_network(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                       values: torch.Tensor, log_probs: torch.Tensor, dones: torch.Tensor, 
                       entropy_bonus: float = 0.01):
        """更新网络"""
        # 计算回报和优势
        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 分离状态特征和地形特征
        state_features = states[:, :13]  # 前13维是基础状态特征
        terrain_features = states[:, 13:].view(-1, 2, 5, 5)  # 后50维是地形特征(2x5x5)
        
        # 训练策略网络
        for _ in range(self.train_pi_iters):
            action_logits, _ = self.network(state_features, terrain_features)
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - log_probs)
            
            # 计算策略损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算熵损失
            entropy_loss = -action_dist.entropy().mean()
            
            # 总损失（带可调节的熵奖励）
            total_loss = policy_loss + entropy_bonus * entropy_loss
            
            # 更新
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # 检查KL散度
            kl_div = (log_probs - new_log_probs).mean()
            if kl_div > self.target_kl:
                break
        
        # 训练价值网络
        for _ in range(self.train_v_iters):
            _, new_values = self.network(state_features, terrain_features)
            value_loss = F.mse_loss(new_values.squeeze(), returns)
            
            self.optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
    def train_episode(self, env) -> Dict:
        """训练一个episode"""
        # 收集数据
        states, actions, rewards, values, log_probs, dones, path, success = self.collect_episode(env)
        
        # 更新网络
        self.update(states, actions, rewards, values, log_probs, dones)
        
        # 记录统计信息
        episode_reward = rewards.sum().item()
        episode_length = len(rewards)
        # success = dones[-1].item() and episode_length < env.max_steps # This line is removed
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.total_episodes += 1
        
        if success:
            self.success_count += 1
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'success': success,
            'success_rate': self.success_count / self.total_episodes
        }
    
    def test_episode(self, env, render: bool = False) -> Dict:
        """测试一个episode"""
        obs, _ = env.reset()
        total_reward = 0
        path = [obs['position'].copy()]
        
        while True:
            if render:
                env.render()
            
            action, _, _ = self.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            path.append(obs['position'].copy())
            
            if done or truncated:
                break
        
        success = done and len(path) < env.max_steps
        
        return {
            'total_reward': total_reward,
            'path_length': len(path),
            'success': success,
            'path': path
        }
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'success_count': self.success_count,
            'total_episodes': self.total_episodes
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.success_count = checkpoint.get('success_count', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)


if __name__ == "__main__":
    # 测试智能体
    from envs.terrain_grid_nav_env import TerrainGridNavEnv
    
    env = TerrainGridNavEnv()
    agent = TerrainPPOAgent()
    
    print("地形PPO智能体测试:")
    print(f"状态维度: 13 (基础) + 50 (地形) = 63")
    print(f"动作维度: 4")
    
    # 测试一个episode
    obs, _ = env.reset()
    action, log_prob, value = agent.get_action(obs)
    print(f"测试动作: {action}, 对数概率: {log_prob.item():.3f}, 价值: {value.item():.3f}")
    
    # 训练一个episode
    result = agent.train_episode(env)
    print(f"训练结果: {result}")
