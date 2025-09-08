#!/usr/bin/env python3
"""
PPO网格导航智能体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List
import random


class PPONetwork(nn.Module):
    """PPO网络：Actor-Critic架构"""
    
    def __init__(self, state_dim: int = 5, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        # 共享特征提取器
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """前向传播"""
        features = self.feature_net(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, state, action=None):
        """获取动作和值"""
        action_logits, value = self.forward(state)
        probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        if action is None:
            action = torch.multinomial(probs, 1)
        
        action_log_prob = log_probs.gather(1, action)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        
        return action, action_log_prob, entropy, value


class PPOGridNavAgent:
    """PPO网格导航智能体"""
    
    def __init__(self, state_dim: int = 5, action_dim: int = 4, lr: float = 3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # PPO参数
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # 训练统计
        self.episode_rewards = []
        self.success_rates = []
        
    def get_state_tensor(self, obs):
        """将观测转换为状态张量"""
        state = np.array([
            obs['position'][0],  # 当前x
            obs['position'][1],  # 当前y
            obs['goal'][0],      # 目标x
            obs['goal'][1],      # 目标y
            obs['distance_to_goal'][0]  # 距离
        ], dtype=np.float32)
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def get_action(self, obs, training=True):
        """获取动作"""
        state = self.get_state_tensor(obs)
        
        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(state)
            return int(action.cpu().numpy()[0])  # 确保返回整数
    
    def collect_episode(self, env, max_steps=200):
        """收集一个episode的数据"""
        obs, _ = env.reset()
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        total_reward = 0
        success = False
        
        for step in range(max_steps):
            state = self.get_state_tensor(obs)
            
            # 获取动作
            action, log_prob, _, value = self.network.get_action_and_value(state)
            
            # 执行动作
            next_obs, reward, done, truncated, info = env.step(action.item())
            
            # 存储数据
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done or truncated)
            
            total_reward += reward
            
            if done and not truncated and info.get('reason') == 'reached_goal':
                success = True
                break
            elif done or truncated:
                break
                
            obs = next_obs
        
        # 转换为张量并分离计算图
        states = torch.cat(states).detach()
        actions = torch.cat(actions).squeeze().detach()  # 确保是1D张量
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.cat(values).squeeze().detach()  # 确保是1D张量
        log_probs = torch.cat(log_probs).squeeze().detach()  # 确保是1D张量
        dones = torch.FloatTensor(dones).to(self.device)  # 转换为浮点张量
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'values': values,
            'log_probs': log_probs,
            'dones': dones,
            'total_reward': total_reward,
            'success': success,
            'episode_length': len(rewards)
        }
    
    def compute_returns_and_advantages(self, rewards, values, dones, gamma=0.99, gae_lambda=0.95):
        """计算回报和优势函数"""
        returns = []
        advantages = []
        
        # 计算GAE
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update_policy(self, episode_data, num_epochs=4):
        """更新策略"""
        states = episode_data['states']
        actions = episode_data['actions']
        old_log_probs = episode_data['log_probs']
        returns, advantages = self.compute_returns_and_advantages(
            episode_data['rewards'], 
            episode_data['values'], 
            episode_data['dones']
        )
        
        # 多轮更新
        for epoch in range(num_epochs):
            # 重新计算当前策略的动作概率和值
            action_logits, values = self.network(states)
            probs = F.softmax(action_logits, dim=-1)
            log_probs = F.log_softmax(action_logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # 计算比率
            ratio = torch.exp(action_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # 总损失
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
    def train_episode(self, env, episode_num):
        """训练一个episode"""
        # 收集数据
        episode_data = self.collect_episode(env)
        
        # 更新策略
        self.update_policy(episode_data)
        
        # 更新统计
        self.episode_rewards.append(episode_data['total_reward'])
        self.success_rates.append(1.0 if episode_data['success'] else 0.0)
        
        # 计算平均统计
        avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
        success_rate = np.mean(self.success_rates[-50:]) if len(self.success_rates) >= 50 else np.mean(self.success_rates)
        
        # 打印进度
        print(f"Episode {episode_num:4d} | "
              f"奖励: {episode_data['total_reward']:6.1f} | "
              f"步数: {episode_data['episode_length']:3d} | "
              f"成功: {'✅' if episode_data['success'] else '❌'} | "
              f"平均奖励: {avg_reward:6.1f} | "
              f"成功率: {success_rate*100:5.1f}%")
        
        return {
            'episode': episode_num,
            'total_reward': episode_data['total_reward'],
            'episode_length': episode_data['episode_length'],
            'success': episode_data['success'],
            'avg_reward': avg_reward,
            'success_rate': success_rate
        }


def test_ppo_agent(env, agent, num_tests=10):
    """测试PPO智能体"""
    print("\n🧪 测试PPO智能体性能...")
    
    successes = 0
    total_steps = 0
    
    for test in range(num_tests):
        obs, _ = env.reset()
        steps = 0
        
        while steps < env.max_steps:
            action = agent.get_action(obs, training=False)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            if done and not truncated and info.get('reason') == 'reached_goal':
                successes += 1
                break
            elif done or truncated:
                break
        
        total_steps += steps
    
    success_rate = successes / num_tests
    avg_steps = total_steps / num_tests
    
    print(f"✅ 测试结果: 成功率 {success_rate*100:.1f}% | 平均步数 {avg_steps:.1f}")
    return success_rate, avg_steps
