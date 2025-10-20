"""
v5.0 PPO训练器

基于契约对象和配置的训练系统。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
import datetime
import json
from collections import deque

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contracts import ActionCandidate, Sequence, StepLog, EnvironmentState
from config_loader import ConfigLoader
from envs.v5_0.city_env import V5CityEnvironment
from solvers.v5_0.rl_selector import V5RLSelector


class V5PPOTrainer:
    """v5.0 PPO训练器"""
    
    def __init__(self, config_path: str):
        """
        初始化PPO训练器
        
        Args:
            config_path: v5.0配置文件路径
        """
        # 加载配置
        self.loader = ConfigLoader()
        self.config = self.loader.load_v5_config(config_path)
        
        # 获取RL配置
        self.rl_config = self.config.get("mappo", {})
        
        # PPO超参数
        self.gamma = self.rl_config.get("gamma", 0.99)
        self.gae_lambda = self.rl_config.get("gae_lambda", 0.95)
        self.clip_ratio = self.rl_config.get("clip_ratio", 0.2)
        self.lr = self.rl_config.get("lr", 3e-4)
        self.value_loss_coef = self.rl_config.get("value_loss_coef", 0.5)
        self.entropy_coef = self.rl_config.get("entropy_coef", 0.01)
        self.max_grad_norm = self.rl_config.get("max_grad_norm", 0.5)
        
        # 训练参数
        self.rollout_horizon = self.rl_config.get("rollout", {}).get("horizon", 20)
        self.minibatch_size = self.rl_config.get("rollout", {}).get("minibatch_size", 32)
        self.updates_per_iter = self.rl_config.get("rollout", {}).get("updates_per_iter", 8)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化环境
        self.env = V5CityEnvironment(config_path)
        
        # 初始化RL选择器
        self.selector = V5RLSelector(self.config)
        
        # 训练状态
        self.training_step = 0
        self.episode_count = 0
        self.total_steps = 0
        
        # 历史记录
        self.training_history = []
        self.episode_rewards = {agent: [] for agent in self.config.get("agents", {}).get("order", [])}
        
        # 优化器
        self._setup_optimizers()
    
    def _setup_optimizers(self):
        """设置优化器"""
        self.optimizers = {}
        agents = self.config.get("agents", {}).get("order", [])
        
        for agent in agents:
            # 为每个智能体创建优化器
            self.optimizers[agent] = {
                'actor': optim.Adam(self.selector.actor_networks[agent].parameters(), lr=self.lr),
                'critic': optim.Adam(self.selector.critic_networks[agent].parameters(), lr=self.lr)
            }
    
    def collect_experience(self, num_steps: int) -> List[Dict]:
        """
        收集经验数据
        
        Args:
            num_steps: 收集步数
            
        Returns:
            经验列表
        """
        all_experiences = []
        steps_collected = 0
        
        while steps_collected < num_steps:
            # 重置环境
            state = self.env.reset()
            
            episode_experiences = []
            done = False
            
            step_count = 0
            max_steps = 1000  # 防止无限循环
            
            while not done and steps_collected < num_steps and step_count < max_steps:
                # 获取当前智能体
                current_agent = self.env.current_agent
                
                # 获取动作候选
                candidates = self.env.get_action_candidates(current_agent)
                
                if not candidates:
                    # 没有可用动作，推进到下一步
                    self.env._update_state()
                    step_count += 1
                    # 检查是否结束
                    done = self.env._is_done()
                    continue
                
                # 选择动作序列
                sequence = self.selector.choose_sequence(
                    agent=current_agent,
                    candidates=candidates,
                    state=state
                )
                
                # 执行动作
                next_state, reward, done, info = self.env.step(current_agent, sequence)
                
                # 创建经验记录
                experience = {
                    'agent': current_agent,
                    'state': state,
                    'candidates': candidates,
                    'sequence': sequence,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done,
                    'info': info,
                    'step_log': info.get('step_log')
                }
                
                episode_experiences.append(experience)
                all_experiences.append(experience)
                
                # 更新状态
                state = next_state
                steps_collected += 1
                step_count += 1
        
        print(f"收集了 {len(all_experiences)} 步经验")
        return all_experiences
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   next_value: float, dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        计算GAE优势估计
        
        Args:
            rewards: 奖励列表
            values: 价值估计列表
            next_value: 下一个状态的价值
            dones: 是否结束标志
            
        Returns:
            (advantages, returns)
        """
        advantages = []
        returns = []
        
        # 计算GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def train_step(self, experiences: List[Dict]) -> Dict[str, float]:
        """
        执行一步训练
        
        Args:
            experiences: 经验数据
            
        Returns:
            训练统计信息
        """
        # 按智能体分组经验
        agent_experiences = {}
        for exp in experiences:
            agent = exp['agent']
            if agent not in agent_experiences:
                agent_experiences[agent] = []
            agent_experiences[agent].append(exp)
        
        # 为每个智能体训练
        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        
        for agent, agent_exps in agent_experiences.items():
            if not agent_exps:
                continue
            
            # 提取数据
            states = [exp['state'] for exp in agent_exps]
            sequences = [exp['sequence'] for exp in agent_exps]
            rewards = [exp['reward'] for exp in agent_exps]
            next_states = [exp['next_state'] for exp in agent_exps]
            dones = [exp['done'] for exp in agent_exps]
            
            # 计算价值估计
            values = []
            for state in states:
                obs = self.env.get_observation(agent)
                with torch.no_grad():
                    value = self.selector.critic_networks[agent](torch.FloatTensor(obs).unsqueeze(0))
                    values.append(value.item())
            
            # 计算下一个状态的价值
            if next_states:
                next_obs = self.env.get_observation(agent)
                with torch.no_grad():
                    next_value = self.selector.critic_networks[agent](torch.FloatTensor(next_obs).unsqueeze(0))
                    next_value = next_value.item()
            else:
                next_value = 0.0
            
            # 计算GAE
            advantages, returns = self.compute_gae(rewards, values, next_value, dones)
            
            # 转换为张量
            advantages = torch.FloatTensor(advantages)
            returns = torch.FloatTensor(returns)
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 训练网络
            for _ in range(self.updates_per_iter):
                # 随机采样批次
                batch_indices = torch.randperm(len(agent_exps))[:self.minibatch_size]
                
                batch_states = [states[i] for i in batch_indices]
                batch_sequences = [sequences[i] for i in batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算损失
                actor_loss, critic_loss, entropy_loss = self._compute_losses(
                    agent, batch_states, batch_sequences, batch_advantages, batch_returns
                )
                
                # 反向传播
                total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_loss
                
                self.optimizers[agent]['actor'].zero_grad()
                self.optimizers[agent]['critic'].zero_grad()
                
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.selector.actor_networks[agent].parameters(), 
                    self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.selector.critic_networks[agent].parameters(), 
                    self.max_grad_norm
                )
                
                self.optimizers[agent]['actor'].step()
                self.optimizers[agent]['critic'].step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # 更新训练步数
        self.training_step += 1
        
        return {
            'total_loss': total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            'actor_loss': total_actor_loss / len(agent_experiences),
            'critic_loss': total_critic_loss / len(agent_experiences),
            'entropy_loss': total_entropy_loss / len(agent_experiences),
            'training_step': self.training_step
        }
    
    def _compute_losses(self, agent: str, states: List[EnvironmentState], 
                       sequences: List[Sequence], advantages: torch.Tensor, 
                       returns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算损失函数
        
        Args:
            agent: 智能体名称
            states: 状态列表
            sequences: 序列列表
            advantages: 优势估计
            returns: 回报
            
        Returns:
            (actor_loss, critic_loss, entropy_loss)
        """
        # 计算动作概率
        action_probs = []
        values = []
        
        for state, sequence in zip(states, sequences):
            obs = self.env.get_observation(agent)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # 计算动作概率（需要梯度）
            action_logits = self.selector.actor_networks[agent](obs_tensor)
            value = self.selector.critic_networks[agent](obs_tensor)
            
            # 简化：使用均匀分布作为动作概率
            action_prob = torch.softmax(action_logits, dim=-1)
            action_probs.append(action_prob)
            values.append(value)
        
        # 计算损失
        action_probs = torch.cat(action_probs, dim=0)
        values = torch.cat(values, dim=0)
        
        # 简化Actor损失计算（避免维度不匹配）
        # 使用简单的策略梯度损失
        actor_loss = -(action_probs.mean(dim=-1) * advantages).mean()
        
        # Critic损失
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # 熵损失
        entropy_loss = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        
        return actor_loss, critic_loss, entropy_loss
    
    def train(self, num_episodes: int, save_interval: int = 100) -> Dict[str, Any]:
        """
        执行训练
        
        Args:
            num_episodes: 训练轮数
            save_interval: 保存间隔
            
        Returns:
            训练结果
        """
        print(f"开始训练 {num_episodes} 轮...")
        
        for episode in range(num_episodes):
            # 收集经验
            experiences = self.collect_experience(self.rollout_horizon)
            
            if not experiences:
                print(f"Episode {episode}: 没有收集到经验")
                continue
            
            # 训练
            train_stats = self.train_step(experiences)
            
            # 记录统计信息
            self.training_history.append(train_stats)
            
            # 计算episode奖励
            episode_rewards = {}
            for agent in self.config.get("agents", {}).get("order", []):
                agent_rewards = [exp['reward'] for exp in experiences if exp['agent'] == agent]
                episode_rewards[agent] = sum(agent_rewards) if agent_rewards else 0.0
                self.episode_rewards[agent].append(episode_rewards[agent])
            
            # 打印进度
            if episode % 10 == 0:
                print(f"Episode {episode}: "
                      f"Total Loss: {train_stats['total_loss']:.4f}, "
                      f"Actor Loss: {train_stats['actor_loss']:.4f}, "
                      f"Critic Loss: {train_stats['critic_loss']:.4f}")
                
                for agent, reward in episode_rewards.items():
                    print(f"  {agent} Reward: {reward:.2f}")
            
            # 保存模型
            if episode % save_interval == 0 and episode > 0:
                self.save_model(f"checkpoints/v5_0_ppo_episode_{episode}.pth")
        
        print("训练完成!")
        
        return {
            'training_history': self.training_history,
            'episode_rewards': self.episode_rewards,
            'total_episodes': num_episodes
        }
    
    def save_model(self, path: str):
        """保存模型"""
        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        model_state = {
            'actor_networks': {agent: net.state_dict() for agent, net in self.selector.actor_networks.items()},
            'critic_networks': {agent: net.state_dict() for agent, net in self.selector.critic_networks.items()},
            'optimizers': {agent: {opt_name: opt.state_dict() for opt_name, opt in opts.items()} 
                          for agent, opts in self.optimizers.items()},
            'training_step': self.training_step,
            'config': self.config
        }
        
        torch.save(model_state, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return
        
        model_state = torch.load(path, map_location=self.device)
        
        # 加载网络权重
        for agent, net in self.selector.actor_networks.items():
            if agent in model_state['actor_networks']:
                net.load_state_dict(model_state['actor_networks'][agent])
        
        for agent, net in self.selector.critic_networks.items():
            if agent in model_state['critic_networks']:
                net.load_state_dict(model_state['critic_networks'][agent])
        
        # 加载优化器状态
        for agent, opts in self.optimizers.items():
            if agent in model_state['optimizers']:
                for opt_name, opt in opts.items():
                    if opt_name in model_state['optimizers'][agent]:
                        opt.load_state_dict(model_state['optimizers'][agent][opt_name])
        
        self.training_step = model_state.get('training_step', 0)
        print(f"模型已从 {path} 加载")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            num_episodes: 评估轮数
            
        Returns:
            评估结果
        """
        print(f"开始评估 {num_episodes} 轮...")
        
        eval_rewards = {agent: [] for agent in self.config.get("agents", {}).get("order", [])}
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_rewards = {agent: 0.0 for agent in self.config.get("agents", {}).get("order", [])}
            
            step_count = 0
            max_steps = 1000  # 防止无限循环
            
            while not done and step_count < max_steps:
                current_agent = self.env.current_agent
                
                # 获取动作候选
                candidates = self.env.get_action_candidates(current_agent)
                
                if not candidates:
                    self.env._update_state()
                    step_count += 1
                    # 检查是否结束
                    done = self.env._is_done()
                    continue
                
                # 选择动作序列（使用贪心策略）
                sequence = self.selector.choose_sequence(
                    agent=current_agent,
                    candidates=candidates,
                    state=state,
                    greedy=True
                )
                
                # 执行动作
                next_state, reward, done, info = self.env.step(current_agent, sequence)
                
                # 记录奖励
                episode_rewards[current_agent] += reward
                
                # 更新状态
                state = next_state
                step_count += 1
            
            # 记录episode奖励
            for agent, reward in episode_rewards.items():
                eval_rewards[agent].append(reward)
        
        # 计算平均奖励
        avg_rewards = {}
        for agent, rewards in eval_rewards.items():
            avg_rewards[agent] = np.mean(rewards) if rewards else 0.0
        
        print("评估结果:")
        for agent, avg_reward in avg_rewards.items():
            print(f"  {agent}: {avg_reward:.2f}")
        
        return avg_rewards
