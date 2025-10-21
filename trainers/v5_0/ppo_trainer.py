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
        self.ppo_config = self.rl_config.get("ppo", {})
        
        # PPO超参数（从 mappo.ppo 读取）
        self.gamma = self.ppo_config.get("gamma", 0.99)
        self.gae_lambda = self.ppo_config.get("gae_lambda", 0.95)
        self.clip_eps = self.ppo_config.get("clip_eps", 0.2)
        self.lr = self.ppo_config.get("lr", 3e-4)
        self.value_loss_coef = self.ppo_config.get("value_coef", 0.5)
        self.entropy_coef = self.ppo_config.get("entropy_coef", 0.01)
        self.max_grad_norm = self.ppo_config.get("max_grad_norm", 0.5)
        
        # 训练参数
        rollout_cfg = self.rl_config.get("rollout", {})
        self.rollout_horizon = rollout_cfg.get("horizon", 20)
        self.minibatch_size = rollout_cfg.get("minibatch_size", 32)
        self.updates_per_iter = rollout_cfg.get("updates_per_iter", 8)
        self.max_updates = rollout_cfg.get("max_updates", 10)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 更新计数器
        self.current_update = 0
        
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
                # 获取当前phase的所有智能体
                phase_agents = self.env.get_phase_agents()
                execution_mode = self.env.get_phase_execution_mode()
                
                # 为每个智能体获取动作候选并用策略选择序列（动态更新候选集）
                phase_sequences = {}
                phase_candidates = {}
                
                for agent in phase_agents:
                    # 获取动作候选（考虑已占用槽位）
                    candidates = self.env.get_action_candidates(agent)
                    phase_candidates[agent] = candidates
                    
                    if candidates:
                        sel = self.selector.select_action(agent, candidates, state, greedy=False)
                        if sel is not None:
                            phase_sequences[agent] = sel['sequence']
                        else:
                            phase_sequences[agent] = None
                    else:
                        phase_sequences[agent] = None
                
                # 执行phase
                next_state, phase_rewards, done, info = self.env.step_phase(phase_agents, phase_sequences)
                
                # 创建经验记录（包含 logprob/value/动作ID 等，并附带 StepLog 与 next_state 用于导出）
                for agent in phase_agents:
                    sel = None
                    if phase_sequences.get(agent) is not None and phase_candidates.get(agent):
                        sel = self.selector.select_action(agent, phase_candidates[agent], state, greedy=True)
                    obs_vec = self.env.get_observation(agent)
                    next_obs_vec = self.env.get_observation(agent)
                    # 匹配该agent的step_log（来自phase_logs）
                    agent_log = None
                    if info.get('phase_logs'):
                        for lg in info['phase_logs']:
                            if getattr(lg, 'agent', None) == agent:
                                agent_log = lg
                                break
                    experience = {
                        'agent': agent,
                        'obs': obs_vec,
                        'action_id': sel['action_id'] if sel else -1,
                        'logprob': sel['logprob'] if sel else 0.0,
                        'value': sel['value'] if sel else 0.0,
                        'reward': phase_rewards.get(agent, 0.0),
                        'next_obs': next_obs_vec,
                        'done': done,
                        'step_log': agent_log,
                        'next_state': next_state,
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
        
        # 为每个智能体训练（标准 PPO 近似）
        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        
        for agent, agent_exps in agent_experiences.items():
            if not agent_exps:
                continue
            
            # 提取张量数据
            obs = torch.FloatTensor([exp['obs'] for exp in agent_exps])
            actions = torch.LongTensor([max(0, exp['action_id']) for exp in agent_exps])
            rewards = torch.FloatTensor([exp['reward'] for exp in agent_exps])
            dones = torch.FloatTensor([1.0 if exp['done'] else 0.0 for exp in agent_exps])
            values_old = torch.FloatTensor([exp['value'] for exp in agent_exps])
            logprobs_old = torch.FloatTensor([exp['logprob'] for exp in agent_exps])
            
            # 引导值：用最后一个样本的 next_obs 估一个 next_value（简化）
            with torch.no_grad():
                last_next = torch.FloatTensor(agent_exps[-1]['next_obs']).unsqueeze(0)
                next_value = self.selector.critic_networks[agent](last_next).squeeze().item()
            
            # 计算GAE
            advantages, returns = self.compute_gae(rewards.tolist(), values_old.tolist(), next_value, dones.tolist())
            
            # 转换为张量
            advantages = torch.FloatTensor(advantages)
            returns = torch.FloatTensor(returns)
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 训练网络
            for _ in range(self.updates_per_iter):
                # 检查是否达到最大更新次数
                if self.current_update >= self.max_updates:
                    print(f"达到最大更新次数: {self.max_updates}")
                    break
                
                # 随机采样批次
                idx = torch.randperm(len(agent_exps))[:min(self.minibatch_size, len(agent_exps))]
                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]
                batch_old_logp = logprobs_old[idx]
                
                # 新分布与 value
                batch_obs_t = batch_obs
                logits = self.selector.actor_networks[agent](batch_obs_t)
                # 掩码（按agent允许动作）
                allow = torch.zeros((logits.size(0), logits.size(1)))
                allowed_ids = self.selector._agent_allowed_actions(agent)
                if allowed_ids:
                    allow[:, allowed_ids] = 1.0
                masked_logits = logits + (allow + 1e-45).log()  # 局部近似mask
                logp_all = torch.log_softmax(masked_logits, dim=-1)
                new_logp = logp_all.gather(1, batch_actions.view(-1,1)).squeeze(1)
                
                values_pred = self.selector.critic_networks[agent](batch_obs_t).squeeze(1)
                
                # PPO目标
                ratio = (new_logp - batch_old_logp).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values_pred, batch_ret)
                entropy = -(torch.softmax(masked_logits, dim=-1) * logp_all).sum(dim=-1).mean()
                entropy_loss = -entropy
                
                total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
                
                # 反向传播
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
                
                # 增加更新计数器
                self.current_update += 1
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.item()
        
        # 更新训练步数
        self.training_step += 1
        
        return {
            'total_loss': total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            'actor_loss': total_actor_loss / len(agent_experiences),
            'critic_loss': total_critic_loss / len(agent_experiences),
            'entropy_loss': total_entropy_loss / len(agent_experiences),
            'training_step': self.training_step
        }
    
    # 旧的简化损失已删除，改用上方标准 PPO 计算
    
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
