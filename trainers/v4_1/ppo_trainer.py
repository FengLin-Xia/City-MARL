#!/usr/bin/env python3
"""
v4.1 PPO训练器
基于现有PPO实现，适配城市仿真环境
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
import hashlib
import glob

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solvers.v4_1.rl_selector import RLPolicySelector
from envs.v4_1.city_env import CityEnvironment


class PPOTrainer:
    """PPO训练器 - 适配城市仿真环境"""
    
    def __init__(self, cfg: Dict):
        """
        初始化PPO训练器
        
        Args:
            cfg: 配置字典，包含RL超参数
        """
        self.cfg = cfg
        self.rl_cfg = cfg.get('growth_v4_1', {}).get('solver', {}).get('rl', {})
        
        # PPO超参数
        self.gamma = self.rl_cfg.get('gamma', 0.99)  # 折扣因子
        self.gae_lambda = self.rl_cfg.get('gae_lambda', 0.95)  # GAE参数
        self.clip_ratio = self.rl_cfg.get('clip_ratio', 0.2)  # PPO裁剪率
        self.lr = self.rl_cfg.get('lr', 3e-4)  # 学习率
        self.value_loss_coef = self.rl_cfg.get('value_loss_coef', 0.5)  # 价值损失系数
        self.entropy_coef = self.rl_cfg.get('entropy_coef', 0.01)  # 熵损失系数
        self.max_grad_norm = self.rl_cfg.get('max_grad_norm', 0.5)  # 梯度裁剪
        self.num_epochs = self.rl_cfg.get('num_epochs', 4)  # 更新轮数
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PPO训练器使用设备: {self.device}")
        
        # 初始化统计信息
        self.training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'total_losses': [],
            'kl_divergences': [],
            'clip_fractions': []
        }
        
        # 创建RL选择器（包含策略网络）
        self.selector = RLPolicySelector(cfg)
        print(f"PPO训练器初始化完成 - 超参数: γ={self.gamma}, λ={self.gae_lambda}, clip={self.clip_ratio}")
    
    def set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def collect_experience(self, env: CityEnvironment, num_steps: int) -> List[Dict]:
        """
        收集经验数据
        
        Args:
            env: 城市环境
            num_steps: 收集步数
            
        Returns:
            经验列表
        """
        all_experiences = []
        steps_collected = 0
        
        while steps_collected < num_steps:
            # 重置环境
            state = env.reset()
            
            episode_experiences = []
            done = False
            
            while not done and steps_collected < num_steps:
                # 获取当前智能体
                current_agent = env.current_agent
                
                # 获取可用动作
                actions, action_feats, mask = env.get_action_pool(current_agent)
                
                if not actions:
                    print(f"    No available actions, ending episode")
                    break
                
                # 使用RL选择器选择动作序列
                _, selected_sequence = self.selector.choose_action_sequence(
                    slots=env.slots,
                    candidates=set(a.footprint_slots[0] for a in actions if a.footprint_slots),
                    occupied=env._get_occupied_slots(),
                    lp_provider=env._create_lp_provider(),
                    agent_types=[current_agent],
                    sizes={current_agent: ['S', 'M', 'L']}
                )
                
                if selected_sequence is None:
                    print(f"    No sequence selected, ending episode")
                    break
                
                # 过滤不允许的动作
                if selected_sequence and selected_sequence.actions:
                    filtered_actions = [a for a in selected_sequence.actions if env.action_allowed(a)]
                    if not filtered_actions:
                        print(f"    All actions filtered out, ending episode")
                        break
                    
                    from logic.v4_enumeration import Sequence
                    # 保存原始的action_index
                    original_action_index = getattr(selected_sequence, 'action_index', -1)
                    
                    # 安全地计算序列属性
                    sum_cost = sum(getattr(a, 'cost', 0.0) for a in filtered_actions)
                    sum_reward = sum(getattr(a, 'reward', 0.0) for a in filtered_actions)
                    sum_prestige = sum(getattr(a, 'prestige', 0.0) for a in filtered_actions)
                    total_score = sum(getattr(a, 'score', 0.0) for a in filtered_actions)
                    
                    selected_sequence = Sequence(
                        actions=filtered_actions,
                        sum_cost=sum_cost,
                        sum_reward=sum_reward,
                        sum_prestige=sum_prestige,
                        score=total_score
                    )
                    # 恢复action_index属性
                    selected_sequence.action_index = original_action_index
                
                # 记录旧策略的动作概率（传入真实动作数量）
                old_log_prob = self._get_action_log_prob(selected_sequence, state, len(actions))
                
                # 执行动作并收集经验
                experience = {
                    'state': state.copy(),
                    'action': selected_sequence,
                    'agent': current_agent,
                    'month': env.current_month,
                    'old_log_prob': old_log_prob,  # 记录旧策略概率
                    'num_actions': len(actions)  # 保存动作数量，确保更新时一致
                }
                
                next_state, reward, done, info = env.step(current_agent, selected_sequence)
                
                experience.update({
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done,
                    'info': info
                })
                
                episode_experiences.append(experience)
                all_experiences.append(experience)
                
                # 更新状态
                state = next_state
                steps_collected += 1
        
        print(f"收集了 {len(all_experiences)} 步经验")
        return all_experiences
    
    def _get_action_log_prob(self, sequence, state, num_actions):
        """获取动作的对数概率
        
        参数:
            sequence: 选择的动作序列
            state: 当前状态
            num_actions: 可用动作的总数（必须与选择时一致）
        """
        action_idx = getattr(sequence, 'action_index', -1)
        if action_idx < 0:
            return torch.tensor(0.0)
        
        # 编码状态
        if sequence.actions:
            first_action = sequence.actions[0]
            state_embed = self.selector._encode_state_for_rl([first_action])
        else:
            state_embed = self.selector._encode_state_for_rl([])
        
        # 获取网络输出
        with torch.no_grad():
            logits = self.selector.actor(state_embed)
            value = self.selector.critic(state_embed)
        
        # 计算动作概率 - 使用真实的动作数量（不再硬编码5）
        num_actions = min(num_actions, self.selector.max_actions)
        valid_logits = logits[0, :num_actions]
        valid_action_idx = min(action_idx, num_actions - 1)
        
        dist = torch.distributions.Categorical(logits=valid_logits.unsqueeze(0))
        log_prob = dist.log_prob(torch.tensor(valid_action_idx).to(self.device))
        
        return log_prob
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool], next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计(GAE) - 参考ppo_grid_nav_agent.py实现
        
        Args:
            rewards: 奖励列表
            values: 价值估计列表
            dones: 终止标志列表
            next_value: 下一个状态的价值
            
        Returns:
            (advantages, returns) 优势函数和回报张量
        """
        advantages = []
        returns = []
        
        # 计算GAE（参考现有实现）
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[i + 1]
            
            # 计算时序差分误差
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]
            
            # 累积GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        # 转换为张量
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def _compute_values(self, experiences: List[Dict]) -> List[float]:
        """
        计算状态价值估计
        
        Args:
            experiences: 经验列表
            
        Returns:
            价值估计列表
        """
        values = []
        
        for exp in experiences:
            state = exp['state']
            sequence = exp['action']  # 这是Sequence对象
            
            # 使用RL选择器的价值网络
            if self.selector.actor is not None:
                # 【MAPPO】获取该经验对应的agent
                agent = exp.get('agent', 'IND')
                if not agent or agent not in self.selector.critics:
                    # 从action推断
                    first_action = sequence.actions[0] if sequence.actions else None
                    agent = first_action.agent if first_action and hasattr(first_action, 'agent') else 'IND'
                
                # 选择该agent的Critic
                critic = self.selector.critics.get(agent, self.selector.critic)
                
                # 编码状态（使用序列中的第一个动作）
                first_action = sequence.actions[0] if sequence.actions else None
                state_embed = self.selector._encode_state_for_rl([first_action]) if first_action else self.selector._encode_state_for_rl([])
                
                # 使用该agent的Critic获取价值估计
                with torch.no_grad():
                    value = critic(state_embed)
                    values.append(value.item())
            else:
                # 回退到奖励估计
                values.append(exp['reward'])
        
        return values
    
    def update_policy(self, experiences: List[Dict]) -> Dict[str, float]:
        """
        更新策略网络 - PPO-Clip算法（MAPPO：分agent更新）
        
        Args:
            experiences: 经验列表
            
        Returns:
            损失统计字典
        """
        if not experiences:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0, 'total_loss': 0.0}
        
        # 【MAPPO】按agent分组经验
        agent_experiences = {}
        for exp in experiences:
            agent = exp.get('agent', 'IND')  # 从经验中获取agent
            if agent not in agent_experiences:
                agent_experiences[agent] = []
            agent_experiences[agent].append(exp)
        
        # 如果经验中没有agent信息，用传统方式（向后兼容）
        if not agent_experiences or all(len(exps)==0 for exps in agent_experiences.values()):
            # 尝试从action中推断agent
            for exp in experiences:
                action = exp['action']
                if action and action.actions:
                    agent = action.actions[0].agent if hasattr(action.actions[0], 'agent') else 'IND'
                else:
                    agent = 'IND'
                if agent not in agent_experiences:
                    agent_experiences[agent] = []
                agent_experiences[agent].append(exp)
        
        # 调试信息
        for agent, exps in agent_experiences.items():
            if exps:
                agent_rewards = [exp['reward'] for exp in exps]
                print(f"    [{agent}] 经验数: {len(exps)}, 奖励: min={min(agent_rewards):.3f}, max={max(agent_rewards):.3f}, mean={np.mean(agent_rewards):.3f}")
        
        # 计算价值估计（使用价值网络）
        values = self._compute_values(experiences)
        
        # 计算GAE
        returns, advantages = self.compute_gae(rewards, values, dones)
        
        # 实现PPO-Clip更新
        print(f"PPO-Clip更新开始 - 经验数: {len(experiences)}, 优势范围: [{advantages.min():.3f}, {advantages.max():.3f}]")
        
        # 初始化损失统计
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divergences = []
        clip_fractions = []
        
        # 多轮更新
        for epoch in range(self.num_epochs):
            # 获取当前策略的动作概率和价值
            current_log_probs = []
            current_values = []
            current_entropies = []
            
            for i, exp in enumerate(experiences):
                state = exp['state']
                sequence = exp['action']  # 这是Sequence对象
                
                # 【MAPPO】获取该经验对应的agent
                agent = exp.get('agent', 'IND')
                if not agent or agent not in self.selector.actors:
                    # 尝试从action推断
                    first_action = sequence.actions[0] if sequence.actions else None
                    agent = first_action.agent if first_action and hasattr(first_action, 'agent') else 'IND'
                
                # 选择该agent的网络
                actor = self.selector.actors.get(agent, self.selector.actor)
                critic = self.selector.critics.get(agent, self.selector.critic)
                
                # 编码状态（使用序列中的第一个动作）
                first_action = sequence.actions[0] if sequence.actions else None
                state_embed = self.selector._encode_state_for_rl([first_action]) if first_action else self.selector._encode_state_for_rl([])
                
                # 使用该agent的网络获取输出
                logits = actor(state_embed)
                value = critic(state_embed)
                
                # 使用真正的概率分布计算动作概率
                action_idx = getattr(sequence, 'action_index', -1)
                if action_idx >= 0:
                    # 使用收集时保存的动作数量，确保与old_log_prob计算时一致
                    num_actions = exp.get('num_actions', len(exp.get('available_actions', [])))
                    if num_actions == 0 or num_actions is None:
                        num_actions = 5  # 回退到默认值
                    
                    num_actions = min(num_actions, self.selector.max_actions)
                    valid_logits = logits[0, :num_actions]
                    valid_action_idx = min(action_idx, num_actions - 1)
                    
                    # 创建动作分布
                    dist = torch.distributions.Categorical(logits=valid_logits.unsqueeze(0))
                    log_prob = dist.log_prob(torch.tensor(valid_action_idx).to(self.device))
                    entropy = dist.entropy()
                else:
                    # 回退到简化计算
                    action_score = sequence.score if hasattr(sequence, 'score') else 1.0
                    log_prob = torch.log(torch.clamp(torch.tensor(action_score), min=1e-8))
                    entropy = -log_prob * torch.exp(log_prob)
                
                current_log_probs.append(log_prob)
                current_values.append(value)
                current_entropies.append(entropy)
            
            # 转换为张量
            current_log_probs = torch.stack(current_log_probs).to(self.device)
            current_values = torch.stack(current_values).squeeze().to(self.device)
            current_entropies = torch.stack(current_entropies).to(self.device)
            
            # 获取旧策略的动作概率（从经验中读取）
            old_log_probs = torch.stack([exp['old_log_prob'] for exp in experiences]).to(self.device)
            
            # 计算策略比率
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # 计算裁剪损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = F.mse_loss(current_values, returns)
            
            # 计算熵损失（鼓励探索）
            entropy = current_entropies.mean()
            
            # 总损失
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss - 
                         self.entropy_coef * entropy)
            
            # 【MAPPO】分别更新各agent的Actor和Critic网络
            # 1. 更新所有agent的Actor（策略网络）
            actor_loss = policy_loss - self.entropy_coef * entropy
            for agent in self.selector.actor_optimizers.keys():
                optimizer = self.selector.actor_optimizers[agent]
                actor_net = self.selector.actors[agent]
                
                optimizer.zero_grad()
            
            actor_loss.backward(retain_graph=True)
            
            for agent in self.selector.actor_optimizers.keys():
                optimizer = self.selector.actor_optimizers[agent]
                actor_net = self.selector.actors[agent]
                
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), self.max_grad_norm)
                optimizer.step()
            
            # 2. 更新所有agent的Critic（价值网络）
            for agent in self.selector.critic_optimizers.keys():
                optimizer = self.selector.critic_optimizers[agent]
                optimizer.zero_grad()
            
            value_loss.backward()
            
            for agent in self.selector.critic_optimizers.keys():
                optimizer = self.selector.critic_optimizers[agent]
                critic_net = self.selector.critics[agent]
                
                torch.nn.utils.clip_grad_norm_(critic_net.parameters(), self.max_grad_norm)
                optimizer.step()
            
            # 计算KL散度和裁剪比例
            with torch.no_grad():
                # 修复KL散度计算：使用正确的近似公式
                # KL(old||new) ≈ E[(ratio - 1) - log(ratio)]
                kl_div = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
            
            # 记录损失
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())  # 记录熵值而不是熵损失
            kl_divergences.append(kl_div.item())
            clip_fractions.append(clip_fraction.item())
            
            print(f"  Epoch {epoch+1}/{self.num_epochs}: "
                  f"policy_loss={policy_loss.item():.4f}, "
                  f"value_loss={value_loss.item():.4f}, "
                  f"entropy={entropy.item():.4f}, "
                  f"kl_div={kl_div.item():.4f}")
        
        # 更新训练统计
        self.training_stats['policy_losses'].extend(policy_losses)
        self.training_stats['value_losses'].extend(value_losses)
        self.training_stats['entropy_losses'].extend(entropy_losses)
        self.training_stats['kl_divergences'].extend(kl_divergences)
        self.training_stats['clip_fractions'].extend(clip_fractions)
        
        # 返回平均损失
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),  # 返回熵值
            'total_loss': np.mean(policy_losses) + self.value_loss_coef * np.mean(value_losses) - self.entropy_coef * np.mean(entropy_losses),
            'kl_divergence': np.mean(kl_divergences),
            'clip_fraction': np.mean(clip_fractions)
        }
    
    def generate_config_hash(self, cfg: Dict) -> str:
        """生成配置的哈希值，用于唯一标识实验"""
        rl_config = cfg.get('growth_v4_1', {}).get('solver', {}).get('rl', {})
        config_str = json.dumps(rl_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def save_training_state(self, path: str, training_step: int):
        """保存训练状态（优化器、步数等）"""
        training_state = {
            'actor_optimizer': self.selector.actor_optimizer.state_dict(),
            'critic_optimizer': self.selector.critic_optimizer.state_dict(),
            'training_step': training_step,
            'hyperparameters': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_ratio': self.clip_ratio,
                'learning_rate': self.lr,
                'value_loss_coef': self.value_loss_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'num_epochs': self.num_epochs,
            },
            'timestamp': datetime.datetime.now().isoformat(),
            'training_stats': self.training_stats
        }
        torch.save(training_state, path)
        print(f"训练状态已保存到: {path}")
    
    def load_training_state(self, path: str):
        """加载训练状态"""
        training_state = torch.load(path, map_location=self.device)
        self.selector.actor_optimizer.load_state_dict(training_state['actor_optimizer'])
        self.selector.critic_optimizer.load_state_dict(training_state['critic_optimizer'])
        self.training_stats = training_state.get('training_stats', self.training_stats)
        print(f"训练状态已从 {path} 加载")
        return training_state
    
    def save_model_with_versioning(self, base_path: str, update: int, cfg: Dict, is_final: bool = False):
        """带版本控制的模型保存"""
        rl_cfg = cfg.get('growth_v4_1', {}).get('solver', {}).get('rl', {})
        experiment_name = rl_cfg.get('experiment_name', 'default')
        config_hash = self.generate_config_hash(cfg)
        
        # 确保保存目录存在
        os.makedirs(base_path, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if is_final:
            model_filename = f"final_model_{experiment_name}_{config_hash}_{timestamp}.pth"
            training_state_filename = f"final_training_state_{experiment_name}_{config_hash}_{timestamp}.pth"
        else:
            model_filename = f"model_{experiment_name}_{config_hash}_{timestamp}_update_{update}.pth"
            training_state_filename = f"training_state_{experiment_name}_{config_hash}_{timestamp}_update_{update}.pth"
        
        model_path = os.path.join(base_path, model_filename)
        training_state_path = os.path.join(base_path, training_state_filename)
        
        # 保存模型权重
        self.selector.save_model(model_path)
        # 保存训练状态
        self.save_training_state(training_state_path, update)
        
        print(f"模型已保存: {model_path}")
        print(f"训练状态已保存: {training_state_path}")
        
        # 清理旧模型（可选）
        keep_count = rl_cfg.get('keep_last_n_models', 5)
        self.cleanup_old_models(base_path, keep_count)
        
        return model_path, training_state_path
    
    def cleanup_old_models(self, base_path: str, keep_count: int):
        """清理旧的模型文件，只保留最近的N个"""
        if keep_count <= 0:
            return
            
        # 查找所有模型文件
        model_pattern = os.path.join(base_path, "model_*_update_*.pth")
        model_files = glob.glob(model_pattern)
        
        if len(model_files) <= keep_count:
            return
        
        # 按修改时间排序
        model_files.sort(key=os.path.getmtime, reverse=True)
        
        # 删除超出保留数量的文件
        for old_file in model_files[keep_count:]:
            try:
                os.remove(old_file)
                # 同时删除对应的训练状态文件
                training_state_file = old_file.replace('model_', 'training_state_')
                if os.path.exists(training_state_file):
                    os.remove(training_state_file)
                print(f"已删除旧模型: {old_file}")
            except OSError as e:
                print(f"删除文件失败 {old_file}: {e}")
    
    def save_model(self, path: str):
        """保存模型（向后兼容）"""
        self.selector.save_model(path)
        print(f"PPO模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型（向后兼容）"""
        self.selector.load_model(path)
        print(f"PPO模型已从 {path} 加载")
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """获取训练统计信息"""
        return self.training_stats.copy()
