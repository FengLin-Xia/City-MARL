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
from enhanced_training_logger import get_training_logger

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
        # 尝试从两个位置读取RL配置
        self.rl_cfg = cfg.get('solver', {}).get('rl', {})
        if not self.rl_cfg:
            self.rl_cfg = cfg.get('growth_v4_1', {}).get('solver', {}).get('rl', {})
        
        # PPO超参数
        self.gamma = self.rl_cfg.get('gamma', 0.99)  # 折扣因子
        self.gae_lambda = self.rl_cfg.get('gae_lambda', 0.95)  # GAE参数
        self.clip_ratio = self.rl_cfg.get('clip_ratio', self.rl_cfg.get('clip_eps', 0.2))  # PPO裁剪率
        self.lr = self.rl_cfg.get('lr', 3e-4)  # 学习率
        self.value_loss_coef = self.rl_cfg.get('value_loss_coef', 0.5)  # 价值损失系数
        self.entropy_coef = self.rl_cfg.get('entropy_coef', 0.01)  # 熵损失系数
        self.max_grad_norm = self.rl_cfg.get('max_grad_norm', 1.0)  # 梯度裁剪
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
        
        # 按照1013-5.md建议：重新初始化actor网络最后一层
        self._reinitialize_actor_last_layers()
    
    def _reinitialize_actor_last_layers(self):
        """按照1013-5.md建议：重新初始化actor网络最后一层"""
        print("重新初始化actor网络最后一层...")
        for agent, actor in self.selector.actors.items():
            # 获取最后一层（输出层）
            last_layer = actor.network[-1]
            # 按照1013-9.md建议：重初始化最后一层（提高gain）
            torch.nn.init.orthogonal_(last_layer.weight, gain=0.5)
            torch.nn.init.zeros_(last_layer.bias)
            print(f"  {agent} actor最后一层已重新初始化")
    
    def set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    

    def _adaptive_kl_adjustment(self, kl_after: float):
        """自适应KL调整（按照1013-7.md建议）"""
        target_kl = 0.02
        
        if kl_after < 0.2 * target_kl:  # 太保守
            # 增大学习率
            for agent in self.selector.actor_optimizers.keys():
                optimizer = self.selector.actor_optimizers[agent]
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 1.5
            print(f"[adaptive] KL too low ({kl_after:.4f} < {0.2 * target_kl:.4f}), increased lr")
            
        elif kl_after > 2.0 * target_kl:  # 太猛
            # 减小学习率
            for agent in self.selector.actor_optimizers.keys():
                optimizer = self.selector.actor_optimizers[agent]
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            print(f"[adaptive] KL too high ({kl_after:.4f} > {2.0 * target_kl:.4f}), decreased lr")

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
                    sizes={'EDU': ['S', 'M', 'L', 'A', 'B', 'C'], 'IND': ['S', 'M', 'L']}
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
                
                # 优先使用采样时保存的log_prob，确保一致性
                if hasattr(selected_sequence, 'old_log_prob') and selected_sequence.old_log_prob is not None:
                    old_log_prob = selected_sequence.old_log_prob
                else:
                    # 回退到重新计算（兼容性）
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
        
        # 注释掉标准化，直接使用原始advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # === 按照1013-9.md建议：临时放大advantages来"点燃"训练 ===
        advantages = 2.0 * advantages
        
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
        
        # 提取数据（修复：重新添加这些定义）
        rewards = [exp['reward'] for exp in experiences]
        dones = [exp['done'] for exp in experiences]
        
        # 计算价值估计（使用价值网络）
        values = self._compute_values(experiences)
        
        # 实现PPO-Clip更新
        print(f"PPO-Clip更新开始 - 经验数: {len(experiences)}")
        
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
                # —— 取缓存（与采样时一致）——
                state_embed = exp.get('state_embed', None)
                if state_embed is None:
                    # 兼容旧数据的兜底（不推荐）
                    sequence = exp['action']
                    first_action = sequence.actions[0] if sequence and sequence.actions else None
                    state_embed = self.selector._encode_state_for_rl([first_action]) if first_action else self.selector._encode_state_for_rl([])
                else:
                    # 从list转换为tensor
                    state_embed = torch.tensor(state_embed, device=self.device, dtype=torch.float32).unsqueeze(0)

                # 选择 agent 对应的网络
                agent = exp.get('agent', 'IND')
                actor = self.selector.actors.get(agent, self.selector.actor)
                critic = self.selector.critics.get(agent, self.selector.critic)

                # === 输入归一化（按照1013-6.md建议） ===
                # 处理embed.std≈52的问题，避免前层饱和
                if i == 0:  # 只在第一个样本计算running stats
                    if not hasattr(self, '_embed_mean'):
                        self._embed_mean = state_embed.mean()
                        self._embed_std = state_embed.std()
                    else:
                        # 更新running stats
                        alpha = 0.1
                        self._embed_mean = alpha * state_embed.mean() + (1-alpha) * self._embed_mean
                        self._embed_std = alpha * state_embed.std() + (1-alpha) * self._embed_std
                
                # 归一化state_embed
                state_embed_normalized = (state_embed - self._embed_mean) / (self._embed_std + 1e-5)
                state_embed_normalized = state_embed_normalized.clamp(-5, 5)
                
                # === BEFORE 测量 ===
                logits_before = actor(state_embed_normalized)     # [1, A]
                value  = critic(state_embed_normalized)
                
                # 诊断关键指标
                if i == 0:  # 只在第一个样本打印
                    std_embed = state_embed.std(dim=-1).mean().item()
                    print(f"[probe] embed.std={std_embed:.4g}")
                if value.dim() > 1: value = value.squeeze()
                if value.dim() == 0: value = value.unsqueeze(0)

                # —— 重放"局部分布"与"局部索引" —— 
                action_idx = int(exp.get('action_index', -1))
                num_actions = exp.get('num_actions', None)
                subset_ids = exp.get('subset_indices', None)  # 可能不存在

                # old_logp 也从 exp 读
                old_logp = exp.get('old_log_prob', None)

                # 统一成张量/基本类型
                if isinstance(subset_ids, list):
                    subset_ids = torch.tensor(subset_ids, device=self.device, dtype=torch.long)
                if old_logp is not None and not torch.is_tensor(old_logp):
                    old_logp = torch.tensor([float(old_logp)], device=self.device, dtype=torch.float32)

                # —— 有效性判断（subset 可选）——
                local_logits = None
                is_valid = True
                
                if num_actions is None or action_idx < 0 or old_logp is None:
                    is_valid = False
                else:
                    if subset_ids is None:
                        # 没有 subset_ids：使用"前K"切片（与采样一致的前K逻辑）
                        k = int(num_actions)
                        if not (0 <= action_idx < k):
                            is_valid = False
                        else:
                            local_logits = logits_before[0, :k]
                    else:
                        # 有 subset_ids：严格按保存顺序切局部 logits
                        k = subset_ids.numel()
                        if (num_actions is not None) and (k != int(num_actions)):
                            is_valid = False
                        elif not (0 <= action_idx < k):
                            is_valid = False
                        else:
                            local_logits = logits_before[0, subset_ids]

                if not is_valid or local_logits is None:
                    log_prob = torch.tensor([float('-inf')], device=self.device)
                    entropy = torch.tensor([0.0], device=self.device)
                else:
                    # 检查logits是否包含NaN或异常值
                    if torch.isnan(local_logits).any() or torch.isinf(local_logits).any():
                        print(f"Warning: Invalid logits detected (NaN or Inf), using fallback")
                        log_prob = torch.tensor([float('-inf')], device=self.device)
                        entropy = torch.tensor([0.0], device=self.device)
                    else:
                        # === 测量BEFORE指标 ===
                        if i == 0:  # 只在第一个样本打印
                            std_logits = local_logits.std().item()
                            action_idx_tensor = torch.tensor(action_idx, device=self.device)
                            dist_before = torch.distributions.Categorical(logits=local_logits)
                            newlp_before = dist_before.log_prob(action_idx_tensor)
                            kl_before = (old_logp - newlp_before).mean().item()
                            
                            # 存储before指标用于后续比较
                            self._before_metrics = {
                                'std_logits': std_logits,
                                'kl_before': kl_before,
                                'local_logits_before': local_logits.clone()
                            }
                            
                            print(f"[probe] embed.std={std_embed:.4g} | loc.std={std_logits:.4g} | KL_before={kl_before:.3g}")
                        
                        # === 按照1013-8.md建议的诊断 ===
                        print("[chk] local_logits.std(before) =", local_logits.std().item())
                        
                        # 温度缩放（如果你要用）
                        tau = 0.5  # 先 0.5，看到KL动就可以再调
                        local_logits = local_logits / tau
                        print("[chk] tau =", tau, " local_logits.std(after) =", local_logits.std().item())
                        
                        dist = torch.distributions.Categorical(logits=local_logits)
                        log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device))
                        entropy = dist.entropy()
                        if log_prob.dim() == 0: log_prob = log_prob.unsqueeze(0)
                        if entropy.dim() == 0: entropy = entropy.unsqueeze(0)
                
                current_log_probs.append(log_prob)
                current_values.append(value)
                current_entropies.append(entropy)
            
            # 转换为张量，确保所有张量形状一致
            if current_log_probs:
                # 确保所有log_prob张量都是1维
                for i, log_prob in enumerate(current_log_probs):
                    if log_prob.dim() == 0:
                        current_log_probs[i] = log_prob.unsqueeze(0)
                    elif log_prob.dim() > 1:
                        current_log_probs[i] = log_prob.squeeze()
                
                # 确保所有value张量都是1维
                for i, val in enumerate(current_values):
                    if val.dim() == 0:
                        current_values[i] = val.unsqueeze(0)
                    elif val.dim() > 1:
                        current_values[i] = val.squeeze()
                
                # 确保所有entropy张量都是1维
                for i, entropy in enumerate(current_entropies):
                    if entropy.dim() == 0:
                        current_entropies[i] = entropy.unsqueeze(0)
                    elif entropy.dim() > 1:
                        current_entropies[i] = entropy.squeeze()
                
                current_log_probs = torch.stack(current_log_probs).to(self.device)
                current_values = torch.stack(current_values).to(self.device)
                current_entropies = torch.stack(current_entropies).to(self.device)
            else:
                print("Warning: No log_probs to stack")
                continue
            
            # 聚合 old_log_prob 的判定（放宽subset要求）
            old_log_probs_list, valid_flags = [], []
            for exp in experiences:
                olp = exp.get('old_log_prob', None)
                aid = exp.get('action_index', -1)
                k   = exp.get('num_actions', None)

                # subset 可缺省
                is_valid = (olp is not None) and (aid is not None) and (int(aid) >= 0) and (k is not None)
                valid_flags.append(is_valid)
                if is_valid:
                    if not torch.is_tensor(olp):
                        olp = torch.tensor([float(olp)], device=self.device, dtype=torch.float32)
                    elif olp.dim() == 0:
                        olp = olp.unsqueeze(0)
                    elif olp.dim() > 1:
                        olp = olp.squeeze()
                    old_log_probs_list.append(olp)

            valid_mask = torch.tensor(valid_flags, device=self.device, dtype=torch.bool)
            if valid_mask.float().mean().item() < 0.5:          # 先放宽阈值，避免全跳过
                print(f"[skip] valid_ratio={valid_mask.float().mean().item():.2f}, skip this mini-batch")
                continue

            old_log_probs = torch.stack(old_log_probs_list).to(self.device)
            
            # 过滤无效样本
            current_log_probs = current_log_probs[valid_mask]
            current_values = current_values[valid_mask]
            current_entropies = current_entropies[valid_mask]
            
            # 为过滤后的样本重新计算advantages和returns
            valid_rewards = []
            valid_values_orig = []
            valid_dones = []
            
            for i, is_valid in enumerate(valid_mask):
                if is_valid:
                    valid_rewards.append(rewards[i])
                    valid_values_orig.append(values[i])
                    valid_dones.append(dones[i])
            
            # 计算GAE
            returns, advantages = self.compute_gae(valid_rewards, valid_values_orig, valid_dones)
            
            # Advantage归一化 - 防止空张量/常数张量归出NaN
            if advantages.numel() == 0:
                continue
            # 注释掉标准化，直接使用原始advantages
            # std = advantages.std(unbiased=False)
            # if torch.isnan(std) or std < 1e-8:
            #     std = torch.tensor(1.0, device=advantages.device)
            # advantages = (advantages - advantages.mean()) / std
            advantages = 2.0 * advantages  # 临时放大验证链路
            advantages = torch.clamp(advantages, -10.0, 10.0)
            
            print(f"Valid samples: {valid_mask.sum().item()}/{len(valid_mask)}")
            
            # 添加调试信息
            print("advantages.mean()", advantages.mean().item())
            print("advantages.std()", advantages.std().item())
            
            # 计算策略比率
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # 添加关键调试信息
            print(f"[debug] current_log_probs range: [{current_log_probs.min().item():.6f}, {current_log_probs.max().item():.6f}]")
            print(f"[debug] old_log_probs range: [{old_log_probs.min().item():.6f}, {old_log_probs.max().item():.6f}]")
            print(f"[debug] log_prob_diff range: [{(current_log_probs - old_log_probs).min().item():.6f}, {(current_log_probs - old_log_probs).max().item():.6f}]")
            print(f"[debug] ratio range: [{ratio.min().item():.6f}, {ratio.max().item():.6f}]")
            
            # 计算裁剪损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 添加调试信息
            print("policy_loss", policy_loss.item() if torch.is_tensor(policy_loss) else policy_loss)
            
            # 计算KL散度和裁剪比例
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
                
                # === 按照1013-9.md建议：不被正负抵消的KL估计 ===
                log_ratio = current_log_probs - old_log_probs
                approx_kl_sym = 0.5 * ((current_log_probs - old_log_probs).pow(2)).mean()
                approx_kl_quad = 0.5 * (log_ratio.pow(2)).mean()
                
                print(f"[kl_fix] approx_kl_sym={approx_kl_sym.item():.6f}, approx_kl_quad={approx_kl_quad.item():.6f}")
            
            # 计算价值损失
            value_loss = F.mse_loss(current_values, returns)
            
            # 计算熵损失（鼓励探索）
            entropy = current_entropies.mean()
            
            # 总损失
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss - 
                         self.entropy_coef * entropy)
            
            # 详细统计信息打印（按照文档建议的格式）
            with torch.no_grad():
                print(f"  Epoch {epoch+1} 详细统计:")
                print(f"    ratio.mean(): {ratio.mean().item():.6f}, ratio.std(): {ratio.std().item():.6f}")
                print(f"    clip_fraction: {clip_fraction.item():.4f}, approx_kl: {approx_kl.item():.6f}")
                print(f"    entropy.mean(): {entropy.item():.6f}")
                
                # 按照文档建议的调试信息格式
                print("old_logp[:5]", old_log_probs[:5].detach().cpu().numpy())
                print("new_logp[:5]", current_log_probs[:5].detach().cpu().numpy())
                print("diff[:5]", (current_log_probs - old_log_probs)[:5].detach().cpu().numpy())
                print("ratio[:5]", ratio[:5].detach().cpu().numpy())
                print("approx_kl", approx_kl.item())
                print("entropy", entropy.item())
                print("valid_ratio", f"{valid_mask.sum().item()}/{len(valid_mask)}")
                
                # 健康检查断言（按照文档建议）
                ratio_mean = ratio.mean().item()
                approx_kl_val = approx_kl.item()
                
                # 断言
                assert torch.isfinite(current_log_probs).all() and torch.isfinite(old_log_probs).all()
                assert current_log_probs.max() <= 0.0 and old_log_probs.max() <= 0.0, "log_prob 应 ≤ 0"
                assert approx_kl < 0.2, f"KL too big {approx_kl.item():.3f}"
                assert valid_mask.float().mean().item() > 0.5, "有效样本比例过低"
                
                # 检查ratio是否在健康范围
                if abs(ratio_mean - 1.0) > 0.5:
                    print(f"Warning: ratio.mean()={ratio_mean:.3f} 偏离1.0太远，可能存在分布不一致")
                
                # 检查KL散度是否过高
                if approx_kl_val > 0.1:
                    print(f"Warning: approx_kl={approx_kl_val:.3f} 过高，可能存在策略更新步长过大")
                
                # 检查entropy是否固定不变
                if abs(entropy.item() - 1.60942) < 1e-5:  # log(5) ≈ 1.60942
                    print(f"Warning: entropy={entropy.item():.5f} 接近log(5)，可能存在分布问题")
            
            # === AFTER 测量 ===
            with torch.no_grad():
                # 重新计算logits after更新
                first_exp = experiences[0]
                first_state_embed = first_exp.get('state_embed', None)
                if first_state_embed is not None and len(first_state_embed) > 0 and hasattr(self, '_before_metrics'):
                    first_agent = first_exp.get('agent', 'IND')
                    first_actor = self.selector.actors.get(first_agent, self.selector.actor)
                    
                    # 使用相同的归一化
                    first_state_embed_normalized = (first_state_embed - self._embed_mean) / (self._embed_std + 1e-5)
                    first_state_embed_normalized = first_state_embed_normalized.clamp(-5, 5)
                    
                    logits_after = first_actor(first_state_embed_normalized)
                    
                    # 计算local_logits after
                    first_action_idx = int(first_exp.get('action_index', -1))
                    first_num_actions = first_exp.get('num_actions', None)
                    first_subset_ids = first_exp.get('subset_indices', None)
                    
                    if first_subset_ids is None:
                        local_logits_after = logits_after[0, :first_num_actions]
                    else:
                        if isinstance(first_subset_ids, list):
                            first_subset_ids = torch.tensor(first_subset_ids, device=self.device, dtype=torch.long)
                        local_logits_after = logits_after[0, first_subset_ids]
                    
                    # 应用温度缩放
                    tau = 0.25
                    local_logits_after = local_logits_after / tau
                    
                    # 计算KL after
                    action_idx_tensor = torch.tensor(first_action_idx, device=self.device)
                    dist_after = torch.distributions.Categorical(logits=local_logits_after)
                    newlp_after = dist_after.log_prob(action_idx_tensor)
                    
                    first_old_logp = first_exp.get('old_log_prob', None)
                    if first_old_logp is not None and not torch.is_tensor(first_old_logp):
                        first_old_logp = torch.tensor([float(first_old_logp)], device=self.device, dtype=torch.float32)
                    
                    kl_after = (first_old_logp - newlp_after).mean().item()
                    
                    # 计算logits变化
                    local_logits_before = self._before_metrics['local_logits_before']
                    dL2 = (local_logits_after - local_logits_before).pow(2).mean().sqrt().item()
                    
                    # 获取before指标
                    kl_before = self._before_metrics['kl_before']
                    
                    print(f"[probe] Δloc_L2={dL2:.4g} | KL_before={kl_before:.3g} | KL_after={kl_after:.3g}")
                    
                    # === 按照1013-8.md建议的AFTER KL诊断 ===
                    print("[chk] KL_after =", kl_after)
                    
                    # === 按照1013-8.md建议：手动扰动参数测试 ===
                    print("[probe] 手动扰动参数测试...")
                    old_params = [p.detach().clone() for p in first_actor.parameters()]
                    with torch.no_grad():
                        for p in first_actor.parameters():
                            p.add_(0.01 * torch.randn_like(p))
                    
                    # 重新前向→算 KL_after
                    with torch.no_grad():
                        logits_jit = first_actor(first_state_embed_normalized)
                        if first_subset_ids is None:
                            local_jit = logits_jit[0, :first_num_actions]
                        else:
                            local_jit = logits_jit[0, first_subset_ids]
                        local_jit = local_jit / tau
                        dist_jit = torch.distributions.Categorical(logits=local_jit)
                        kl_jit = (first_old_logp - dist_jit.log_prob(action_idx_tensor)).mean().item()
                    print("[probe] KL after param jitter =", kl_jit)
                    
                    # 还原
                    for p, q in zip(first_actor.parameters(), old_params):
                        p.copy_(q)
                    
                    # === 按照1013-9.md建议：校验最后一层真的在更新 ===
                    print("[probe] 校验最后一层更新...")
                    with torch.no_grad():
                        local1 = first_actor(first_state_embed_normalized)[0, :first_num_actions]
                    
                    # 这里应该是在step之后，但我们先记录local1，在step后再比较
                    # 暂时跳过，因为我们需要在step前后对比
                    
                    # 自适应KL调整
                    self._adaptive_kl_adjustment(kl_after)
            
            # 【MAPPO】分别更新各agent的Actor和Critic网络
            # 1. 更新所有agent的Actor（策略网络）
            actor_loss = policy_loss - self.entropy_coef * entropy
            for agent in self.selector.actor_optimizers.keys():
                optimizer = self.selector.actor_optimizers[agent]
                actor_net = self.selector.actors[agent]
                
                # === 按照1013-8.md建议：检查最后一层梯度 ===
                print(f"\n[grad] 检查 {agent} actor最后一层梯度:")
                for name, p in actor_net.named_parameters():
                    if "network.2" in name:  # 最后一层 (network.2是第三层，即输出层)
                        g = (p.grad.norm().item() if p.grad is not None else 0.0)
                        print(f"[grad] {name}: grad_norm={g:.3e}")
                
                # 记录step前的权重（只记录最后一层）
                last_params_before = {}
                for name, p in actor_net.named_parameters():
                    if "network.2" in name:  # 最后一层
                        last_params_before[name] = p.detach().clone()
                
                optimizer.zero_grad()
            
            actor_loss.backward(retain_graph=True)
            
            # 1) 梯度范数检查
            total_grad = 0.0
            nz = 0
            for agent in self.selector.actors.keys():
                actor_net = self.selector.actors[agent]
                for p in actor_net.parameters():
                    if p.grad is not None:
                        g = p.grad.data
                        total_grad += g.norm().item()
                        nz += (g.abs() > 0).sum().item()
            print(f"[dbg] actor grad_norm_sum={total_grad:.6f}, grad_nonzero={nz}")
            
            # 验证优化器 param group（从共享→MAPPO 常见漏绑）
            for agent in self.selector.actor_optimizers.keys():
                optimizer = self.selector.actor_optimizers[agent]
                actor_net = self.selector.actors[agent]
                
                names = set()
                for i, g in enumerate(optimizer.param_groups):
                    cnt = sum(p.numel() for p in g['params'])
                    print(f"[opt] {agent} group#{i} lr={g['lr']} params={cnt}")
                    for p in g['params']:
                        names.add(id(p))
                # 粗暴检查：actor 的任意一个参数 id 是否在 names 里
                example_param_id = id(next(actor_net.parameters()))
                print(f"[opt] {agent} actor params bound:", example_param_id in names)
            
            for agent in self.selector.actor_optimizers.keys():
                optimizer = self.selector.actor_optimizers[agent]
                actor_net = self.selector.actors[agent]
                
                # 2) 参数是否真的在变
                old_params = [p.detach().clone() for p in actor_net.parameters()]
                
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), self.max_grad_norm)
                optimizer.step()
                
                diff = sum((p.detach() - q).abs().sum().item() for p,q in zip(actor_net.parameters(), old_params))
                print(f"[dbg] {agent} actor param_diff={diff:.6f}")
                
                # === 按照1013-8.md建议：检查最后一层参数更新 ===
                diff_sum = 0.0
                for name, p in actor_net.named_parameters():
                    if name in last_params_before:
                        diff = (p.detach() - last_params_before[name]).abs().sum().item()
                        diff_sum += diff
                print(f"[upd] {agent} last_layer_param_diff={diff_sum:.6f}")
                
                # 清除记录，避免跨batch累积
                last_params_before.clear()
            
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
            
            # 记录详细的训练指标到增强记录器
            logger = get_training_logger()
            if logger:
                epoch_metrics = {
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy_loss': entropy.item(),
                    'total_loss': total_loss.item(),
                    'kl_divergence': kl_div.item(),
                    'clip_fraction': clip_fraction.item(),
                    'entropy': entropy.item(),
                    'advantages_mean': advantages.mean().item(),
                    'advantages_std': advantages.std().item(),
                    'ratio_mean': ratio.mean().item(),
                    'ratio_std': ratio.std().item()
                }
                logger.record_epoch(epoch, epoch_metrics)
            
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
