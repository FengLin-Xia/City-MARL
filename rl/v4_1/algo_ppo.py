"""
v4.1 PPO/MAPPO训练算法
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from .models import MultiAgentActorCritic, StateEncoder, ActionEncoder
from .buffers import MultiAgentRolloutBuffer
from .utils import (
    masked_sample, masked_log_prob, compute_gae_batch, 
    normalize_advantages, clip_grad_norm, categorical_entropy,
    set_seed, get_device
)


class PPOTrainer:
    """PPO训练器 - 单智能体版本"""
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.rl_cfg = cfg['solver']['rl']
        self.device = get_device()
        
        # 设置随机种子
        set_seed(self.rl_cfg['seed'])
        
        # 创建模型
        self._build_models()
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.rl_cfg['lr'])
        
        # 创建经验缓冲区
        self.buffer = MultiAgentRolloutBuffer(
            capacity=self.rl_cfg['rollout_steps'],
            agents=self.rl_cfg['agents'],
            state_dim=512,
            action_feat_dim=64,
            max_actions=1000,
            device=self.device
        )
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'kl_divergences': []
        }
    
    def _build_models(self):
        """构建模型"""
        # 状态编码器
        state_encoder = StateEncoder(
            grid_size=200,
            grid_channels=5,
            global_stats_dim=10,
            hidden_dim=256,
            output_dim=512
        )
        
        # 动作编码器
        action_encoder = ActionEncoder(
            action_feat_dim=20,
            hidden_dim=128,
            output_dim=64
        )
        
        # 多智能体Actor-Critic
        self.model = MultiAgentActorCritic(
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            agents=self.rl_cfg['agents'],
            state_dim=512,
            action_feat_dim=64,
            hidden_dim=256,
            max_actions=1000
        ).to(self.device)
    
    def collect_rollout(self, env) -> Dict:
        """收集一个rollout的经验"""
        self.buffer.clear()
        
        # 重置环境
        state = env.reset()
        done = False
        step_count = 0
        
        while step_count < self.rl_cfg['rollout_steps'] and not done:
            for agent in self.rl_cfg['agents']:
                # 获取动作池和特征
                actions, mask, feats = self._get_action_pool(state, agent)
                
                if len(actions) == 0:
                    continue
                
                # 编码状态和动作特征
                state_embed = self._encode_state(state)
                action_feats = self._encode_action_features(feats)
                
                # 获取策略和价值
                with torch.no_grad():
                    logits, value = self.model(
                        state_embed, state_embed, action_feats, agent
                    )
                    
                    # 采样动作
                    action_idx, log_prob = masked_sample(logits, mask)
                
                # 执行动作
                next_state, reward, done, info = env.step(agent, actions[action_idx])
                
                # 添加到缓冲区
                self.buffer.add(
                    agent=agent,
                    state=state_embed,
                    action_feats=action_feats,
                    action_mask=mask,
                    action=action_idx,
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                    done=done,
                    global_state=state_embed
                )
                
                state = next_state
                step_count += 1
                
                if done:
                    break
        
        return {
            'steps_collected': step_count,
            'buffer_size': len(self.buffer)
        }
    
    def update_policy(self) -> Dict:
        """更新策略"""
        update_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'kl_divergence': 0.0,
            'grad_norm': 0.0
        }
        
        # 计算优势值
        advantages = {}
        returns = {}
        
        for agent in self.rl_cfg['agents']:
            adv, ret = self.buffer.compute_advantages(
                agent=agent,
                gamma=self.rl_cfg['gamma'],
                gae_lambda=self.rl_cfg['gae_lambda'],
                normalize=True
            )
            advantages[agent] = adv
            returns[agent] = ret
        
        # PPO更新
        for epoch in range(self.rl_cfg['K_epochs']):
            for agent in self.rl_cfg['agents']:
                # 获取批量数据
                batch = self.buffer.get_batch(
                    agent=agent, 
                    batch_size=self.rl_cfg['mini_batch_size']
                )
                
                if len(batch) == 0:
                    continue
                
                # 重新计算策略
                logits, values = self.model(
                    batch['states'],
                    batch['states'],  # 单智能体时全局状态=局部状态
                    batch['action_feats'],
                    agent
                )
                
                # 计算新的对数概率
                new_log_probs = masked_log_prob(logits, batch['action_masks'], batch['actions'])
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch['log_probs'])
                
                # 计算优势（使用全局优势）
                global_advantages = advantages[agent][:len(batch['actions'])]
                global_advantages = global_advantages.to(self.device)
                
                # PPO损失
                surr1 = ratio * global_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.rl_cfg['clip_eps'], 
                    1 + self.rl_cfg['clip_eps']
                ) * global_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values.squeeze(-1), returns[agent][:len(batch['actions'])])
                
                # 熵损失
                entropy = categorical_entropy(logits, batch['action_masks']).mean()
                entropy_loss = -self.rl_cfg['ent_coef'] * entropy
                
                # 总损失
                total_loss = policy_loss + self.rl_cfg['vf_coef'] * value_loss + entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = clip_grad_norm(self.model, max_norm=0.5)
                self.optimizer.step()
                
                # 统计
                update_stats['policy_loss'] += policy_loss.item()
                update_stats['value_loss'] += value_loss.item()
                update_stats['entropy_loss'] += entropy_loss.item()
                update_stats['grad_norm'] += grad_norm
        
        # 平均统计
        num_epochs = self.rl_cfg['K_epochs'] * len(self.rl_cfg['agents'])
        for key in update_stats:
            update_stats[key] /= num_epochs
        
        return update_stats
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """评估模型性能"""
        eval_results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': 0.0,
            'avg_reward': 0.0
        }
        
        # TODO: 实现评估逻辑
        # 这里需要调用环境进行评估
        
        return eval_results
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cfg': self.cfg,
            'training_stats': self.training_stats
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
    
    def _get_action_pool(self, state, agent):
        """获取动作池 - 需要与环境交互"""
        # TODO: 实现动作池获取逻辑
        pass
    
    def _encode_state(self, state):
        """编码状态 - 需要与环境交互"""
        # TODO: 实现状态编码逻辑
        pass
    
    def _encode_action_features(self, feats):
        """编码动作特征"""
        # TODO: 实现动作特征编码逻辑
        pass
    
    def set_seed(self, seed: int):
        """设置随机种子"""
        set_seed(seed)


class MAPPOTrainer:
    """MAPPO训练器 - 多智能体版本"""
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.rl_cfg = cfg['solver']['rl']
        self.device = get_device()
        
        # 设置随机种子
        set_seed(self.rl_cfg['seed'])
        
        # 创建模型
        self._build_models()
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.model.actors.parameters(), lr=self.rl_cfg['lr'])
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=self.rl_cfg['lr'])
        
        # 创建经验缓冲区
        self.buffer = MultiAgentRolloutBuffer(
            capacity=self.rl_cfg['rollout_steps'],
            agents=self.rl_cfg['agents'],
            state_dim=512,
            action_feat_dim=64,
            max_actions=1000,
            device=self.device
        )
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'cooperation_rewards': []
        }
    
    def _build_models(self):
        """构建模型 - 与PPO相同"""
        # 状态编码器
        state_encoder = StateEncoder(
            grid_size=200,
            grid_channels=5,
            global_stats_dim=10,
            hidden_dim=256,
            output_dim=512
        )
        
        # 动作编码器
        action_encoder = ActionEncoder(
            action_feat_dim=20,
            hidden_dim=128,
            output_dim=64
        )
        
        # 多智能体Actor-Critic
        self.model = MultiAgentActorCritic(
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            agents=self.rl_cfg['agents'],
            state_dim=512,
            action_feat_dim=64,
            hidden_dim=256,
            max_actions=1000
        ).to(self.device)
    
    def collect_rollout(self, env) -> Dict:
        """收集rollout - 与PPO相同"""
        # TODO: 实现MAPPO特定的rollout收集
        pass
    
    def update_policy(self) -> Dict:
        """MAPPO更新策略"""
        # TODO: 实现MAPPO特定的策略更新
        # 主要区别：
        # 1. 集中式critic使用全局状态
        # 2. 协作奖励计算
        # 3. 独立的actor和critic优化器
        pass
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """评估模型性能"""
        # TODO: 实现MAPPO评估
        pass
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'cfg': self.cfg,
            'training_stats': self.training_stats
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
    
    def set_seed(self, seed: int):
        """设置随机种子"""
        set_seed(seed)

