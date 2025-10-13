"""
v4.1 经验回放缓冲区
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class RolloutBuffer:
    """PPO/MAPPO经验回放缓冲区"""
    
    def __init__(self, 
                 capacity: int,
                 num_agents: int = 2,
                 state_dim: int = 512,
                 action_feat_dim: int = 64,
                 max_actions: int = 1000,
                 device: torch.device = None):
        
        self.capacity = capacity
        self.num_agents = num_agents
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 存储轨迹数据
        self.data = {
            'states': [],           # 状态编码
            'action_feats': [],     # 动作特征
            'action_masks': [],     # 动作掩码
            'actions': [],          # 选择的动作索引
            'log_probs': [],        # 动作对数概率
            'rewards': [],          # 奖励
            'values': [],           # 状态价值
            'dones': [],            # 终止标志
            'agents': [],           # 智能体标识
        }
        
        # 当前轨迹长度
        self.ptr = 0
        self.size = 0
        
        # 预分配张量（可选，用于性能优化）
        self.preallocated = False
    
    def add(self, 
            state: torch.Tensor,
            action_feats: torch.Tensor,
            action_mask: torch.Tensor,
            action: torch.Tensor,
            log_prob: torch.Tensor,
            reward: float,
            value: torch.Tensor,
            done: bool,
            agent: str):
        """添加一个时间步的经验"""
        
        # 确保数据在正确设备上
        state = state.to(self.device)
        action_feats = action_feats.to(self.device)
        action_mask = action_mask.to(self.device)
        action = action.to(self.device)
        log_prob = log_prob.to(self.device)
        value = value.to(self.device)
        
        # 添加到缓冲区
        self.data['states'].append(state)
        self.data['action_feats'].append(action_feats)
        self.data['action_masks'].append(action_mask)
        self.data['actions'].append(action)
        self.data['log_probs'].append(log_prob)
        self.data['rewards'].append(reward)
        self.data['values'].append(value)
        self.data['dones'].append(done)
        self.data['agents'].append(agent)
        
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)
    
    def get_batch(self, batch_size: int = None) -> Dict[str, torch.Tensor]:
        """获取批量数据"""
        if batch_size is None:
            batch_size = self.size
        
        # 随机采样索引
        indices = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        
        batch = {}
        for key in self.data:
            if key == 'agents':
                batch[key] = [self.data[key][i] for i in indices]
            else:
                batch[key] = torch.stack([self.data[key][i] for i in indices])
        
        return batch
    
    def get_rollout(self) -> Dict[str, torch.Tensor]:
        """获取完整轨迹数据"""
        rollout = {}
        for key in self.data:
            if key == 'agents':
                rollout[key] = self.data[key]
            else:
                rollout[key] = torch.stack(self.data[key])
        
        return rollout
    
    def compute_advantages(self, 
                          gamma: float = 0.99,
                          gae_lambda: float = 0.95,
                          normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算优势值和回报"""
        
        if self.size == 0:
            return torch.tensor([]), torch.tensor([])
        
        # 转换为张量
        rewards = torch.tensor(self.data['rewards'], dtype=torch.float32, device=self.device)
        values = torch.stack(self.data['values']).squeeze(-1)
        dones = torch.tensor(self.data['dones'], dtype=torch.bool, device=self.device)
        
        # 计算GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        advantage = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                td_error = rewards[t] - values[t]
            else:
                td_error = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            
            advantage = td_error + gamma * gae_lambda * advantage * (1 - dones[t])
            advantages[t] = advantage
            returns[t] = advantage + values[t]
        
        # 标准化优势值
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def clear(self):
        """清空缓冲区"""
        for key in self.data:
            self.data[key].clear()
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size


class MultiAgentRolloutBuffer:
    """多智能体经验回放缓冲区"""
    
    def __init__(self, 
                 capacity: int,
                 agents: List[str],
                 state_dim: int = 512,
                 action_feat_dim: int = 64,
                 max_actions: int = 1000,
                 device: torch.device = None):
        
        self.capacity = capacity
        self.agents = agents
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 为每个智能体创建独立的缓冲区
        self.buffers = {
            agent: RolloutBuffer(
                capacity=capacity,
                num_agents=1,
                state_dim=state_dim,
                action_feat_dim=action_feat_dim,
                max_actions=max_actions,
                device=device
            )
            for agent in agents
        }
        
        # 全局状态缓冲区（用于集中式critic）
        self.global_buffer = RolloutBuffer(
            capacity=capacity,
            num_agents=len(agents),
            state_dim=state_dim,
            action_feat_dim=action_feat_dim,
            max_actions=max_actions,
            device=device
        )
    
    def add(self, 
            agent: str,
            state: torch.Tensor,
            action_feats: torch.Tensor,
            action_mask: torch.Tensor,
            action: torch.Tensor,
            log_prob: torch.Tensor,
            reward: float,
            value: torch.Tensor,
            done: bool,
            global_state: Optional[torch.Tensor] = None):
        """添加经验到指定智能体的缓冲区"""
        
        # 添加到智能体缓冲区
        self.buffers[agent].add(
            state=state,
            action_feats=action_feats,
            action_mask=action_mask,
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
            agent=agent
        )
        
        # 添加到全局缓冲区（用于集中式critic）
        if global_state is not None:
            self.global_buffer.add(
                state=global_state,
                action_feats=action_feats,
                action_mask=action_mask,
                action=action,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=done,
                agent=agent
            )
    
    def get_batch(self, agent: str, batch_size: int = None) -> Dict[str, torch.Tensor]:
        """获取指定智能体的批量数据"""
        return self.buffers[agent].get_batch(batch_size)
    
    def get_global_batch(self, batch_size: int = None) -> Dict[str, torch.Tensor]:
        """获取全局批量数据"""
        return self.global_buffer.get_batch(batch_size)
    
    def compute_advantages(self, 
                          agent: str,
                          gamma: float = 0.99,
                          gae_lambda: float = 0.95,
                          normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算指定智能体的优势值"""
        return self.buffers[agent].compute_advantages(gamma, gae_lambda, normalize)
    
    def clear(self):
        """清空所有缓冲区"""
        for buffer in self.buffers.values():
            buffer.clear()
        self.global_buffer.clear()
    
    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers.values())

