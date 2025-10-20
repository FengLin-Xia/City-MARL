"""
v5.0 RL选择器

基于契约对象和配置的RL策略选择器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contracts import ActionCandidate, Sequence, EnvironmentState
from config_loader import ConfigLoader


class V5ActorNetwork(nn.Module):
    """v5.0 Actor网络"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 9):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class V5CriticNetwork(nn.Module):
    """v5.0 Critic网络"""
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class V5RLSelector:
    """v5.0 RL选择器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化RL选择器
        
        Args:
            config: v5.0配置
        """
        self.config = config
        self.loader = ConfigLoader()
        
        # 获取智能体配置
        self.agents = config.get("agents", {}).get("order", [])
        
        # 网络参数
        self.obs_size = 64  # 观察空间大小
        self.hidden_size = 128
        self.action_size = 9  # 动作空间大小（0-8）
        
        # 初始化网络
        self.actor_networks = {}
        self.critic_networks = {}
        
        for agent in self.agents:
            self.actor_networks[agent] = V5ActorNetwork(
                input_size=self.obs_size,
                hidden_size=self.hidden_size,
                output_size=self.action_size
            )
            self.critic_networks[agent] = V5CriticNetwork(
                input_size=self.obs_size,
                hidden_size=self.hidden_size
            )
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 将网络移到设备
        for agent in self.agents:
            self.actor_networks[agent] = self.actor_networks[agent].to(self.device)
            self.critic_networks[agent] = self.critic_networks[agent].to(self.device)
    
    def choose_sequence(self, agent: str, candidates: List[ActionCandidate], 
                       state: EnvironmentState, greedy: bool = False) -> Optional[Sequence]:
        """
        选择动作序列
        
        Args:
            agent: 智能体名称
            candidates: 动作候选列表
            state: 环境状态
            greedy: 是否使用贪心策略
            
        Returns:
            选择的序列
        """
        if not candidates:
            return None
        
        # 获取智能体的可用动作ID
        agent_config = self.config.get("agents", {}).get("defs", {}).get(agent, {})
        available_action_ids = agent_config.get("action_ids", [])
        
        # 过滤候选动作
        valid_candidates = [c for c in candidates if c.id in available_action_ids]
        
        if not valid_candidates:
            return None
        
        # 选择动作
        if greedy:
            # 贪心策略：选择第一个有效动作
            chosen_action = valid_candidates[0]
        else:
            # 随机策略：随机选择一个有效动作
            chosen_action = np.random.choice(valid_candidates)
        
        # 创建序列
        sequence = Sequence(
            agent=agent,
            actions=[chosen_action.id]
        )
        
        return sequence
    
    def get_action_probabilities(self, agent: str, candidates: List[ActionCandidate], 
                                state: EnvironmentState) -> torch.Tensor:
        """
        获取动作概率分布
        
        Args:
            agent: 智能体名称
            candidates: 动作候选列表
            state: 环境状态
            
        Returns:
            动作概率分布
        """
        if not candidates:
            return torch.zeros(self.action_size)
        
        # 获取智能体的可用动作ID
        agent_config = self.config.get("agents", {}).get("defs", {}).get(agent, {})
        available_action_ids = agent_config.get("action_ids", [])
        
        # 创建动作掩码
        action_mask = torch.zeros(self.action_size)
        for action_id in available_action_ids:
            action_mask[action_id] = 1.0
        
        # 获取网络输出
        with torch.no_grad():
            obs = self._encode_state(state)
            logits = self.actor_networks[agent](obs)
            
            # 应用掩码
            masked_logits = logits * action_mask
            
            # 计算概率
            probs = F.softmax(masked_logits, dim=-1)
        
        return probs
    
    def get_value(self, agent: str, state: EnvironmentState) -> float:
        """
        获取状态价值
        
        Args:
            agent: 智能体名称
            state: 环境状态
            
        Returns:
            状态价值
        """
        with torch.no_grad():
            obs = self._encode_state(state)
            value = self.critic_networks[agent](obs)
            return value.item()
    
    def _encode_state(self, state: EnvironmentState) -> torch.Tensor:
        """
        编码状态为观察向量
        
        Args:
            state: 环境状态
            
        Returns:
            观察向量
        """
        # 简化实现：返回固定长度的观察向量
        obs = torch.zeros(self.obs_size)
        
        # 填充基础信息
        obs[0] = state.month
        obs[1] = len(state.buildings)
        obs[2] = len(state.slots)
        
        # 填充预算信息
        for i, (agent, budget) in enumerate(state.budgets.items()):
            if i < 3:  # 最多3个智能体
                obs[3 + i] = budget
        
        # 填充地价信息
        if state.land_prices is not None:
            flat_prices = state.land_prices.flatten()
            obs[6:18] = torch.from_numpy(flat_prices[:12]).float()
        
        return obs.to(self.device)
    
    def save_networks(self, path: str):
        """保存网络权重"""
        torch.save({
            'actor_networks': {agent: net.state_dict() for agent, net in self.actor_networks.items()},
            'critic_networks': {agent: net.state_dict() for agent, net in self.critic_networks.items()}
        }, path)
    
    def load_networks(self, path: str):
        """加载网络权重"""
        if not os.path.exists(path):
            print(f"网络权重文件不存在: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载actor网络
        for agent, net in self.actor_networks.items():
            if agent in checkpoint['actor_networks']:
                net.load_state_dict(checkpoint['actor_networks'][agent])
        
        # 加载critic网络
        for agent, net in self.critic_networks.items():
            if agent in checkpoint['critic_networks']:
                net.load_state_dict(checkpoint['critic_networks'][agent])
        
        print(f"网络权重已从 {path} 加载")
