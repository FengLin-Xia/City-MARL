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
import torch.distributions as D
from utils.logger_factory import get_logger, topic_enabled, sampling_allows


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
        self.logger = get_logger("policy")
    
    def _agent_allowed_actions(self, agent: str) -> List[int]:
        agent_config = self.config.get("agents", {}).get("defs", {}).get(agent, {})
        return list(agent_config.get("action_ids", []))

    def _candidate_ids(self, candidates: List[ActionCandidate]) -> List[int]:
        return [c.id for c in candidates]

    def _masked_logits(self, agent: str, state: EnvironmentState, allowed_ids: List[int]) -> torch.Tensor:
        obs = self._encode_state(state)
        logits = self.actor_networks[agent](obs)
        mask = torch.full((self.action_size,), float('-inf'), device=self.device)
        if allowed_ids:
            mask[allowed_ids] = 0.0
        masked_logits = logits + mask
        return masked_logits

    def select_action(self, agent: str, candidates: List[ActionCandidate], state: EnvironmentState, greedy: bool = False) -> Optional[Dict[str, Any]]:
        """基于策略从候选中选择动作，返回包含 logprob/value 的信息"""
        if not candidates:
            return None
        # 候选内索引化：仅对当前候选集合建分布
        obs = self._encode_state(state)
        logits_full = self.actor_networks[agent](obs)
        # 提取对应候选的 logit，按候选顺序组成向量
        cand_ids_list = [c.id for c in candidates]
        logits = logits_full[cand_ids_list]
        # 温度采样（从配置取 mappo.exploration.temperature，若无则1.0）
        temp = float(self.config.get('mappo', {}).get('exploration', {}).get('temperature', 1.0))
        logits = logits / max(temp, 1e-6)
        log_probs_vec = torch.log_softmax(logits, dim=-1)
        probs_vec = torch.softmax(logits, dim=-1)
        if greedy:
            idx = int(torch.argmax(probs_vec).item())
        else:
            dist = D.Categorical(probs_vec)
            idx = int(dist.sample().item())
        action_id = cand_ids_list[idx]
        chosen_logprob = log_probs_vec[idx].detach().item()
        with torch.no_grad():
            value = self.critic_networks[agent](self._encode_state(state)).squeeze().item()

        # 找到对应候选
        chosen_cand = candidates[idx] if 0 <= idx < len(candidates) else None
        if chosen_cand is None:
            return None

        sequence = Sequence(agent=agent, actions=[action_id])

        # 选择日志（受配置开关与采样控制）
        if topic_enabled("policy_select") and sampling_allows(agent, getattr(state, 'month', None), None):
            # 提取允许集合上的 top3 概率
            topk = []
            vals, idxs = torch.topk(probs_vec, k=min(3, probs_vec.numel()))
            for v, ix in zip(vals.tolist(), idxs.tolist()):
                topk.append((int(cand_ids_list[ix]), round(float(v), 4)))
            # 计算熵
            p = probs_vec
            entropy = float(-(p * (p + 1e-8).log()).sum().item())
            self.logger.info(
                f"policy_select agent={agent} month={getattr(state, 'month', '?')} action_id={action_id} logp={round(chosen_logprob,4)} value={round(value,3)} top3={topk} H={round(entropy,4)}")
        return {
            'sequence': sequence,
            'action_id': action_id,
            'logprob': chosen_logprob,
            'value': value,
            'probs': probs_vec.detach().cpu().numpy(),
        }

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
