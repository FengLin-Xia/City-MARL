"""
v4.1 RL工具函数
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """
    带掩码的softmax
    
    Args:
        logits: [B, K] 或 [K] - 动作logits
        mask: [B, K] 或 [K] - 动作掩码 (1=合法, 0=非法)
        dim: softmax维度
        eps: 数值稳定性参数
    
    Returns:
        probs: [B, K] 或 [K] - 归一化后的概率分布
    """
    # 将非法动作的logits设为负无穷
    masked_logits = logits + (mask.float().log() * 0.0).masked_fill(mask == 0, -1e9)
    
    # 计算softmax
    probs = F.softmax(masked_logits, dim=dim)
    
    # 确保概率和为1（数值稳定性）
    probs = probs / (probs.sum(dim=dim, keepdim=True) + eps)
    
    return probs


def masked_sample(logits: torch.Tensor, mask: torch.Tensor, greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    带掩码的动作采样
    
    Args:
        logits: [B, K] 或 [K] - 动作logits
        mask: [B, K] 或 [K] - 动作掩码 (1=合法, 0=非法)
        greedy: 是否使用贪心选择
    
    Returns:
        action_idx: [B] 或 scalar - 选中的动作索引
        log_prob: [B] 或 scalar - 动作的对数概率
    """
    probs = masked_softmax(logits, mask)
    
    if greedy:
        # 贪心选择
        action_idx = torch.argmax(probs, dim=-1)
    else:
        # 随机采样
        action_idx = torch.distributions.Categorical(probs).sample()
    
    # 计算对数概率
    log_prob = torch.log(probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1) + 1e-12)
    
    return action_idx, log_prob


def masked_log_prob(logits: torch.Tensor, mask: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
    """
    计算带掩码的动作对数概率
    
    Args:
        logits: [B, K] - 动作logits
        mask: [B, K] - 动作掩码
        action_idx: [B] - 动作索引
    
    Returns:
        log_prob: [B] - 对数概率
    """
    probs = masked_softmax(logits, mask)
    log_prob = torch.log(probs.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1) + 1e-12)
    return log_prob


def compute_gae(rewards: List[float], 
                values: List[float], 
                next_values: List[float],
                dones: List[bool],
                gamma: float = 0.99,
                gae_lambda: float = 0.95) -> Tuple[List[float], List[float]]:
    """
    计算广义优势估计 (GAE)
    
    Args:
        rewards: 奖励序列
        values: 状态价值序列
        next_values: 下一状态价值序列
        dones: 终止标志序列
        gamma: 折扣因子
        gae_lambda: GAE参数
    
    Returns:
        advantages: 优势值序列
        returns: 回报序列
    """
    advantages = []
    returns = []
    
    # 计算TD误差
    td_errors = []
    for i in range(len(rewards)):
        if dones[i]:
            td_error = rewards[i] - values[i]
        else:
            td_error = rewards[i] + gamma * next_values[i] - values[i]
        td_errors.append(td_error)
    
    # 计算GAE优势
    advantage = 0
    for i in reversed(range(len(rewards))):
        if dones[i]:
            advantage = 0
        advantage = td_errors[i] + gamma * gae_lambda * advantage
        advantages.insert(0, advantage)
    
    # 计算回报
    for i in range(len(rewards)):
        returns.append(advantages[i] + values[i])
    
    return advantages, returns


def compute_gae_batch(rewards: torch.Tensor,
                     values: torch.Tensor,
                     next_values: torch.Tensor,
                     dones: torch.Tensor,
                     gamma: float = 0.99,
                     gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量计算GAE（GPU加速版本）
    
    Args:
        rewards: [T, B] - 奖励
        values: [T, B] - 状态价值
        next_values: [T, B] - 下一状态价值
        dones: [T, B] - 终止标志
        gamma: 折扣因子
        gae_lambda: GAE参数
    
    Returns:
        advantages: [T, B] - 优势值
        returns: [T, B] - 回报
    """
    T, B = rewards.shape
    device = rewards.device
    
    # 计算TD误差
    td_errors = torch.zeros_like(rewards)
    for t in range(T):
        if t == T - 1:
            td_errors[t] = rewards[t] - values[t]
        else:
            td_errors[t] = rewards[t] + gamma * next_values[t + 1] * (1 - dones[t]) - values[t]
    
    # 计算GAE优势
    advantages = torch.zeros_like(rewards)
    advantage = torch.zeros(B, device=device)
    
    for t in reversed(range(T)):
        advantage = td_errors[t] + gamma * gae_lambda * advantage * (1 - dones[t])
        advantages[t] = advantage
    
    # 计算回报
    returns = advantages + values
    
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    标准化优势值
    
    Args:
        advantages: [T, B] - 优势值
        eps: 数值稳定性参数
    
    Returns:
        normalized_advantages: [T, B] - 标准化后的优势值
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)


def clip_grad_norm(model: torch.nn.Module, max_norm: float = 0.5) -> float:
    """
    梯度裁剪
    
    Args:
        model: 模型
        max_norm: 最大梯度范数
    
    Returns:
        grad_norm: 裁剪前的梯度范数
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def categorical_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    计算带掩码的分类熵
    
    Args:
        logits: [B, K] - 动作logits
        mask: [B, K] - 动作掩码
    
    Returns:
        entropy: [B] - 熵值
    """
    probs = masked_softmax(logits, mask)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def get_device() -> torch.device:
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

