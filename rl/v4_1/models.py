"""
v4.1 RL模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class StateEncoder(nn.Module):
    """状态编码器 - 处理栅格数据和全局统计"""
    
    def __init__(self, 
                 grid_size: int = 200,
                 grid_channels: int = 5,  # 占用、功能、锁期、地价、河距
                 global_stats_dim: int = 10,  # 月份、预算、计数等
                 hidden_dim: int = 256,
                 output_dim: int = 512):
        super().__init__()
        
        self.grid_size = grid_size
        self.grid_channels = grid_channels
        self.global_stats_dim = global_stats_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 栅格编码器 - 使用CNN处理空间信息
        self.grid_encoder = nn.Sequential(
            # 第一层：200x200 -> 100x100
            nn.Conv2d(grid_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 第二层：100x100 -> 50x50
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 第三层：50x50 -> 25x25
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # 第四层：25x25 -> 12x12
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )
        
        # 全局统计编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(global_stats_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, grid_data: torch.Tensor, global_stats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_data: [B, grid_channels, grid_size, grid_size]
            global_stats: [B, global_stats_dim]
        Returns:
            state_embed: [B, output_dim]
        """
        # 编码栅格数据
        grid_feat = self.grid_encoder(grid_data)  # [B, hidden_dim]
        
        # 编码全局统计
        global_feat = self.global_encoder(global_stats)  # [B, hidden_dim//4]
        
        # 融合特征
        combined = torch.cat([grid_feat, global_feat], dim=-1)
        state_embed = self.fusion(combined)
        
        return state_embed


class ActionEncoder(nn.Module):
    """动作编码器 - 处理动作特征"""
    
    def __init__(self, 
                 action_feat_dim: int = 20,  # 槽位坐标、地价统计等
                 hidden_dim: int = 128,
                 output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(action_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, action_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_feats: [B, K, action_feat_dim] - K是动作数量
        Returns:
            encoded_actions: [B, K, output_dim]
        """
        return self.encoder(action_feats)


class Actor(nn.Module):
    """策略网络 - 输出动作概率分布"""
    
    def __init__(self, 
                 state_dim: int = 512,
                 action_feat_dim: int = 64,
                 hidden_dim: int = 256,
                 max_actions: int = 1000):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_feat_dim = action_feat_dim
        self.max_actions = max_actions
        
        # 状态-动作交互层
        self.interaction = nn.Sequential(
            nn.Linear(state_dim + action_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # 每个动作一个logit
        )
    
    def forward(self, state_embed: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embed: [B, state_dim]
            action_feats: [B, K, action_feat_dim]
        Returns:
            logits: [B, K]
        """
        B, K, _ = action_feats.shape
        
        # 广播状态特征到每个动作
        state_expanded = state_embed.unsqueeze(1).expand(B, K, self.state_dim)
        
        # 拼接状态和动作特征
        combined = torch.cat([state_expanded, action_feats], dim=-1)
        
        # 计算每个动作的logit
        logits = self.interaction(combined).squeeze(-1)  # [B, K]
        
        return logits


class Critic(nn.Module):
    """价值网络 - 评估状态价值（MAPPO用）"""
    
    def __init__(self, 
                 state_dim: int = 512,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embed: [B, state_dim] - 全局状态编码
        Returns:
            value: [B, 1]
        """
        return self.value_net(state_embed)


class MultiAgentActorCritic(nn.Module):
    """多智能体Actor-Critic网络"""
    
    def __init__(self, 
                 state_encoder: StateEncoder,
                 action_encoder: ActionEncoder,
                 agents: List[str] = ['EDU', 'IND'],
                 state_dim: int = 512,
                 action_feat_dim: int = 64,
                 hidden_dim: int = 256,
                 max_actions: int = 1000):
        super().__init__()
        
        self.agents = agents
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        
        # 为每个智能体创建独立的Actor
        self.actors = nn.ModuleDict({
            agent: Actor(state_dim, action_feat_dim, hidden_dim, max_actions)
            for agent in agents
        })
        
        # 集中式Critic（MAPPO）
        self.critic = Critic(state_dim, hidden_dim)
    
    def forward(self, 
                grid_data: torch.Tensor,
                global_stats: torch.Tensor,
                action_feats: torch.Tensor,
                agent: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            grid_data: [B, grid_channels, grid_size, grid_size]
            global_stats: [B, global_stats_dim]
            action_feats: [B, K, action_feat_dim]
            agent: 智能体名称
        Returns:
            logits: [B, K] - 动作概率分布
            value: [B, 1] - 状态价值
        """
        # 编码状态
        state_embed = self.state_encoder(grid_data, global_stats)
        
        # 编码动作特征
        encoded_actions = self.action_encoder(action_feats)
        
        # 获取对应智能体的策略
        logits = self.actors[agent](state_embed, encoded_actions)
        
        # 计算状态价值
        value = self.critic(state_embed)
        
        return logits, value

