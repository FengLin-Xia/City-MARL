#!/usr/bin/env python3
"""
地形道路规划策略网络
支持处理复杂的地形观察空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import gymnasium as gym

class TerrainEncoder(nn.Module):
    """地形编码器 - 处理高程图和地形类型图"""
    
    def __init__(self, grid_size: Tuple[int, int], hidden_dim: int = 128):
        super().__init__()
        self.grid_size = grid_size
        
        # 高程图编码器 (CNN)
        self.height_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # 自适应池化到固定大小
            nn.Flatten()
        )
        
        # 地形类型编码器 (CNN)
        self.terrain_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        
        # 道路网络编码器 (CNN)
        self.road_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        
        # 位置编码器 (MLP)
        self.position_encoder = nn.Sequential(
            nn.Linear(4, 64),  # agent_pos + target_pos
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 资源编码器 (MLP)
        self.resource_encoder = nn.Sequential(
            nn.Linear(3, 32),  # resources
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # 特征融合层
        # 计算实际的特征维度
        # 每个编码器输出: 128 * 8 * 8 = 8192 (AdaptiveAvgPool2d((8, 8)))
        encoder_features = 128 * 8 * 8
        total_features = encoder_features * 3 + 128 + 64  # height + terrain + road + position + resource
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # 处理高程图
        height_map = observation['height_map'].unsqueeze(1)  # 添加通道维度
        height_features = self.height_encoder(height_map)
        
        # 处理地形类型图
        terrain_map = observation['terrain_map'].unsqueeze(1).float()
        terrain_features = self.terrain_encoder(terrain_map)
        
        # 处理道路网络图
        road_map = observation['road_map'].unsqueeze(1).float()
        road_features = self.road_encoder(road_map)
        
        # 处理位置信息
        agent_pos = observation['agent_pos'].float()
        target_pos = observation['target_pos'].float()
        position_input = torch.cat([agent_pos, target_pos], dim=-1)
        position_features = self.position_encoder(position_input)
        
        # 处理资源信息
        resources = observation['resources'].float()
        resource_features = self.resource_encoder(resources)
        
        # 融合所有特征
        combined_features = torch.cat([
            height_features, terrain_features, road_features,
            position_features, resource_features
        ], dim=-1)
        
        return self.fusion_layer(combined_features)

class TerrainPolicyNetwork(nn.Module):
    """地形道路规划策略网络"""
    
    def __init__(self, 
                 grid_size: Tuple[int, int],
                 action_space: gym.spaces.Discrete,
                 hidden_dim: int = 256,
                 use_attention: bool = True):
        super().__init__()
        
        self.grid_size = grid_size
        self.action_space = action_space
        self.num_actions = action_space.n
        self.use_attention = use_attention
        
        # 地形编码器
        self.terrain_encoder = TerrainEncoder(grid_size, hidden_dim)
        
        # 策略头 (Actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_actions)
        )
        
        # 价值头 (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 注意力机制 (可选)
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 编码地形信息
        terrain_features = self.terrain_encoder(observation)
        
        # 应用注意力机制 (可选)
        if self.use_attention:
            # 将特征重塑为序列形式用于注意力
            terrain_features = terrain_features.unsqueeze(1)  # [batch, 1, features]
            attended_features, _ = self.attention(terrain_features, terrain_features, terrain_features)
            terrain_features = attended_features.squeeze(1)  # [batch, features]
        
        # 策略头
        action_logits = self.policy_head(terrain_features)
        
        # 价值头
        value = self.value_head(terrain_features)
        
        return action_logits, value
    
    def get_action(self, observation: Dict[str, torch.Tensor], 
                   deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """获取动作"""
        with torch.no_grad():
            action_logits, value = self.forward(observation)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                # 使用softmax采样
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).squeeze(-1)
            
            return action.item(), action_logits, value

class TerrainValueNetwork(nn.Module):
    """地形价值网络 (用于DQN等算法)"""
    
    def __init__(self, 
                 grid_size: Tuple[int, int],
                 action_space: gym.spaces.Discrete,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.grid_size = grid_size
        self.action_space = action_space
        self.num_actions = action_space.n
        
        # 地形编码器
        self.terrain_encoder = TerrainEncoder(grid_size, hidden_dim)
        
        # Q值头
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_actions)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        terrain_features = self.terrain_encoder(observation)
        q_values = self.q_head(terrain_features)
        return q_values
    
    def get_action(self, observation: Dict[str, torch.Tensor], 
                   epsilon: float = 0.0) -> int:
        """获取动作 (epsilon-greedy)"""
        with torch.no_grad():
            q_values = self.forward(observation)
            
            if np.random.random() < epsilon:
                # 随机动作
                return np.random.randint(self.num_actions)
            else:
                # 最优动作
                return torch.argmax(q_values, dim=-1).item()

class TerrainActorCritic(nn.Module):
    """地形Actor-Critic网络 (用于A2C, PPO等算法)"""
    
    def __init__(self, 
                 grid_size: Tuple[int, int],
                 action_space: gym.spaces.Discrete,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.policy_network = TerrainPolicyNetwork(grid_size, action_space, hidden_dim)
        self.value_network = TerrainValueNetwork(grid_size, action_space, hidden_dim)
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        action_logits, value = self.policy_network(observation)
        return action_logits, value
    
    def get_action(self, observation: Dict[str, torch.Tensor], 
                   deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """获取动作"""
        return self.policy_network.get_action(observation, deterministic)
    
    def evaluate_actions(self, observation: Dict[str, torch.Tensor], 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作 (用于PPO等算法)"""
        action_logits, value = self.policy_network(observation)
        
        # 计算动作概率
        action_probs = F.softmax(action_logits, dim=-1)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        
        # 获取选中动作的log概率
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算熵
        entropy = -(action_probs * action_log_probs).sum(dim=-1)
        
        return selected_log_probs, value.squeeze(-1), entropy
