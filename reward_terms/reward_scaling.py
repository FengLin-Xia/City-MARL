"""
奖励缩放模块

实现奖励的缩放和裁剪。
"""

from typing import Dict, Any
import numpy as np
from contracts import EnvironmentState


class RewardScalingTerm:
    """奖励缩放项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("reward_scaling", {})
        self.enabled = self.config.get("enabled", False)
        self.reward_scale = self.config.get("reward_scale", 3000.0)
        self.reward_clip = self.config.get("reward_clip", 1.0)
        self.target_range = self.config.get("target_range", [-1.0, 1.0])
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算奖励缩放
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            缩放后的奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 获取原始奖励
        raw_reward = self._get_raw_reward(prev_state, state, action_id)
        
        # 应用缩放
        scaled_reward = raw_reward / self.reward_scale
        
        # 应用裁剪
        clipped_reward = np.clip(scaled_reward, -self.reward_clip, self.reward_clip)
        
        return clipped_reward
    
    def _get_raw_reward(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """获取原始奖励"""
        # 简化实现：返回固定值
        # 实际实现需要根据state和action_id计算原始奖励
        return 1000.0  # 示例值
