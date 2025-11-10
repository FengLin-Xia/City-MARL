"""
动作多样性奖励项

鼓励智能体选择不同的动作类型，提高探索性。
"""

from typing import Dict, Any, List, Deque
from collections import deque, Counter
import math
from contracts import EnvironmentState


class ActionDiversityRewardTerm:
    """动作多样性奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_size = config.get("window", 12)
        self.epsilon = config.get("epsilon", 1e-06)
        
        # 为每个智能体维护动作历史窗口
        self.action_history: Dict[str, Deque[int]] = {}
        
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算动作多样性奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            多样性奖励值
        """
        # 获取当前智能体
        agent = getattr(state, 'current_agent', 'UNKNOWN')
        
        # 初始化动作历史
        if agent not in self.action_history:
            self.action_history[agent] = deque(maxlen=self.window_size)
        
        # 添加当前动作到历史
        self.action_history[agent].append(action_id)
        
        # 计算动作分布的熵
        if len(self.action_history[agent]) < 2:
            return 0.0
        
        # 统计动作频次
        action_counts = Counter(self.action_history[agent])
        total_actions = len(self.action_history[agent])
        
        # 计算熵
        entropy = 0.0
        for count in action_counts.values():
            prob = count / total_actions
            if prob > self.epsilon:
                entropy -= prob * math.log(prob)
        
        # 归一化熵（除以最大可能熵）
        max_entropy = math.log(len(action_counts))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        return normalized_entropy

