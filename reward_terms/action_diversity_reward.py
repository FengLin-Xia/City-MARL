"""
动作多样性奖励项

鼓励智能体选择不同的动作类型，提高探索性。
"""

from typing import Dict, Any, List, Deque, Optional
from collections import deque, Counter
import math
from contracts import EnvironmentState


class ActionDiversityRewardTerm:
    """动作多样性奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        # 从配置中获取参数
        reward_config = config.get("reward_terms", {})
        term_params = config.get("term_params", {})
        diversity_config = term_params.get("action_diversity", {})
        
        self.enabled = diversity_config.get("enabled", True)
        self.window_size = diversity_config.get("window", 24)
        self.epsilon = diversity_config.get("epsilon", 1e-06)
        self.reward_scale = diversity_config.get("reward_scale", 50.0)
        self.novelty_bonus = diversity_config.get("novelty_bonus", 10.0)
        self.repetition_penalty = diversity_config.get("repetition_penalty", -5.0)
        
        # 频率奖励参数
        self.frequency_enabled = diversity_config.get("frequency_enabled", True)
        self.frequency_scale = diversity_config.get("frequency_scale", 50.0)
        self.middle_action_bonus = diversity_config.get("middle_action_bonus", 30.0)
        self.unlock_action_bonus = diversity_config.get("unlock_action_bonus", 20.0)
        self.first_time_bonus = diversity_config.get("first_time_bonus", 15.0)
        
        # 针对动作9和10的特殊奖励（新增）
        self.action9_bonus = diversity_config.get("action9_bonus", 200.0)
        self.action10_bonus = diversity_config.get("action10_bonus", 100.0)
        
        # 为每个智能体维护动作历史窗口（用于熵计算和新动作奖励）
        self.action_history: Dict[str, Deque[int]] = {}
        
        # 为每个智能体维护全局动作历史（用于频率奖励）
        self.global_action_history: Dict[str, List[int]] = {}
        
        # 定义"中间动作"和解锁动作
        self.middle_actions = [7, 10]  # COUNCIL_B 和 IND_B
        self.unlock_actions = [9, 10, 11]
        
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int, agent: str = None) -> float:
        """
        计算动作多样性奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            agent: 智能体名称（可选，如果提供则优先使用）
            
        Returns:
            多样性奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 优先使用传入的agent参数
        if agent is None:
            agent = getattr(state, 'current_agent', 'UNKNOWN')
        
        if agent == 'UNKNOWN':
            return 0.0
        
        # 初始化动作历史
        if agent not in self.action_history:
            self.action_history[agent] = deque(maxlen=self.window_size)
        
        # 检查是否是新动作（在窗口内首次出现）
        is_novel = action_id not in self.action_history[agent]
        
        # 检查是否是重复动作（与上一个动作相同）
        is_repetition = False
        if len(self.action_history[agent]) > 0:
            is_repetition = self.action_history[agent][-1] == action_id
        
        # 添加当前动作到历史
        self.action_history[agent].append(action_id)
        
        # 如果历史不足，只给予新动作奖励
        if len(self.action_history[agent]) < 2:
            if is_novel:
                return self.novelty_bonus
            return 0.0
        
        # 计算动作分布的熵
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
        
        # 基础熵奖励（缩放）
        entropy_reward = normalized_entropy * self.reward_scale
        
        # 新动作奖励
        novelty_reward = self.novelty_bonus if is_novel else 0.0
        
        # 重复惩罚
        repetition_penalty = self.repetition_penalty if is_repetition else 0.0
        
        # 频率奖励（新增）
        frequency_bonus = 0.0
        if self.frequency_enabled:
            frequency_bonus = self._compute_frequency_bonus(agent, action_id, state)
        
        # 总奖励 = 熵奖励 + 新动作奖励 + 重复惩罚 + 频率奖励
        total_reward = entropy_reward + novelty_reward + repetition_penalty + frequency_bonus
        
        return total_reward
    
    def _compute_frequency_bonus(self, agent: str, action_id: int, 
                                  state: EnvironmentState) -> float:
        """
        计算基于频率的奖励
        
        Args:
            agent: 智能体名称
            action_id: 动作ID
            state: 当前状态
            
        Returns:
            频率奖励值
        """
        # 初始化全局历史
        if agent not in self.global_action_history:
            self.global_action_history[agent] = []
        
        # 计算全局频率（在添加当前动作之前）
        global_counts = Counter(self.global_action_history[agent])
        total_global_actions = len(self.global_action_history[agent])
        
        if total_global_actions == 0:
            # 首次选择，给予最大奖励
            action_frequency = 0.0
        else:
            action_frequency = global_counts.get(action_id, 0) / total_global_actions
        
        # 基础频率奖励（频率越低，奖励越高）
        frequency_bonus = (1.0 - action_frequency) * self.frequency_scale
        
        # 特殊动作奖励
        # 1. "中间动作"奖励
        if action_id in self.middle_actions:
            frequency_bonus += self.middle_action_bonus
        
        # 2. 解锁动作奖励
        if action_id in self.unlock_actions and state.month >= 35:
            frequency_bonus += self.unlock_action_bonus
        
        # 3. 首次选择奖励
        if action_frequency == 0.0:
            frequency_bonus += self.first_time_bonus
        
        # 4. 针对动作9和10的特殊奖励（新增）
        if action_id == 9:
            frequency_bonus += self.action9_bonus
        elif action_id == 10:
            frequency_bonus += self.action10_bonus
        
        # 添加当前动作到全局历史（在计算奖励之后）
        self.global_action_history[agent].append(action_id)
        
        return frequency_bonus

