"""
河流溢价奖励模块

实现河流附近的溢价奖励计算。
"""

from typing import Dict, Any
import math
from contracts import EnvironmentState


class RiverPremiumRewardTerm:
    """河流溢价奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("river_premium", {})
        self.enabled = self.config.get("enabled", False)
        self.river_half_distance = self.config.get("river_half_distance", 120.0)
        self.max_premium_cap = self.config.get("max_premium_cap", 10000.0)
        self.premium_rates = self.config.get("premium_rates", {"IND": 0.20, "EDU": 0.15})
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算河流溢价奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            河流溢价奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 获取当前智能体类型
        agent_type = self._get_agent_type_from_action(action_id)
        if not agent_type:
            return 0.0
        
        # 获取溢价率
        premium_rate = self.premium_rates.get(agent_type, 0.0)
        if premium_rate <= 0:
            return 0.0
        
        # 计算到河流的距离
        river_distance = self._get_river_distance(state, action_id)
        if river_distance is None:
            return 0.0
        
        # 计算衰减因子
        decay = self._calculate_decay(river_distance)
        
        # 计算基础收益
        base_revenue = self._get_base_revenue(action_id)
        if base_revenue <= 0:
            return 0.0
        
        # 计算河流溢价
        raw_premium = base_revenue * premium_rate * decay
        river_premium = min(raw_premium, self.max_premium_cap)
        
        return river_premium
    
    def _get_agent_type_from_action(self, action_id: int) -> str:
        """从动作ID推断智能体类型"""
        if 0 <= action_id <= 2:
            return "EDU"
        elif 3 <= action_id <= 5:
            return "IND"
        elif 6 <= action_id <= 8:
            return "COUNCIL"
        else:
            return None
    
    def _get_river_distance(self, state: EnvironmentState, action_id: int) -> float:
        """获取到河流的距离"""
        # 简化实现：返回固定距离
        # 实际实现需要根据state和action_id计算到河流的实际距离
        return 50.0  # 示例值
    
    def _calculate_decay(self, distance: float) -> float:
        """计算衰减因子"""
        if distance <= 0:
            return 1.0
        
        # 使用指数衰减：decay = 2^(-distance / half_distance)
        decay = 2 ** (-distance / self.river_half_distance)
        return decay
    
    def _get_base_revenue(self, action_id: int) -> float:
        """获取基础收益"""
        # 简化实现：返回固定值
        # 实际实现需要根据action_id获取具体的基础收益
        return 100.0  # 示例值
