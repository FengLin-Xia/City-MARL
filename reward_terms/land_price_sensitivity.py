"""
地价敏感度奖励模块

实现地价对奖励的调节。
"""

from typing import Dict, Any
from contracts import EnvironmentState


class LandPriceSensitivityTerm:
    """地价敏感度奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("land_price_sensitivity", {})
        self.enabled = self.config.get("enabled", False)
        self.reward_lp_k = self.config.get("reward_lp_k", {"IND": 0.25, "EDU": 0.10})
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算地价敏感度奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            地价敏感度奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 获取当前智能体类型
        agent_type = self._get_agent_type_from_action(action_id)
        if not agent_type:
            return 0.0
        
        # 获取地价敏感度系数
        lp_k = self.reward_lp_k.get(agent_type, 0.0)
        if lp_k <= 0:
            return 0.0
        
        # 获取地价指数
        lp_idx = self._get_land_price_index(state, action_id)
        if lp_idx is None:
            return 0.0
        
        # 计算地价敏感度奖励
        # 地价越高，奖励越高
        land_price_sensitivity_reward = lp_idx * lp_k
        
        return land_price_sensitivity_reward
    
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
    
    def _get_land_price_index(self, state: EnvironmentState, action_id: int) -> float:
        """获取地价指数"""
        # 简化实现：返回固定值
        # 实际实现需要根据state和action_id计算具体的地价指数
        return 50.0  # 示例值
