"""
地价货币化奖励模块

实现地价对成本/收益的影响，EDU获得地价，IND支付地价。
"""

from typing import Dict, Any
from contracts import EnvironmentState


class LandPriceMonetizationTerm:
    """地价货币化奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("land_price_monetization", {})
        self.enabled = self.config.get("enabled", False)
        self.land_price_base = self.config.get("land_price_base", 11)
        self.edu_receives_land = self.config.get("edu_receives_land", True)
        self.ind_pays_land = self.config.get("ind_pays_land", True)
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算地价货币化奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            地价货币化奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 获取当前智能体类型
        agent_type = self._get_agent_type_from_action(action_id)
        if not agent_type:
            return 0.0
        
        # 计算地价指数
        lp_idx = self._get_land_price_index(state, action_id)
        if lp_idx is None:
            return 0.0
        
        # 计算地价价值
        lp_value = lp_idx * self.land_price_base
        
        # 根据智能体类型计算奖励
        if agent_type == "EDU" and self.edu_receives_land:
            # EDU获得地价
            return lp_value
        elif agent_type == "IND" and self.ind_pays_land:
            # IND支付地价
            return -lp_value
        else:
            return 0.0
    
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
