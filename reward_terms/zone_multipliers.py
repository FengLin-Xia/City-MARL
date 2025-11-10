"""
区位乘子奖励模块

实现区位差异对奖励的影响。
"""

from typing import Dict, Any
from contracts import EnvironmentState


class ZoneMultipliersTerm:
    """区位乘子奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("zone_multipliers", {})
        self.enabled = self.config.get("enabled", False)
        self.m_zone = self.config.get("m_zone", {"near": 1.2, "mid": 1.0, "far": 0.8})
        self.m_adj = self.config.get("m_adj", {"adjacent": 1.1, "non_adjacent": 1.0})
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算区位乘子奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            区位乘子奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 获取区位信息
        zone = self._get_zone(state, action_id)
        is_adjacent = self._is_adjacent(state, action_id)
        
        # 计算区位乘子
        zone_multiplier = self.m_zone.get(zone, 1.0)
        adj_multiplier = self.m_adj.get("adjacent" if is_adjacent else "non_adjacent", 1.0)
        
        # 总乘子
        total_multiplier = zone_multiplier * adj_multiplier
        
        # 返回乘子调整值（相对于1.0的差异）
        multiplier_adjustment = total_multiplier - 1.0
        
        return multiplier_adjustment
    
    def _get_zone(self, state: EnvironmentState, action_id: int) -> str:
        """获取区位"""
        # 简化实现：返回固定区位
        # 实际实现需要根据state和action_id计算具体区位
        return "mid"
    
    def _is_adjacent(self, state: EnvironmentState, action_id: int) -> bool:
        """判断是否相邻"""
        # 简化实现：返回固定值
        # 实际实现需要根据state和action_id判断是否与现有建筑相邻
        return False

