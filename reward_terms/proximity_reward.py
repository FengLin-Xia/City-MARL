"""
邻近性奖励模块

实现邻近性奖励/惩罚机制。
"""

from typing import Dict, Any
import math
from contracts import EnvironmentState


class ProximityRewardTerm:
    """邻近性奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("proximity_reward", {})
        self.enabled = self.config.get("enabled", False)
        self.proximity_threshold = self.config.get("proximity_threshold", 15.0)
        self.proximity_reward = self.config.get("proximity_reward", 900.0)
        self.distance_penalty_coef = self.config.get("distance_penalty_coef", 0.6)
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算邻近性奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            邻近性奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 获取动作位置
        action_position = self._get_action_position(state, action_id)
        if not action_position:
            return 0.0
        
        # 计算邻近性奖励
        proximity_bonus = self._calculate_proximity_bonus(action_position, state)
        
        # 计算距离惩罚
        distance_penalty = self._calculate_distance_penalty(action_position, state)
        
        # 总邻近性奖励
        total_proximity_reward = proximity_bonus - distance_penalty
        
        return total_proximity_reward
    
    def _get_action_position(self, state: EnvironmentState, action_id: int) -> tuple:
        """获取动作位置"""
        # 简化实现：返回固定位置
        # 实际实现需要根据state和action_id确定具体位置
        return (100.0, 100.0)
    
    def _calculate_proximity_bonus(self, position: tuple, state: EnvironmentState) -> float:
        """计算邻近性奖励"""
        proximity_bonus = 0.0
        
        # 检查与现有建筑的距离
        for building_type in ["public", "industrial"]:
            buildings = state.buildings.get(building_type, [])
            for building in buildings:
                building_position = self._get_building_position(building)
                if building_position:
                    distance = self._calculate_distance(position, building_position)
                    
                    # 在邻近阈值内给予奖励
                    if distance <= self.proximity_threshold:
                        proximity_bonus += self.proximity_reward
        
        return proximity_bonus
    
    def _calculate_distance_penalty(self, position: tuple, state: EnvironmentState) -> float:
        """计算距离惩罚"""
        distance_penalty = 0.0
        
        # 计算到所有建筑的距离惩罚
        for building_type in ["public", "industrial"]:
            buildings = state.buildings.get(building_type, [])
            for building in buildings:
                building_position = self._get_building_position(building)
                if building_position:
                    distance = self._calculate_distance(position, building_position)
                    
                    # 距离越远惩罚越大
                    if distance > self.proximity_threshold:
                        penalty = (distance - self.proximity_threshold) * self.distance_penalty_coef
                        distance_penalty += penalty
        
        return distance_penalty
    
    def _get_building_position(self, building: Dict[str, Any]) -> tuple:
        """获取建筑位置"""
        if 'xy' in building:
            return tuple(building['xy'])
        return None
    
    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> float:
        """计算两点间距离"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)
