"""
协作奖励模块

实现智能体间的协作奖励，包括功能互补和空间协调。
"""

from typing import Dict, Any, List, Tuple
import math
from contracts import EnvironmentState


class CooperationRewardTerm:
    """协作奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("cooperation_reward", {})
        self.enabled = self.config.get("enabled", False)
        self.lambda_coef = self.config.get("lambda", 0.1)
        
        # 功能互补配置
        self.functional_config = self.config.get("functional_complement", {})
        self.edu_ind_bonus = self.functional_config.get("edu_ind_bonus", 0.05)
        self.ind_edu_bonus = self.functional_config.get("ind_edu_bonus", 0.05)
        
        # 空间协调配置
        self.spatial_config = self.config.get("spatial_coordination", {})
        self.optimal_distance_min = self.spatial_config.get("optimal_distance_min", 5)
        self.optimal_distance_max = self.spatial_config.get("optimal_distance_max", 20)
        self.coordination_bonus = self.spatial_config.get("coordination_bonus", 0.02)
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算协作奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            协作奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 获取当前智能体类型
        agent_type = self._get_agent_type_from_action(action_id)
        if not agent_type:
            return 0.0
        
        # 计算功能互补奖励
        functional_bonus = self._calculate_functional_complement(agent_type, state)
        
        # 计算空间协调奖励
        spatial_bonus = self._calculate_spatial_coordination(action_id, state)
        
        # 总协作奖励
        total_cooperation_bonus = functional_bonus + spatial_bonus
        
        # 应用lambda系数
        cooperation_reward = self.lambda_coef * total_cooperation_bonus
        
        return cooperation_reward
    
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
    
    def _calculate_functional_complement(self, agent_type: str, state: EnvironmentState) -> float:
        """计算功能互补奖励"""
        functional_bonus = 0.0
        
        if agent_type == "EDU":
            # EDU建筑越多，IND的奖励越高
            ind_buildings = self._count_buildings(state, "industrial")
            functional_bonus += ind_buildings * self.edu_ind_bonus
        elif agent_type == "IND":
            # IND建筑越多，EDU的奖励越高
            edu_buildings = self._count_buildings(state, "public")
            functional_bonus += edu_buildings * self.ind_edu_bonus
        
        return functional_bonus
    
    def _calculate_spatial_coordination(self, action_id: int, state: EnvironmentState) -> float:
        """计算空间协调奖励"""
        spatial_bonus = 0.0
        
        # 获取动作对应的槽位位置
        action_position = self._get_action_position(action_id, state)
        if not action_position:
            return 0.0
        
        # 计算与其他建筑的距离协调性
        for building_type in ["public", "industrial"]:
            buildings = state.buildings.get(building_type, [])
            for building in buildings:
                building_position = self._get_building_position(building)
                if building_position:
                    distance = self._calculate_distance(action_position, building_position)
                    
                    # 适中距离给予奖励
                    if self.optimal_distance_min <= distance <= self.optimal_distance_max:
                        spatial_bonus += self.coordination_bonus
        
        return spatial_bonus
    
    def _count_buildings(self, state: EnvironmentState, building_type: str) -> int:
        """计算指定类型的建筑数量"""
        if not hasattr(state, 'buildings') or not state.buildings:
            return 0
        
        buildings = state.buildings.get(building_type, [])
        return len(buildings)
    
    def _get_action_position(self, action_id: int, state: EnvironmentState) -> Tuple[float, float]:
        """获取动作对应的位置"""
        # 简化实现：返回随机位置
        # 实际实现需要根据action_id和state确定具体位置
        return (100.0, 100.0)
    
    def _get_building_position(self, building: Dict[str, Any]) -> Tuple[float, float]:
        """获取建筑位置"""
        if 'xy' in building:
            return tuple(building['xy'])
        return None
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点间距离"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)

