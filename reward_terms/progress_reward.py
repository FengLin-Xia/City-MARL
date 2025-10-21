"""
进度奖励模块

基于已有建筑数量给予奖励，鼓励智能体建造更多建筑。
"""

from typing import Dict, Any
from contracts import EnvironmentState


class ProgressRewardTerm:
    """进度奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("progress_reward", {})
        self.enabled = self.config.get("enabled", False)
        self.building_count_multiplier = self.config.get("building_count_multiplier", 0.5)
        self.agents_config = self.config.get("agents", {})
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算进度奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            进度奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 获取当前智能体类型
        agent_type = self._get_agent_type_from_action(action_id)
        if not agent_type:
            return 0.0
        
        # 获取智能体配置
        agent_config = self.agents_config.get(agent_type, {})
        building_type = agent_config.get("building_type")
        if not building_type:
            return 0.0
        
        # 计算当前建筑数量
        current_building_count = self._count_buildings(state, building_type)
        
        # 计算进度奖励
        progress_reward = current_building_count * self.building_count_multiplier
        
        return progress_reward
    
    def _get_agent_type_from_action(self, action_id: int) -> str:
        """从动作ID推断智能体类型"""
        # 根据v5.0的动作映射：0-2 EDU, 3-5 IND, 6-8 COUNCIL
        if 0 <= action_id <= 2:
            return "EDU"
        elif 3 <= action_id <= 5:
            return "IND"
        elif 6 <= action_id <= 8:
            return "COUNCIL"
        else:
            return None
    
    def _count_buildings(self, state: EnvironmentState, building_type: str) -> int:
        """计算指定类型的建筑数量"""
        if not hasattr(state, 'buildings') or not state.buildings:
            return 0
        
        buildings = state.buildings.get(building_type, [])
        return len(buildings)
