"""
建筑规模奖励模块

实现建筑规模奖励，鼓励建造M/L型建筑。
"""

from typing import Dict, Any
from contracts import EnvironmentState


class BuildingSizeBonusTerm:
    """建筑规模奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("building_size_bonus", {})
        self.enabled = self.config.get("enabled", False)
        self.size_bonus_rates = self.config.get("size_bonus_rates", {"S": 0, "M": 0.1, "L": 0.2})
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算建筑规模奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            建筑规模奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 获取动作大小
        action_size = self._get_action_size(action_id)
        if not action_size:
            return 0.0
        
        # 获取规模奖励率
        size_bonus_rate = self.size_bonus_rates.get(action_size, 0.0)
        
        # 计算规模奖励
        size_bonus = size_bonus_rate * 100.0  # 基础奖励值
        
        return size_bonus
    
    def _get_action_size(self, action_id: int) -> str:
        """获取动作大小"""
        if action_id in [0, 3, 6]:  # S型
            return "S"
        elif action_id in [1, 4, 7]:  # M型
            return "M"
        elif action_id in [2, 5, 8]:  # L型
            return "L"
        else:
            return None
