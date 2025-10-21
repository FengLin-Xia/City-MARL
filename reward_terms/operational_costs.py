"""
运营成本奖励模块

实现月度运营成本计算。
"""

from typing import Dict, Any
from contracts import EnvironmentState


class OperationalCostsTerm:
    """运营成本奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("operational_costs", {})
        self.enabled = self.config.get("enabled", False)
        self.opex_rates = self.config.get("opex_rates", {
            "IND": {"S": 100, "M": 180, "L": 300},
            "EDU": {"S": 70, "M": 120, "L": 190}
        })
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算运营成本奖励（负值）
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            运营成本奖励值（负值）
        """
        if not self.enabled:
            return 0.0
        
        # 获取当前智能体类型
        agent_type = self._get_agent_type_from_action(action_id)
        if not agent_type:
            return 0.0
        
        # 获取动作大小
        action_size = self._get_action_size(action_id)
        if not action_size:
            return 0.0
        
        # 获取运营成本率
        opex_rates = self.opex_rates.get(agent_type, {})
        opex_rate = opex_rates.get(action_size, 0)
        
        # 运营成本为负值
        operational_cost = -opex_rate
        
        return operational_cost
    
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
    
    def _get_action_size(self, action_id: int) -> str:
        """获取动作大小"""
        # 根据v5.0的动作映射推断大小
        if action_id in [0, 3, 6]:  # S型
            return "S"
        elif action_id in [1, 4, 7]:  # M型
            return "M"
        elif action_id in [2, 5, 8]:  # L型
            return "L"
        else:
            return None
