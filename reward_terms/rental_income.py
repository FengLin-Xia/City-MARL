"""
租金收入奖励模块

实现EDU的租金收入机制。
"""

from typing import Dict, Any
from contracts import EnvironmentState


class RentalIncomeTerm:
    """租金收入奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("rental_income", {})
        self.enabled = self.config.get("enabled", False)
        self.rent_rates = self.config.get("rent_rates", {"S": 25, "M": 45, "L": 70})
        self.edu_only = self.config.get("edu_only", True)
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算租金收入奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            租金收入奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 获取当前智能体类型
        agent_type = self._get_agent_type_from_action(action_id)
        if not agent_type:
            return 0.0
        
        # 只有EDU获得租金收入
        if self.edu_only and agent_type != "EDU":
            return 0.0
        
        # 获取动作大小
        action_size = self._get_action_size(action_id)
        if not action_size:
            return 0.0
        
        # 获取租金率
        rent_rate = self.rent_rates.get(action_size, 0)
        
        return rent_rate
    
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
        if action_id in [0, 3, 6]:  # S型
            return "S"
        elif action_id in [1, 4, 7]:  # M型
            return "M"
        elif action_id in [2, 5, 8]:  # L型
            return "L"
        else:
            return None
