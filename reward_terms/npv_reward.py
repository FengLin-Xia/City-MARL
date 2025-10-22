"""
NPV计算奖励模块

实现净现值计算，考虑未来收益的现值。
"""

from typing import Dict, Any, Tuple
import numpy as np
from contracts import EnvironmentState


class NPVRewardTerm:
    """NPV计算奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("npv_calculation", {})
        self.enabled = self.config.get("enabled", False)
        self.expected_lifetime = self.config.get("expected_lifetime", 12)
        self.discount_rate = self.config.get("discount_rate", 0.05)
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算NPV奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            NPV奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 获取动作参数
        action_params = self._get_action_params(action_id)
        if not action_params:
            return 0.0
        
        # 计算建造成本
        build_cost = action_params.get("cost", 0.0)
        if build_cost <= 0:
            return 0.0
        
        # 计算月度收益
        monthly_reward = action_params.get("reward", 0.0)
        if monthly_reward <= 0:
            return 0.0
        
        # 计算未来收益现值
        future_income_pv = self._calculate_present_value(monthly_reward)
        
        # NPV = 未来收益现值 - 建造成本
        npv = future_income_pv - build_cost
        
        return npv
    
    def _get_action_params(self, action_id: int) -> Dict[str, Any]:
        """获取动作参数"""
        # 这里需要从配置中获取动作参数
        # 简化实现，返回默认值
        return {
            "cost": 1000.0,
            "reward": 100.0
        }
    
    def _calculate_present_value(self, monthly_income: float) -> float:
        """计算现值"""
        if self.discount_rate <= 0:
            # 无折现
            return monthly_income * self.expected_lifetime
        
        # 使用年金现值公式
        # PV = PMT * [(1 - (1 + r)^-n) / r]
        # 其中 PMT = monthly_income, r = discount_rate/12, n = expected_lifetime
        monthly_rate = self.discount_rate / 12
        if monthly_rate == 0:
            return monthly_income * self.expected_lifetime
        
        pv_factor = (1 - (1 + monthly_rate) ** (-self.expected_lifetime)) / monthly_rate
        present_value = monthly_income * pv_factor
        
        return present_value

