"""
v5.0 奖励计算器

基于v4.1逻辑的完整reward计算系统。
"""

from typing import Dict, List, Any, Tuple
import numpy as np

from contracts import ActionCandidate, EnvironmentState, RewardTerms
from .v5_land_price_calculator import V5LandPriceCalculator
from .v5_proximity_calculator import V5ProximityCalculator
from .v5_river_calculator import V5RiverCalculator
from .v5_size_calculator import V5SizeCalculator


class V5RewardCalculator:
    """v5.0奖励计算器 - 参照v4.1的_calc_crp逻辑"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化奖励计算器
        
        Args:
            config: v5.0配置
        """
        self.config = config
        self.reward_config = config.get("reward_terms", {})
        
        # 初始化子计算器
        self.land_price_calculator = V5LandPriceCalculator(config)
        self.proximity_calculator = V5ProximityCalculator(config)
        self.river_calculator = V5RiverCalculator(config)
        self.size_calculator = V5SizeCalculator(config)
        
        # 获取配置参数
        self.enabled = self.reward_config.get("enabled", True)
        self.debug_mode = self.reward_config.get("debug_mode", False)
        self.precision = self.reward_config.get("precision", "float")
        self.rounding_mode = self.reward_config.get("rounding_mode", "nearest")
        
        # 组件启用状态
        self.components = self.reward_config.get("components", {
            "land_price": True,
            "proximity": True,
            "river": True,
            "size_bonus": True
        })
    
    def calculate_reward(self, action: ActionCandidate, state: EnvironmentState) -> RewardTerms:
        """
        计算完整的reward - 参照v4.1的_calc_crp方法
        
        Args:
            action: 动作候选
            state: 环境状态
            
        Returns:
            RewardTerms: 奖励分项
        """
        if not self.enabled:
            return RewardTerms(
                base_reward=0.0,
                land_price_reward=0.0,
                proximity_reward=0.0,
                river_premium=0.0,
                size_bonus=0.0,
                cost=0.0,
                total=0.0
            )
        
        # 1. 基础成本计算
        base_cost = self._calculate_base_cost(action, state)
        
        # 2. 基础收入计算
        base_reward = self._calculate_base_reward(action, state)
        
        # 3. 地价敏感度计算
        land_price_reward = 0.0
        if self.components.get("land_price", True):
            land_price_reward = self.land_price_calculator.calculate_land_price_sensitivity(action, state)
        
        # 4. 邻近性奖励计算
        proximity_reward = 0.0
        if self.components.get("proximity", True):
            proximity_reward = self.proximity_calculator.calculate_proximity_reward(action, state)
        
        # 5. 河流溢价计算
        river_premium = 0.0
        if self.components.get("river", True):
            river_premium = self.river_calculator.calculate_river_premium(action, state)
        
        # 6. 规模奖励计算
        size_bonus = 0.0
        if self.components.get("size_bonus", True):
            size_bonus = self.size_calculator.calculate_size_bonus(action, state)
        
        # 计算总奖励 - 采用 budget - cost + reward 公式
        # 获取智能体预算
        agent_budget = self._get_agent_budget(action, state)
        
        # 计算剩余预算
        remaining_budget = agent_budget - base_cost
        
        # 计算总奖励: 剩余预算 + 各种奖励
        total_reward = remaining_budget + base_reward + land_price_reward + proximity_reward + river_premium + size_bonus
        
        # 应用舍入
        if self.rounding_mode == "nearest":
            total_reward = round(total_reward)
            base_reward = round(base_reward)
            land_price_reward = round(land_price_reward)
            proximity_reward = round(proximity_reward)
            river_premium = round(river_premium)
            size_bonus = round(size_bonus)
            base_cost = round(base_cost)
        
        # 创建奖励分项
        reward_terms = RewardTerms(
            revenue=base_reward,
            cost=-base_cost,  # 成本为负值
            prestige=0.0,  # 声望暂时设为0
            proximity=proximity_reward,
            diversity=0.0,  # 多样性暂时设为0
            other={
                "land_price_reward": land_price_reward,
                "river_premium": river_premium,
                "size_bonus": size_bonus,
                "remaining_budget": remaining_budget,
                "agent_budget": agent_budget,
                "total": total_reward
            }
        )
        
        if self.debug_mode:
            print(f"[REWARD_DEBUG] Action {action.id}: base={base_reward}, land_price={land_price_reward}, "
                  f"proximity={proximity_reward}, river={river_premium}, size={size_bonus}, "
                  f"cost={base_cost}, total={total_reward}")
        
        return reward_terms
    
    def _calculate_base_cost(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """计算基础成本"""
        # 从配置中获取动作参数
        action_params = self.config.get("action_params", {}).get(str(action.id), {})
        base_cost = action_params.get("cost", 0.0)  # 使用配置中的cost字段
        
        return base_cost
    
    def _calculate_base_reward(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """计算基础收入"""
        # 从配置中获取动作参数
        action_params = self.config.get("action_params", {}).get(str(action.id), {})
        base_reward = action_params.get("reward", 0.0)  # 使用配置中的reward字段
        
        return base_reward
    
    def _get_agent_budget(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """获取智能体预算"""
        # 从环境状态中获取智能体预算
        if hasattr(state, 'budgets') and state.budgets:
            # 根据动作ID确定智能体类型
            if action.id in [0, 1, 2]:  # EDU
                return state.budgets.get('EDU', 0.0)
            elif action.id in [3, 4, 5]:  # IND
                return state.budgets.get('IND', 0.0)
            elif action.id in [6, 7, 8]:  # COUNCIL
                return state.budgets.get('COUNCIL', 0.0)
        
        # 默认预算
        return 20000.0
