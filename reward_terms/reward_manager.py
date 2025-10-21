"""
奖励管理器

统一管理所有奖励项的计算。
"""

from typing import Dict, Any, List
from contracts import EnvironmentState

# 导入所有奖励项
from .npv_reward import NPVRewardTerm
from .progress_reward import ProgressRewardTerm
from .cooperation_reward import CooperationRewardTerm
from .land_price_monetization import LandPriceMonetizationTerm
from .river_premium_reward import RiverPremiumRewardTerm
from .operational_costs import OperationalCostsTerm
from .rental_income import RentalIncomeTerm
from .proximity_reward import ProximityRewardTerm
from .zone_multipliers import ZoneMultipliersTerm
from .land_price_sensitivity import LandPriceSensitivityTerm
from .building_size_bonus import BuildingSizeBonusTerm
from .reward_scaling import RewardScalingTerm


class RewardManager:
    """奖励管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化所有奖励项
        self.reward_terms = {
            "npv": NPVRewardTerm(config),
            "progress": ProgressRewardTerm(config),
            "cooperation": CooperationRewardTerm(config),
            "land_price_monetization": LandPriceMonetizationTerm(config),
            "river_premium": RiverPremiumRewardTerm(config),
            "operational_costs": OperationalCostsTerm(config),
            "rental_income": RentalIncomeTerm(config),
            "proximity": ProximityRewardTerm(config),
            "zone_multipliers": ZoneMultipliersTerm(config),
            "land_price_sensitivity": LandPriceSensitivityTerm(config),
            "building_size_bonus": BuildingSizeBonusTerm(config),
            "reward_scaling": RewardScalingTerm(config)
        }
    
    def compute_total_reward(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算总奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            总奖励值
        """
        total_reward = 0.0
        
        # 计算所有奖励项
        for term_name, term in self.reward_terms.items():
            if term.enabled:
                reward = term.compute(prev_state, state, action_id)
                total_reward += reward
                
                # 调试信息
                if abs(reward) > 0.01:  # 只记录有意义的奖励
                    print(f"[Reward Debug] {term_name}: {reward:.3f}")
        
        return total_reward
    
    def get_reward_breakdown(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> Dict[str, float]:
        """
        获取奖励分解
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            奖励分解字典
        """
        breakdown = {}
        
        for term_name, term in self.reward_terms.items():
            if term.enabled:
                reward = term.compute(prev_state, state, action_id)
                breakdown[term_name] = reward
        
        return breakdown
