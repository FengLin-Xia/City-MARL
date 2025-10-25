"""
v5.0 地价敏感度计算器

参照v4.1的LP计算逻辑。
"""

from typing import Dict, Any
import numpy as np

from contracts import ActionCandidate, EnvironmentState


class V5LandPriceCalculator:
    """地价相关计算 - 参照v4.1的LP计算逻辑"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化地价计算器
        
        Args:
            config: v5.0配置
        """
        self.config = config
        self.land_price_config = config.get("reward_terms", {}).get("land_price", {})
        
        # 获取配置参数
        self.lp_idx_min = self.land_price_config.get("LP_idx_min", 10)
        self.lp_idx_max = self.land_price_config.get("LP_idx_max", 100)
        self.land_price_base = self.land_price_config.get("LandPriceBase", 11.0)
        self.reward_lp_k = self.land_price_config.get("RewardLP_k", {"IND": 0.25, "EDU": 0.10, "COUNCIL": 0.10})
    
    def calculate_land_price_sensitivity(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """
        计算地价敏感度奖励 - 参照v4.1的LP_norm和k_lp逻辑
        
        Args:
            action: 动作候选
            state: 环境状态
            
        Returns:
            float: 地价敏感度奖励
        """
        try:
            # 1. 计算LP_norm (地价归一化值)
            lp_norm = self._calculate_lp_norm(action, state)
            
            # 2. 计算k_lp (地价敏感度系数)
            agent = self._get_agent_from_action(action)
            k_lp = self.reward_lp_k.get(agent, 0.1)
            
            # 3. 计算reward *= (1 + k_lp * LP_norm)
            land_price_reward = lp_norm * k_lp * 100.0  # 基础倍数
            
            return land_price_reward
            
        except Exception as e:
            print(f"[LAND_PRICE_ERROR] 计算地价敏感度失败: {e}")
            return 0.0
    
    def _calculate_lp_norm(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """计算地价归一化值"""
        try:
            # 获取动作的槽位信息
            if hasattr(action, 'meta') and 'slots' in action.meta:
                slots = action.meta['slots']
            else:
                return 0.0
            
            if not slots:
                return 0.0
            
            # 计算槽位的地价
            total_land_price = 0.0
            for slot_id in slots:
                # 这里需要从环境中获取地价信息
                # 简化实现：使用槽位坐标计算地价
                land_price = self._get_land_price_at_slot(slot_id, state)
                total_land_price += land_price
            
            # 归一化到[0, 1]范围
            lp_norm = min(1.0, max(0.0, total_land_price / 1000.0))
            
            return lp_norm
            
        except Exception as e:
            print(f"[LAND_PRICE_ERROR] 计算LP_norm失败: {e}")
            return 0.0
    
    def _get_land_price_at_slot(self, slot_id: str, state: EnvironmentState) -> float:
        """获取槽位的地价"""
        try:
            # 这里需要从环境中获取地价信息
            # 简化实现：返回随机地价
            import random
            return random.uniform(50.0, 200.0)
        except Exception:
            return 100.0
    
    def _get_agent_from_action(self, action: ActionCandidate) -> str:
        """从动作ID推断智能体类型"""
        action_id = action.id
        
        if action_id in [0, 1, 2]:
            return "EDU"
        elif action_id in [3, 4, 5]:
            return "IND"
        elif action_id in [6, 7, 8]:
            return "COUNCIL"
        else:
            return "UNKNOWN"




