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
        """v5.0 完整 CRP 计算 (cost / reward / prestige)"""
        print(f"[V5RewardCalculator] 被调用了！action_id={action.id}, state_month={state.month}")
        
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

        # ==== 读取动作与环境元数据 ====
        meta: Dict[str, Any] = getattr(action, "meta", {}) or {}
        zone = meta.get("zone", "far")               # near / mid / far
        land_price_norm = float(meta.get("land_price_norm", 0.0))  # 0~1
        river_dist_m = float(meta.get("river_dist_m", 1e9))        # 米
        adj = int(meta.get("adj", 0))                # 0/1 是否邻接

        # ==== 读取静态动作参数 ====
        ap = self.config.get("action_params", {}).get(str(action.id), {})
        base_cost = ap.get("base_cost", 0.0)
        base_reward = ap.get("base_reward", 0.0)
        opex = ap.get("opex", 0.0)
        rent = ap.get("rent", 0.0)
        land_price_k = ap.get("land_price_k", 0.0)
        river_pct = ap.get("river_pct", 0.0)
        size_bonus_cfg = ap.get("size_bonus", 0.0)

        # ==== 区位常数 ====
        zone_cost_tbl = self.reward_config.get("zone_cost", {"near": 200, "mid": 100, "far": 0})
        zone_reward_tbl = self.reward_config.get("zone_reward", {"near": 80, "mid": 40, "far": 0})
        zone_cost = zone_cost_tbl.get(zone, 0.0)
        zone_reward = zone_reward_tbl.get(zone, 0.0)

        # ==== 成本计算 ====
        land_price_cost = land_price_norm * 1000.0
        cost = base_cost + zone_cost + land_price_cost

        # ==== 收益各项 ====
        # 河流溢价
        river_premium = 0.0
        if self.components.get("river", True):
            river_premium = self.river_calculator.calculate_river_premium(action, state)
        # 基础 reward_base
        reward_base = base_reward + zone_reward - opex + river_premium
        # 租金逻辑
        if action.id in [3, 4, 5]:  # IND pays rent
            reward_base -= rent
        elif action.id in [0, 1, 2]:  # EDU receives rent
            reward_base += rent
        # land_price 放大
        reward_lp_factor = 1.0 + land_price_k * land_price_norm
        reward = reward_base * reward_lp_factor

        # ==== 规模/邻近奖励 ====
        proximity_bonus = 0.0
        if self.components.get("proximity", True):
            proximity_bonus = self.proximity_calculator.calculate_proximity_reward(action, state)
        size_bonus = 0.0
        if self.components.get("size_bonus", True):
            size_bonus = size_bonus_cfg  # 已由表给定

        total_reward = reward + proximity_bonus + size_bonus

        # ==== prestige 计算（简要） ====
        prestige = ap.get("prestige_base", 0.0)
        if zone == "near":
            prestige += 1.0
        prestige += adj * 1.0
        pollution_penalty = ap.get("pollution", 0.0)
        prestige -= pollution_penalty

        # ==== 构造返回 ====
        return RewardTerms(
            revenue=total_reward,
            cost=-cost,
            prestige=prestige,
            proximity=proximity_bonus,
            diversity=0.0,
            other={
                "river_premium": river_premium,
                "land_price_cost": land_price_cost,
                "zone_cost": zone_cost,
                "zone_reward": zone_reward,
                "size_bonus": size_bonus,
                "base_reward": base_reward,
                "opex": opex,
                "rent": rent,
                "land_price_k": land_price_k,
                "river_pct": river_pct,
                "zone": zone,
                "land_price_norm": land_price_norm,
                "river_dist_m": river_dist_m,
                "adj": adj,
                "total": total_reward
            }
        )
    
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
