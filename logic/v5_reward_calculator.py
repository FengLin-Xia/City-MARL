"""
v5.0 奖励计算器

基于v4.1逻辑的完整reward计算系统。
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

from contracts import ActionCandidate, EnvironmentState, RewardTerms
from .v5_land_price_calculator import V5LandPriceCalculator
from .v5_proximity_calculator import V5ProximityCalculator
from .v5_river_calculator import V5RiverCalculator
from .v5_size_calculator import V5SizeCalculator
from reward_terms.action_diversity_reward import ActionDiversityRewardTerm


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
        self.diversity_calculator = ActionDiversityRewardTerm(config)
        
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
            "size_bonus": True,
            "diversity": True,
            "hub_distance": True  # 新增：Hub距离奖励
        })
        
        # Hub3距离奖励配置
        hub_distance_config = config.get("reward_mechanisms", {}).get("hub_distance_reward", {})
        self.hub_distance_enabled = hub_distance_config.get("enabled", True)
        self.hub_distance_config = hub_distance_config
        
        # 维护前一个状态用于多样性计算（按agent）
        self.prev_states: Dict[str, Optional[EnvironmentState]] = {}
    
    def calculate_reward(self, action: ActionCandidate, state: EnvironmentState) -> RewardTerms:
        """v5.0 完整 CRP 计算 (cost / reward / prestige)"""
        logger.info(f"[V5RewardCalculator] 被调用了！action_id={action.id}, state_month={state.month}")
        
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

        # ==== 工业集群协同奖励 ====
        cluster_bonus = self._calculate_industrial_cluster_bonus(action, state)
        
        # ==== Hub3距离奖励（新增） ====
        hub3_distance_reward = 0.0
        if self.components.get("hub_distance", True):
            hub3_distance_reward = self._calculate_hub3_distance_reward(action, state)
        
        total_reward = reward + proximity_bonus + size_bonus + cluster_bonus + hub3_distance_reward

        # ==== 多样性奖励 ====
        diversity_reward = 0.0
        if self.components.get("diversity", True):
            # 获取当前智能体（从action.meta中获取，因为state可能没有current_agent字段）
            agent = meta.get("agent", "UNKNOWN")
            if agent == "UNKNOWN":
                # 尝试从state中获取（如果state有current_agent属性）
                agent = getattr(state, 'current_agent', "UNKNOWN")
            
            # 获取前一个状态
            prev_state = self.prev_states.get(agent, None)
            
            # 计算多样性奖励（如果没有prev_state，使用当前state作为prev_state）
            if prev_state is None:
                # 第一次调用，使用当前state作为prev_state（多样性计算器会处理这种情况）
                diversity_reward = self.diversity_calculator.compute(state, state, action.id, agent=agent)
            else:
                diversity_reward = self.diversity_calculator.compute(prev_state, state, action.id, agent=agent)
            
            # 更新prev_state（为下次调用准备）
            # 注意：这里保存的是当前state，下次调用时作为prev_state使用
            self.prev_states[agent] = state
            
            if abs(diversity_reward) > 0.01:
                logger.info(f"[V5RewardCalculator] 多样性奖励: agent={agent}, action_id={action.id}, diversity_reward={diversity_reward:.3f}")

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
            diversity=diversity_reward,
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
                "cluster_bonus": cluster_bonus,
                "hub3_distance_reward": hub3_distance_reward,  # 新增
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
    
    def _calculate_industrial_cluster_bonus(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """
        计算工业集群协同奖励
        
        Args:
            action: 动作候选
            state: 环境状态
            
        Returns:
            协同奖励值
        """
        # 获取工业集群配置
        cluster_config = self.config.get("reward_mechanisms", {}).get("industrial_cluster_bonus", {})
        if not cluster_config.get("enabled", False):
            return 0.0
        
        # 只对传统工业建筑计算协同奖励
        affects_actions = cluster_config.get("affects_actions", [3, 4, 5])
        if action.id not in affects_actions:
            return 0.0
        
        # 检查是否有高级工业建筑激活协同效应
        trigger_actions = cluster_config.get("trigger_actions", [9, 10, 11])
        if not self._is_cluster_bonus_active(state, trigger_actions):
            return 0.0
        
        # 计算基础奖励
        ap = self.config.get("action_params", {}).get(str(action.id), {})
        base_reward = ap.get("base_reward", 100.0)
        
        # 应用协同加成
        bonus_rate = cluster_config.get("bonus_rate", 0.15)
        cluster_bonus = base_reward * bonus_rate
        
        logger.info(f"[V5RewardCalculator] 工业集群协同奖励: action_id={action.id}, base_reward={base_reward}, bonus_rate={bonus_rate}, cluster_bonus={cluster_bonus}")
        
        return cluster_bonus
    
    def _calculate_hub3_distance_reward(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """
        计算基于Hub3距离的奖励
        
        规则：
        - 动作9（IND_A）：离Hub3越远，奖励越大
        - 动作11（IND_C）：离Hub3越近，奖励越大
        - 动作10（IND_B）：中性或根据距离调整
        
        Args:
            action: 动作候选
            state: 环境状态
            
        Returns:
            Hub3距离奖励值
        """
        if not self.hub_distance_enabled:
            return 0.0
        
        action_id = action.id
        
        # 只处理动作9、10、11
        if action_id not in [9, 10, 11]:
            return 0.0
        
        # 获取距离信息（从meta中获取）
        meta = getattr(action, "meta", {}) or {}
        hub3_distance = float(meta.get("hub3_distance", float('inf')))
        
        # 调试日志：输出距离信息
        logger.info(f"[HubDistanceReward] action_id={action_id}, hub3_distance={hub3_distance}, meta_keys={list(meta.keys())}")
        
        if hub3_distance == float('inf') or hub3_distance < 0:
            logger.warning(f"[HubDistanceReward] 距离无效: action_id={action_id}, hub3_distance={hub3_distance}")
            return 0.0
        
        # 获取配置参数
        action9_config = self.hub_distance_config.get("action9", {
            "far_bonus_scale": 20.0,      # 离Hub3远的奖励系数
            "near_penalty_scale": -50.0,   # 离Hub3近的惩罚系数
            "threshold": 10.0              # 距离阈值
        })
        action10_config = self.hub_distance_config.get("action10", {
            "bonus_scale": 0.0,  # 中性，无奖励
            "threshold": 10.0
        })
        action11_config = self.hub_distance_config.get("action11", {
            "near_bonus_scale": 100.0,     # 离Hub3近的奖励系数（从配置读取，默认100.0）
            "far_penalty_scale": -20.0,    # 离Hub3远的惩罚系数
            "threshold": 10.0              # 距离阈值
        })
        
        # 根据动作ID和距离计算奖励
        if action_id == 9:
            # 动作9：离Hub3越远，奖励越大
            threshold = action9_config.get("threshold", 10.0)
            far_bonus_scale = action9_config.get("far_bonus_scale", 20.0)
            near_penalty_scale = action9_config.get("near_penalty_scale", -50.0)
            
            if hub3_distance < threshold:
                # 离Hub3近，扣分（距离越近，扣分越多）
                penalty = (threshold - hub3_distance) * abs(near_penalty_scale) / threshold
                reward = -penalty
            else:
                # 离Hub3远，加分（距离越远，加分越多）
                bonus = (hub3_distance - threshold) * far_bonus_scale / 10.0
                reward = bonus
                
        elif action_id == 11:
            # 动作11：离Hub3越近，奖励越大
            threshold = action11_config.get("threshold", 10.0)
            near_bonus_scale = action11_config.get("near_bonus_scale", 50.0)
            far_penalty_scale = action11_config.get("far_penalty_scale", -20.0)
            
            if hub3_distance < threshold:
                # 离Hub3近，加分（距离越近，加分越多）
                bonus = (threshold - hub3_distance) * near_bonus_scale / threshold
                reward = bonus
            else:
                # 离Hub3远，扣分（距离越远，扣分越多）
                penalty = (hub3_distance - threshold) * abs(far_penalty_scale) / 10.0
                reward = -penalty
                
        else:  # action_id == 10
            # 动作10：中性，无奖励
            reward = 0.0
        
        # 始终输出日志（用于调试）
        threshold_str = f"{threshold:.1f}" if action_id in [9, 11] else "N/A"
        logger.info(f"[HubDistanceReward] action_id={action_id}, distance={hub3_distance:.2f}, reward={reward:.3f}, threshold={threshold_str}")
        
        return reward
    
    def _is_cluster_bonus_active(self, state: EnvironmentState, trigger_actions: List[int]) -> bool:
        """
        检查协同效应是否激活
        
        Args:
            state: 环境状态
            trigger_actions: 触发动作列表
            
        Returns:
            是否激活协同效应
        """
        if not hasattr(state, 'building_registry') or not state.building_registry:
            return False
        
        # 检查是否有高级工业建筑
        advanced_count = sum(
            state.building_registry.action_counts.get(aid, 0)
            for aid in trigger_actions
        )
        
        logger.info(f"[V5RewardCalculator] 检查协同效应: trigger_actions={trigger_actions}, advanced_count={advanced_count}")
        
        return advanced_count > 0
