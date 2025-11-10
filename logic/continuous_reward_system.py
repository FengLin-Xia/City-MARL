"""
v5.0 持续奖励系统

实现持续奖励与协同效应机制，包含增量奖励和潜势塑形。
"""

from typing import Dict, List, Any, Optional
import logging
from contracts import EnvironmentState, BuildingRegistry

logger = logging.getLogger(__name__)


class ContinuousRewardSystem:
    """持续奖励系统 - 实现增量奖励和潜势塑形机制"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化持续奖励系统
        
        Args:
            config: v5.0配置
        """
        self.config = config
        self.continuous_config = config.get("continuous_rewards", {})
        self.synergy_config = config.get("synergy_effects", {})
        
        # 获取配置参数
        self.gamma = self.continuous_config.get("gamma", 0.99)
        self.enabled = self.continuous_config.get("enabled", True)
        
        # 历史月度总收益（用于增量奖励和潜势塑形）
        self.monthly_totals = {}  # agent -> monthly_total
        
        # 动作奖励率缓存
        self.action_reward_rates = {}
        self._initialize_action_reward_rates()
        
        logger.info(f"ContinuousRewardSystem initialized: enabled={self.enabled}, gamma={self.gamma}")
    
    def _initialize_action_reward_rates(self):
        """初始化动作奖励率"""
        action_params = self.config.get("action_params", {})
        
        for action_id_str, params in action_params.items():
            action_id = int(action_id_str)
            base_reward = params.get("base_reward", 100.0)
            self.action_reward_rates[action_id] = base_reward
        
        logger.info(f"Initialized action reward rates: {self.action_reward_rates}")
    
    def calculate_monthly_total_rewards(self, state: EnvironmentState) -> Dict[str, float]:
        """
        计算所有agent的月度总收益（包含协同效应）
        
        Args:
            state: 环境状态
            
        Returns:
            每个agent的月度总收益
        """
        if not self.enabled:
            return {}
        
        total_rewards = {}
        
        # 获取所有agent
        agents = list(state.budgets.keys())
        
        for agent in agents:
            agent_total = 0.0
            
            # 获取该agent的所有建筑
            if state.building_registry and agent in state.building_registry.buildings_by_agent:
                agent_buildings = state.building_registry.buildings_by_agent[agent]
                
                for building_id in agent_buildings:
                    if building_id in state.building_registry.buildings:
                        building = state.building_registry.buildings[building_id]
                        action_id = building.action_id
                        
                        # 计算基础奖励率
                        base_rate = self.action_reward_rates.get(action_id, 0.0)
                        
                        # 应用协同效应
                        synergy_multiplier = self.calculate_synergy_multiplier(action_id, state)
                        
                        # 最终月度收益
                        monthly_income = base_rate * synergy_multiplier
                        agent_total += monthly_income
            
            total_rewards[agent] = agent_total
        
        logger.debug(f"Calculated monthly total rewards: {total_rewards}")
        return total_rewards
    
    def calculate_delta_reward(self, agent: str, monthly_total_t: float) -> float:
        """
        计算增量奖励
        
        Args:
            agent: 智能体名称
            monthly_total_t: 当前月度总收益
            
        Returns:
            增量奖励
        """
        if not self.enabled:
            return 0.0
        
        if agent not in self.monthly_totals:
            self.monthly_totals[agent] = 0.0
        
        monthly_total_prev = self.monthly_totals[agent]
        delta_reward = monthly_total_t - monthly_total_prev
        
        # 更新历史记录
        self.monthly_totals[agent] = monthly_total_t
        
        logger.debug(f"Delta reward for {agent}: {monthly_total_prev} -> {monthly_total_t} = {delta_reward}")
        return delta_reward
    
    def calculate_potential_shaping(self, agent: str, monthly_total_t: float) -> float:
        """
        计算潜势塑形奖励
        
        Args:
            agent: 智能体名称
            monthly_total_t: 当前月度总收益
            
        Returns:
            潜势塑形奖励
        """
        if not self.enabled:
            return 0.0
        
        if agent not in self.monthly_totals:
            return 0.0
        
        monthly_total_prev = self.monthly_totals[agent]
        
        # 潜势函数
        phi_s = monthly_total_prev / (1 - self.gamma)
        phi_sp = monthly_total_t / (1 - self.gamma)
        
        # 塑形项
        shaping_reward = self.gamma * phi_sp - phi_s
        
        logger.debug(f"Potential shaping for {agent}: phi_s={phi_s:.2f}, phi_sp={phi_sp:.2f}, shaping={shaping_reward:.2f}")
        return shaping_reward
    
    def calculate_synergy_multiplier(self, action_id: int, state: EnvironmentState) -> float:
        """
        计算协同效应倍数
        
        Args:
            action_id: 动作ID
            state: 环境状态
            
        Returns:
            协同效应倍数
        """
        if not self.enabled:
            return 1.0
        
        multiplier = 1.0
        
        # 工业集群协同效应
        industrial_cluster_config = self.synergy_config.get("industrial_cluster", {})
        if industrial_cluster_config.get("enabled", False):
            affects_actions = industrial_cluster_config.get("affects_actions", [3, 4, 5])
            trigger_actions = industrial_cluster_config.get("trigger_actions", [9, 10, 11])
            
            if action_id in affects_actions:
                if self._is_synergy_active(state, trigger_actions):
                    bonus_rate = industrial_cluster_config.get("bonus_rate", 0.15)
                    multiplier += bonus_rate
                    
                    logger.debug(f"Synergy bonus applied: action_id={action_id}, bonus_rate={bonus_rate}, multiplier={multiplier}")
        
        return multiplier
    
    def _is_synergy_active(self, state: EnvironmentState, trigger_actions: List[int]) -> bool:
        """
        检查协同效应是否激活
        
        Args:
            state: 环境状态
            trigger_actions: 触发动作列表
            
        Returns:
            是否激活协同效应
        """
        if not state.building_registry:
            return False
        
        # 检查是否有触发动作被建造
        for action_id in trigger_actions:
            if state.building_registry.action_counts.get(action_id, 0) > 0:
                return True
        
        return False
    
    def activate_synergy(self, action_id: int, state: EnvironmentState):
        """
        激活协同效应
        
        Args:
            action_id: 动作ID
            state: 环境状态
        """
        if not self.enabled:
            return
        
        # 工业集群协同效应
        industrial_cluster_config = self.synergy_config.get("industrial_cluster", {})
        if industrial_cluster_config.get("enabled", False):
            trigger_actions = industrial_cluster_config.get("trigger_actions", [9, 10, 11])
            
            if action_id in trigger_actions:
                state.synergy_activations["industrial_cluster"] = True
                logger.info(f"工业集群协同效应已激活，动作{action_id}被建造")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        获取状态摘要
        
        Returns:
            状态摘要
        """
        return {
            "enabled": self.enabled,
            "gamma": self.gamma,
            "monthly_totals": self.monthly_totals.copy(),
            "action_reward_rates": self.action_reward_rates.copy()
        }

