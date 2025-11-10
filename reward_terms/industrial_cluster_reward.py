"""
工业集群协同奖励模块

实现高级工业建筑对传统工业建筑的全局奖励加成。
"""

from typing import Dict, Any, List
from contracts import EnvironmentState


class IndustrialClusterRewardTerm:
    """工业集群协同奖励项"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward_mechanisms", {}).get("industrial_cluster_bonus", {})
        self.enabled = self.config.get("enabled", False)
        self.bonus_rate = self.config.get("bonus_rate", 0.15)
        self.affects_actions = self.config.get("affects_actions", [3, 4, 5])
        self.trigger_actions = self.config.get("trigger_actions", [9, 10, 11])
        self.global_effect = self.config.get("global_effect", True)
    
    def compute(self, prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float:
        """
        计算工业集群协同奖励
        
        Args:
            prev_state: 前一个状态
            state: 当前状态
            action_id: 动作ID
            
        Returns:
            协同奖励值
        """
        if not self.enabled:
            return 0.0
        
        # 只对传统工业建筑计算协同奖励
        if action_id not in self.affects_actions:
            return 0.0
        
        # 检查是否有高级工业建筑激活协同效应
        if not self._is_cluster_bonus_active(state):
            return 0.0
        
        # 计算基础奖励
        base_reward = self._get_base_reward(action_id)
        
        # 应用协同加成
        cluster_bonus = base_reward * self.bonus_rate
        
        return cluster_bonus
    
    def _is_cluster_bonus_active(self, state: EnvironmentState) -> bool:
        """检查协同效应是否激活"""
        # 从状态中获取建筑注册表信息
        if hasattr(state, 'building_registry') and state.building_registry:
            advanced_count = sum(
                state.building_registry.action_counts.get(aid, 0) 
                for aid in self.trigger_actions
            )
            return advanced_count > 0
        
        return False
    
    def _get_base_reward(self, action_id: int) -> float:
        """获取基础奖励值"""
        # 从配置中获取基础奖励，如果没有则使用默认值
        action_params = self.config.get("action_params", {}).get(str(action_id), {})
        base_reward = action_params.get("base_reward", 100.0)  # 默认基础奖励100
        return base_reward
