"""
v5.0 动作打分器

基于契约对象的动作评分系统。
"""

from typing import Dict, List, Any, Callable
import numpy as np

from contracts import ActionCandidate, RewardTerms, Sequence
from config_loader import ConfigLoader


class V5ActionScorer:
    """v5.0动作打分器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化打分器
        
        Args:
            config: v5.0配置
        """
        self.config = config
        self.action_params = config.get("action_params", {})
        self.rules_config = config.get("rules", {})
        
    def score_action(self, candidate: ActionCandidate, 
                    state: Dict[str, Any]) -> RewardTerms:
        """
        为动作候选打分
        
        Args:
            candidate: 动作候选
            state: 环境状态
            
        Returns:
            奖励分项
        """
        action_id = candidate.id
        meta = candidate.meta
        
        # 获取动作参数
        action_params = self.action_params.get(str(action_id), {})
        
        # 基础奖励计算
        base_reward = self._calculate_base_reward(action_params, meta, state)
        cost = self._calculate_cost(action_params, meta, state)
        prestige = self._calculate_prestige(action_params, meta, state)
        
        # 空间奖励
        proximity_reward = self._calculate_proximity_reward(candidate, state)
        diversity_reward = self._calculate_diversity_reward(candidate, state)
        
        # 其他奖励
        other_rewards = self._calculate_other_rewards(candidate, state)
        
        return RewardTerms(
            revenue=base_reward,
            cost=cost,
            prestige=prestige,
            proximity=proximity_reward,
            diversity=diversity_reward,
            other=other_rewards
        )
    
    def _calculate_base_reward(self, action_params: Dict[str, Any], 
                              meta: Dict[str, Any], state: Dict[str, Any]) -> float:
        """计算基础奖励"""
        base_reward = action_params.get("reward", 0.0)
        
        # 根据地价调整
        lp_norm = meta.get("lp_norm", 0.0)
        lp_multiplier = 1.0 + lp_norm * 0.5  # 地价越高，奖励越高
        
        return base_reward * lp_multiplier
    
    def _calculate_cost(self, action_params: Dict[str, Any], 
                       meta: Dict[str, Any], state: Dict[str, Any]) -> float:
        """计算成本"""
        base_cost = action_params.get("cost", 0.0)
        
        # 根据地价调整成本
        lp_norm = meta.get("lp_norm", 0.0)
        lp_multiplier = 1.0 + lp_norm * 0.3  # 地价越高，成本越高
        
        return base_cost * lp_multiplier
    
    def _calculate_prestige(self, action_params: Dict[str, Any], 
                           meta: Dict[str, Any], state: Dict[str, Any]) -> float:
        """计算声望"""
        base_prestige = action_params.get("prestige", 0.0)
        
        # 根据区域调整声望
        zone = meta.get("zone", "default")
        zone_multiplier = self._get_zone_multiplier(zone)
        
        return base_prestige * zone_multiplier
    
    def _calculate_proximity_reward(self, candidate: ActionCandidate, 
                                   state: Dict[str, Any]) -> float:
        """计算邻近奖励"""
        # 简化实现：基于已有建筑密度
        slots = candidate.meta.get("slots", [])
        if not slots:
            return 0.0
        
        # 计算邻近建筑数量
        proximity_count = 0
        for slot_id in slots:
            slot_neighbors = self._get_slot_neighbors(slot_id)
            for neighbor_id in slot_neighbors:
                if self._is_slot_occupied(neighbor_id, state):
                    proximity_count += 1
        
        # 邻近奖励：邻近建筑越多，奖励越高
        return proximity_count * 0.1
    
    def _calculate_diversity_reward(self, candidate: ActionCandidate, 
                                   state: Dict[str, Any]) -> float:
        """计算多样性奖励"""
        # 简化实现：基于建筑类型多样性
        agent = candidate.meta.get("agent", "")
        action_id = candidate.id
        
        # 检查是否增加了建筑类型多样性
        existing_types = self._get_existing_building_types(state)
        new_type = f"{agent}_{action_id}"
        
        if new_type not in existing_types:
            return 0.2  # 新类型奖励
        else:
            return 0.0
    
    def _calculate_other_rewards(self, candidate: ActionCandidate, 
                               state: Dict[str, Any]) -> Dict[str, float]:
        """计算其他奖励"""
        other_rewards = {}
        
        # 区域奖励
        zone = candidate.meta.get("zone", "default")
        if zone != "default":
            other_rewards["zone_bonus"] = 0.1
        
        # 时间奖励
        month = state.get("month", 0)
        if month > 12:  # 后期奖励
            other_rewards["late_game_bonus"] = 0.05
        
        return other_rewards
    
    def _get_zone_multiplier(self, zone: str) -> float:
        """获取区域乘数"""
        zone_multipliers = {
            "hub": 1.5,
            "river": 1.3,
            "center": 1.2,
            "default": 1.0
        }
        return zone_multipliers.get(zone, 1.0)
    
    def _get_slot_neighbors(self, slot_id: str) -> List[str]:
        """获取槽位邻居"""
        # 简化实现：返回空列表
        # 实际实现需要根据槽位配置获取邻居
        return []
    
    def _is_slot_occupied(self, slot_id: str, state: Dict[str, Any]) -> bool:
        """检查槽位是否被占用"""
        buildings = state.get("buildings", [])
        for building in buildings:
            if slot_id in building.get("slots", []):
                return True
        return False
    
    def _get_existing_building_types(self, state: Dict[str, Any]) -> set:
        """获取现有建筑类型"""
        buildings = state.get("buildings", [])
        types = set()
        for building in buildings:
            agent = building.get("agent", "")
            action_id = building.get("action_id", "")
            types.add(f"{agent}_{action_id}")
        return types
    
    def calculate_sequence_score(self, sequences: List[Sequence], 
                                state: Dict[str, Any]) -> List[float]:
        """
        计算序列得分
        
        Args:
            sequences: 序列列表
            state: 环境状态
            
        Returns:
            得分列表
        """
        scores = []
        
        for sequence in sequences:
            total_score = 0.0
            
            # 使用get_legacy_ids()兼容AtomicAction
            legacy_ids = sequence.get_legacy_ids()
            for action_id in legacy_ids:
                # 获取动作参数
                action_params = self.action_params.get(str(action_id), {})
                
                # 计算动作得分
                reward = action_params.get("reward", 0.0)
                cost = action_params.get("cost", 0.0)
                prestige = action_params.get("prestige", 0.0)
                
                # 简化得分计算
                action_score = reward - cost + prestige * 10
                total_score += action_score
            
            scores.append(total_score)
        
        return scores
