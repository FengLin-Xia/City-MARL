"""
v5.0 序列选择器

基于契约对象的序列选择系统。
"""

from typing import Dict, List, Any, Optional
import random

from contracts import Sequence, StepLog
from config_loader import ConfigLoader


class V5SequenceSelector:
    """v5.0序列选择器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化选择器
        
        Args:
            config: v5.0配置
        """
        self.config = config
        self.constraints = config.get("constraints", {})
        self.agents_config = config.get("agents", {})
        
    def select_sequence(self, agent: str, candidates: List[Sequence], 
                       state: Dict[str, Any], mode: str = "greedy") -> Optional[Sequence]:
        """
        选择最优序列
        
        Args:
            agent: 智能体名称
            candidates: 候选序列列表
            state: 环境状态
            mode: 选择模式 ("greedy", "random", "weighted")
            
        Returns:
            选中的序列，如果没有合法序列则返回None
        """
        if not candidates:
            return None
        
        # 过滤合法序列
        valid_sequences = self._filter_valid_sequences(agent, candidates, state)
        if not valid_sequences:
            return None
        
        # 根据模式选择
        if mode == "greedy":
            return self._select_greedy(valid_sequences, state)
        elif mode == "random":
            return self._select_random(valid_sequences)
        elif mode == "weighted":
            return self._select_weighted(valid_sequences, state)
        else:
            return self._select_greedy(valid_sequences, state)
    
    def _filter_valid_sequences(self, agent: str, candidates: List[Sequence], 
                               state: Dict[str, Any]) -> List[Sequence]:
        """过滤合法序列"""
        valid_sequences = []
        
        for sequence in candidates:
            if self._is_sequence_valid(agent, sequence, state):
                valid_sequences.append(sequence)
        
        return valid_sequences
    
    def _is_sequence_valid(self, agent: str, sequence: Sequence, 
                          state: Dict[str, Any]) -> bool:
        """检查序列是否合法"""
        # 检查智能体匹配
        if sequence.agent != agent:
            return False
        
        # 检查动作数量限制
        agent_config = self.agents_config.get("defs", {}).get(agent, {})
        max_actions = agent_config.get("constraints", {}).get("max_actions_per_turn", 1)
        if len(sequence.actions) > max_actions:
            return False
        
        # 检查预算限制
        if not self._check_budget_constraint(agent, sequence, state):
            return False
        
        # 检查空间约束
        if not self._check_spatial_constraint(sequence, state):
            return False
        
        return True
    
    def _check_budget_constraint(self, agent: str, sequence: Sequence, 
                                state: Dict[str, Any]) -> bool:
        """检查预算约束"""
        budgets = state.get("budgets", {})
        agent_budget = budgets.get(agent, 0.0)
        
        total_cost = 0.0
        for action_id in sequence.actions:
            action_params = self.config.get("action_params", {}).get(str(action_id), {})
            cost = action_params.get("cost", 0.0)
            total_cost += cost
        
        return total_cost <= agent_budget
    
    def _check_spatial_constraint(self, sequence: Sequence, 
                                 state: Dict[str, Any]) -> bool:
        """检查空间约束"""
        occupied_slots = set()
        buildings = state.get("buildings", [])
        for building in buildings:
            occupied_slots.update(building.get("slots", []))
        
        # 简化实现：假设所有动作都使用不同槽位
        # 实际实现需要根据动作类型检查空间冲突
        return True
    
    def _select_greedy(self, sequences: List[Sequence], 
                      state: Dict[str, Any]) -> Sequence:
        """贪心选择"""
        if not sequences:
            return None
        
        # 计算每个序列的得分
        scores = []
        for sequence in sequences:
            score = self._calculate_sequence_score(sequence, state)
            scores.append(score)
        
        # 选择得分最高的序列
        best_idx = scores.index(max(scores))
        return sequences[best_idx]
    
    def _select_random(self, sequences: List[Sequence]) -> Sequence:
        """随机选择"""
        if not sequences:
            return None
        return random.choice(sequences)
    
    def _select_weighted(self, sequences: List[Sequence], 
                        state: Dict[str, Any]) -> Sequence:
        """加权选择"""
        if not sequences:
            return None
        
        # 计算得分和权重
        scores = []
        for sequence in sequences:
            score = self._calculate_sequence_score(sequence, state)
            scores.append(max(score, 0.001))  # 避免负权重
        
        # 归一化权重
        total_score = sum(scores)
        if total_score == 0:
            return random.choice(sequences)
        
        weights = [score / total_score for score in scores]
        
        # 加权随机选择
        return random.choices(sequences, weights=weights)[0]
    
    def _calculate_sequence_score(self, sequence: Sequence, 
                                state: Dict[str, Any]) -> float:
        """计算序列得分"""
        total_score = 0.0
        
        for action_id in sequence.actions:
            # 获取动作参数
            action_params = self.config.get("action_params", {}).get(str(action_id), {})
            
            # 基础得分
            reward = action_params.get("reward", 0.0)
            cost = action_params.get("cost", 0.0)
            prestige = action_params.get("prestige", 0.0)
            
            # 计算动作得分
            action_score = reward - cost + prestige * 10
            total_score += action_score
        
        return total_score
    
    def create_step_log(self, step: int, agent: str, sequence: Sequence, 
                       reward_terms: Dict[str, float], 
                       budget_snapshot: Optional[Dict[str, float]] = None) -> StepLog:
        """
        创建步骤日志
        
        Args:
            step: 步骤编号
            agent: 智能体名称
            sequence: 执行的序列
            reward_terms: 奖励分项
            budget_snapshot: 预算快照
            
        Returns:
            步骤日志
        """
        return StepLog(
            t=step,
            agent=agent,
            chosen=sequence.actions,
            reward_terms=reward_terms,
            budget_snapshot=budget_snapshot
        )
