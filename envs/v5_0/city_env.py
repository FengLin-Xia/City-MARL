"""
v5.0 城市环境包装器

基于契约对象和配置的环境系统。
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import deque

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from contracts import ActionCandidate, Sequence, StepLog, EnvironmentState, CandidateIndex, AtomicAction
from config_loader import ConfigLoader
from scheduler import PhaseCycleScheduler
from .budget_pool import BudgetPoolManager
from logic.v5_enumeration import V5ActionEnumerator
from logic.v5_scorer import V5ActionScorer
from logic.v5_selector import V5SequenceSelector
from utils.logger_factory import get_logger, topic_enabled, sampling_allows


class V5CityEnvironment:
    """v5.0城市环境包装器"""
    
    def __init__(self, config_path: str):
        """初始化环境
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = get_logger("env")
        self.config_path = config_path
        
        # 加载配置
        loader = ConfigLoader()
        self.config = loader.load_v5_config(config_path)
        self.agents_config = self.config.get("agents", {})
        self.scheduler_config = self.config.get("scheduler", {})
        self.budget_config = self.config.get("budget", {})
        
        # 初始化调度器
        self.scheduler = PhaseCycleScheduler(self.scheduler_config.get("params", {}))
        
        # 初始化预算管理器
        self.budget_manager = BudgetPoolManager(self.budget_config)
        
        # 初始化动作枚举器、评分器、选择器
        self.enumerator = V5ActionEnumerator(self.config)
        self.scorer = V5ActionScorer(self.config)
        self.selector = V5SequenceSelector(self.config)
        
        # 环境状态
        self.current_month = 0
        self.current_step = 0
        self.total_months = self.config.get("simulation", {}).get("total_months", 30)
        self.agents = list(self.agents_config.get("defs", {}).keys())
        
        # 预算状态
        self.budgets: Dict[str, float] = {}
        self.budget_history: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        
        # 占用槽位追踪
        self.occupied_slots: Set[str] = set()
        
        # 候选快照（用于执行时回溯）
        self._last_candidates: Dict[str, List[ActionCandidate]] = {}
        self._last_cand_idx: Dict[str, CandidateIndex] = {}
        
        # 历史记录
        self.step_logs: List[StepLog] = []
        self.env_states: List[EnvironmentState] = []
        
        self.logger.info(f"v5.0 环境初始化完成，agents={self.agents}, total_months={self.total_months}")
    
    def reset(self) -> Dict[str, Any]:
        """重置环境到初始状态
        
        Returns:
            初始观测
        """
        self.current_month = 0
        self.current_step = 0
        
        # 重置预算
        self.budget_manager.reset_all_pools()
        self.budgets = {agent: self.budget_manager.get_remaining_budget(agent) for agent in self.agents}
        self.budget_history = {agent: [self.budgets[agent]] for agent in self.agents}
        
        # 清空槽位
        self.occupied_slots.clear()
        
        # 清空历史
        self.step_logs.clear()
        self.env_states.clear()
        
        # 清空快照
        self._last_candidates.clear()
        self._last_cand_idx.clear()
        
        self.logger.info(f"环境重置，初始预算: {self.budgets}")
        
        return self._get_observation()
    
    def step(self, sequences: Dict[str, Sequence]) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """执行一个时间步
        
        Args:
            sequences: 各agent的动作序列
            
        Returns:
            (observation, rewards, done, info)
        """
        self.current_step += 1
        
        # 执行动作并计算奖励
        rewards = {}
        reward_terms_all = {}
        
        for agent, sequence in sequences.items():
            reward, reward_terms = self._execute_agent_sequence(agent, sequence)
            rewards[agent] = reward
            reward_terms_all[agent] = reward_terms
        
        # 更新预算历史
        for agent in self.agents:
            self.budget_history[agent].append(self.budgets[agent])
        
        # 记录状态
        env_state = EnvironmentState(
            month=self.current_month,
            step=self.current_step,
            budgets=self.budgets.copy(),
            occupied_slots=list(self.occupied_slots),
            meta={"reward_terms": reward_terms_all}
        )
        self.env_states.append(env_state)
        
        # 检查是否结束
        done = self.current_month >= self.total_months
        
        obs = self._get_observation()
        info = {
            "month": self.current_month,
            "step": self.current_step,
            "budgets": self.budgets.copy(),
            "reward_terms": reward_terms_all
        }
        
        return obs, rewards, done, info
    
    def advance_month(self) -> None:
        """推进到下一个月"""
        self.current_month += 1
        
        # 按配置注入预算
        for agent in self.agents:
            self.budget_manager.inject_monthly(agent, self.current_month)
            self.budgets[agent] = self.budget_manager.get_remaining_budget(agent)
        
        if topic_enabled("budget"):
            self.logger.info(f"月度推进到 {self.current_month}，新预算: {self.budgets}")
    
    def get_action_candidates(self, agent: str) -> List[ActionCandidate]:
        """获取agent的动作候选（v5.0单动作模式）
        
        Args:
            agent: 智能体名称
            
        Returns:
            候选动作列表
        """
        candidates = self.enumerator.enumerate_actions(
            agent=agent,
            occupied_slots=self.occupied_slots,
            lp_provider=self._get_land_price,
            budget=self.budgets.get(agent, 0),
            current_month=self.current_month
        )
        
        # 缓存候选
        self._last_candidates[agent] = candidates
        
        if topic_enabled("candidates"):
            self.logger.info(f"枚举候选: agent={agent}, count={len(candidates)}")
        
        return candidates
    
    def get_action_candidates_with_index(self, agent: str) -> Tuple[List[ActionCandidate], CandidateIndex]:
        """获取agent的动作候选和索引（v5.1多动作模式）
        
        Args:
            agent: 智能体名称
            
        Returns:
            (候选动作列表, 候选索引)
        """
        candidates, cand_idx = self.enumerator.enumerate_with_index(
            agent=agent,
            occupied_slots=self.occupied_slots,
            lp_provider=self._get_land_price,
            budget=self.budgets.get(agent, 0),
            current_month=self.current_month
        )
        
        # 缓存候选和索引
        self._last_candidates[agent] = candidates
        self._last_cand_idx[agent] = cand_idx
        
        if topic_enabled("candidates"):
            self.logger.info(f"枚举候选(多动作): agent={agent}, points={len(cand_idx.points)}, total_candidates={len(candidates)}")
        
        return candidates, cand_idx
    
    def _execute_agent_sequence(self, agent: str, sequence: Sequence) -> Tuple[float, Dict[str, float]]:
        """执行agent的动作序列并计算奖励
        
        Args:
            agent: 智能体名称
            sequence: 动作序列
            
        Returns:
            (总奖励, 奖励明细)
        """
        reward = 0.0
        reward_terms = {}
        
        if not sequence or not sequence.actions:
            return reward, reward_terms
        
        # 兼容性层：检查是否是新版AtomicAction
        if sequence.actions and isinstance(sequence.actions[0], AtomicAction):
            # 新版多动作模式
            for atomic_action in sequence.actions:
                action_reward, action_terms = self._execute_action_atomic(agent, atomic_action)
                reward += action_reward
                for key, val in action_terms.items():
                    reward_terms[key] = reward_terms.get(key, 0) + val
        else:
            # 旧版单动作模式（通过compatibility layer转换）
            legacy_ids = sequence.get_legacy_ids()
            for action_id in legacy_ids:
                action_reward, action_terms = self._execute_action_legacy(agent, action_id)
                reward += action_reward
                for key, val in action_terms.items():
                    reward_terms[key] = reward_terms.get(key, 0) + val
        
        return reward, reward_terms
    
    def _execute_action_atomic(self, agent: str, atomic_action: AtomicAction) -> Tuple[float, Dict[str, float]]:
        """执行单个原子动作（v5.1）
        
        Args:
            agent: 智能体名称
            atomic_action: 原子动作
            
        Returns:
            (奖励, 奖励明细)
        """
        # 从meta中获取原始action_id
        action_id = atomic_action.meta.get('action_id', atomic_action.atype)
        
        # 获取候选
        cand = self._get_candidate_from_snapshot(agent, action_id)
        if not cand:
            self.logger.warning(f"未找到action_id={action_id}的候选")
            return 0.0, {}
        
        # 扣除预算
        cost = cand.meta.get("cost", 0)
        if cost > 0:
            self.budget_manager.spend(agent, cost)
            self.budgets[agent] = self.budget_manager.get_remaining_budget(agent)
        
        # 标记槽位为已占用
        for slot_id in cand.meta.get("slots", []):
            self.occupied_slots.add(slot_id)
            if topic_enabled("occupied_slots"):
                self.logger.info(f"[SLOT_MARK] agent={agent} slot={slot_id} action_id={action_id}")
        
        # 计算奖励
        reward, reward_terms = self._compute_reward(agent, cand)
        
        return reward, reward_terms
    
    def _execute_action_legacy(self, agent: str, action_id: int) -> Tuple[float, Dict[str, float]]:
        """执行单个动作（v5.0兼容）
        
        Args:
            agent: 智能体名称
            action_id: 动作ID
            
        Returns:
            (奖励, 奖励明细)
        """
        # 获取候选
        cand = self._get_candidate_from_snapshot(agent, action_id)
        if not cand:
            self.logger.warning(f"未找到action_id={action_id}的候选")
            return 0.0, {}
        
        # 扣除预算
        cost = cand.meta.get("cost", 0)
        if cost > 0:
            self.budget_manager.spend(agent, cost)
            self.budgets[agent] = self.budget_manager.get_remaining_budget(agent)
        
        # 标记槽位为已占用
        for slot_id in cand.meta.get("slots", []):
            self.occupied_slots.add(slot_id)
            if topic_enabled("occupied_slots"):
                self.logger.info(f"[SLOT_MARK] agent={agent} slot={slot_id} action_id={action_id}")
        
        # 计算奖励
        reward, reward_terms = self._compute_reward(agent, cand)
        
        return reward, reward_terms
    
    def _update_occupied_slots_from_snapshot(self, agent: str, sequence: Sequence) -> None:
        """使用候选快照更新已占用槽位（兼容AtomicAction）"""
        if not sequence or not sequence.actions:
            return
        
        # 调试：检查数据类型
        if topic_enabled("occupied_slots"):
            self.logger.info(f"[SLOT_DEBUG] _update_occupied_slots agent={agent}")
            self.logger.info(f"[SLOT_DEBUG]   sequence.actions type: {type(sequence.actions)}")
            self.logger.info(f"[SLOT_DEBUG]   sequence.actions[0] type: {type(sequence.actions[0]) if sequence.actions else 'empty'}")
        
        # 获取legacy IDs（兼容新旧版本）
        legacy_ids = sequence.get_legacy_ids()
        if topic_enabled("occupied_slots"):
            self.logger.info(f"[SLOT_DEBUG]   legacy_ids: {legacy_ids}")
        
        for action_id in legacy_ids:
            cand = self._get_candidate_from_snapshot(agent, action_id)
            if topic_enabled("occupied_slots"):
                self.logger.info(f"[SLOT_DEBUG]   action_id={action_id}, found_cand={cand is not None}")
            if not cand:
                if topic_enabled("occupied_slots"):
                    self.logger.warning(f"[SLOT_DEBUG]     WARNING: No candidate found for action_id={action_id}")
                continue
            slots = cand.meta.get("slots", [])
            if topic_enabled("occupied_slots"):
                self.logger.info(f"[SLOT_DEBUG]     slots to occupy: {slots}")
            for slot_id in slots:
                self.occupied_slots.add(slot_id)
                if topic_enabled("occupied_slots"):
                    self.logger.info(f"[SLOT_DEBUG]     Added slot {slot_id} to occupied_slots")
                    self.logger.info(f"occupied agent={agent} slot={slot_id} month={self.current_month} step={self.current_step}")

    def _build_slot_positions_from_snapshot(self, agent: str, sequence: Sequence) -> List[Dict[str, Any]]:
        """根据候选快照构建槽位位置信息（兼容AtomicAction）"""
        positions: List[Dict[str, Any]] = []
        if not sequence or not sequence.actions:
            return positions
        
        # 获取legacy IDs（兼容新旧版本）
        legacy_ids = sequence.get_legacy_ids()
        
        for action_id in legacy_ids:
            cand = self._get_candidate_from_snapshot(agent, action_id)
            if not cand:
                continue
            
            for slot_id in cand.meta.get("slots", []):
                slot_info = self.enumerator.slots.get(slot_id)
                if slot_info:
                    positions.append({
                        "slot_id": slot_id,
                        "x": slot_info.x,
                        "y": slot_info.y,
                        "z": slot_info.z,
                        "action_id": action_id
                    })
        
        return positions
    
    def _get_candidate_from_snapshot(self, agent: str, action_id: int) -> Optional[ActionCandidate]:
        """从快照中获取候选
        
        Args:
            agent: 智能体名称
            action_id: 动作ID
            
        Returns:
            候选动作或None
        """
        candidates = self._last_candidates.get(agent, [])
        for cand in candidates:
            if cand.action_id == action_id:
                return cand
        return None
    
    def _compute_reward(self, agent: str, candidate: ActionCandidate) -> Tuple[float, Dict[str, float]]:
        """计算动作奖励
        
        Args:
            agent: 智能体名称
            candidate: 候选动作
            
        Returns:
            (总奖励, 奖励明细)
        """
        # 使用scorer计算奖励
        reward = self.scorer.compute_reward(agent, candidate, self.current_month)
        reward_terms = candidate.meta.get("reward_terms", {})
        
        if topic_enabled("reward_terms"):
            self.logger.info(f"奖励计算: agent={agent}, action={candidate.action_id}, reward={reward:.2f}, terms={reward_terms}")
        
        return reward, reward_terms
    
    def _get_land_price(self, x: float, y: float) -> float:
        """获取指定位置的地价
        
        Args:
            x, y: 坐标
            
        Returns:
            地价值
        """
        # 简化实现：返回固定值
        # 实际应该调用land_price模块
        return 1.0
    
    def _get_observation(self) -> Dict[str, Any]:
        """获取当前观测
        
        Returns:
            观测字典
        """
        return {
            "month": self.current_month,
            "step": self.current_step,
            "budgets": self.budgets.copy(),
            "occupied_count": len(self.occupied_slots),
            "agents": self.agents
        }
    
    def get_state_for_agent(self, agent: str) -> Dict[str, Any]:
        """获取agent的状态向量
        
        Args:
            agent: 智能体名称
            
        Returns:
            状态字典
        """
        return {
            "month": self.current_month,
            "step": self.current_step,
            "budget": self.budgets.get(agent, 0),
            "occupied_count": len(self.occupied_slots),
            "total_slots": len(self.enumerator.slots)
        }
    
    def render(self, mode='human') -> None:
        """渲染环境（当前为占位实现）"""
        pass
    
    def close(self) -> None:
        """关闭环境"""
        self.logger.info("环境关闭")
