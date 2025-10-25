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
from logic.v5_reward_calculator import V5RewardCalculator
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
        
        # 初始化动作枚举器、评分器、选择器、奖励计算器
        self.enumerator = V5ActionEnumerator(self.config)
        self.scorer = V5ActionScorer(self.config)
        self.selector = V5SequenceSelector(self.config)
        self.reward_calculator = V5RewardCalculator(self.config)
        
        # 加载槽位数据
        self._load_slots_data()
        
        # 初始化高斯地价场系统（与v4.1相同）
        self._initialize_land_price_system()
        
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
    
    def _load_slots_data(self) -> None:
        """加载槽位数据到枚举器"""
        try:
            # 从配置获取槽位数据路径
            slots_config = self.config.get("slots", {})
            slots_path = slots_config.get("path", "")
            
            # 处理路径变量替换
            if slots_path.startswith("${paths."):
                # 从配置中获取路径
                paths_config = self.config.get("paths", {})
                path_key = slots_path.replace("${paths.", "").replace("}", "")
                slots_path = paths_config.get(path_key, "")
            
            if not slots_path:
                self.logger.warning("未找到槽位数据路径，使用默认槽位")
                # 创建一些默认槽位用于测试
                default_slots = []
                for i in range(100):  # 创建100个测试槽位
                    default_slots.append({
                        "id": f"slot_{i}",
                        "x": i % 10,
                        "y": i // 10,
                        "neighbors": [],
                        "building_level": 3
                    })
                self.enumerator.load_slots(default_slots)
                self.logger.info(f"加载了 {len(default_slots)} 个默认槽位")
            else:
                # 从文件加载槽位数据
                import os
                slots_file = slots_path  # 直接使用路径，不需要expandvars
                if os.path.exists(slots_file):
                    with open(slots_file, 'r') as f:
                        slots_data = []
                        for i, line in enumerate(f):
                            if line.strip():
                                # 解析格式: x, y, angle, building_level
                                parts = line.strip().split(',')
                                if len(parts) >= 2:
                                    x = float(parts[0].strip())
                                    y = float(parts[1].strip())
                                    angle = float(parts[2].strip()) if len(parts) > 2 else 0.0  # 解析角度
                                    building_level = int(parts[3].strip()) if len(parts) > 3 else 3  # 解析建筑等级
                                    
                                    # 生成槽位ID
                                    slot_id = f"slot_{i}"
                                    
                                    slots_data.append({
                                        "id": slot_id,
                                        "x": x,
                                        "y": y,
                                        "angle": angle,  # 添加角度信息
                                        "neighbors": [],
                                        "building_level": building_level
                                    })
                        self.enumerator.load_slots(slots_data)
                        self.logger.info(f"从文件加载了 {len(slots_data)} 个槽位")
                else:
                    self.logger.warning(f"槽位文件不存在: {slots_file}，使用默认槽位")
                    # 使用默认槽位
                    default_slots = []
                    for i in range(100):
                        default_slots.append({
                            "id": f"slot_{i}",
                            "x": i % 10,
                            "y": i // 10,
                            "neighbors": [],
                            "building_level": 3
                        })
                    self.enumerator.load_slots(default_slots)
                    self.logger.info(f"加载了 {len(default_slots)} 个默认槽位")
        except Exception as e:
            self.logger.error(f"加载槽位数据失败: {e}")
            # 使用最小默认槽位
            default_slots = [{"id": f"slot_{i}", "x": i, "y": 0, "neighbors": [], "building_level": 3} for i in range(10)]
            self.enumerator.load_slots(default_slots)
            self.logger.info(f"使用最小默认槽位: {len(default_slots)} 个")
    
    def reset(self) -> EnvironmentState:
        """重置环境到初始状态
        
        Returns:
            EnvironmentState: 初始环境状态对象
        """
        self.current_month = 0
        self.current_step = 0
        
        # 重置预算（每个episode都重新初始化预算）
        self.budget_manager = BudgetPoolManager(self.config)
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
        
        # 返回EnvironmentState对象而不是字典
        return EnvironmentState(
            month=self.current_month,
            land_prices=self._get_actual_land_prices(),
            buildings=self._get_actual_buildings(),
            budgets=self.budgets.copy(),
            slots=self._get_actual_slots()
        )
    
    def step(self, sequences: Dict[str, Sequence]) -> Tuple[EnvironmentState, Dict[str, float], bool, Dict[str, Any]]:
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
        
        # 检查是否结束
        done = self.current_month >= self.total_months
        
        info = {
            "month": self.current_month,
            "step": self.current_step,
            "budgets": self.budgets.copy(),
            "reward_terms": reward_terms_all
        }
        
        # 创建新的环境状态
        new_state = EnvironmentState(
            month=self.current_month,
            step=self.current_step,
            budgets=self.budgets.copy(),
            occupied_slots=list(self.occupied_slots),
            land_price_field=self.land_price_system.get_land_price_field() if self.land_price_system else None
        )
        
        return new_state, rewards, done, info
    
    def advance_month(self) -> None:
        """推进到下一个月"""
        self.current_month += 1
        
        # 简化版本：暂时不注入月度预算，保持现有预算
        if topic_enabled("budget"):
            self.logger.info(f"月度推进到 {self.current_month}，当前预算: {self.budgets}")
    
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
        
        # 使用V5RewardCalculator动态计算成本和奖励
        print(f"[DEBUG] 准备调用V5RewardCalculator，action_id={action_id}")
        current_state = self._get_current_environment_state()
        print(f"[DEBUG] 环境状态: month={current_state.month}, budgets={current_state.budgets}")
        
        reward_terms = self.reward_calculator.calculate_reward(
            action=cand,
            state=current_state
        )
        
        print(f"[DEBUG] V5RewardCalculator返回: cost={reward_terms.cost}, revenue={reward_terms.revenue}")
        
        cost = reward_terms.cost
        reward = reward_terms.revenue
        
        # 直接更新预算：先扣除成本，再增加奖励
        old_budget = self.budgets[agent]
        if cost > 0:
            self.budgets[agent] -= cost
        if reward > 0:
            self.budgets[agent] += reward
            
        # 同步到预算管理器
        self.budget_manager.set_budget(agent, self.budgets[agent])
        
        # 预算流调试日志
        if topic_enabled("budget_flow"):
            self.logger.info(f"[BUDGET_FLOW] {agent}: {old_budget:.1f} -> {self.budgets[agent]:.1f} (cost: {cost:.1f}, reward: {reward:.1f})")
        
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
        
        # 使用V5RewardCalculator动态计算成本和奖励
        print(f"[DEBUG] 准备调用V5RewardCalculator，action_id={action_id}")
        current_state = self._get_current_environment_state()
        print(f"[DEBUG] 环境状态: month={current_state.month}, budgets={current_state.budgets}")
        
        reward_terms = self.reward_calculator.calculate_reward(
            action=cand,
            state=current_state
        )
        
        print(f"[DEBUG] V5RewardCalculator返回: cost={reward_terms.cost}, revenue={reward_terms.revenue}")
        
        cost = reward_terms.cost
        reward = reward_terms.revenue
        
        # 直接更新预算：先扣除成本，再增加奖励
        old_budget = self.budgets[agent]
        if cost > 0:
            self.budgets[agent] -= cost
        if reward > 0:
            self.budgets[agent] += reward
            
        # 同步到预算管理器
        self.budget_manager.set_budget(agent, self.budgets[agent])
        
        # 预算流调试日志
        if topic_enabled("budget_flow"):
            self.logger.info(f"[BUDGET_FLOW] {agent}: {old_budget:.1f} -> {self.budgets[agent]:.1f} (cost: {cost:.1f}, reward: {reward:.1f})")
        
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
        
        # 优先使用多动作模式的候选索引（更准确）
        if agent in self._last_cand_idx and self._last_cand_idx[agent]:
            cand_idx = self._last_cand_idx[agent]
            
            for action in sequence.actions:
                if hasattr(action, 'point') and action.point < len(cand_idx.points):
                    point_id = cand_idx.points[action.point]
                    slots = cand_idx.point_to_slots.get(point_id, [])
                    
                    for slot_id in slots:
                        slot_info = self.enumerator.slots.get(slot_id)
                        if slot_info:
                            positions.append({
                                "slot_id": slot_id,
                                "x": slot_info.x,
                                "y": slot_info.y,
                                "z": 0.0,  # 使用默认z坐标
                                "angle": getattr(slot_info, 'angle', 0.0),  # 添加角度信息
                                "action_id": action.meta.get('action_id', -1)
                            })
        else:
            # 回退到单动作模式的方法
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
                            "z": 0.0,  # 使用默认z坐标
                            "angle": getattr(slot_info, 'angle', 0.0),  # 添加角度信息
                            "action_id": action_id
                        })
        
        return positions
    
    def _get_current_state_dict(self) -> Dict[str, Any]:
        """获取当前状态字典，用于V5RewardCalculator"""
        return {
            "month": self.current_month,
            "step": self.current_step,
            "budgets": self.budgets.copy(),
            "occupied_slots": self.occupied_slots.copy(),
            "land_price_field": self.land_price_system.get_land_price_field() if self.land_price_system else None,
            "agents": self.agents
        }
    
    def _get_current_environment_state(self) -> EnvironmentState:
        """获取当前环境状态，用于V5RewardCalculator"""
        return EnvironmentState(
            month=self.current_month,
            land_prices=self._get_actual_land_prices(),
            buildings=self._get_actual_buildings(),
            budgets=self.budgets.copy(),
            slots=self._get_actual_slots()
        )
    
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
            if cand.id == action_id:
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
        # 获取当前状态字典
        current_state = self._get_current_state_dict()
        
        # 使用scorer计算奖励
        reward_terms = self.scorer.score_action(candidate, current_state)
        
        # 计算总奖励（收入 - 成本 + 其他奖励）
        total_reward = reward_terms.revenue - reward_terms.cost + reward_terms.prestige + reward_terms.proximity + reward_terms.diversity
        
        # 添加其他奖励项
        if reward_terms.other:
            total_reward += sum(reward_terms.other.values())
        
        # 转换为字典格式
        reward_dict = reward_terms.to_dict()
        
        if topic_enabled("reward_terms"):
            self.logger.info(f"奖励计算: agent={agent}, action={candidate.id}, reward={total_reward:.2f}, terms={reward_dict}")
        
        return total_reward, reward_dict
    
    def _get_current_state_dict(self) -> Dict[str, Any]:
        """获取当前状态字典（用于奖励计算）
        
        Returns:
            状态字典
        """
        return {
            "month": self.current_month,
            "step": self.current_step,
            "budgets": self.budgets.copy(),
            "occupied_slots": list(self.occupied_slots),
            "agents": self.agents,
            "land_prices": self._get_actual_land_prices(),
            "buildings": self._get_actual_buildings(),
            "slots": self._get_actual_slots()
        }
    
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
    
    def get_phase_agents(self) -> List[str]:
        """获取当前阶段的活跃智能体
        
        Returns:
            当前阶段的智能体列表
        """
        return self.scheduler.get_active_agents(self.current_month)
    
    def get_phase_execution_mode(self) -> str:
        """获取当前阶段的执行模式
        
        Returns:
            执行模式："concurrent" 或 "sequential"
        """
        return self.scheduler.get_execution_mode(self.current_month)
    
    def get_observation(self, agent: str) -> np.ndarray:
        """获取指定智能体的观察（数值向量）
        
        Args:
            agent: 智能体名称
            
        Returns:
            数值化的观察向量
        """
        state = self.get_state_for_agent(agent)
        return self._vectorize_observation(state)
    
    def _get_actual_land_prices(self) -> np.ndarray:
        """获取实际地价数据"""
        # 从地价系统获取实际数据
        # 这里需要根据实际的地价系统实现
        # 暂时返回一个合理的默认值
        map_size = self.config.get("city", {}).get("map_size", [200, 200])
        return np.ones((map_size[0], map_size[1]), dtype=np.float32)
    
    def _get_actual_buildings(self) -> List[Dict[str, Any]]:
        """获取实际建筑数据"""
        # 从建筑系统获取实际数据
        # 这里需要根据实际的建筑系统实现
        buildings = []
        for slot_id in self.occupied_slots:
            slot_info = self.enumerator.slots.get(slot_id)
            if slot_info:
                buildings.append({
                    "slot_id": slot_id,
                    "x": slot_info.x,
                    "y": slot_info.y,
                    "z": self._get_default_z_coordinate(),  # 从配置获取z坐标
                    "type": "building"  # 简化实现
                })
        return buildings
    
    def _get_default_z_coordinate(self) -> float:
        """获取默认z坐标（从配置读取）
        
        Returns:
            z坐标值
        """
        coordinates_config = self.config.get("env", {}).get("coordinates", {})
        return coordinates_config.get("default_z", 0.0)
    
    def _get_actual_slots(self) -> List[Dict[str, Any]]:
        """获取实际槽位数据"""
        # 从槽位系统获取实际数据
        slots = []
        for slot_id, slot_info in self.enumerator.slots.items():
            slots.append({
                "id": slot_id,
                "x": slot_info.x,
                "y": slot_info.y,
                "z": self._get_default_z_coordinate(),  # 从配置获取z坐标
                "occupied": slot_id in self.occupied_slots
            })
        return slots
    
    def _vectorize_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """将状态字典转换为数值向量"""
        # 根据网络期望的输入维度创建观察向量
        obs_dim = 64  # 网络期望的输入维度
        
        # 基础特征（前10维）
        features = []
        features.append(float(state.get("month", 0)))
        features.append(float(state.get("step", 0)))
        features.append(float(state.get("budget", 0)))
        features.append(float(state.get("occupied_count", 0)))
        features.append(float(state.get("total_slots", 0)))
        
        # 添加预算历史特征（最近5步）
        agent = "IND"  # 简化实现，实际应该根据agent参数
        budget_history = self.budget_history.get(agent, [])
        for i in range(5):
            if i < len(budget_history):
                features.append(float(budget_history[-(i+1)]))
            else:
                features.append(0.0)
        
        # 扩展到目标维度（用零填充）
        while len(features) < obs_dim:
            features.append(0.0)
        
        # 截断到目标维度
        features = features[:obs_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _get_land_price(self, slot_id: str) -> float:
        """获取槽位的地价
        
        Args:
            slot_id: 槽位ID
            
        Returns:
            地价
        """
        # 获取槽位信息
        if slot_id in self.enumerator.slots:
            slot = self.enumerator.slots[slot_id]
            
            # 使用高斯地价场系统（与v4.1相同）
            if hasattr(self, 'land_price_system'):
                price = self.land_price_system.get_land_price([slot.x, slot.y])
                return max(0.0, min(1.0, float(price)))
            else:
                # 如果没有地价系统，使用配置中的基础地价
                land_price_config = self.config.get("env", {}).get("land_price", {})
                base_price = land_price_config.get("base", 1.0)
                return base_price
        else:
            # 默认地价
            return 1.0
    
    def _initialize_land_price_system(self) -> None:
        """初始化高斯地价场系统"""
        try:
            from logic.enhanced_sdf_system import GaussianLandPriceSystem
            
            # 获取地价系统配置
            land_price_config = self.config.get("land_price", {}).get("gaussian_system", {})
            
            # 创建地价系统
            self.land_price_system = GaussianLandPriceSystem(self.config)
            
            # 获取交通枢纽位置（从配置或使用默认值）
            hubs = self._get_transport_hubs()
            map_size = self._get_map_size()
            
            # 初始化系统
            self.land_price_system.initialize_system(hubs, map_size)
            
            self.logger.info(f"高斯地价场系统初始化成功，枢纽数量: {len(hubs)}")
            
        except Exception as e:
            self.logger.warning(f"高斯地价场系统初始化失败: {e}，将使用简化地价")
            self.land_price_system = None
    
    def _get_transport_hubs(self) -> List[List[int]]:
        """获取交通枢纽位置"""
        # 从配置获取枢纽位置，或使用默认值
        hubs_config = self.config.get("transport_hubs", [])
        if hubs_config:
            return hubs_config
        
        # 默认枢纽位置（基于地图中心）
        map_size = self._get_map_size()
        center_x, center_y = map_size[0] // 2, map_size[1] // 2
        return [
            [center_x - 50, center_y - 50],
            [center_x + 50, center_y - 50], 
            [center_x, center_y + 50]
        ]
    
    def _get_map_size(self) -> List[int]:
        """获取地图尺寸"""
        # 从配置获取地图尺寸，或使用默认值
        map_config = self.config.get("map", {})
        width = map_config.get("width", 256)
        height = map_config.get("height", 256)
        return [width, height]
    
    def step_phase(self, phase_agents: List[str], phase_sequences: Dict[str, Sequence]) -> Tuple[EnvironmentState, Dict[str, float], bool, Dict[str, Any]]:
        """执行一个阶段的动作
        
        Args:
            phase_agents: 当前阶段的智能体列表
            phase_sequences: 各智能体的动作序列
            
        Returns:
            (observation, rewards, done, info)
        """
        # 首先推进到下一个月（这是正确的架构）
        month_advanced = False
        if self.current_month < self.total_months:
            old_month = self.current_month
            self.advance_month()
            month_advanced = True
            if topic_enabled("phase_switch"):
                self.logger.info(f"月份推进: {old_month} -> {self.current_month}")
        
        if topic_enabled("environment_step"):
            self.logger.info(f"[ENV_STEP] 执行阶段动作: agents={phase_agents}, sequences={list(phase_sequences.keys())}")
            self.logger.info(f"[ENV_STEP] 当前状态: month={self.current_month}, step={self.current_step}")
        
        # 执行动作并计算奖励
        rewards = {}
        reward_terms_all = {}
        
        for agent in phase_agents:
            if agent in phase_sequences:
                sequence = phase_sequences[agent]
                if topic_enabled("action_execution"):
                    self.logger.info(f"[ACTION_EXEC] 执行智能体 {agent} 的动作序列: {sequence}")
                reward, reward_terms = self._execute_agent_sequence(agent, sequence)
                rewards[agent] = reward
                reward_terms_all[agent] = reward_terms
                if topic_enabled("action_execution"):
                    self.logger.info(f"[ACTION_EXEC] 智能体 {agent} 获得奖励: {reward}")
            else:
                if topic_enabled("action_execution"):
                    self.logger.info(f"[ACTION_EXEC] 智能体 {agent} 没有动作序列")
                rewards[agent] = 0.0
                reward_terms_all[agent] = {}
        
        # 更新预算历史
        for agent in self.agents:
            self.budget_history[agent].append(self.budgets[agent])
        
        # 删除重复的EnvironmentState创建
        
        # 检查是否结束
        done = self.current_month >= self.total_months
        
        # 创建新的环境状态对象（月份级别）
        new_state = EnvironmentState(
            month=self.current_month,
            land_prices=self._get_actual_land_prices(),
            buildings=self._get_actual_buildings(),
            budgets=self.budgets.copy(),
            slots=self._get_actual_slots()
        )
        
        # 只有在月份推进时才记录环境状态
        if month_advanced:
            self.env_states.append(new_state)
        
        # 为每个agent创建StepLog记录（不管是否有动作）
        for agent in phase_agents:
            # 检查是否有动作
            has_actions = False
            chosen_actions = []
            slot_positions = []
            
            if agent in phase_sequences and phase_sequences[agent]:
                sequence = phase_sequences[agent]
                if sequence.actions:
                    has_actions = True
                    chosen_actions = sequence.get_legacy_ids()
                    slot_positions = self._build_slot_positions_from_snapshot(agent, sequence)
            
            # 为无动作的agent提供默认的slot_positions（避免export严格模式错误）
            if not slot_positions:
                slot_positions = [{
                    "slot_id": "default",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "angle": 0.0,
                    "action_id": -1
                }]
            
            # 创建StepLog（有动作或无动作都记录）
            step_log = StepLog(
                t=self.current_month,
                agent=agent,
                chosen=chosen_actions,  # 有动作时记录动作ID，无动作时为空列表
                reward_terms=reward_terms_all.get(agent, {}),
                budget_snapshot=self.budgets.copy(),
                slot_positions=slot_positions
            )
            
            if topic_enabled("candidates") and has_actions:
                self.logger.info(f"[SLOT_POSITIONS] agent={agent}, found={len(slot_positions)} positions")
            
            # 存储StepLog
            self.step_logs.append(step_log)
        
        # 收集所有StepLog用于info（为所有智能体创建）
        phase_logs = []
        for agent in phase_agents:
            # 检查是否有动作
            has_actions = False
            chosen_actions = []
            slot_positions = []
            
            if agent in phase_sequences and phase_sequences[agent]:
                sequence = phase_sequences[agent]
                if sequence.actions:
                    has_actions = True
                    chosen_actions = sequence.get_legacy_ids()
                    slot_positions = self._build_slot_positions_from_snapshot(agent, sequence)
            
            # 为无动作的agent提供默认的slot_positions（避免export严格模式错误）
            if not slot_positions:
                slot_positions = [{
                    "slot_id": "default",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "angle": 0.0,
                    "action_id": -1
                }]
            
            # 创建StepLog（有动作或无动作都记录）
            step_log = StepLog(
                t=self.current_month,
                agent=agent,
                chosen=chosen_actions,
                reward_terms=reward_terms_all.get(agent, {}),
                budget_snapshot=self.budgets.copy(),
                slot_positions=slot_positions
            )
            phase_logs.append(step_log)
        
        
        info = {
            "month": self.current_month,
            "step": self.current_step,
            "budgets": self.budgets.copy(),
            "reward_terms": reward_terms_all,
            "step_log": phase_logs[0] if phase_logs else None,
            "phase_logs": phase_logs
        }
        
        return new_state, rewards, done, info
    
    
    def _calculate_action_reward(self, agent: str, action_id: int, action: Any) -> Tuple[float, Dict[str, float]]:
        """
        计算单个动作的奖励
        
        Args:
            agent: 智能体名称
            action_id: 动作ID
            action: 动作对象
            
        Returns:
            (reward, reward_terms): 奖励和奖励分项
        """
        # 获取动作参数
        action_params = self.config.get("action_params", {}).get(str(action_id), {})
        
        # 基础奖励计算
        base_reward = action_params.get("base_reward", 0.0)
        cost = action_params.get("cost", 0.0)
        prestige = action_params.get("prestige", 0.0)
        
        # 计算地价奖励
        land_price_reward = self._calculate_land_price_reward(action, agent)
        
        # 计算邻近性奖励
        proximity_reward = self._calculate_proximity_reward(action, agent)
        
        # 计算总奖励
        total_reward = base_reward + land_price_reward + proximity_reward - cost
        
        # 构建奖励分项
        reward_terms = {
            "base_reward": base_reward,
            "land_price_reward": land_price_reward,
            "proximity_reward": proximity_reward,
            "cost": -cost,
            "prestige": prestige,
            "total": total_reward
        }
        
        return total_reward, reward_terms
    
    def _calculate_land_price_reward(self, action: Any, agent: str) -> float:
        """计算地价奖励"""
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
                if slot_id in self.enumerator.slots:
                    slot = self.enumerator.slots[slot_id]
                    # 获取地价（简化计算）
                    land_price = self.land_price_system.get_land_price_at_position(
                        slot.x, slot.y
                    )
                    total_land_price += land_price
            
            # 地价奖励 = 地价 * 系数
            land_price_coeff = self.config.get("reward_terms", {}).get("land_price_coeff", 0.01)
            return total_land_price * land_price_coeff
            
        except Exception as e:
            self.logger.warning(f"计算地价奖励失败: {e}")
            return 0.0
    
    def _calculate_proximity_reward(self, action: Any, agent: str) -> float:
        """计算邻近性奖励"""
        try:
            # 获取动作的槽位信息
            if hasattr(action, 'meta') and 'slots' in action.meta:
                slots = action.meta['slots']
            else:
                return 0.0
            
            if not slots:
                return 0.0
            
            # 计算邻近性奖励（简化实现）
            proximity_coeff = self.config.get("reward_terms", {}).get("proximity_coeff", 0.1)
            return len(slots) * proximity_coeff
            
        except Exception as e:
            self.logger.warning(f"计算邻近性奖励失败: {e}")
            return 0.0
    
    def close(self) -> None:
        """关闭环境"""
        self.logger.info("环境关闭")
