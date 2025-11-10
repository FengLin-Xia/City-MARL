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

from contracts import ActionCandidate, Sequence, StepLog, EnvironmentState, CandidateIndex, AtomicAction, BuildingRegistry
from config_loader import ConfigLoader
from scheduler import PhaseCycleScheduler
from .budget_pool import BudgetPoolManager
from logic.v5_enumeration import V5ActionEnumerator
from logic.v5_selector import V5SequenceSelector
from logic.v5_reward_calculator import V5RewardCalculator
from utils.logger_factory import get_logger, topic_enabled, sampling_allows

# 导入中间件
from action_mw.unlock_gate import UnlockGateMW
from envs.land_price_evo import LandPriceEvo


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
        
        # 初始化动作枚举器、选择器、奖励计算器
        self.enumerator = V5ActionEnumerator(self.config)
        self.selector = V5SequenceSelector(self.config)
        self.reward_calculator = V5RewardCalculator(self.config)
        
        # 初始化持续奖励系统
        from logic.continuous_reward_system import ContinuousRewardSystem
        self.continuous_reward_system = ContinuousRewardSystem(self.config)
        
        # 加载槽位数据
        self._load_slots_data()
        
        # 初始化地价演化系统（整合原有高斯地价系统）
        self._initialize_land_price_evolution()
        
        # 初始化解锁中间件
        self.unlock_middleware = UnlockGateMW()
        
        # 环境状态
        self.current_month = 0
        self.current_step = 0
        # 优先读取 env.time_model.total_steps，回退到 simulation.total_months（保持兼容）
        time_model_cfg = self.config.get("env", {}).get("time_model", {})
        self.total_months = int(time_model_cfg.get(
            "total_steps",
            self.config.get("simulation", {}).get("total_months", 30)
        ))
        self.agents = list(self.agents_config.get("defs", {}).keys())
        
        # 预算状态
        self.budgets: Dict[str, float] = {}
        self.budget_history: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        
        # 占用槽位追踪
        self.occupied_slots: Set[str] = set()
        
        # 建筑注册表
        self.building_registry = BuildingRegistry()
        self.global_occupied_slots: Set[str] = set()
        
        # 候选快照（用于执行时回溯）
        self._last_candidates: Dict[str, List[ActionCandidate]] = {}
        self._last_cand_idx: Dict[str, CandidateIndex] = {}
        
        # 当前智能体槽位占用跟踪
        self._current_agent_occupied_slots: Dict[str, Set[str]] = {}
        
        # 全局槽位ID占用跟踪（基于slot_id去重）
        self._global_occupied_slot_ids: Set[str] = set()
        
        # 全局坐标占用跟踪（基于浮点坐标去重）
        self._global_occupied_coordinates: Set[Tuple[float, float]] = set()

        # 新增：已建设（执行期真正落地）的槽位集合，用于区分“执行占位”与“非执行占位”
        self._built_slot_ids: Set[str] = set()
        
        # 历史记录
        self.step_logs: List[StepLog] = []
        self.env_states: List[EnvironmentState] = []
        
        self.logger.info(f"v5.0 环境初始化完成，agents={self.agents}, total_months={self.total_months}")
    
    def _is_slot_id_occupied(self, slot_id: str) -> bool:
        """检查槽位ID是否已被占用
        
        Args:
            slot_id: 要检查的槽位ID
            
        Returns:
            是否已被占用
        """
        return slot_id in self._global_occupied_slot_ids
    
    def _mark_slot_id_occupied(self, slot_id: str) -> None:
        """标记槽位ID为已占用
        
        Args:
            slot_id: 要标记的槽位ID
        """
        self._global_occupied_slot_ids.add(slot_id)
        print(f"[SLOT_ID_MARK] 标记槽位 {slot_id} 为已占用")
        try:
            self.logger.warning(f"[OCCUPY_MARK] slot_id={slot_id} month={self.current_month}")
        except Exception:
            pass
    
    def _is_coordinate_occupied(self, x: float, y: float) -> bool:
        """检查坐标是否已被占用（基于浮点坐标）
        
        Args:
            x, y: 要检查的坐标
            
        Returns:
            是否已被占用
        """
        coord = (round(x, 1), round(y, 1))  # 保留1位小数精度
        return coord in self._global_occupied_coordinates
    
    def _mark_coordinate_occupied(self, x: float, y: float) -> None:
        """标记坐标为已占用（基于浮点坐标）
        
        Args:
            x, y: 要标记的坐标
        """
        coord = (round(x, 1), round(y, 1))  # 保留1位小数精度
        self._global_occupied_coordinates.add(coord)
        print(f"[COORDINATE_MARK] 标记坐标 {coord} 为已占用")
    
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
        self.global_occupied_slots.clear()
        
        # 清空历史
        self.step_logs.clear()
        self.env_states.clear()
        
        # 清空快照
        self._last_candidates.clear()
        self._last_cand_idx.clear()
        
        # 清空当前智能体槽位占用
        self._current_agent_occupied_slots.clear()
        
        # 清空全局槽位ID占用
        self._global_occupied_slot_ids.clear()
        
        # 清空全局坐标占用
        self._global_occupied_coordinates.clear()

        # 清空已建设槽位集合
        self._built_slot_ids.clear()
        
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
            # 应用中间件处理序列（包括解锁中间件）
            processed_sequence = self._apply_middleware(agent, sequence)
            
            # 修复：在动作执行前更新槽位状态，防止重复选择
            self._update_occupied_slots_for_agent(agent, processed_sequence)
            
            reward, reward_terms = self._execute_agent_sequence(agent, processed_sequence)
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
            land_price_field=self.land_price_evo.get_land_price_field() if hasattr(self, 'land_price_evo') and self.land_price_evo else (self.land_price_system.get_land_price_field() if self.land_price_system else None)
        )
        
        return new_state, rewards, done, info
    
    def advance_month(self) -> None:
        """推进到下一个月"""
        self.current_month += 1
        
        # 修复：清空当前月份的槽位占用
        self.occupied_slots.clear()
        
        # 清空当前智能体槽位占用
        self._current_agent_occupied_slots.clear()
        
        # 保持全局槽位ID占用（禁止跨月重复选择）
        # self._global_occupied_slot_ids.clear()  # 注释掉，禁止跨月重复选择
        # self._global_occupied_coordinates.clear()  # 注释掉，禁止跨月重复选择
        print(f"[CROSS_MONTH_DEDUP] 跨月推进，保持全局槽位占用: {len(self._global_occupied_slot_ids)} 个槽位, {len(self._global_occupied_coordinates)} 个坐标")
        
        # 修复：保持全局槽位占用，禁止跨月重复选择
        # 注意：如果希望槽位跨月保持占用，应该注释掉下面这行
        # self.global_occupied_slots.clear()  # 注释掉，禁止跨月重复选择
        
        # 简化版本：暂时不注入月度预算，保持现有预算
        if topic_enabled("budget"):
            self.logger.info(f"月度推进到 {self.current_month}，当前预算: {self.budgets}")
        
        # 新增：计算月度总收益并更新budget
        self._update_monthly_rewards()
        
        if topic_enabled("occupied_slots"):
            self.logger.info(f"月度重置：清空槽位占用状态，occupied_slots={len(self.occupied_slots)}, global_occupied_slots={len(self.global_occupied_slots)}")
        
        # 更新地价演化系统
        self._update_land_price_evolution()
    
    def _update_monthly_rewards(self):
        """更新月度奖励并更新budget"""
        # 获取当前环境状态
        current_state = self._get_current_environment_state()
        
        # 计算月度总收益
        monthly_totals = self.continuous_reward_system.calculate_monthly_total_rewards(current_state)
        
        # 更新budget（包含持续收益）
        for agent in self.agents:
            monthly_income = monthly_totals.get(agent, 0.0)
            self.budgets[agent] += monthly_income
            
            if topic_enabled("budget_flow"):
                self.logger.info(f"[BUDGET_FLOW] {agent}: 持续收益 +{monthly_income:.1f}")
        
        # 更新环境状态中的月度奖励
        current_state.monthly_rewards = monthly_totals.copy()
        
        # 记录历史
        for agent in self.agents:
            if agent not in current_state.monthly_totals_history:
                current_state.monthly_totals_history[agent] = []
            current_state.monthly_totals_history[agent].append(monthly_totals.get(agent, 0.0))
    
    def get_action_candidates(self, agent: str) -> List[ActionCandidate]:
        """获取agent的动作候选（v5.0单动作模式）
        
        Args:
            agent: 智能体名称
            
        Returns:
            候选动作列表
        """
        # 修复：使用全局去重状态而不是当前月份的occupied_slots
        global_occupied = self._global_occupied_slot_ids.union(self.occupied_slots)
        # 获取解锁状态
        print(f"[DEBUG] Before _get_unlocked_actions for agent {agent}, month {self.current_month}")
        unlocked_actions = self._get_unlocked_actions(agent)
        print(f"[DEBUG] After _get_unlocked_actions, unlocked_actions={unlocked_actions}")
        
        candidates = self.enumerator.enumerate_actions(
            agent=agent,
            occupied_slots=global_occupied,
            lp_provider=self._get_land_price,
            budget=self.budgets.get(agent, 0),
            current_month=self.current_month,
            unlocked_actions=unlocked_actions
        )
        
        # 新增：过滤全局已占用的槽位
        filtered_candidates = []
        print(f"[CANDIDATE_FILTER] Agent {agent}, total candidates before filtering: {len(candidates)}")
        print(f"[CANDIDATE_FILTER] Global occupied slot_ids: {len(self._global_occupied_slot_ids)}")
        
        for candidate in candidates:
            slots = candidate.meta.get("slots", [])
            # 检查是否有槽位已被占用（使用我们的去重机制）
            if not any(self._is_slot_id_occupied(slot_id) for slot_id in slots):
                filtered_candidates.append(candidate)
            else:
                print(f"[CANDIDATE_FILTER] Filtered out candidate {candidate.id} with slots {slots}")
        
        print(f"[CANDIDATE_FILTER] Agent {agent}, filtered candidates: {len(filtered_candidates)}")
        
        # 缓存过滤后的候选
        self._last_candidates[agent] = filtered_candidates
        
        if topic_enabled("candidates"):
            self.logger.info(f"枚举候选: agent={agent}, count={len(filtered_candidates)}")
        
        return filtered_candidates
    
    def get_action_candidates_with_index(self, agent: str) -> Tuple[List[ActionCandidate], CandidateIndex]:
        """获取agent的动作候选和索引（v5.1多动作模式）
        
        Args:
            agent: 智能体名称
            
        Returns:
            (候选动作列表, 候选索引)
        """
        # 修复：使用全局去重状态而不是当前月份的occupied_slots
        global_occupied = self._global_occupied_slot_ids.union(self.occupied_slots)
        # 获取解锁状态
        print(f"[DEBUG] get_action_candidates_with_index: Before _get_unlocked_actions for agent {agent}, month {self.current_month}")
        unlocked_actions = self._get_unlocked_actions(agent)
        print(f"[DEBUG] get_action_candidates_with_index: After _get_unlocked_actions, unlocked_actions={unlocked_actions}")
        
        candidates, cand_idx = self.enumerator.enumerate_with_index(
            agent=agent,
            occupied_slots=global_occupied,
            lp_provider=self._get_land_price,
            budget=self.budgets.get(agent, 0),
            current_month=self.current_month,
            unlocked_actions=unlocked_actions
        )
        
        # 修复：过滤全局已占用的槽位，包括当前智能体已选择的槽位
        filtered_candidates = []
        print(f"[CANDIDATE_FILTER] Agent {agent}, total candidates before filtering: {len(candidates)}")
        print(f"[CANDIDATE_FILTER] Global occupied slot_ids: {len(self._global_occupied_slot_ids)}")
        
        # 获取当前智能体已选择的槽位
        current_agent_occupied = set()
        for other_agent in self.agents:
            if other_agent != agent:
                # 从其他智能体的历史记录中获取已占用的槽位
                if hasattr(self, '_agent_occupied_slots') and other_agent in self._agent_occupied_slots:
                    current_agent_occupied.update(self._agent_occupied_slots[other_agent])
        
        # 修复：添加当前智能体在当前月份已选择的槽位
        if hasattr(self, '_current_agent_occupied_slots') and agent in self._current_agent_occupied_slots:
            current_agent_occupied.update(self._current_agent_occupied_slots[agent])
        
        for candidate in candidates:
            slots = candidate.meta.get("slots", [])
            
            # 检查槽位是否可用（使用统一的slot_id检查）
            slots_available = True
            for slot_id in slots:
                print(f"[SLOT_ID_CHECK] Checking candidate {candidate.id} slot {slot_id}")
                print(f"[SLOT_ID_CHECK] Global occupied slot_ids: {self._global_occupied_slot_ids}")
                
                # 检查是否被全局占用或当前智能体占用
                if (self._is_slot_id_occupied(slot_id) or 
                    slot_id in current_agent_occupied):
                    slots_available = False
                    print(f"[SLOT_ID_FILTER] Filtered out candidate {candidate.id} with slot_id {slot_id}")
                    print(f"[SLOT_ID_FILTER] Reason: slot_id_occupied={self._is_slot_id_occupied(slot_id)}, current_agent_occupied={slot_id in current_agent_occupied}")
                    break
            
            if slots_available:
                filtered_candidates.append(candidate)
        
        print(f"[CANDIDATE_FILTER] Agent {agent}, filtered candidates: {len(filtered_candidates)}")
        
        # 缓存过滤后的候选和索引
        self._last_candidates[agent] = filtered_candidates
        self._last_cand_idx[agent] = cand_idx
        
        if topic_enabled("candidates"):
            self.logger.info(f"枚举候选(多动作): agent={agent}, points={len(cand_idx.points)}, total_candidates={len(filtered_candidates)}")
        
        return filtered_candidates, cand_idx
    
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
                    if isinstance(val, (int, float)):
                        reward_terms[key] = reward_terms.get(key, 0) + val
                    else:
                        # 对于非数值类型，直接赋值（不累加）
                        reward_terms[key] = val
        else:
            # 旧版单动作模式（通过compatibility layer转换）
            legacy_ids = sequence.get_legacy_ids()
            for action_id in legacy_ids:
                action_reward, action_terms = self._execute_action_legacy(agent, action_id)
                reward += action_reward
                for key, val in action_terms.items():
                    if isinstance(val, (int, float)):
                        reward_terms[key] = reward_terms.get(key, 0) + val
                    else:
                        # 对于非数值类型，直接赋值（不累加）
                        reward_terms[key] = val
        
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
        # 注意：V5RewardCalculator返回的cost是负数，需要直接加（负数相加等于减法）
        old_budget = self.budgets[agent]
        if cost != 0:
            self.budgets[agent] += cost  # cost已经是负数，直接加就等于减法
        if reward > 0:
            self.budgets[agent] += reward
            
        # 同步到预算管理器
        self.budget_manager.set_budget(agent, self.budgets[agent])
        
        # 预算流调试日志
        if topic_enabled("budget_flow"):
            self.logger.info(f"[BUDGET_FLOW] {agent}: {old_budget:.1f} -> {self.budgets[agent]:.1f} (cost: {cost:.1f}, reward: {reward:.1f})")
        
        # 标记槽位为已占用（执行成功后才写入全局集）
        # 日志：记录EXEC标记时使用的数据源
        slots_from_meta = cand.meta.get("slots", [])
        point_id_from_meta = cand.meta.get("point_id", None)
        slots_from_idx = []
        
        # 尝试从cand_idx获取slots
        if agent in self._last_cand_idx and self._last_cand_idx[agent]:
            cand_idx = self._last_cand_idx[agent]
            if hasattr(atomic_action, 'point') and atomic_action.point < len(cand_idx.points):
                point_id_from_action = cand_idx.points[atomic_action.point]
                slots_from_idx = cand_idx.point_to_slots.get(point_id_from_action, [])
                try:
                    self.logger.warning(f"[EXEC_SLOTS] agent={agent} action_id={action_id} month={self.current_month} point={atomic_action.point} point_id_from_action={point_id_from_action} point_id_from_meta={point_id_from_meta} slots_from_meta={slots_from_meta} slots_from_idx={slots_from_idx}")
                except Exception:
                    pass
        
        # 修复：优先使用slots_from_idx（来自action.point），而不是slots_from_meta（来自cand.meta）
        # 这样可以确保EXEC标记的槽位与实际执行的槽位一致，与slot_positions构建使用同一数据源
        if slots_from_idx:
            # 优先使用slots_from_idx（从cand_idx.point_to_slots获取，与slot_positions构建一致）
            slots_to_mark = slots_from_idx
        else:
            # 回退到使用slots_from_meta（如果没有cand_idx，比如单动作模式）
            slots_to_mark = slots_from_meta
        
        for slot_id in slots_to_mark:
            self.occupied_slots.add(slot_id)
            # 修复：添加到当前智能体槽位占用
            if agent not in self._current_agent_occupied_slots:
                self._current_agent_occupied_slots[agent] = set()
            self._current_agent_occupied_slots[agent].add(slot_id)
            
            # 修复：添加到全局槽位ID占用
            self._mark_slot_id_occupied(slot_id)
            print(f"[SLOT_ID_MARK] agent={agent} slot_id={slot_id}")
            try:
                self.logger.warning(f"[OCCUPY_FROM_EXEC] agent={agent} slot_id={slot_id} month={self.current_month}")
            except Exception:
                pass

            # 新增：记录为"已建设"占位，用于与非执行占位区分
            self._built_slot_ids.add(slot_id)
            
            if topic_enabled("occupied_slots"):
                self.logger.info(f"[SLOT_MARK] agent={agent} slot={slot_id} action_id={action_id}")
        
        # 更新建筑注册表
        self._update_building_registry(agent, action_id, cand, reward)
        
        # 计算奖励
        reward, reward_terms = self._compute_reward(agent, cand)
        
        return reward, reward_terms
    
    def _update_building_registry(self, agent: str, action_id: int, candidate: ActionCandidate, reward: float) -> None:
        """更新建筑注册表"""
        # 生成建筑ID
        building_id = f"{agent}_{action_id}_{self.current_month}_{len(self.building_registry.buildings)}"
        
        # 获取位置信息
        position = (0.0, 0.0)  # 简化实现，实际可以从candidate.meta中获取
        if "slots" in candidate.meta and candidate.meta["slots"]:
            slot_id = candidate.meta["slots"][0]
            # 从枚举器的槽位数据中获取位置
            if hasattr(self.enumerator, 'slots') and self.enumerator.slots:
                for slot in self.enumerator.slots:
                    # 检查slot是字典还是字符串
                    if isinstance(slot, dict) and slot.get("id") == slot_id:
                        position = (slot.get("x", 0.0), slot.get("y", 0.0))
                        break
                    elif isinstance(slot, str) and slot == slot_id:
                        # 如果是字符串，使用默认位置
                        position = (0.0, 0.0)
                        break
        
        # 创建建筑信息
        from contracts import BuildingInfo
        building_info = BuildingInfo(
            building_id=building_id,
            action_id=action_id,
            agent=agent,
            month=self.current_month,
            position=position,
            cost=abs(candidate.meta.get("cost", 0)),  # 成本（正值）
            reward=reward
        )
        
        # 更新注册表
        self.building_registry.buildings[building_id] = building_info
        self.building_registry.action_counts[action_id] = self.building_registry.action_counts.get(action_id, 0) + 1
        
        # 新增：按agent和动作类型分组
        if agent not in self.building_registry.buildings_by_agent:
            self.building_registry.buildings_by_agent[agent] = []
        self.building_registry.buildings_by_agent[agent].append(building_id)
        
        if action_id not in self.building_registry.buildings_by_action:
            self.building_registry.buildings_by_action[action_id] = []
        self.building_registry.buildings_by_action[action_id].append(building_id)
        
        # 激活协同效应
        self.continuous_reward_system.activate_synergy(action_id, self._get_current_environment_state())
        
        # 调试日志
        if topic_enabled("building_registry"):
            self.logger.info(f"[BUILDING_REGISTRY] Added building: {building_id}, action_id={action_id}, count={self.building_registry.action_counts[action_id]}")
    
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
        # 注意：V5RewardCalculator返回的cost是负数，需要直接加（负数相加等于减法）
        old_budget = self.budgets[agent]
        if cost != 0:
            self.budgets[agent] += cost  # cost已经是负数，直接加就等于减法
        if reward > 0:
            self.budgets[agent] += reward
            
        # 同步到预算管理器
        self.budget_manager.set_budget(agent, self.budgets[agent])
        
        # 预算流调试日志
        if topic_enabled("budget_flow"):
            self.logger.info(f"[BUDGET_FLOW] {agent}: {old_budget:.1f} -> {self.budgets[agent]:.1f} (cost: {cost:.1f}, reward: {reward:.1f})")
        
        # 标记槽位为已占用（执行成功后才写入全局集）
        for slot_id in cand.meta.get("slots", []):
            self.occupied_slots.add(slot_id)
            # 修复：添加到当前智能体槽位占用
            if agent not in self._current_agent_occupied_slots:
                self._current_agent_occupied_slots[agent] = set()
            self._current_agent_occupied_slots[agent].add(slot_id)
            
            # 修复：添加到全局槽位ID占用
            self._mark_slot_id_occupied(slot_id)
            print(f"[SLOT_ID_MARK] agent={agent} slot_id={slot_id}")
            try:
                self.logger.warning(f"[OCCUPY_FROM_EXEC] agent={agent} slot_id={slot_id} month={self.current_month}")
            except Exception:
                pass

            # 新增：记录为“已建设”占位，用于与非执行占位区分
            self._built_slot_ids.add(slot_id)
            
            if topic_enabled("occupied_slots"):
                self.logger.info(f"[SLOT_MARK] agent={agent} slot={slot_id} action_id={action_id}")
        
        # 计算奖励
        reward, reward_terms = self._compute_reward(agent, cand)
        
        return reward, reward_terms
    
    def _update_occupied_slots_for_agent(self, agent: str, sequence: Sequence) -> None:
        """更新槽位占用状态，供后续智能体使用（修复：只写入临时集，不写入全局集）"""
        print(f"[SLOT_UPDATE_DEBUG] Called for agent {agent}, sequence: {sequence}")
        if not sequence or not sequence.actions:
            print(f"[SLOT_UPDATE_DEBUG] No sequence or actions for agent {agent}")
            return
        
        # 从AtomicAction.meta中获取槽位ID信息
        for i, action in enumerate(sequence.actions):
            print(f"[SLOT_UPDATE_DEBUG] Action {i}: {action}")
            print(f"[SLOT_UPDATE_DEBUG] Action meta: {action.meta}")
            slots = action.meta.get("slots", [])
            print(f"[SLOT_UPDATE_DEBUG] Slots from meta: {slots}")
            if slots:
                print(f"[SLOT_UPDATE_DEBUG] Agent {agent} selected slots: {slots}")
                for slot_id in slots:
                    # 更新当前月份的槽位占用
                    self.occupied_slots.add(slot_id)
                    # 更新全局槽位占用
                    self.global_occupied_slots.add(slot_id)
                    # 更新全局槽位ID占用
                    self._mark_slot_id_occupied(slot_id)
                    print(f"[SLOT_UPDATE_DEBUG] Added slot {slot_id} to occupied_slots (total: {len(self.occupied_slots)})")
                    try:
                        self.logger.warning(f"[OCCUPY_FROM_SEQUENCE] agent={agent} slot_id={slot_id} month={self.current_month}")
                    except Exception:
                        pass
                    if topic_enabled("occupied_slots"):
                        self.logger.info(f"[SLOT_UPDATE] {agent}占用槽位 {slot_id}")
            else:
                print(f"[SLOT_UPDATE_DEBUG] No slots found in action {i} for agent {agent}")
    
    def _update_global_occupied_slots(self, agent: str, sequence: Sequence) -> None:
        """更新全局已占用槽位状态（新增方法）"""
        if not sequence or not sequence.actions:
            return
        
        # 获取动作的legacy IDs
        legacy_ids = sequence.get_legacy_ids()
        
        print(f"[GLOBAL_SLOT_DEBUG] Agent {agent}, legacy_ids={legacy_ids}, month={self.current_month}")
        
        for action_id in legacy_ids:
            cand = self._get_candidate_from_snapshot(agent, action_id)
            if not cand:
                print(f"[GLOBAL_SLOT_DEBUG] No candidate found for action_id={action_id}")
                continue
            
            slots = cand.meta.get("slots", [])
            print(f"[GLOBAL_SLOT_DEBUG] Action {action_id} has slots: {slots}")
            for slot_id in slots:
                self.global_occupied_slots.add(slot_id)
                # 修复：添加到全局槽位ID占用
                self._mark_slot_id_occupied(slot_id)
                print(f"[GLOBAL_SLOT_DEBUG] Added slot {slot_id} to global_occupied_slots (total: {len(self.global_occupied_slots)})")
                if topic_enabled("occupied_slots"):
                    self.logger.info(f"[GLOBAL_SLOT] {agent}占用槽位 {slot_id}")
    
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
                # 修复：添加到全局槽位ID占用
                self._mark_slot_id_occupied(slot_id)
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
                    
                    # 日志：记录slot_positions构建时使用的数据源
                    action_id = action.meta.get('action_id', action.atype if hasattr(action, 'atype') else -1)
                    try:
                        self.logger.warning(f"[SLOT_POS_BUILD] agent={agent} action_id={action_id} point={action.point} point_id={point_id} slots={slots}")
                    except Exception:
                        pass
                    
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
            "land_price_field": self.land_price_evo.get_land_price_field() if hasattr(self, 'land_price_evo') and self.land_price_evo else (self.land_price_system.get_land_price_field() if self.land_price_system else None),
            "agents": self.agents
        }
    
    def _get_current_environment_state(self) -> EnvironmentState:
        """获取当前环境状态，用于V5RewardCalculator"""
        return EnvironmentState(
            month=self.current_month,
            land_prices=self._get_actual_land_prices(),
            buildings=self._get_actual_buildings(),
            budgets=self.budgets.copy(),
            slots=self._get_actual_slots(),
            building_registry=self.building_registry
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
        """计算动作奖励 - 使用新的持续奖励系统
        
        Args:
            agent: 智能体名称
            candidate: 候选动作
            
        Returns:
            (总奖励, 奖励明细)
        """
        # 获取当前环境状态
        current_state = self._get_current_environment_state()
        
        # 现有动作奖励计算
        reward_terms = self.reward_calculator.calculate_reward(candidate, current_state)
        action_reward = reward_terms.revenue + reward_terms.cost + reward_terms.prestige + reward_terms.proximity + reward_terms.diversity
        
        # 添加其他奖励项（只计算数值类型，排除非奖励项）
        if reward_terms.other:
            # 排除不应该累加到奖励中的值
            exclude_keys = {
                'river_dist_m',      # 距离值，不是奖励
                'land_price_k',      # 系数，不是奖励
                'river_pct',         # 百分比，不是奖励
                'zone',              # 字符串，不是数值
                'land_price_norm',   # 归一化值，不是奖励
                'adj'                # 标志位，不是奖励
            }
            
            numeric_values = [v for k, v in reward_terms.other.items() 
                             if isinstance(v, (int, float)) and k not in exclude_keys]
            action_reward += sum(numeric_values)
        
        # 计算月度总收益
        monthly_totals = self.continuous_reward_system.calculate_monthly_total_rewards(current_state)
        monthly_total_t = monthly_totals.get(agent, 0.0)
        
        # 计算增量奖励
        delta_reward = self.continuous_reward_system.calculate_delta_reward(agent, monthly_total_t)
        
        # 计算潜势塑形
        shaping_reward = self.continuous_reward_system.calculate_potential_shaping(agent, monthly_total_t)
        
        # 总训练奖励
        total_reward = action_reward + delta_reward + shaping_reward
        
        # 转换为字典格式
        reward_dict = reward_terms.to_dict()
        reward_dict.update({
            "action_reward": action_reward,
            "delta_reward": delta_reward,
            "shaping_reward": shaping_reward,
            "monthly_total": monthly_total_t
        })
        
        # 日志输出
        if topic_enabled("reward_terms"):
            self.logger.info(f"奖励计算: agent={agent}, action={candidate.id}, "
                            f"action_reward={action_reward:.2f}, "
                            f"delta_reward={delta_reward:.2f}, "
                            f"shaping_reward={shaping_reward:.2f}, "
                            f"monthly_total={monthly_total_t:.2f}, "
                            f"total={total_reward:.2f}")
        
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
        return self._vectorize_observation(state, agent)
    
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
    
    def _vectorize_observation(self, state: Dict[str, Any], agent: str) -> np.ndarray:
        """将状态字典转换为数值向量
        
        Args:
            state: 状态字典
            agent: 智能体名称（用于获取预算历史）
            
        Returns:
            数值化的观察向量
        """
        # 根据网络期望的输入维度创建观察向量
        obs_dim = 64  # 网络期望的输入维度
        
        # 预算归一化参数：使用对数归一化 log(budget + 1) / log(max_budget + 1)
        # max_budget设置为100万，覆盖配置中的初始预算（5万）以及运行中的增长
        max_budget = 1000000.0
        log_max = np.log(max_budget + 1.0)
        
        # 槽位归一化参数：使用对数归一化
        # max_slots设置为1000，覆盖实际槽位数（203）以及可能的增长
        max_slots = 1000.0
        log_max_slots = np.log(max_slots + 1.0)
        
        # 月份归一化参数：线性归一化到[0, 1]
        max_months = float(self.total_months)
        
        # 基础特征（前10维）
        features = []
        
        # 修复：对month进行归一化（线性归一化）
        raw_month = float(state.get("month", 0))
        normalized_month = raw_month / max_months if max_months > 0 else 0.0
        features.append(normalized_month)
        
        # step通常较小，暂时不归一化
        features.append(float(state.get("step", 0)))
        
        # 修复：对budget进行归一化（对数归一化），并clip到[0, 1]
        raw_budget = float(state.get("budget", 0))
        if raw_budget > 0:
            normalized_budget = np.log(raw_budget + 1.0) / log_max
            normalized_budget = min(normalized_budget, 1.0)  # clip到[0, 1]
        else:
            normalized_budget = 0.0
        features.append(normalized_budget)
        
        # 修复：对occupied_count进行归一化（归一化为占用率）
        raw_occupied = float(state.get("occupied_count", 0))
        raw_total_slots = float(state.get("total_slots", 0))
        if raw_total_slots > 0:
            normalized_occupied = raw_occupied / raw_total_slots  # 占用率[0, 1]
        else:
            normalized_occupied = 0.0
        features.append(normalized_occupied)
        
        # 修复：对total_slots进行归一化（对数归一化）
        raw_total_slots = float(state.get("total_slots", 0))
        if raw_total_slots > 0:
            normalized_total_slots = np.log(raw_total_slots + 1.0) / log_max_slots
        else:
            normalized_total_slots = 0.0
        features.append(normalized_total_slots)
        
        # 添加预算历史特征（最近5步）
        # 修复：使用传入的agent参数，而不是硬编码
        budget_history = self.budget_history.get(agent, [])
        for i in range(5):
            if i < len(budget_history):
                # 修复：对预算历史也进行归一化，并clip到[0, 1]
                raw_budget_hist = float(budget_history[-(i+1)])
                if raw_budget_hist > 0:
                    normalized_budget_hist = np.log(raw_budget_hist + 1.0) / log_max
                    normalized_budget_hist = min(normalized_budget_hist, 1.0)  # clip到[0, 1]
                else:
                    normalized_budget_hist = 0.0
                features.append(normalized_budget_hist)
            else:
                features.append(0.0)
        
        # 扩展到目标维度（用零填充）
        while len(features) < obs_dim:
            features.append(0.0)
        
        # 截断到目标维度
        features = features[:obs_dim]
        
        # 调试：检查各维度的值（仅在第一次或异常时输出）
        obs_array = np.array(features, dtype=np.float32)
        if obs_array.max() > 100:
            # 输出前15个维度的值（基础特征 + 预算历史）
            dim_names = ["month", "step", "budget(norm)", "occupied_count", "total_slots", 
                        "budget_hist_1", "budget_hist_2", "budget_hist_3", "budget_hist_4", "budget_hist_5"]
            self.logger.warning(f"[OBS_DIM_CHECK] agent={agent} obs_max={obs_array.max():.2f}")
            for i, dim_name in enumerate(dim_names):
                if i < len(features):
                    self.logger.warning(f"[OBS_DIM_CHECK]   dim_{i}({dim_name})={features[i]:.2f}")
        
        return obs_array
    
    def _get_unlocked_actions(self, agent: str) -> Set[int]:
        """
        获取智能体已解锁的动作
        
        Args:
            agent: 智能体名称
            
        Returns:
            已解锁的动作集合
        """
        print(f"[DEBUG] _get_unlocked_actions called for agent {agent}, month {self.current_month}")
        print(f"[DEBUG] config keys: {self.config.keys()}")
        print(f"[DEBUG] action_unlocks in config: {'action_unlocks' in self.config}")
        try:
            # 直接检查解锁配置，而不是依赖中间件
            unlock_config = self.config.get("action_unlocks", {})
            rules = unlock_config.get("rules", [])
            print(f"[DEBUG] rules: {rules}")
            
            self.logger.info(f"[UNLOCK_DEBUG] Agent {agent}, month {self.current_month}, rules: {rules}")
            
            unlocked_actions = set()

            for rule in rules:
                # 检查智能体匹配
                rule_agent = rule.get("agent")
                if rule_agent and rule_agent != agent:
                    continue

                # 检查时间条件
                after_month = rule.get("after_month", 0)
                if self.current_month < after_month:
                    continue

                # 检查预算条件（如果存在）
                passed = True

                if "budget_threshold" in rule:
                    agent_budget = self.budgets.get(agent, 0)
                    budget_threshold = rule["budget_threshold"]
                    passed &= (agent_budget >= budget_threshold)

                if "total_budget_threshold" in rule:
                    total_budget = sum(self.budgets.values())
                    total_threshold = rule["total_budget_threshold"]
                    passed &= (total_budget >= total_threshold)

                # 如果条件满足，收集解锁动作
                if passed:
                    unlock_actions = rule.get("unlock_action_ids", [])
                    unlocked_actions.update(unlock_actions)
            
            # 检查是否有针对该智能体的解锁规则
            has_agent_rules = any(
                rule.get("agent") == agent for rule in rules
            )
            
            # 计算并返回 基础动作 ∪ 解锁动作
            agent_config = self.agents_config.get("defs", {}).get(agent, {})
            all_actions = set(agent_config.get("action_ids", []))

            # 被规则标记为“需要解锁”的动作集合（视为非基础动作）
            locked_actions = set()
            for rule in rules:
                rule_agent = rule.get("agent")
                if rule_agent and rule_agent != agent:
                    continue
                unlock_actions = rule.get("unlock_action_ids", [])
                locked_actions.update(unlock_actions)

            basic_actions = all_actions - locked_actions

            if has_agent_rules:
                # 调试模式：如果解锁了动作，只返回解锁的动作（替换而不是合并）
                # 这样可以验证解锁机制是否生效：解锁后应该无法执行基础动作
                if unlocked_actions:
                    # 解锁后，只返回解锁的动作
                    self.logger.info(
                        f"[UNLOCK_DEBUG] Has rules for {agent}. month={self.current_month} basic={sorted(list(basic_actions))} "
                        f"unlocked={sorted(list(unlocked_actions))} [DEBUG MODE] returning only unlocked actions"
                    )
                    print(
                        f"[DEBUG] _get_unlocked_actions [DEBUG MODE] returning only unlocked: {unlocked_actions}"
                    )
                    return unlocked_actions
                else:
                    # 未解锁时，返回基础动作
                    self.logger.info(
                        f"[UNLOCK_DEBUG] Has rules for {agent}. month={self.current_month} basic={sorted(list(basic_actions))} "
                        f"unlocked={sorted(list(unlocked_actions))} returning basic actions only"
                    )
                    print(
                        f"[DEBUG] _get_unlocked_actions returning basic: {basic_actions}"
                    )
                    return basic_actions
            
            # 如果没有针对该智能体的解锁规则，返回所有动作
            if not has_agent_rules:
                agent_config = self.agents_config.get("defs", {}).get(agent, {})
                all_actions = set(agent_config.get("action_ids", []))
                self.logger.info(f"[UNLOCK_DEBUG] No rules for agent, returning all: {all_actions}")
                print(f"[DEBUG] _get_unlocked_actions returning all actions: {all_actions}")
                return all_actions
            
            print(f"[DEBUG] _get_unlocked_actions returning unlocked_actions: {unlocked_actions}")
            return unlocked_actions
            
        except Exception as e:
            import traceback
            self.logger.error(f"获取解锁状态失败: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"[ERROR] 获取解锁状态失败: {e}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            # 如果获取失败，返回所有动作（不进行解锁过滤）
            agent_config = self.agents_config.get("defs", {}).get(agent, {})
            return set(agent_config.get("action_ids", []))
    
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
            
            # 使用地价演化系统（整合原有系统）
            if hasattr(self, 'land_price_evo') and self.land_price_evo:
                price = self.land_price_evo.get_land_price([slot.x, slot.y])
                return max(0.0, min(1.0, float(price)))
            elif hasattr(self, 'land_price_system'):
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
    
    def _initialize_land_price_evolution(self) -> None:
        """初始化地价演化系统"""
        try:
            from envs.land_price_evo import LandPriceEvo
            
            # 获取地图尺寸，使用与原有系统一致的200x200
            map_size = self._get_map_size()
            # 使用200x200与原有系统保持一致
            grid_shape = (200, 200)  # (height, width)
            
            # 创建地价演化系统
            self.land_price_evo = LandPriceEvo(self.config, grid_shape)
            
            self.logger.info(f"地价演化系统初始化成功，网格尺寸: {grid_shape}")
            
        except Exception as e:
            self.logger.error(f"地价演化系统初始化失败: {e}")
            # 不抛出异常，允许系统在没有演化功能的情况下运行
            self.land_price_evo = None
    
    def _update_land_price_evolution(self) -> None:
        """更新地价演化系统"""
        if self.land_price_evo is None:
            return
        
        try:
            # 更新地价场
            updated_land_price = self.land_price_evo.update_if_needed(self.current_month)
            
            # 更新环境状态中的地价场
            if hasattr(self, 'land_price_grid'):
                self.land_price_grid = updated_land_price
            
            # 记录激活的Hub
            active_hubs = self.land_price_evo.get_active_hubs()
            if active_hubs:
                self.logger.info(f"月份 {self.current_month}: 激活Hub数量 {len(active_hubs)}")
                
        except Exception as e:
            self.logger.error(f"地价演化更新失败: {e}")
    
    def _apply_middleware(self, agent: str, sequence: Sequence) -> Sequence:
        """
        应用中间件处理序列
        
        Args:
            agent: 智能体名称
            sequence: 原始序列
            
        Returns:
            处理后的序列
        """
        if not hasattr(self, 'config'):
            return sequence
        
        # 获取中间件配置
        middleware_config = self.config.get("action_mw", [])
        if not middleware_config:
            return sequence
        
        # 创建环境状态对象（用于中间件）
        state = EnvironmentState(
            month=self.current_month,
            land_prices=self._get_actual_land_prices(),
            buildings=self._get_actual_buildings(),
            budgets=self.budgets.copy(),
            slots=self._get_actual_slots()
        )
        # 添加配置到状态对象
        state.cfg = self.config
        state.t = self.current_month  # 添加时间信息
        state.month = self.current_month  # 添加月份信息
        state.agents = self.agents  # 添加智能体列表
        
        processed_sequence = sequence
        
        # 应用每个中间件
        for middleware_name in middleware_config:
            try:
                if middleware_name == "unlock.gate":
                    # 使用预先初始化的解锁中间件
                    if hasattr(self, 'unlock_middleware'):
                        processed_sequence = self.unlock_middleware.apply(processed_sequence, state)
                        self.logger.debug(f"应用解锁中间件: {len(sequence.actions)} -> {len(processed_sequence.actions)}")
                    else:
                        # 备用：动态创建
                        from action_mw.unlock_gate import UnlockGateMW
                        middleware = UnlockGateMW()
                        processed_sequence = middleware.apply(processed_sequence, state)
                        self.logger.debug(f"应用解锁中间件（动态创建）: {len(sequence.actions)} -> {len(processed_sequence.actions)}")
                
                elif middleware_name == "candidate_range":
                    from action_mw.candidate_range import CandidateRangeMiddleware
                    middleware = CandidateRangeMiddleware(self.config)
                    processed_sequence = middleware.apply(processed_sequence, state)
                    self.logger.debug(f"应用候选范围中间件: {sequence} -> {processed_sequence}")
                
                elif middleware_name == "river_restriction":
                    from action_mw.river_restriction import RiverRestrictionMiddleware
                    middleware = RiverRestrictionMiddleware(self.config)
                    processed_sequence = middleware.apply(processed_sequence, state)
                    self.logger.debug(f"应用河流限制中间件: {sequence} -> {processed_sequence}")
                
                # 可以添加更多中间件...
                
            except Exception as e:
                self.logger.error(f"中间件 {middleware_name} 应用失败: {e}")
                # 继续处理其他中间件
        
        return processed_sequence
    
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
        print("=== STEP_PHASE_CALLED ===")
        print(f"phase_agents: {phase_agents}")
        print(f"phase_sequences: {phase_sequences}")
        
        # 测试：添加一个简单的print语句，不受topic控制
        print("TEST: This should always print")
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
        
        print(f"[STEP_PHASE_DEBUG] phase_agents: {phase_agents}")
        print(f"[STEP_PHASE_DEBUG] phase_sequences keys: {list(phase_sequences.keys())}")
        
        for agent in phase_agents:
            print(f"[STEP_PHASE_DEBUG] Processing agent: {agent}")
            if agent in phase_sequences:
                sequence = phase_sequences[agent]
                # 空序列保护：当选择器未产出序列时，跳过中间件与执行
                if sequence is None or (hasattr(sequence, 'actions') and not sequence.actions):
                    self.logger.warning(f"[PIPELINE_SKIP] agent={agent} month={self.current_month} reason=no_sequence")
                    rewards[agent] = 0.0
                    reward_terms_all[agent] = {}
                    continue
                print(f"[STEP_PHASE_DEBUG] Agent {agent} has sequence: {sequence}")
                
                # 应用中间件处理序列
                processed_sequence = self._apply_middleware(agent, sequence)
                print(f"[STEP_PHASE_DEBUG] Agent {agent} processed sequence: {processed_sequence}")
                
                # 注意：动作序列生成时已经应用了去重过滤，无需重新获取
                if topic_enabled("action_execution"):
                    self.logger.info(f"[ACTION_EXEC] 执行智能体 {agent} 的动作序列: {processed_sequence}")
                reward, reward_terms = self._execute_agent_sequence(agent, processed_sequence)
                rewards[agent] = reward
                reward_terms_all[agent] = reward_terms
                
                # 修复：立即更新槽位占用状态，供后续智能体使用
                # 使用processed_sequence而不是原始sequence，确保只有实际执行的槽位被标记
                print(f"[STEP_PHASE_DEBUG] Calling _update_occupied_slots_for_agent for {agent}")
                self._update_occupied_slots_for_agent(agent, processed_sequence)
                
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

        # 月度占位汇总：高优先级，用于观察占位规模是否逼近阈值，并区分执行/非执行占位
        try:
            total_occ = len(self._global_occupied_slot_ids)
            built_occ = len(self._built_slot_ids)
            non_exec_set = list(self._global_occupied_slot_ids - self._built_slot_ids)
            sample_non_exec = non_exec_set[:5]
            self.logger.warning(
                f"[OCCUPIED_SUMMARY] month={self.current_month} occupied_ids={total_occ} built_ids={built_occ} "
                f"non_exec_ids={total_occ - built_occ} sample_non_exec={sample_non_exec}"
            )
        except Exception:
            pass
        
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
