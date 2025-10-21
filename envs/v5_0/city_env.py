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

from contracts import ActionCandidate, Sequence, StepLog, EnvironmentState
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
        """
        初始化环境
        
        Args:
            config_path: v5.0配置文件路径
        """
        # 加载配置
        self.loader = ConfigLoader()
        self.config = self.loader.load_v5_config(config_path)
        
        # 初始化调度器
        scheduler_config = self.loader.get_scheduler_config()
        self.scheduler = PhaseCycleScheduler(scheduler_config["params"])
        
        # 初始化枚举器、打分器、选择器
        self.enumerator = V5ActionEnumerator(self.config)
        self.scorer = V5ActionScorer(self.config)
        self.selector = V5SequenceSelector(self.config)
        self.logger = get_logger("env")
        # 每个agent最近一次候选快照（用于StepLog和占用更新，不再二次枚举）
        self._last_candidates: Dict[str, List[ActionCandidate]] = {}
        
        # 初始化预算池管理器
        self.budget_pool_manager = BudgetPoolManager(self.config)
        
        # 环境状态
        self.current_step = 0
        self.current_month = 0
        self.current_agent = None
        self.agent_turn = 0
        
        # 建筑和槽位状态
        self.buildings = []
        self.slots = {}
        self.occupied_slots = set()
        
        # 预算系统
        self.budgets = {}
        self.budget_history = {}
        
        # 地价系统
        self.land_price_field = None
        
        # 历史记录
        self.step_logs = []
        self.episode_history = []
        
        # 初始化环境
        self._initialize_environment()
    
    def _initialize_environment(self):
        """初始化环境"""
        # 加载槽位数据
        self._load_slots()
        
        # 初始化预算
        self._initialize_budgets()
        
        # 初始化地价系统
        self._initialize_land_price_system()
        
        # 初始化历史记录
        self._initialize_history()
    
    def _load_slots(self):
        """加载槽位数据"""
        slots_path = self.loader.get_paths_config().get("slots_txt")
        if not slots_path or not os.path.exists(slots_path):
            # 使用默认槽位数据
            self.slots = self._create_default_slots()
        else:
            # 从文件加载槽位
            self.slots = self._load_slots_from_file(slots_path)
        
        # 更新枚举器的槽位数据
        slots_data = [{"id": sid, "x": slot["x"], "y": slot["y"], 
                      "neighbors": slot["neighbors"], "building_level": slot["building_level"]}
                     for sid, slot in self.slots.items()]
        self.enumerator.load_slots(slots_data)
    
    def _create_default_slots(self):
        """创建默认槽位"""
        slots = {}
        map_size = self.config.get("city", {}).get("map_size", [200, 200])
        
        # 创建网格槽位
        for x in range(0, map_size[0], 10):
            for y in range(0, map_size[1], 10):
                slot_id = f"slot_{x}_{y}"
                slots[slot_id] = {
                    "id": slot_id,
                    "x": x, "y": y,
                    "neighbors": [],
                    "building_level": 5
                }
        
        return slots
    
    def _load_slots_from_file(self, path: str):
        """从文件加载槽位"""
        slots = {}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析槽位数据：x, y, angle, building_level
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        x = float(parts[0])
                        y = float(parts[1])
                        angle = float(parts[2]) if len(parts) > 2 else 0.0
                        building_level = int(parts[3]) if len(parts) > 3 else 5
                        
                        slot_id = f"slot_{idx}"
                        slots[slot_id] = {
                            "id": slot_id,
                            "x": x,
                            "y": y,
                            "angle": angle,
                            "neighbors": [],
                            "building_level": building_level
                        }
            
            if topic_enabled("candidates"):
                self.logger.info(f"slots_loaded count={len(slots)} path={path}")
            return slots
        
        except Exception as e:
            self.logger.warning(f"slots_load_failed path={path} error={e}")
            return self._create_default_slots()
    
    def _initialize_budgets(self):
        """初始化预算系统"""
        ledger_config = self.config.get("ledger", {})
        initial_budgets = ledger_config.get("initial_budget", {})
        
        self.budgets = dict(initial_budgets)
        
        # 初始化预算历史
        for agent in self.budgets.keys():
            self.budget_history[agent] = []
    
    def _initialize_land_price_system(self):
        """初始化地价系统"""
        # 简化实现：创建默认地价场
        map_size = self.config.get("city", {}).get("map_size", [200, 200])
        self.land_price_field = np.ones((map_size[1], map_size[0]), dtype=np.float32)
    
    def _initialize_history(self):
        """初始化历史记录"""
        agents = self.config.get("agents", {}).get("order", [])
        for agent in agents:
            self.episode_history.append([])
    
    def reset(self) -> EnvironmentState:
        """重置环境"""
        # 重置状态
        self.current_step = 0
        self.current_month = 0
        self.agent_turn = 0
        
        # 设置初始智能体
        active_agents = self.scheduler.get_active_agents(0)
        if active_agents:
            self.current_agent = active_agents[0]
        else:
            self.current_agent = "EDU"  # 默认智能体
        
        # 清空建筑和槽位
        self.buildings.clear()
        self.occupied_slots.clear()
        
        # 重置预算
        self._initialize_budgets()
        
        # 重置预算池
        self.budget_pool_manager.reset_all_pools()
        
        # 清空历史记录
        self.step_logs.clear()
        self.episode_history.clear()
        
        # 返回初始状态
        return self._get_current_state()
    
    def step(self, agent: str, sequence: Optional[Sequence]) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """执行一步"""
        if agent != self.current_agent:
            raise ValueError(f"当前轮次智能体不匹配: 期望{self.current_agent}, 得到{agent}")
        
        # 执行序列
        reward = 0.0
        reward_terms = {}
        
        if sequence and sequence.actions:
            for action_id in sequence.actions:
                # 执行动作
                action_reward, action_terms = self._execute_action(agent, action_id)
                reward += action_reward
                reward_terms.update(action_terms)
            
            # 更新已占用槽位（使用候选快照）
            self._update_occupied_slots_from_snapshot(agent, sequence)
        
        # 创建步骤日志
        step_log = self.selector.create_step_log(
            step=self.current_step,
            agent=agent,
            sequence=sequence,
            reward_terms=reward_terms,
            budget_snapshot=dict(self.budgets)
        )
        
        # 添加槽位位置信息（使用候选快照）
        if sequence and sequence.actions:
            step_log.slot_positions = self._build_slot_positions_from_snapshot(agent, sequence)
        self.step_logs.append(step_log)
        
        # 更新状态
        self._update_state()
        
        # 检查是否结束
        done = self._is_done()
        
        # 获取新状态
        next_state = self._get_current_state()
        
        return next_state, reward, done, {"step_log": step_log}
    
    def step_phase(self, phase_agents: List[str], phase_sequences: Dict[str, Optional[Sequence]]) -> Tuple[EnvironmentState, Dict[str, float], bool, Dict[str, Any]]:
        """执行一个phase（支持并发执行多个agent）"""
        phase_rewards = {}
        phase_logs = []
        
        # 获取当前phase的执行模式
        execution_mode = self.scheduler.get_execution_mode(self.current_step)
        
        if execution_mode == "concurrent":
            # 并发执行多个agent - 需要动态更新候选集
            for agent in phase_agents:
                if agent in phase_sequences and phase_sequences[agent]:
                    sequence = phase_sequences[agent]
                    # 执行单个agent
                    agent_reward, agent_terms = self._execute_agent_sequence(agent, sequence)
                    phase_rewards[agent] = agent_reward
                    
                    # 更新已占用槽位（防止后续智能体选择同一槽位）
                    self._update_occupied_slots_from_snapshot(agent, sequence)
                    
                    # 创建步骤日志
                    step_log = self.selector.create_step_log(
                        step=self.current_step,
                        agent=agent,
                        sequence=sequence,
                        reward_terms=agent_terms,
                        budget_snapshot=dict(self.budgets)
                    )
                    # 使用候选快照填充坐标
                    step_log.slot_positions = self._build_slot_positions_from_snapshot(agent, sequence)
                    phase_logs.append(step_log)
        else:
            # 顺序执行多个agent - 需要动态更新候选集
            for agent in phase_agents:
                if agent in phase_sequences and phase_sequences[agent]:
                    sequence = phase_sequences[agent]
                    # 执行单个agent
                    agent_reward, agent_terms = self._execute_agent_sequence(agent, sequence)
                    phase_rewards[agent] = agent_reward
                    
                    # 更新已占用槽位（防止后续智能体选择同一槽位）
                    self._update_occupied_slots_from_snapshot(agent, sequence)
                    
                    # 创建步骤日志
                    step_log = self.selector.create_step_log(
                        step=self.current_step,
                        agent=agent,
                        sequence=sequence,
                        reward_terms=agent_terms,
                        budget_snapshot=dict(self.budgets)
                    )
                    # 使用候选快照填充坐标
                    step_log.slot_positions = self._build_slot_positions_from_snapshot(agent, sequence)
                    phase_logs.append(step_log)
        
        # 添加所有日志
        self.step_logs.extend(phase_logs)
        
        # 更新状态
        self._update_state()
        
        # 检查是否结束
        done = self._is_done()
        
        # 获取新状态
        next_state = self._get_current_state()
        
        return next_state, phase_rewards, done, {"phase_logs": phase_logs}
    
    def _execute_agent_sequence(self, agent: str, sequence: Sequence) -> Tuple[float, Dict[str, float]]:
        """执行单个agent的序列"""
        reward = 0.0
        reward_terms = {}
        
        if sequence and sequence.actions:
            for action_id in sequence.actions:
                # 执行动作
                action_reward, action_terms = self._execute_action(agent, action_id)
                reward += action_reward
                reward_terms.update(action_terms)
        
        return reward, reward_terms
    
    def _update_occupied_slots_from_snapshot(self, agent: str, sequence: Sequence) -> None:
        """使用候选快照更新已占用槽位"""
        if not sequence or not sequence.actions:
            return
        for action_id in sequence.actions:
            cand = self._get_candidate_from_snapshot(agent, action_id)
            if not cand:
                continue
            for slot_id in cand.meta.get("slots", []):
                self.occupied_slots.add(slot_id)
                if topic_enabled("occupied_slots") and sampling_allows(agent, self.current_month, self.current_step):
                    self.logger.info(f"occupied agent={agent} slot={slot_id} month={self.current_month} step={self.current_step}")

    def _build_slot_positions_from_snapshot(self, agent: str, sequence: Sequence) -> List[Dict[str, Any]]:
        """根据候选快照构建槽位位置信息"""
        positions: List[Dict[str, Any]] = []
        if not sequence or not sequence.actions:
            return positions
        for action_id in sequence.actions:
            cand = self._get_candidate_from_snapshot(agent, action_id)
            if not cand:
                continue
            for slot_id in cand.meta.get("slots", []):
                if slot_id in self.slots:
                    slot = self.slots[slot_id]
                    positions.append({
                        'slot_id': slot_id,
                        'x': slot['x'],
                        'y': slot['y'],
                        'angle': slot.get('angle', 0.0)
                    })
        return positions

    def _get_candidate_from_snapshot(self, agent: str, action_id: int) -> Optional[ActionCandidate]:
        """从最近候选快照中查找指定动作候选"""
        snapshot = self._last_candidates.get(agent) or []
        for cand in snapshot:
            if cand.id == action_id:
                return cand
        # 兜底：尝试刷新一次（并记录）
        cands = self.get_action_candidates(agent)
        for cand in cands:
            if cand.id == action_id:
                return cand
        return None
    
    def _execute_action(self, agent: str, action_id: int) -> Tuple[float, Dict[str, float]]:
        """执行单个动作"""
        # 获取动作参数
        action_params = self.loader.get_action_params(action_id)
        if not action_params:
            return 0.0, {}
        
        # 计算成本
        cost = action_params.get("cost", 0.0)
        
        # 检查预算（使用预算池管理器）
        if not self.budget_pool_manager.can_afford(agent, cost):
            return 0.0, {"cost": 0.0, "budget_error": "insufficient_funds"}
        
        # 扣除成本（从预算池扣除）
        if not self.budget_pool_manager.deduct(agent, cost):
            return 0.0, {"cost": 0.0, "budget_error": "deduction_failed"}
        
        # 计算奖励
        reward = action_params.get("reward", 0.0)
        prestige = action_params.get("prestige", 0.0)
        
        # 创建建筑记录
        building = {
            "agent": agent,
            "action_id": action_id,
            "month": self.current_month,
            "cost": cost,
            "reward": reward,
            "prestige": prestige
        }
        self.buildings.append(building)
        
        # 计算奖励分项
        reward_terms = {
            "revenue": reward,
            "cost": -cost,
            "prestige": prestige
        }
        
        return reward, reward_terms
    
    def _update_state(self):
        """更新环境状态"""
        # 更新步骤
        self.current_step += 1
        
        # 检查是否需要切换智能体
        if self._should_switch_agent():
            self._switch_agent()
        
        # 检查是否需要切换月份
        if self._should_switch_month():
            self._switch_month()
    
    def _should_switch_agent(self) -> bool:
        """检查是否需要切换智能体"""
        # 获取当前阶段的活跃智能体
        active_agents = self.scheduler.get_active_agents(self.current_step)
        
        # 如果当前智能体不在活跃列表中，需要切换
        if self.current_agent not in active_agents:
            return True
        
        # 如果当前阶段有多个智能体，需要轮换
        if len(active_agents) > 1:
            return True
        
        # 否则不需要切换
        return False
    
    def get_phase_agents(self) -> List[str]:
        """获取当前phase的所有智能体"""
        return self.scheduler.get_active_agents(self.current_step)
    
    def get_phase_execution_mode(self) -> str:
        """获取当前phase的执行模式"""
        return self.scheduler.get_execution_mode(self.current_step)
    
    def _should_switch_month(self) -> bool:
        """检查是否需要切换月份"""
        # 每1步切换一个月 (符合total_steps配置)
        return True
    
    def _switch_agent(self):
        """切换智能体"""
        # 获取当前阶段的活跃智能体
        active_agents = self.scheduler.get_active_agents(self.current_step)
        
        if not active_agents:
            return
        
        # 如果当前智能体不在活跃列表中，选择第一个
        if self.current_agent not in active_agents:
            self.current_agent = active_agents[0]
            self.agent_turn = 0
            return
        
        # 轮换到下一个活跃智能体
        current_index = active_agents.index(self.current_agent)
        next_index = (current_index + 1) % len(active_agents)
        self.current_agent = active_agents[next_index]
        self.agent_turn = next_index
    
    def _switch_month(self):
        """切换月份"""
        self.current_month += 1
    
    def _is_done(self) -> bool:
        """检查是否结束"""
        max_months = self.config.get("env", {}).get("time_model", {}).get("total_steps", 30)
        return self.current_month >= max_months
    
    def _get_current_state(self) -> EnvironmentState:
        """获取当前状态"""
        return EnvironmentState(
            month=self.current_month,
            land_prices=self.land_price_field,
            buildings=self.buildings.copy(),
            budgets=dict(self.budgets),
            slots=list(self.slots.values())
        )
    
    def get_observation(self, agent: str) -> np.ndarray:
        """获取智能体观察"""
        # 简化实现：返回固定长度的观察向量
        obs_size = 64  # 可配置
        obs = np.zeros(obs_size, dtype=np.float32)
        
        # 填充基础信息
        obs[0] = self.current_month
        obs[1] = self.budgets.get(agent, 0.0)
        obs[2] = len(self.buildings)
        
        # 填充地价信息
        if self.land_price_field is not None:
            obs[3:15] = self.land_price_field.flatten()[:12]
        
        # 追加：候选统计与预算（增强信号）
        try:
            cands = self.get_action_candidates(agent)
        except Exception:
            cands = []
        num_cands = float(len(cands))
        obs[15] = num_cands
        # 近似的候选质量（使用 meta.lp_norm 均值，若无则0）
        if cands:
            lp_vals = [float(ci.meta.get('lp_norm', 0.0)) for ci in cands]
            obs[16] = float(np.mean(lp_vals))
            obs[17] = float(np.max(lp_vals))
        else:
            obs[16] = 0.0
            obs[17] = 0.0
        
        return obs
    
    def get_action_candidates(self, agent: str) -> List[ActionCandidate]:
        """获取动作候选"""
        # 获取智能体预算
        budget = self.budgets.get(agent, 0.0)
        
        # 模拟地价提供函数
        def lp_provider(slot_id):
            return 0.5  # 简化实现
        
        # 枚举动作候选
        candidates = self.enumerator.enumerate_actions(
            agent=agent,
            occupied_slots=self.occupied_slots,
            lp_provider=lp_provider,
            budget=budget,
            current_month=self.current_month
        )
        # 缓存候选快照
        self._last_candidates[agent] = candidates
        return candidates
    
    def get_available_actions(self, agent: str) -> List[int]:
        """获取可用动作ID列表"""
        candidates = self.get_action_candidates(agent)
        return [c.id for c in candidates]
    
    def is_action_valid(self, agent: str, action_id: int) -> bool:
        """检查动作是否有效"""
        available_actions = self.get_available_actions(agent)
        return action_id in available_actions
    
    def get_reward_terms(self, agent: str, action_id: int) -> Dict[str, float]:
        """获取奖励分项"""
        # 获取动作参数
        action_params = self.loader.get_action_params(action_id)
        if not action_params:
            return {}
        
        return {
            "revenue": action_params.get("reward", 0.0),
            "cost": action_params.get("cost", 0.0),
            "prestige": action_params.get("prestige", 0.0)
        }
    
    def get_step_logs(self) -> List[StepLog]:
        """获取步骤日志"""
        return self.step_logs.copy()
    
    def export_txt(self, output_path: str):
        """导出txt文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for step_log in self.step_logs:
                f.write(f"{step_log.t},{step_log.agent},{','.join(map(str, step_log.chosen))}\\n")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_steps": len(self.step_logs),
            "total_buildings": len(self.buildings),
            "budgets": dict(self.budgets),
            "month": self.current_month
        }
