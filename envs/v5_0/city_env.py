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
from logic.v5_enumeration import V5ActionEnumerator
from logic.v5_scorer import V5ActionScorer
from logic.v5_selector import V5SequenceSelector


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
        # 简化实现：返回默认槽位
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
        
        # 创建步骤日志
        step_log = self.selector.create_step_log(
            step=self.current_step,
            agent=agent,
            sequence=sequence,
            reward_terms=reward_terms,
            budget_snapshot=dict(self.budgets)
        )
        self.step_logs.append(step_log)
        
        # 更新状态
        self._update_state()
        
        # 检查是否结束
        done = self._is_done()
        
        # 获取新状态
        next_state = self._get_current_state()
        
        return next_state, reward, done, {"step_log": step_log}
    
    def _execute_action(self, agent: str, action_id: int) -> Tuple[float, Dict[str, float]]:
        """执行单个动作"""
        # 获取动作参数
        action_params = self.loader.get_action_params(action_id)
        if not action_params:
            return 0.0, {}
        
        # 计算成本
        cost = action_params.get("cost", 0.0)
        
        # 检查预算
        if cost > self.budgets.get(agent, 0.0):
            return 0.0, {"cost": 0.0}
        
        # 扣除成本
        self.budgets[agent] -= cost
        
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
        # 简化实现：每个智能体执行一步后切换
        return True
    
    def _should_switch_month(self) -> bool:
        """检查是否需要切换月份"""
        # 简化实现：每12步切换一个月
        return self.current_step % 12 == 0
    
    def _switch_agent(self):
        """切换智能体"""
        # 获取下一个活跃智能体
        active_agents = self.scheduler.get_active_agents(self.current_step)
        if not active_agents:
            return
        
        # 轮换智能体
        self.agent_turn = (self.agent_turn + 1) % len(active_agents)
        self.current_agent = active_agents[self.agent_turn]
    
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
            budget=budget
        )
        
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
