"""
v4.1 城市环境包装器
将现有的v4.0城市模拟系统包装成RL训练环境
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import deque

from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.v4_enumeration import V4Planner, SlotNode, _auto_fill_neighbors_4n, Action, Sequence
from enhanced_city_simulation_v4_0 import (
    load_slots_from_points_file, 
    min_dist_to_hubs, 
    load_river_coords,
    river_center_y_from_coords,
    build_river_components,
    compute_R,
    ring_candidates,
    _get_river_buffer_px
)


class CityEnvironment:
    """城市环境包装器 - 适配RL训练"""
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.rl_cfg = cfg['solver']['rl']
        
        # 基础配置
        self.sim_cfg = cfg.get('simulation', {})
        self.total_months = int(self.sim_cfg.get('total_months', 20))
        self.city_cfg = cfg.get('city', {})
        self.map_size = self.city_cfg.get('map_size', [200, 200])
        self.hubs = self.city_cfg.get('transport_hubs', [[125, 75], [112, 121]])
        
        # Budget系统配置
        self.budget_cfg = cfg.get('budget_system', {'enabled': False})
        if self.budget_cfg.get('enabled', False):
            initial_budgets = self.budget_cfg.get('initial_budgets', {'IND': 5000, 'EDU': 4000})
            
            # 【共享预算机制】Council和EDU共享预算
            self.budgets = dict(initial_budgets)
            
            # 如果存在Council，将其预算合并到EDU
            if 'Council' in self.budgets and 'EDU' in self.budgets:
                council_budget = self.budgets.pop('Council', 0)
                self.budgets['EDU'] += council_budget
                print(f"[Budget] Council和EDU共享预算 - 总预算: {self.budgets['EDU']}")
            
            # 为所有智能体创建预算历史
            self.budget_history = {agent: [] for agent in self.rl_cfg['agents']}
            
            # 为Council创建虚拟预算（用于记录，实际使用EDU预算）
            if 'Council' in self.rl_cfg['agents']:
                self.budget_history['Council'] = []
            
            print(f"[Budget] 系统已启用 - IND: {self.budgets.get('IND', 0)}, EDU: {self.budgets.get('EDU', 0)}")
        else:
            self.budgets = None
            self.budget_history = None
        
        # v4.1配置
        self.v4_cfg = cfg.get('growth_v4_1', {})
        if not self.v4_cfg:
            # 回退到v4.0配置
            self.v4_cfg = cfg.get('growth_v4_0', {})
            if not self.v4_cfg:
                raise ValueError("growth_v4_0或growth_v4_1配置未找到")
        
        # 初始化环境组件
        self._initialize_environment()
        
        # RL状态
        self.current_month = 0
        self.current_agent = self.rl_cfg['agents'][0]  # 当前决策智能体
        self.agent_turn = 0  # 智能体轮次
        self._council_execution_phase = None  # Council执行阶段标记
        
        # 状态缓存
        self.state_cache = {}
        self.action_cache = {}
        
        # 历史记录
        self.episode_history = []
        self.monthly_rewards = {agent: [] for agent in self.rl_cfg['agents']}
        
        # 【月度收益机制】在营资产管理
        self.active_assets = {agent: [] for agent in self.rl_cfg['agents']}
        self.monthly_income_history = {agent: [] for agent in self.rl_cfg['agents']}
        
    def _initialize_environment(self):
        """初始化环境组件"""
        # 加载槽位
        slots_source = self.v4_cfg.get('slots', {}).get('path', 'slots_with_angle.txt')
        self.slots = load_slots_from_points_file(slots_source, self.map_size)
        
        # 初始化地价系统
        self.land_price_system = GaussianLandPriceSystem(self.cfg)
        self.land_price_system.initialize_system(self.hubs, self.map_size)
        
        # 初始化v4规划器（用于参数化模式对比）
        self.param_planner = V4Planner(self.cfg)
        
        # 建筑状态
        self.buildings = {'public': [], 'industrial': []}
        
        # 河流和区域分割（用于同侧过滤）
        self.river_coords = load_river_coords(self.cfg)
        self._setup_river_components()
        
    def _setup_river_components(self):
        """设置河流区域分割（恢复v4.0的两侧分开建设功能）"""
        # 计算河流中心线
        self.river_center_y = river_center_y_from_coords(self.river_coords)
        
        # 基于河流缓冲的区域分割
        buffer_px = _get_river_buffer_px(self.cfg, default_px=2.0)
        self.comp_grid = build_river_components(self.map_size, self.river_coords, buffer_px)
        
        # 计算hub所在的连通域
        self.hub_components = []
        for hub in self.hubs:
            hub_comp = self._get_component_of_xy(hub[0], hub[1])
            self.hub_components.append(hub_comp)
    
    def _get_component_of_xy(self, x: float, y: float) -> int:
        """获取坐标(x,y)所在的连通域ID"""
        xi, yi = int(round(x)), int(round(y))
        if yi < 0 or yi >= int(self.map_size[1]) or xi < 0 or xi >= int(self.map_size[0]):
            return -1
        return int(self.comp_grid[yi][xi])
    
    def action_allowed(self, action: Action) -> bool:
        """检查动作是否被允许（两侧分开建设约束）"""
        if not action.footprint_slots:
            return True
        
        # 现在候选槽位已经在枚举阶段过滤了，这里可以简化或移除
        # 保留作为双重检查
        return True
        
        # 原始过滤逻辑（暂时注释掉）
        # if not hasattr(self, 'hub_components') or len(self.hub_components) < 2:
        #     print(f"    Debug: hub_components未正确设置，跳过动作过滤")
        #     return True
        
        # print(f"    Debug: 检查动作 {action.agent}, hub_components={self.hub_components}")
            
        # # 检查动作的每个槽位是否与hub在同一连通域
        # for slot_id in action.footprint_slots:
        #     slot = self.slots.get(slot_id)
        #     if slot is None:
        #         return False
            
        #     # 使用浮点坐标
        #     xx = float(getattr(slot, 'fx', slot.x))
        #     yy = float(getattr(slot, 'fy', slot.y))
        #     slot_comp = self._get_component_of_xy(xx, yy)
            
        #     # 检查是否与对应hub在同一连通域
        #     if action.agent == 'EDU' and len(self.hub_components) >= 1:
        #         if slot_comp != self.hub_components[0]:
        #             return False
        #     elif action.agent == 'IND' and len(self.hub_components) >= 2:
        #         if slot_comp != self.hub_components[1]:
        #             return False
        
        # return True
        
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 重置状态
        self.current_month = 0
        
        # 支持first_agent配置
        first_agent = self.v4_cfg.get('enumeration', {}).get('first_agent', None)
        if first_agent and first_agent in self.rl_cfg['agents']:
            self.agent_turn = self.rl_cfg['agents'].index(first_agent)
            self.current_agent = first_agent
        else:
            self.agent_turn = 0
            self.current_agent = self.rl_cfg['agents'][0]
        
        # 清空建筑状态
        self.buildings = {'public': [], 'industrial': []}
        
        # 重置Budget
        if self.budgets is not None:
            self.budgets = dict(self.budget_cfg.get('initial_budgets', {'IND': 5000, 'EDU': 4000}))
            for agent in self.rl_cfg['agents']:
                self.budget_history[agent].clear()
        
        # 清空缓存和历史
        self.state_cache.clear()
        self.action_cache.clear()
        self.episode_history.clear()
        for agent in self.rl_cfg['agents']:
            self.monthly_rewards[agent].clear()
        
        # 初始化地价系统
        self.land_price_system.initialize_system(self.hubs, self.map_size)
        
        # 返回初始状态
        return self._get_current_state()
    
    def step(self, agent: str, sequence: Optional[Sequence]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """执行动作序列（恢复v4.0的序列执行机制）"""
        if agent != self.current_agent:
            raise ValueError(f"当前轮次智能体不匹配: 期望{self.current_agent}, 得到{agent}")
        
        # 记录序列
        self.action_cache[agent] = sequence
        
        # 计算序列总奖励
        total_reward = 0.0
        
        # 执行序列中的所有动作
        if sequence and sequence.actions:
            for action in sequence.actions:
                # 执行建筑放置（先放置，以便计算收益）
                self._place_building(agent, action)
                
                # 【月度收益机制】更新budget：扣除建造成本
                if self.budgets is not None:
                    build_cost = float(action.cost) if action.cost is not None else 0.0
                    
                    # 【共享预算机制】Council使用EDU的预算
                    if agent == 'Council':
                        self.budgets['EDU'] -= build_cost
                        # 同时更新Council的预算历史（用于记录）
                        if 'Council' in self.budget_history:
                            self.budget_history['Council'].append({
                                'month': self.current_month,
                                'cost': -build_cost,
                                'remaining': self.budgets['EDU']
                            })
                    else:
                        self.budgets[agent] -= build_cost
                
                # 计算单个动作奖励（已包含monthly_income）
                action_reward = self._calculate_reward(agent, action)
                total_reward += action_reward
                
                # 记录单个动作历史
                self.episode_history.append({
                    'month': self.current_month,
                    'agent': agent,
                    'action': action,
                    'reward': action_reward,
                    'buildings': self.buildings.copy()
                })
        else:
            # 空序列：只有月度收益（无建造）
            monthly_income = self._calculate_monthly_income(agent)
            total_reward = monthly_income / self.rl_cfg.get('reward_scale', 500.0)
        
        # 切换到下一个月（序列执行完成后）
        next_state, done, info = self._advance_turn()
        
        return next_state, total_reward, done, info
    
    def _advance_turn(self):
        """推进到下一个回合 - 交替模式：奇数月IND，偶数月EDU+Council"""
        # 推进到下一个月份
        self.current_month += 1
        
        # 检查是否达到最大月份
        if self.current_month >= self.total_months:
            return self._get_current_state(), True, {'reason': 'max_months_reached'}
        
        # 【交替模式】根据月份决定智能体
        if self.current_month % 2 == 1:  # 奇数月：IND
            self.current_agent = 'IND'
            self.agent_turn = 0
            self._council_execution_phase = None
        else:  # 偶数月：EDU和Council
            # 偶数月开始时，先切换到EDU
            self.current_agent = 'EDU'
            self.agent_turn = 1
            self._council_execution_phase = 'EDU'
        
        # 更新地价系统
        if hasattr(self.land_price_system, 'update_month'):
            self.land_price_system.update_month(self.current_month)
        
        return self._get_current_state(), False, {'month': self.current_month, 'agent': self.current_agent}
    
    def _get_current_state(self) -> Dict[str, Any]:
        """获取当前环境状态"""
        state = {
            # 基础信息
            'month': self.current_month,
            'current_agent': self.current_agent,
            'agent_turn': self.agent_turn,
            
            # 地图信息
            'map_size': self.map_size,
            'hubs': self.hubs,
            'river_coords': self.river_coords,
            
            # 建筑状态
            'buildings': self.buildings.copy(),
            'occupied_slots': self._get_occupied_slots(),
            
            # 候选区域
            'candidate_slots': self._get_candidate_slots(),
            
            # 地价信息
            'land_price_field': self._get_land_price_field(),
            
            # 统计信息
            'monthly_stats': self._get_monthly_stats()
        }
        
        return state
    
    def _get_occupied_slots(self) -> Set[str]:
        """获取已占用的槽位"""
        occupied = set()
        xy_to_sid = {}
        
        # 建立坐标到槽位ID的映射
        for sid, slot in self.slots.items():
            xy_to_sid[(slot.x, slot.y)] = sid
        
        # 检查已占用槽位
        for building_type in ['public', 'industrial']:
            for building in self.buildings[building_type]:
                xy = building.get('xy', [0, 0])
                sid = xy_to_sid.get((int(round(xy[0])), int(round(xy[1]))))
                if sid is not None:
                    occupied.add(sid)
        
        # 【新增】检查槽位的occupied_by属性
        for sid, slot in self.slots.items():
            if hasattr(slot, 'occupied_by') and slot.occupied_by:
                occupied.add(sid)
                print(f"[DEBUG] 槽位{sid}被{slot.occupied_by}占用")
        
        return occupied
    
    def _get_candidate_slots(self) -> Set[str]:
        """获取候选槽位（根据当前智能体过滤到对应连通域）"""
        # 使用v4.0的候选区域计算逻辑
        all_candidates = ring_candidates(
            self.slots, 
            self.hubs, 
            self.current_month, 
            self.v4_cfg.get('hubs', {}), 
            tol=1.0
        )
        # print(f"    [Debug] ring_candidates返回: {len(all_candidates)}个槽位")
        
        # 【新增】为EDU agent添加对岸槽位（用于A/B/C动作）
        if self.current_agent == 'EDU':
            other_side_slots = self._get_other_side_slots()
            
            # 【修复】只添加距离hub合理的对岸槽位
            valid_other_side_slots = self._filter_other_side_slots_by_distance(other_side_slots, all_candidates)
            
            all_candidates = all_candidates | set(valid_other_side_slots)
        
        # 【修正顺序】先应用河流连通域过滤，再应用邻近性约束
        # 根据当前智能体过滤到对应连通域
        if hasattr(self, 'hub_components') and len(self.hub_components) >= 2:
            agent_idx = self.rl_cfg['agents'].index(self.current_agent)
            
            # Council智能体特殊处理：使用EDU的连通域（因为Council负责A/B/C教育建筑）
            if self.current_agent == 'Council':
                # Council使用EDU的连通域（索引1）
                expected_comp = self.hub_components[1] if len(self.hub_components) > 1 else self.hub_components[0]
            else:
                expected_comp = self.hub_components[agent_idx]
            
            # 河流连通域过滤：EDU的A/B/C绕过，Council完全绕过，其他都受限制
            if self.current_agent == 'Council':
                # Council智能体：检查是否达到启动月份
                council_cfg = self.v4_cfg.get('evaluation', {}).get('council', {})
                start_after_month = council_cfg.get('start_after_month', 0)
                
                if self.current_month < start_after_month:
                    print(f"    [Council Filter] Council智能体延迟启动: 当前月份{self.current_month} < 启动月份{start_after_month}，跳过")
                    all_candidates = set()  # 返回空候选集，跳过本次执行
                else:
                    # Council智能体：完全绕过河流过滤，可以跨河放置A/B/C
                    print(f"    [River Filter] Council agent: 完全绕过河流过滤，保留所有槽位")
                    # all_candidates 已经包含所有槽位，不进行任何过滤
                    
                    # Council特殊过滤：只保留building_level=3的槽位
                    # 原因：building_level=3只能建S尺寸，IND不会在这些槽位放M/L，避免抢占
                    level_3_candidates = set()
                    for slot_id in all_candidates:
                        slot = self.slots.get(slot_id)
                        if slot and getattr(slot, 'building_level', 3) == 3:
                            level_3_candidates.add(slot_id)
                    
                    print(f"    [Council Filter] building_level=3过滤: {len(all_candidates)} -> {len(level_3_candidates)}个槽位")
                    all_candidates = level_3_candidates
            elif self.current_agent == 'EDU':
                # EDU agent：A/B/C绕过河流过滤，S/M/L仍受限制
                # 由于在槽位候选阶段无法区分具体尺寸，我们采用以下策略：
                # 1. 先获取同侧槽位（S/M/L用）
                # 2. 再添加对岸槽位（A/B/C用）
                same_side_candidates = set()
                other_side_candidates = set()
                
                # 获取河流信息用于对岸判断
                rivers = self.v4_cfg.get('terrain_features', {}).get('rivers', [])
                coords = rivers[0].get('coordinates', []) if rivers else []
                center_y = None
                edu_hub_y = None
                
                if coords:
                    y_coords = [point[1] for point in coords]
                    center_y = sum(y_coords) / len(y_coords)
                    hubs = self.v4_cfg.get('city', {}).get('transport_hubs', [[125, 75], [112, 121]])
                    edu_hub_y = hubs[1][1] if len(hubs) > 1 else hubs[0][1]
                
                for slot_id in all_candidates:
                    slot = self.slots.get(slot_id)
                    if slot is None:
                        continue
                        
                    x = float(getattr(slot, 'fx', slot.x))
                    y = float(getattr(slot, 'fy', slot.y))
                    slot_comp = self._get_component_of_xy(x, y)
                    
                    # 先检查是否在对岸（不依赖于连通域）
                    if center_y is not None and edu_hub_y is not None:
                        is_other_side = (y > center_y) != (edu_hub_y > center_y)
                        if is_other_side:
                            other_side_candidates.add(slot_id)
                        elif slot_comp == expected_comp:
                            same_side_candidates.add(slot_id)
                    else:
                        # 如果没有河流信息，按连通域判断
                        if slot_comp == expected_comp:
                            same_side_candidates.add(slot_id)
                
                # 【关键修复】EDU agent保留所有槽位（同侧+对岸），不进行河流过滤
                # 因为A/B/C需要绕过河流过滤，S/M/L的过滤在动作枚举阶段处理
                print(f"    [River Filter] EDU agent: 保留所有槽位，不进行河流过滤")
                # all_candidates 已经包含同侧和对岸槽位，直接使用，不进行任何过滤
            else:
                # IND agent：所有尺寸都受河流过滤限制
                filtered_candidates = set()
                for slot_id in all_candidates:
                    slot = self.slots.get(slot_id)
                    if slot is None:
                        continue
                        
                    x = float(getattr(slot, 'fx', slot.x))
                    y = float(getattr(slot, 'fy', slot.y))
                    slot_comp = self._get_component_of_xy(x, y)
                    
                    if slot_comp == expected_comp:
                        filtered_candidates.add(slot_id)
                
                print(f"    [River Filter] {self.current_agent} agent: {len(all_candidates)} -> {len(filtered_candidates)} candidates (连通域 {expected_comp})")
                all_candidates = filtered_candidates
        else:
            print(f"    [River Filter] hub_components未设置，返回所有候选槽位: {len(all_candidates)}")
        
        # 【新增】邻近性约束：优先选择靠近已有建筑的槽位（在河流过滤之后，只在同一连通域内）
        proximity_cfg = self.v4_cfg.get('proximity_constraint', {})
        if proximity_cfg.get('enabled', False) and self.current_month >= proximity_cfg.get('apply_after_month', 1):
            from enhanced_city_simulation_v4_0 import filter_near_buildings
            # 只使用当前agent类型的建筑作为参考（避免跨连通域）
            agent_type = 'industrial' if self.current_agent == 'IND' else 'public'
            agent_buildings = self.buildings.get(agent_type, [])
            # 放宽/跳过 A/B/C 的邻近性阈值（仅 EDU 生效）
            max_dist = float(proximity_cfg.get('max_distance', 10.0))
            skip_abcs = bool(proximity_cfg.get('edu_abcs_skip', False))
            if self.current_agent == 'EDU' and skip_abcs:
                # A/B/C 跳过邻近性；S/M/L 仍应用
                # 将候选拆为 A/B/C 与其他，再只对其他应用邻近性过滤
                from logic.v4_enumeration import ActionEnumerator
                # 此处仅按槽ID无法区分尺寸，保守做法：跳过邻近性（只对 EDU 生效）
                pass
            else:
                if self.current_agent == 'EDU' and proximity_cfg.get('edu_abcs_relax', False):
                    max_dist = float(proximity_cfg.get('edu_abcs_max_distance', max_dist))
                all_candidates = filter_near_buildings(
                    all_candidates,
                    self.slots,
                    agent_buildings,
                    max_distance=max_dist,
                    min_candidates=int(proximity_cfg.get('min_candidates', 5))
                )
        
        # 【新增】过滤已占用的槽位
        occupied_slots = self._get_occupied_slots()
        if occupied_slots:
            # print(f"    [Debug] 过滤已占用槽位: {len(occupied_slots)}个")
            all_candidates = all_candidates - occupied_slots
            # print(f"    [Debug] 过滤后候选槽位: {len(all_candidates)}个")
        
        return all_candidates
    
    def _get_land_price_field(self) -> np.ndarray:
        """获取地价场"""
        # 更新地价系统
        self.land_price_system.update_land_price_field(self.current_month, self.buildings)
        
        # 提取地价场
        W, H = self.map_size
        land_price_field = np.zeros((H, W), dtype=np.float32)
        
        for y in range(H):
            for x in range(W):
                price = self.land_price_system.get_land_price([x, y])
                land_price_field[y, x] = max(0.0, min(1.0, float(price)))
        
        return land_price_field
    
    def _get_monthly_stats(self) -> Dict[str, Any]:
        """获取月度统计信息"""
        stats = {
            'total_buildings': sum(len(buildings) for buildings in self.buildings.values()),
            'public_buildings': len(self.buildings['public']),
            'industrial_buildings': len(self.buildings['industrial']),
            'monthly_rewards': {agent: rewards.copy() for agent, rewards in self.monthly_rewards.items()},
            'episode_length': len(self.episode_history)
        }
        
        return stats
    
    def _calculate_reward(self, agent: str, action: Action) -> float:
        """计算奖励（固定NPV机制）"""
        # 【固定NPV机制】核心改造
        # 只评估"建造动作本身"的价值，不包含被动收益
        
        # 1. 计算建造成本
        build_cost = float(action.cost) if action.cost is not None else 0.0
        
        if build_cost > 0:  # 有建造
            # 2. 计算未来收益（固定回报期）
            expected_lifetime = self.rl_cfg.get('expected_lifetime', 12)
            monthly_reward = float(action.reward) if action.reward is not None else 0.0
            future_income = monthly_reward * expected_lifetime
            
            # 3. NPV = 未来收益 - 成本
            npv = future_income - build_cost
            
            # 4. 进度奖励
            progress_reward = 0.0
            if agent == 'EDU':
                progress_reward = len(self.buildings['public']) * 0.5
            elif agent == 'IND':
                progress_reward = len(self.buildings['industrial']) * 0.5
            
            # 5. 协作奖励（已禁用）
            cooperation_bonus = 0.0
            if self.rl_cfg.get('cooperation_lambda', 0) > 0:
                cooperation_bonus = self._calculate_cooperation_reward(agent, action)
            
            # 6. Budget惩罚（软约束）
            budget_penalty = 0.0
            if self.budgets is not None:
                # 预估建造后的budget
                if agent == 'Council':
                    # Council使用EDU的预算
                    budget_after = self.budgets['EDU'] - build_cost
                else:
                    budget_after = self.budgets[agent] - build_cost
                
                # 负债惩罚
                if budget_after < 0:
                    debt_penalty_coef = self.budget_cfg.get('debt_penalty_coef', 0.1)
                    budget_penalty = abs(budget_after) * debt_penalty_coef
                
                # 破产检测
                bankruptcy_threshold = self.budget_cfg.get('bankruptcy_threshold', -5000)
                if budget_after < bankruptcy_threshold:
                    bankruptcy_penalty = abs(self.budget_cfg.get('bankruptcy_penalty', -100.0))
                    budget_penalty += bankruptcy_penalty
            
            # 7. 总奖励 = NPV + 进度 - 惩罚
            total_reward = npv + progress_reward + cooperation_bonus - budget_penalty
        else:
            # 空序列（不建造）：无reward
            total_reward = 0.0
        
        # 记录奖励
        self.monthly_rewards[agent].append(total_reward)
        
        # 8. 奖励缩放到[-1, 1]范围
        scale_factor = self.rl_cfg.get('reward_scale', 3000.0)
        scaled_reward = total_reward / scale_factor
        
        # Reward clipping
        clip_value = self.rl_cfg.get('reward_clip', 1.0)
        scaled_reward = np.clip(scaled_reward, -clip_value, clip_value)
        
        
        # 调试信息
        if abs(scaled_reward) > 1:
            # print(f"    [Reward Debug] {agent}: npv={npv if build_cost > 0 else 0:.1f}, progress={progress_reward if build_cost > 0 else 0:.1f}, total={total_reward:.1f}, scaled={scaled_reward:.3f}")
            pass
        
        return scaled_reward
    
    def _calculate_cooperation_reward(self, agent: str, action: Action) -> float:
        """计算协作奖励"""
        cooperation_lambda = self.rl_cfg.get('cooperation_lambda', 0)
        if cooperation_lambda <= 0:
            return 0.0
        
        cooperation_bonus = 0.0
        
        # 1. 功能互补奖励：EDU和IND的协调
        if agent == 'EDU':
            # EDU建筑越多，IND的奖励越高
            ind_buildings = len(self.buildings['industrial'])
            cooperation_bonus += ind_buildings * 0.05
        elif agent == 'IND':
            # IND建筑越多，EDU的奖励越高
            edu_buildings = len(self.buildings['public'])
            cooperation_bonus += edu_buildings * 0.05
        
        # 2. 空间协调奖励：距离其他建筑的协调性
        if action.footprint_slots:
            first_slot_id = action.footprint_slots[0]
            slot = self.slots.get(first_slot_id)
            if slot:
                # 计算与其他建筑的距离协调性
                for building_type in ['public', 'industrial']:
                    for building in self.buildings[building_type]:
                        if 'footprint_slots' in building:
                            for other_slot_id in building['footprint_slots']:
                                other_slot = self.slots.get(other_slot_id)
                                if other_slot:
                                    # 计算距离
                                    distance = ((slot.x - other_slot.x) ** 2 + (slot.y - other_slot.y) ** 2) ** 0.5
                                    # 适中距离给予奖励（不要太近也不要太远）
                                    if 5 <= distance <= 20:
                                        cooperation_bonus += 0.02
        
        return cooperation_lambda * cooperation_bonus
    
    def _calculate_monthly_income(self, agent: str) -> float:
        """计算agent的月度收益（所有在营建筑的累加）"""
        if agent not in self.active_assets:
            return 0.0
        
        total_income = sum([asset['monthly_income'] for asset in self.active_assets[agent]])
        return float(total_income)
    
    def _calculate_building_monthly_income(self, agent: str, action: Action) -> float:
        """计算单个建筑的月度收入（使用动作枚举阶段计算的reward值）"""
        # 使用动作枚举阶段已经计算好的reward值作为月度收入
        # 这个值已经考虑了建筑类型、大小、位置、地价等因素
        return float(action.reward) if action.reward is not None else 0.0
    
    def _place_building(self, agent: str, action: Action):
        """放置建筑"""
        # 根据智能体类型和动作确定建筑类型
        if agent == 'EDU':
            building_type = 'public'
        elif agent == 'IND':
            building_type = 'industrial'
        elif agent == 'Council':
            # Council智能体放置A/B/C尺寸的教育建筑（与EDU相同类型但不同尺寸）
            building_type = 'public'
        else:
            raise ValueError(f"未知的智能体类型: {agent}")
        
        # 获取建筑位置（使用footprint的第一个槽位）
        if not action.footprint_slots:
            return
        
        first_slot_id = action.footprint_slots[0]
        slot = self.slots.get(first_slot_id)
        if slot is None:
            return
        
        # 创建建筑记录
        building = {
            'xy': [slot.x, slot.y],
            'agent': agent,
            'size': action.size,
            'month': self.current_month,
            'score': action.score,
            'footprint_slots': action.footprint_slots.copy()
        }
        
        # 添加到建筑列表
        self.buildings[building_type].append(building)
        
        # 【月度收益机制】记录为在营资产
        # 使用正确的月度收入计算
        monthly_income = self._calculate_building_monthly_income(agent, action)
        asset = {
            'size': action.size,
            'monthly_income': monthly_income,
            'cost': float(action.cost) if action.cost is not None else 0.0,
            'built_month': self.current_month,
            'building_id': len(self.active_assets[agent])  # 唯一ID
        }
        self.active_assets[agent].append(asset)
    
    def _advance_turn(self) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """推进回合（支持Turn-Based和Multi-Agent模式）"""
        # 【月度收益机制】每月开始时，为所有agent累加月度收益到budget
        if self.budgets is not None:
            for ag in self.rl_cfg['agents']:
                monthly_income = self._calculate_monthly_income(ag)
                
                # 【共享预算机制】Council的收益也加到EDU预算中
                if ag == 'Council':
                    self.budgets['EDU'] += monthly_income
                    # 同时更新Council的预算历史（用于记录）
                    if 'Council' in self.budget_history:
                        self.budget_history['Council'].append({
                            'month': self.current_month,
                            'income': monthly_income,
                            'remaining': self.budgets['EDU']
                        })
                else:
                    self.budgets[ag] += monthly_income
                
                # 记录月度收益历史
                self.monthly_income_history[ag].append(monthly_income)
        
        # 检查是否启用turn-based模式
        turn_based = self.v4_cfg.get('enumeration', {}).get('turn_based', False)
        custom_order = self.v4_cfg.get('enumeration', {}).get('custom_execution_order', {})
        
        if turn_based and custom_order.get('enabled', False):
            # 自定义执行顺序模式：IND单月，EDU+Council双月
            pattern = custom_order.get('pattern', 'IND_single_EDU_Council_pair')
            
            if pattern == 'IND_single_EDU_Council_pair':
                # 模式：IND单月，EDU+Council双月
                if self.current_agent == 'IND':
                    # IND执行后，进入下个月，切换到EDU
                    self.current_month += 1
                    self.current_agent = 'EDU'
                    self.agent_turn = 1  # EDU的索引
                    self._council_execution_phase = 'EDU'  # 标记当前是EDU阶段
                elif self.current_agent == 'EDU':
                    # EDU执行后，同月切换到Council
                    self.current_agent = 'Council'
                    self.agent_turn = 2  # Council的索引
                    self._council_execution_phase = 'Council'  # 标记当前是Council阶段
                elif self.current_agent == 'Council':
                    # Council执行后，进入下个月，切换到IND
                    self.current_month += 1
                    self.current_agent = 'IND'
                    self.agent_turn = 0  # IND的索引
                    self._council_execution_phase = None  # 重置阶段标记
            else:
                # 默认turn-based模式
                self.current_month += 1
                self.agent_turn = (self.agent_turn + 1) % len(self.rl_cfg['agents'])
                self.current_agent = self.rl_cfg['agents'][self.agent_turn]
        elif turn_based:
            # 标准Turn-Based模式：每月一个agent，轮流行动
            # 先进入下个月
            self.current_month += 1
            
            # 再轮换到下一个agent
            self.agent_turn = (self.agent_turn + 1) % len(self.rl_cfg['agents'])
            self.current_agent = self.rl_cfg['agents'][self.agent_turn]
        else:
            # Multi-Agent模式（原v4.1）：每月两个agent依次行动
            # 先轮换agent
            self.agent_turn = (self.agent_turn + 1) % len(self.rl_cfg['agents'])
            self.current_agent = self.rl_cfg['agents'][self.agent_turn]
            
            # 如果轮换回第一个智能体，进入下个月
            if self.agent_turn == 0:
                self.current_month += 1
        
        # 检查是否完成整个episode
        done = self.current_month >= self.total_months
        print(f"    Agent switched to {self.current_agent}, month={self.current_month}/{self.total_months}, done={done}")
        
        if done:
            info = {
                'episode_complete': True,
                'final_stats': self._get_monthly_stats(),
                'total_rewards': {
                    agent: sum(rewards) for agent, rewards in self.monthly_rewards.items()
                }
            }
        else:
            info = {'agent_switched': True, 'current_agent': self.current_agent}
        
        # 返回下一个状态
        next_state = self._get_current_state()
        
        return next_state, done, info
    
    def get_action_pool(self, agent: str) -> Tuple[List[Action], torch.Tensor, torch.Tensor]:
        """获取动作池、掩码和特征"""
        # 设置当前智能体（用于对岸槽位添加逻辑）
        self.current_agent = agent
        print(f"[DEBUG] 为{agent}智能体生成动作池...")
        # 获取候选槽位
        candidates = self._get_candidate_slots()
        occupied = self._get_occupied_slots()
        print(f"[DEBUG] 候选槽位数量: {len(candidates)}, 已占用槽位数量: {len(occupied)}")
        
        # 创建动作枚举器
        from logic.v4_enumeration import ActionEnumerator
        enumerator = ActionEnumerator(self.slots)
        
        # 枚举动作 - 支持Council智能体
        if agent == 'Council':
            # Council智能体：只负责A/B/C
            actions = enumerator.enumerate_actions(
                candidates=candidates,
                occupied=occupied,
                agent_types=[agent],
                sizes={'Council': ['A', 'B', 'C']},
                lp_provider=self._create_lp_provider(),
                adjacency='4-neighbor',
                caps=self.v4_cfg.get('enumeration', {}).get('caps', {})
            )
        else:
            # EDU/IND智能体：保持原有逻辑
            if agent == 'EDU':
                # EDU只负责S/M/L，不再负责A/B/C
                sizes = {'EDU': ['S', 'M', 'L']}
            else:
                # IND保持原有逻辑
                sizes = {'IND': ['S', 'M', 'L']}
            
            actions = enumerator.enumerate_actions(
                candidates=candidates,
                occupied=occupied,
                agent_types=[agent],
                sizes=sizes,
                lp_provider=self._create_lp_provider(),
                adjacency='4-neighbor',
                caps=self.v4_cfg.get('enumeration', {}).get('caps', {})
            )
        
        # 设置当前动作池，用于河流过滤判断
        self.current_actions = actions
        
        # 统计A/B/C动作数量
        abc_actions = [a for a in actions if a.size in ['A', 'B', 'C']]
        print(f"[DEBUG] 动作池生成完成: 总动作数={len(actions)}, A/B/C动作数={len(abc_actions)}")
        if abc_actions:
            abc_examples = [f"{a.size}({a.footprint_slots[0] if a.footprint_slots else 'N/A'})" for a in abc_actions[:5]]
            print(f"[DEBUG] A/B/C动作示例: {abc_examples}")
        
        if not actions:
            return [], torch.tensor([]), torch.tensor([])
        
        # 生成动作特征和掩码
        action_feats, mask = self._extract_action_features(actions)
        
        return actions, action_feats, mask
    
    def _create_lp_provider(self):
        """创建地价提供器"""
        def lp_provider(slot_id: str) -> float:
            slot = self.slots.get(slot_id)
            if slot is None:
                return 0.0
            price = self.land_price_system.get_land_price([slot.x, slot.y])
            return max(0.0, min(1.0, float(price)))
        
        return lp_provider
    
    def _get_other_side_slots(self) -> List[str]:
        """获取对岸槽位"""
        other_side_slots = []
        try:
            # print(f"    [Debug] 开始获取对岸槽位...")
            # 获取河流信息
            rivers = self.v4_cfg.get('terrain_features', {}).get('rivers', [])
            coords = rivers[0].get('coordinates', []) if rivers else []
            if not coords:
                # print(f"    [Debug] 没有河流坐标数据，尝试从文件加载...")
                # 尝试从文件加载河流坐标
                try:
                    from envs.v4_1.city_env import load_river_coords
                    coords = load_river_coords(self.v4_cfg)
                    # print(f"    [Debug] 从文件加载河流坐标: {len(coords)}个点")
                except Exception as e:
                    # print(f"    [Debug] 从文件加载河流坐标失败: {e}")
                    return other_side_slots
            
            if not coords:
                # print(f"    [Debug] 仍然没有河流坐标数据")
                return other_side_slots
            
            # 计算河流中心线
            y_coords = [point[1] for point in coords]
            center_y = sum(y_coords) / len(y_coords)
            
            # 获取EDU hub位置
            hubs = self.v4_cfg.get('city', {}).get('transport_hubs', [[125, 75], [112, 121]])
            edu_hub_y = hubs[1][1] if len(hubs) > 1 else hubs[0][1]
            
            # 检查所有槽位是否在对岸
            # print(f"    [Debug] 河流中心线: {center_y}, EDU hub: {edu_hub_y}")
            for slot_id, slot in self.slots.items():
                y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
                # 对岸判断：EDU hub在河流下方，对岸槽位在河流上方
                is_other_side = (y > center_y) != (edu_hub_y > center_y)
                if is_other_side:
                    other_side_slots.append(slot_id)
                    # if len(other_side_slots) <= 5:  # 只打印前5个
                    #     print(f"    [Debug] 对岸槽位: {slot_id}, y={y:.1f}")
            
            # print(f"    [Debug] 总槽位数: {len(self.slots)}, 对岸槽位数: {len(other_side_slots)}")
        except Exception as e:
            print(f"获取对岸槽位失败: {e}")
            import traceback
            traceback.print_exc()
        
        return other_side_slots
    
    def _filter_other_side_slots_by_distance(self, other_side_slots: List[str], base_candidates: Set[str]) -> List[str]:
        """筛选距离hub合理的对岸槽位"""
        valid_slots = []
        
        try:
            # 获取当前月份的距离约束
            from enhanced_city_simulation_v4_0 import compute_R
            R_prev, R_curr = compute_R(self.current_month, self.v4_cfg.get('hubs', {}), True)
            # print(f"    [Debug] 当前月份{self.current_month}的距离约束: R_curr={R_curr:.1f}")
            
            # 获取EDU hub位置
            hubs = self.v4_cfg.get('city', {}).get('transport_hubs', [[125, 75], [112, 121]])
            edu_hub = hubs[1] if len(hubs) > 1 else hubs[0]  # EDU hub
            
            for slot_id in other_side_slots:
                slot = self.slots.get(slot_id)
                if slot is None:
                    continue
                
                # 计算到EDU hub的距离
                x = float(getattr(slot, 'fx', getattr(slot, 'x', 0.0)))
                y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
                
                from enhanced_city_simulation_v4_0 import min_dist_to_hubs
                distance = min_dist_to_hubs(x, y, [edu_hub])
                
                # 检查距离是否合理（使用与ring_candidates相同的约束）
                if distance <= (R_curr + 1.0):  # 使用相同的tol=1.0
                    valid_slots.append(slot_id)
                    # if len(valid_slots) <= 3:  # 只打印前3个
                    #     print(f"    [Debug] 有效对岸槽位: {slot_id}, 距离={distance:.1f}")
                else:
                    # if len(valid_slots) <= 3:  # 只打印前3个被过滤的
                    #     print(f"    [Debug] 距离过远的对岸槽位: {slot_id}, 距离={distance:.1f} > {R_curr:.1f}")
                    pass
            
            # print(f"    [Debug] 对岸槽位距离筛选: {len(other_side_slots)} -> {len(valid_slots)}")
            pass
            
        except Exception as e:
            print(f"对岸槽位距离筛选失败: {e}")
            import traceback
            traceback.print_exc()
            # 如果筛选失败，返回空列表，避免添加不合理的槽位
            return []
        
        return valid_slots
    
    def _extract_action_features(self, actions: List[Action]) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取动作特征"""
        if not actions:
            return torch.tensor([]), torch.tensor([])
        
        features = []
        for action in actions:
            # 基础特征
            feat = [
                float(action.score),  # 动作得分
                float(action.cost),   # 成本
                float(action.reward), # 奖励
                float(action.prestige), # 声望
                len(action.footprint_slots),  # 占用槽位数量
            ]
            
            # 位置特征
            if action.footprint_slots:
                first_slot = self.slots.get(action.footprint_slots[0])
                if first_slot:
                    feat.extend([
                        first_slot.x / self.map_size[0],  # 归一化x坐标
                        first_slot.y / self.map_size[1],  # 归一化y坐标
                    ])
                else:
                    feat.extend([0.0, 0.0])
            else:
                feat.extend([0.0, 0.0])
            
            # 地价特征
            if action.footprint_slots:
                prices = []
                for slot_id in action.footprint_slots:
                    price = self._create_lp_provider()(slot_id)
                    prices.append(price)
                feat.extend([
                    np.mean(prices),  # 平均地价
                    np.std(prices),   # 地价标准差
                    np.min(prices),   # 最低地价
                    np.max(prices),   # 最高地价
                ])
            else:
                feat.extend([0.0, 0.0, 0.0, 0.0])
            
            # 填充到固定长度
            while len(feat) < 20:
                feat.append(0.0)
            
            features.append(feat[:20])  # 截断到20维
        
        # 转换为张量
        action_feats = torch.tensor(features, dtype=torch.float32)
        mask = torch.ones(len(actions), dtype=torch.bool)
        
        return action_feats, mask
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """渲染环境（可选）"""
        # 这里可以实现可视化逻辑
        pass
    
    def advance_to_next_month(self):
        """推进到下一个月份 - 交替模式：奇数月IND，偶数月EDU+Council"""
        self.current_month += 1
        
        # 检查是否超过总月份数
        if self.current_month >= self.total_months:
            return False  # 表示episode应该结束
        
        # 【交替模式】根据月份决定智能体
        if self.current_month % 2 == 1:  # 奇数月：IND
            self.current_agent = 'IND'
            self.agent_turn = 0
            self._council_execution_phase = None
        else:  # 偶数月：EDU和Council
            # 偶数月开始时，先切换到EDU
            self.current_agent = 'EDU'
            self.agent_turn = 1
            self._council_execution_phase = 'EDU'
        
        # 更新地价系统
        if hasattr(self.land_price_system, 'update_month'):
            self.land_price_system.update_month(self.current_month)
        
        return True  # 表示成功推进到下一月
    
    def switch_to_council_in_same_month(self):
        """在同一个月内从EDU切换到Council（仅偶数月使用）"""
        if self.current_month % 2 == 0 and self._council_execution_phase == 'EDU':
            self.current_agent = 'Council'
            self.agent_turn = 2
            self._council_execution_phase = 'Council'
            return True
        return False
    
    def close(self):
        """关闭环境"""
        pass
