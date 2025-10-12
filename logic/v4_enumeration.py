"""
v4.0 架构：动作枚举器 / 打分器 / 序列选择器

本模块提供：
- ActionEnumerator: 基于外部槽位与 R(m) 候选集合，枚举 EDU/IND 的 S/M/L 合法动作
- ActionScorer: 依据 PRD 的公式与权重，计算 cost/reward/prestige 与综合得分
- SequenceSelector: 穷举/束搜索 长度≤L 的不冲突动作序列，并选择最优序列

API 预期与配置：
- 配置入口：growth_v4_0
- 槽位来源：外部文件（JSON），或由调用方传入含邻接关系的 slots 字典
- 适配 4-neighbor 邻接；IND(M)=1×2 相邻对；IND(L)=2×2 区块；EDU(S/M/L)=占位 S=1, M=1(可配置), L=1(可配置)

注意：
- 该模块为独立逻辑层，主流程负责：R(m) 计算、LP(x,y,m) 更新与归一化、锁定/重判、状态落地
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Iterable
import json
import math


# -----------------------------
# 数据结构
# -----------------------------

@dataclass
class SlotNode:
    slot_id: str
    x: int
    y: int
    # 原始浮点坐标（用于输出/精度保留，不影响4邻接整数栅格）
    fx: Optional[float] = None
    fy: Optional[float] = None
    neighbors: List[str] = field(default_factory=list)
    terrain_mask: Optional[str] = None
    road_dist: Optional[float] = None
    occupied_by: Optional[str] = None
    reserved_in_turn: bool = False
    building_level: int = 3  # 建筑等级：3=只能建S, 4=可建S/M, 5=可建S/M/L


@dataclass
class Action:
    agent: str               # 'EDU' | 'IND'
    size: str                # 'S' | 'M' | 'L'
    footprint_slots: List[str]
    zone: Optional[str]      # 预留：策略/分区，可为 None
    LP_norm: float           # [0,1] 的本地地价强度（由调用方传入或本模块估算）
    adjacency: Dict[str, int]
    cost: float = 0.0
    reward: float = 0.0
    prestige: float = 0.0
    score: float = 0.0


@dataclass
class Sequence:
    actions: List[Action]
    sum_cost: float
    sum_reward: float
    sum_prestige: float
    score: float


# -----------------------------
# 工具函数
# -----------------------------

def load_slots_from_file(path: str) -> Dict[str, SlotNode]:
    """从 JSON 文件加载槽位，要求包含 id,x,y 以及可选邻接信息。
    若无 neighbors，则按 4-neighbor 自动补齐。
    JSON 结构示例：[{"id":"s_1","x":10,"y":20,"neighbors":["s_2","s_3"]}, ...]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    by_id: Dict[str, SlotNode] = {}
    for obj in data:
        node = SlotNode(
            slot_id=str(obj['id']),
            x=int(obj['x']),
            y=int(obj['y']),
            neighbors=list(obj.get('neighbors', [])),
            terrain_mask=obj.get('terrain_mask'),
            road_dist=float(obj.get('road_dist', 0.0)) if obj.get('road_dist') is not None else None,
            occupied_by=obj.get('occupied_by')
        )
        by_id[node.slot_id] = node
    # 自动补邻接（4-neighbor）
    if all(len(n.neighbors) == 0 for n in by_id.values()):
        _auto_fill_neighbors_4n(by_id)
    return by_id


def _auto_fill_neighbors_4n(by_id: Dict[str, SlotNode]) -> None:
    """若未提供邻接，按 4-neighbor 自动生成（基于坐标相邻）。"""
    coords: Dict[Tuple[int, int], str] = {(n.x, n.y): nid for nid, n in by_id.items()}
    for nid, node in by_id.items():
        neighbors: List[str] = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            key = (node.x + dx, node.y + dy)
            sid = coords.get(key)
            if sid is not None:
                neighbors.append(sid)
        node.neighbors = neighbors


def within_ring_R(dist_to_hub: float, R_prev: float, R_curr: float, tol: float) -> bool:
    return (dist_to_hub <= (R_curr + tol)) and (dist_to_hub > (R_prev - tol))


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# -----------------------------
# 动作枚举器
# -----------------------------

class ActionEnumerator:
    """根据 PRD v4.0，按月生成合法动作 A(m)。

    - 输入：
      - slots: Dict[str, SlotNode]
      - occupied: Set[str] 已占用槽位 id
      - candidates: Set[str] 候选槽位 id（R(m) 内且未被过滤）
      - agent_types: List[str] 如 ['EDU','IND']
      - sizes: Dict[str, List[str]] 如 {'EDU':['S','M','L'],'IND':['S','M','L']}
      - lp_provider: callable(slot_id)->LP_norm(0~1)
      - caps:  例如 top_slots_per_agent_size 用于裁剪枚举量
    """

    def __init__(self, slots: Dict[str, SlotNode]):
        self.slots = slots
    
    def _calculate_zone_for_slots(self, footprint_slots: List[str]) -> str:
        """计算槽位集合的zone（near/mid/far）
        
        基于槽位到最近的hub的距离来计算zone
        """
        if not footprint_slots:
            return 'mid'
        
        # 获取所有hub位置（从slots中推断或使用固定位置）
        hub_positions = []
        
        # 方法1：从slots中寻找hub标记的槽位
        for slot_id, slot in self.slots.items():
            if hasattr(slot, 'is_hub') and slot.is_hub:
                hub_positions.append((slot.x, slot.y))
        
        # 方法2：如果没有找到hub标记，使用固定位置（从配置中获取）
        if not hub_positions:
            # 使用默认hub位置（这些应该从配置中获取）
            hub_positions = [(122, 80), (112, 121)]
        
        # 计算每个槽位到最近hub的距离
        min_distances = []
        for slot_id in footprint_slots:
            slot = self.slots.get(slot_id)
            if slot is None:
                continue
                
            x = getattr(slot, 'fx', getattr(slot, 'x', 0))
            y = getattr(slot, 'fy', getattr(slot, 'y', 0))
            
            min_dist_to_hub = float('inf')
            for hub_x, hub_y in hub_positions:
                dist = ((x - hub_x) ** 2 + (y - hub_y) ** 2) ** 0.5
                min_dist_to_hub = min(min_dist_to_hub, dist)
            
            if min_dist_to_hub != float('inf'):
                min_distances.append(min_dist_to_hub)
        
        if not min_distances:
            return 'mid'
        
        # 使用平均距离
        avg_distance = sum(min_distances) / len(min_distances)
        
        # 根据距离阈值确定zone
        if avg_distance <= 20:
            return 'near'
        elif avg_distance <= 50:
            return 'mid'
        else:
            return 'far'

    def enumerate_actions(
        self,
        candidates: Set[str],
        occupied: Set[str],
        agent_types: List[str],
        sizes: Dict[str, List[str]],
        lp_provider,
        adjacency: str = '4-neighbor',
        caps: Optional[Dict] = None,
    ) -> List[Action]:
        caps = caps or {}
        actions: List[Action] = []
        # 预先过滤可用槽位
        free_ids: Set[str] = {sid for sid in candidates if (sid not in occupied and sid in self.slots and not self.slots[sid].reserved_in_turn)}

        # EDU/IND 的 S/M/L
        for agent in agent_types:
            for size in sizes.get(agent, []):
                # 旧逻辑（已注释）：IND M/L需要相邻对/区块
                # if agent == 'IND' and size in ('M', 'L'):
                #     feats = self._enumerate_ind_footprints(size, free_ids)
                # else:
                #     feats = self._enumerate_single_slots(free_ids)
                
                # 新逻辑：IND根据building_level过滤，EDU保持不变
                if agent == 'IND':
                    feats = self._enumerate_ind_by_level(size, free_ids)
                else:
                    feats = self._enumerate_single_slots(free_ids)

                # 生成动作
                for fp in feats:
                    lp_vals = [float(lp_provider(sid)) for sid in fp]
                    lp_norm = float(sum(lp_vals) / max(1, len(lp_vals)))
                    # 计算zone（基于槽位位置和hub距离）
                    zone = self._calculate_zone_for_slots(fp)
                    
                    act = Action(
                        agent=agent,
                        size=size,
                        footprint_slots=list(fp),
                        zone=zone,
                        LP_norm=lp_norm,
                        adjacency={'footprint': len(fp)}
                    )
                    actions.append(act)

                # 裁剪 Top-K（可选）
                top_caps = caps.get('top_slots_per_agent_size', {})
                limit = int(top_caps.get(agent, {}).get(size, 0)) if top_caps else 0
                if limit > 0 and len(actions) > limit:
                    # 按 LP_norm 选前 K（此处先粗选，后续打分再精排）
                    subset = [a for a in actions if a.agent == agent and a.size == size]
                    subset.sort(key=lambda a: a.LP_norm, reverse=True)
                    keep = set(id(a) for a in subset[:limit])
                    actions = [a for a in actions if (a.agent != agent or a.size != size or id(a) in keep)]

        return actions

    def _enumerate_single_slots(self, free_ids: Set[str]) -> List[List[str]]:
        return [[sid] for sid in free_ids]

    def _enumerate_ind_by_level(self, size: str, free_ids: Set[str]) -> List[List[str]]:
        """
        根据槽位的building_level属性枚举IND建筑（新逻辑）
        
        规则：
        - S型：所有等级（3/4/5）都可以建
        - M型：只有等级4和5可以建
        - L型：只有等级5可以建
        
        所有IND建筑都是单槽位（不再需要相邻对/区块）
        """
        result = []
        
        for slot_id in free_ids:
            slot = self.slots.get(slot_id)
            if slot is None:
                continue
            
            # 获取槽位的建筑等级
            level = getattr(slot, 'building_level', 3)
            
            # 根据size和level判断是否可以建造
            if size == 'S':
                # S型：所有等级都可以
                result.append([slot_id])
            elif size == 'M':
                # M型：只有等级4和5
                if level >= 4:
                    result.append([slot_id])
            elif size == 'L':
                # L型：只有等级5
                if level >= 5:
                    result.append([slot_id])
        
        return result

    def _enumerate_ind_footprints(self, size: str, free_ids: Set[str]) -> List[List[str]]:
        """IND M=1×2 相邻对；IND L=2×2 区块。"""
        if size == 'M':
            return self._enumerate_adjacent_pairs(free_ids)
        if size == 'L':
            return self._enumerate_2x2_blocks(free_ids)
        return self._enumerate_single_slots(free_ids)

    def _enumerate_adjacent_pairs(self, free_ids: Set[str]) -> List[List[str]]:
        res: List[List[str]] = []
        seen_pairs: Set[Tuple[str, str]] = set()
        for sid in free_ids:
            if sid in self.slots:
                for nb in self.slots[sid].neighbors:
                    if nb in free_ids:
                        key = tuple(sorted((sid, nb)))
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            res.append([key[0], key[1]])
        return res

    def _enumerate_2x2_blocks(self, free_ids: Set[str]) -> List[List[str]]:
        res: List[List[str]] = []
        # 用坐标判断 2×2 ：(x,y),(x+1,y),(x,y+1),(x+1,y+1)
        by_coord: Dict[Tuple[int, int], str] = {}
        for nid in free_ids:
            if nid in self.slots:
                n = self.slots[nid]
                by_coord[(n.x, n.y)] = nid
        for (x, y), sid in by_coord.items():
            a = sid
            b = by_coord.get((x + 1, y))
            c = by_coord.get((x, y + 1))
            d = by_coord.get((x + 1, y + 1))
            if b and c and d:
                block = [a, b, c, d]
                res.append(block)
        return res


# -----------------------------
# 打分器（评估系统接口）
# -----------------------------

class ActionScorer:
    """根据 PRD 的 EDU/IND 公式计算 cost/reward/prestige，并做归一化与加权。

    normalize: per-month-pool-minmax（按当月动作池做最小-最大归一化）
    objective: {'EDU':{'w_r':..,'w_p':..,'w_c':..}, 'IND':{...}}
    """

    def __init__(self, objective: Dict[str, Dict[str, float]], normalize: str = 'per-month-pool-minmax', eval_params: Optional[Dict] = None, slots: Optional[Dict[str, SlotNode]] = None):
        self.objective = objective
        self.normalize = normalize
        self.slots = slots  # 用于邻近性奖励计算
        # 默认评估参数，可被配置覆盖
        self.params = self._build_default_params()
        if isinstance(eval_params, dict):
            for k, v in eval_params.items():
                if isinstance(v, dict) and k in self.params and isinstance(self.params[k], dict):
                    self.params[k].update(v)
                else:
                    self.params[k] = v

    def score_actions(self, actions: List[Action], river_distance_provider=None, buildings=None) -> List[Action]:
        # 1) 先计算原始 cost/reward/prestige
        for a in actions:
            self._calc_crp(a, river_distance_provider=river_distance_provider, buildings=buildings)

        # 2) 归一化（各自维度 min-max）
        costs = [a.cost for a in actions]
        rewards = [a.reward for a in actions]
        prestiges = [a.prestige for a in actions]
        c_min, c_max = (min(costs) if costs else 0.0), (max(costs) if costs else 1.0)
        r_min, r_max = (min(rewards) if rewards else 0.0), (max(rewards) if rewards else 1.0)
        p_min, p_max = (min(prestiges) if prestiges else 0.0), (max(prestiges) if prestiges else 1.0)

        def norm(v, lo, hi):
            if hi - lo <= 1e-9:
                # 当所有值相同时，返回0.5而不是0.0，避免完全抹平差异
                return 0.5
            return (v - lo) / (hi - lo)

        # 3) 按 agent 使用不同权重
        for a in actions:
            w = self.objective.get(a.agent, {"w_r": 0.5, "w_p": 0.3, "w_c": 0.2})
            nr = norm(a.reward, r_min, r_max)
            np_ = norm(a.prestige, p_min, p_max)
            nc = norm(a.cost, c_min, c_max)
            a.score = float(w.get('w_r', 0.5)) * nr + float(w.get('w_p', 0.3)) * np_ - float(w.get('w_c', 0.2)) * nc
        return actions

    def _calc_crp(self, a: Action, river_distance_provider=None, buildings=None) -> None:
        """PRD 正式实现（units: cost=M£, reward=k£/mo, prestige=—）。

        EDU:
          cost = (Base_EDU[size]+Add_EDU[size]) × LP_norm + ZoneAdd[zone]
          reward = (α × Capacity[size]) × m_zone × m_adj − OPEX_EDU[size]
          prestige = PrestigeBase[size] + I(zone==near) + I(adj) − β × Pollution[size]
        IND:
          cost = (Base_IND[size]+Add_IND[size]) × LP_norm + ZoneAdd[zone]
          reward = ((p_market × u × Capacity[size]) / 1000) × m_zone × m_adj − c_opex × GFA_k[size] + s_zone[zone]
          prestige = PrestigeBase[size] + I(zone==near) + I(adj) − 0.2 × Pollution[size]
        
        新增（邻近性奖励）：
          如果buildings参数提供，计算到最近建筑的距离，给予邻近奖励或距离惩罚
        """
        P = self.params
        size = (a.size or 'S')
        agent = (a.agent or 'EDU')
        zone = (a.zone or 'mid')
        lp = self._clamp(a.LP_norm, 0.1, 1.0)

        # --- INT 版地价与河流溢价计算 ---
        # LP_idx: 10..100，线性映射自 LP_norm（可由配置覆盖）
        lp_idx_min = int(P.get('LP_idx_min', 10))
        lp_idx_max = int(P.get('LP_idx_max', 100))
        lp_idx = int(round(lp_idx_min + (lp_idx_max - lp_idx_min) * lp))
        lp_idx = max(lp_idx_min, min(lp_idx_max, lp_idx))

        # LandPriceBase: kGBP per LP_idx point → LP_value in kGBP
        land_price_base = float(P.get('LandPriceBase', 11.0))
        LP_value = float(lp_idx) * land_price_base

        # river distance（米）
        river_dist_m = 0.0
        if callable(river_distance_provider):
            try:
                # 取 footprint 内最小距离
                dists = []
                for sid in (a.footprint_slots or []):
                    d = river_distance_provider(str(sid))
                    if d is not None:
                        dists.append(float(d))
                if dists:
                    river_dist_m = max(0.0, min(dists))
            except Exception:
                river_dist_m = 0.0

        # 衰减：2^(- d / RiverD_half_m)
        half_m = float(P.get('RiverD_half_m', 120.0))
        if half_m <= 1e-9:
            decay = 0.0
        else:
            decay = 2.0 ** (-(max(0.0, river_dist_m) / half_m))
        # Max premium pct
        rpct_map = P.get('RiverPmax_pct', {'IND': 20.0, 'EDU': 15.0})
        rpct = float(rpct_map.get(agent, 0.0)) / 100.0
        # Zone revenue base（kGBP）
        ZR = P.get('ZR', {'near': 80, 'mid': 40, 'far': 0})
        RevBase_IND = P.get('RevBase_IND', {'S': 180, 'M': 320, 'L': 520})
        RevBase_EDU = P.get('RevBase_EDU', {'S': 140, 'M': 260, 'L': 420})
        if agent == 'IND':
            base_for_premium = float(RevBase_IND.get(size, 0.0)) + float(ZR.get(zone, 0.0))
        else:
            base_for_premium = float(RevBase_EDU.get(size, 0.0)) + float(ZR.get(zone, 0.0))
        raw_river_prem = base_for_premium * rpct * decay
        # round & clamp
        cap_k = float(P.get('RiverPremiumCap_kGBP', 10000.0))
        if str(P.get('RoundMode', 'nearest')).lower() == 'nearest':
            river_premium = float(int(round(raw_river_prem)))
        else:
            river_premium = raw_river_prem
        river_premium = max(0.0, min(cap_k, river_premium))

        # 邻接：优先读动作标志；否则 footprint>1 视为相邻
        is_adj = 0
        if isinstance(a.adjacency, dict):
            if 'adjacency' in a.adjacency:
                is_adj = 1 if self._bool(a.adjacency.get('adjacency')) else 0
            elif 'footprint' in a.adjacency:
                is_adj = 1 if int(a.adjacency.get('footprint', 1)) > 1 else 0
        if not is_adj and len(a.footprint_slots) > 1:
            is_adj = 1

        # 乘子
        m_zone = float(P.get('m_zone', {}).get(zone, 1.0))
        m_adj = float(P.get('m_adj', {}).get('on' if is_adj else 'off', 1.0))

        # 区位附加（成本/收入）与 OPEX、租金、基表（kGBP）
        ZC = P.get('ZC', {'near': 200, 'mid': 100, 'far': 0})
        OPEX_IND = P.get('OPEX_IND', {'S': 100, 'M': 180, 'L': 300})
        OPEX_EDU = P.get('OPEX_EDU', {'S': 70, 'M': 120, 'L': 190})
        Rent = P.get('Rent', {'S': 25, 'M': 45, 'L': 70})
        Adj_bonus = float(P.get('Adj_bonus', 0))
        # Base costs
        BaseCost_IND = P.get('BaseCost_IND', {'S': 900, 'M': 1500, 'L': 2400})
        BaseCost_EDU = P.get('BaseCost_EDU', {'S': 700, 'M': 1200, 'L': 1900})

        if agent == 'EDU':
            # 建造成本（含地价货币化）
            cost = float(BaseCost_EDU.get(size, 0.0)) + float(ZC.get(zone, 0.0)) + LP_value
            # 月度收入（build + land）
            rev_build = float(RevBase_EDU.get(size, 0.0)) + float(ZR.get(zone, 0.0)) + Adj_bonus * is_adj - float(OPEX_EDU.get(size, 0.0)) + river_premium
            rev_land = float(Rent.get(size, 0.0))
            reward = rev_build + rev_land
            # 声望先延用旧表（如需可改）
            pres0 = float(P.get('PrestigeBase_EDU', {'S': 0.2, 'M': 0.6, 'L': 1.0}).get(size, 0.0))
            prestige = pres0
        else:
            cost = float(BaseCost_IND.get(size, 0.0)) + float(ZC.get(zone, 0.0)) + LP_value
            rev = float(RevBase_IND.get(size, 0.0)) + float(ZR.get(zone, 0.0)) + Adj_bonus * is_adj - float(OPEX_IND.get(size, 0.0)) - float(Rent.get(size, 0.0)) + river_premium
            reward = rev
            pres0 = float(P.get('PrestigeBase_IND', {'S': 0.2, 'M': 0.1, 'L': -0.1}).get(size, 0.0))
            prestige = pres0

        # --- LP 正相关收益增益（INT 版）---
        # reward *= (1 + k * LP_norm)
        kmap = P.get('RewardLP_k', {'IND': 0.25, 'EDU': 0.10})
        try:
            k_lp = float(kmap.get(agent, 0.0))
        except Exception:
            k_lp = 0.0
        reward = reward * (1.0 + k_lp * lp)

        # 整数化（与 RoundMode 一致，默认 nearest）
        if str(P.get('RoundMode', 'nearest')).lower() == 'nearest':
            reward = float(int(round(reward)))

        # --- 邻近性奖励/惩罚（新增）---
        if buildings and len(buildings) > 0:
            # 获取动作的槽位位置
            if a.footprint_slots and len(a.footprint_slots) > 0:
                slot_id = a.footprint_slots[0]
                slot = self.slots.get(slot_id)
                if slot:
                    sx = float(getattr(slot, 'fx', slot.x))
                    sy = float(getattr(slot, 'fy', slot.y))
                    
                    # 计算到最近建筑的距离
                    min_dist = float('inf')
                    for b in buildings:
                        bxy = b.get('xy', [0, 0])
                        dist = math.hypot(sx - float(bxy[0]), sy - float(bxy[1]))
                        min_dist = min(min_dist, dist)
                    
                    # 邻近奖励/距离惩罚
                    proximity_threshold = float(P.get('proximity_threshold', 10.0))
                    proximity_reward_val = float(P.get('proximity_reward', 50.0))
                    distance_penalty_coef = float(P.get('distance_penalty_coef', 2.0))
                    
                    if min_dist <= proximity_threshold:
                        # 邻近奖励（距离越近，奖励越高）
                        proximity_bonus = proximity_reward_val * (1.0 - min_dist / proximity_threshold)
                        reward = reward + proximity_bonus
                    else:
                        # 距离惩罚（距离越远，惩罚越大）
                        distance_penalty = (min_dist - proximity_threshold) * distance_penalty_coef
                        reward = reward - distance_penalty
        
        # --- Size Bonus（鼓励建造M/L型建筑）---
        size_bonus_cfg = P.get('size_bonus', {})
        if size_bonus_cfg and size in size_bonus_cfg:
            size_bonus = float(size_bonus_cfg.get(size, 0))
            reward = reward + size_bonus

        a.cost = float(cost)
        a.reward = float(reward)
        a.prestige = float(prestige)

    # ---- helpers & defaults ----
    def _clamp(self, v: float, lo: float, hi: float) -> float:
        try:
            v = float(v)
        except Exception:
            v = lo
        return max(lo, min(hi, v))

    def _bool(self, v) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        try:
            return bool(int(v))
        except Exception:
            return str(v).lower() in ('true', 'yes', 'y', 'on', '1')

    def _build_default_params(self) -> Dict:
        # 可由 cfg.growth_v4_0.evaluation 覆盖
        return {
            # --- EDU ---
            'Base_EDU': {'S': 1.2, 'M': 2.8, 'L': 5.5},
            'Add_EDU':  {'S': 0.4, 'M': 0.9, 'L': 1.6},
            'Capacity_EDU': {'S': 60, 'M': 120, 'L': 240},
            'OPEX_EDU':     {'S': 0.20, 'M': 0.35, 'L': 0.55},
            'PrestigeBase_EDU': {'S': 0.2, 'M': 0.6, 'L': 1.0},
            'Pollution_EDU':    {'S': 0.2, 'M': 0.4, 'L': 0.6},

            # --- IND ---
            'Base_IND': {'S': 1.0, 'M': 2.2, 'L': 4.5},
            'Add_IND':  {'S': 0.5, 'M': 1.0, 'L': 2.0},
            'Capacity_IND': {'S': 80, 'M': 200, 'L': 500},
            'GFA_k':        {'S': 1.0, 'M': 2.0, 'L': 4.0},
            'PrestigeBase_IND': {'S': 0.2, 'M': 0.1, 'L': -0.1},
            'Pollution_IND':    {'S': 0.6, 'M': 0.9, 'L': 1.2},

            # --- Shared / scalars ---
            'ZoneAdd': {'near': 0.8, 'mid': 0.3, 'far': 0.0},
            's_zone':  {'near': 0.5, 'mid': 0.2, 'far': 0.0},
            'm_zone':  {'near': 1.10, 'mid': 1.00, 'far': 0.90},
            'm_adj':   {'on': 1.10, 'off': 1.00},

            'alpha': 0.08,
            'beta':  0.25,
            'p_market': 12.0,
            'u': 0.85,
            'c_opex': 0.30,
        }


# -----------------------------
# 序列选择器
# -----------------------------

class SequenceSelector:
    """在动作池上生成长度≤L的无冲突序列，并计算序列得分。

    支持：
    - 穷举（当动作池很小）
    - 束搜索（beam search）以控制扩展数量
    - 冲突规则：同一序列内 footprint_slots 不能交叠
    """

    def __init__(self, length_max: int = 5, beam_width: int = 16, max_expansions: int = 2000):
        self.length_max = int(max(1, length_max))
        self.beam_width = int(max(1, beam_width))
        self.max_expansions = int(max(1, max_expansions))

    def choose_best_sequence(
        self,
        actions: List[Action],
        objective: Optional[Dict[str, float]] = None
    ) -> Sequence:
        # 预处理：按得分排序，便于束搜索
        acts = list(actions)
        acts.sort(key=lambda a: a.score, reverse=True)

        # 初始化 beam（每个元素：序列、占用集合、累计指标）
        BeamState = Tuple[List[Action], Set[str], float, float, float, float]  # actions, used, cost, reward, prestige, score
        beam: List[BeamState] = [([], set(), 0.0, 0.0, 0.0, 0.0)]
        expansions = 0

        for depth in range(self.length_max):
            new_beam: List[BeamState] = []
            for seq, used, c_sum, r_sum, p_sum, s_sum in beam:
                # 扩展一个动作
                for a in acts:
                    if self._conflict(a, used):
                        continue
                    n_used = used | set(a.footprint_slots)
                    n_seq = seq + [a]
                    n_c = c_sum + a.cost
                    n_r = r_sum + a.reward
                    n_p = p_sum + a.prestige
                    # 简化序列得分：动作得分之和（也可使用统一归一化规则）
                    n_s = s_sum + a.score
                    new_beam.append((n_seq, n_used, n_c, n_r, n_p, n_s))
                    expansions += 1
                    if expansions >= self.max_expansions:
                        break
                if expansions >= self.max_expansions:
                    break
            if not new_beam:
                break
            # 束截断
            new_beam.sort(key=lambda st: st[5], reverse=True)
            beam = new_beam[: self.beam_width]
            if expansions >= self.max_expansions:
                break

        # 允许空序列（Skip）
        best = max(beam, key=lambda st: st[5]) if beam else ([], set(), 0.0, 0.0, 0.0, 0.0)
        seq, used, c_sum, r_sum, p_sum, s_sum = best
        return Sequence(actions=seq, sum_cost=c_sum, sum_reward=r_sum, sum_prestige=p_sum, score=s_sum)

    def _conflict(self, action: Action, used: Set[str]) -> bool:
        for sid in action.footprint_slots:
            if sid in used:
                return True
        return False


# -----------------------------
# 便捷入口（供主流程使用）
# -----------------------------

class V4Planner:
    """封装：动作池生成 → 打分 → 序列选择 的一步式调用。"""

    def __init__(self, cfg: Dict):
        enum_cfg = cfg.get('growth_v4_0', {}).get('enumeration', {})
        self.length_max = int(enum_cfg.get('length_max', 5))
        self.use_skip = bool(enum_cfg.get('use_skip', True))
        self.search_mode = str(enum_cfg.get('search_mode', 'exhaustive'))
        self.beam_width = int(enum_cfg.get('beam_width', 16))
        self.max_expansions = int(enum_cfg.get('max_expansions', 2000))
        self.caps = enum_cfg.get('caps', {})
        obj = enum_cfg.get('objective', {})
        self.objective = {
            'EDU': obj.get('EDU', {'w_r': 0.3, 'w_p': 0.6, 'w_c': 0.1}),
            'IND': obj.get('IND', {'w_r': 0.6, 'w_p': 0.2, 'w_c': 0.2}),
        }
        self.normalize = str(obj.get('normalize', 'per-month-pool-minmax'))
        # 读取评估参数（可覆盖默认表）
        self.eval_params = cfg.get('growth_v4_0', {}).get('evaluation', {})

        self.scorer = ActionScorer(self.objective, self.normalize, eval_params=self.eval_params, slots=None)
        self.selector = SequenceSelector(self.length_max, self.beam_width, self.max_expansions)
        self.slots = None  # 将在plan()中设置

    def plan(
        self,
        slots: Dict[str, SlotNode],
        candidates: Set[str],
        occupied: Set[str],
        lp_provider,
        river_distance_provider=None,
        agent_types: Optional[List[str]] = None,
        sizes: Optional[Dict[str, List[str]]] = None,
        buildings: Optional[List[Dict]] = None,
    ) -> Tuple[List[Action], Sequence]:
        agent_types = agent_types or ['EDU', 'IND']
        sizes = sizes or {'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L']}

        # 设置slots到scorer（用于邻近性奖励计算）
        if self.scorer.slots is None:
            self.scorer.slots = slots
        
        enumerator = ActionEnumerator(slots)
        actions = enumerator.enumerate_actions(
            candidates=candidates,
            occupied=occupied,
            agent_types=agent_types,
            sizes=sizes,
            lp_provider=lp_provider,
            adjacency='4-neighbor',
            caps=self.caps,
        )

        scored = self.scorer.score_actions(actions, river_distance_provider=river_distance_provider, buildings=buildings)
        best_seq = self.selector.choose_best_sequence(scored)
        return scored, best_seq


