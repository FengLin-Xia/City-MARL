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
    neighbors: List[str] = field(default_factory=list)
    terrain_mask: Optional[str] = None
    road_dist: Optional[float] = None
    occupied_by: Optional[str] = None
    reserved_in_turn: bool = False


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
        free_ids: Set[str] = {sid for sid in candidates if (sid not in occupied and not self.slots[sid].reserved_in_turn)}

        # EDU/IND 的 S/M/L
        for agent in agent_types:
            for size in sizes.get(agent, []):
                if agent == 'IND' and size in ('M', 'L'):
                    feats = self._enumerate_ind_footprints(size, free_ids)
                else:
                    feats = self._enumerate_single_slots(free_ids)

                # 生成动作
                for fp in feats:
                    lp_vals = [float(lp_provider(sid)) for sid in fp]
                    lp_norm = float(sum(lp_vals) / max(1, len(lp_vals)))
                    act = Action(
                        agent=agent,
                        size=size,
                        footprint_slots=list(fp),
                        zone=None,
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
            for nb in self.slots[sid].neighbors:
                if nb in free_ids:
                    key = tuple(sorted((sid, nb)))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        res.append([key[0], key[1]])
        return res

    def _enumerate_2x2_blocks(self, free_ids: Set[str]) -> List[List[str]]:
        """宽容版 2×2 检测：估计步长(step_x, step_y)+容差 ε，近似匹配四点方块。
        注意：使用 free_ids 限定仅在可用槽位上枚举。
        """
        coords: List[Tuple[str, float, float]] = [
            (sid, float(self.slots[sid].x), float(self.slots[sid].y)) for sid in free_ids
            if sid in self.slots
        ]
        if len(coords) < 4:
            return []

        # 估计水平/垂直步长（使用近邻统计的中位数）
        def median(lst: List[float]) -> float:
            if not lst:
                return 1.0
            s = sorted(lst)
            m = len(s) // 2
            return (s[m] if len(s) % 2 == 1 else 0.5 * (s[m - 1] + s[m]))

        # 搜集正向最近 dx（|dy| 小）与正向最近 dy（|dx| 小）
        dx_samples: List[float] = []
        dy_samples: List[float] = []
        # 容许的小跨越阈值（以像素计），用于判定“近似水平/垂直”
        small = 2.0
        for i, (_, x0, y0) in enumerate(coords):
            best_dx = 1e9
            best_dy = 1e9
            for j, (_, x1, y1) in enumerate(coords):
                if i == j:
                    continue
                dx = x1 - x0
                dy = y1 - y0
                # 水平邻近（dy 很小、dx>0 最小）
                if abs(dy) <= small and dx > 0 and dx < best_dx:
                    best_dx = dx
                # 垂直邻近（dx 很小、dy>0 最小）
                if abs(dx) <= small and dy > 0 and dy < best_dy:
                    best_dy = dy
            if best_dx < 1e9:
                dx_samples.append(best_dx)
            if best_dy < 1e9:
                dy_samples.append(best_dy)

        step_x = median(dx_samples)
        step_y = median(dy_samples)
        # 容差：允许 35% 偏差
        eps = 0.35 * max(1.0, min(step_x, step_y))

        # 辅助：在 coords 中寻找接近 (tx,ty) 的槽位 sid，阈值 eps
        def find_near(tx: float, ty: float, exclude: Set[str]) -> Optional[str]:
            best_sid = None
            best_d2 = eps * eps
            for sid, x, y in coords:
                if sid in exclude:
                    continue
                d2 = (x - tx) * (x - tx) + (y - ty) * (y - ty)
                if d2 <= best_d2:
                    best_sid = sid
                    best_d2 = d2
            return best_sid

        seen_blocks: Set[Tuple[str, str, str, str]] = set()
        res: List[List[str]] = []
        for sid_a, x0, y0 in coords:
            used = {sid_a}
            # 目标四点：A(x0,y0), B(x0+step_x,y0), C(x0,y0+step_y), D(x0+step_x,y0+step_y)
            sid_b = find_near(x0 + step_x, y0, used)
            if not sid_b:
                continue
            used.add(sid_b)
            sid_c = find_near(x0, y0 + step_y, used)
            if not sid_c:
                continue
            used.add(sid_c)
            sid_d = find_near(x0 + step_x, y0 + step_y, used)
            if not sid_d:
                continue
            block = tuple(sorted([sid_a, sid_b, sid_c, sid_d]))
            if block in seen_blocks:
                continue
            seen_blocks.add(block)
            res.append([sid_a, sid_b, sid_c, sid_d])
        return res


# -----------------------------
# 打分器（评估系统接口）
# -----------------------------

class ActionScorer:
    """根据 PRD 的 EDU/IND 公式计算 cost/reward/prestige，并做归一化与加权。

    normalize: per-month-pool-minmax（按当月动作池做最小-最大归一化）
    objective: {'EDU':{'w_r':..,'w_p':..,'w_c':..}, 'IND':{...}}
    """

    def __init__(self, objective: Dict[str, Dict[str, float]], normalize: str = 'per-month-pool-minmax'):
        self.objective = objective
        self.normalize = normalize

    def score_actions(self, actions: List[Action]) -> List[Action]:
        # 1) 先计算原始 cost/reward/prestige
        for a in actions:
            self._calc_crp(a)

        # 2) 归一化（各自维度 min-max）
        costs = [a.cost for a in actions]
        rewards = [a.reward for a in actions]
        prestiges = [a.prestige for a in actions]
        c_min, c_max = (min(costs) if costs else 0.0), (max(costs) if costs else 1.0)
        r_min, r_max = (min(rewards) if rewards else 0.0), (max(rewards) if rewards else 1.0)
        p_min, p_max = (min(prestiges) if prestiges else 0.0), (max(prestiges) if prestiges else 1.0)

        def norm(v, lo, hi):
            if hi - lo <= 1e-9:
                return 0.0
            return (v - lo) / (hi - lo)

        # 3) 按 agent 使用不同权重
        for a in actions:
            w = self.objective.get(a.agent, {"w_r": 0.5, "w_p": 0.3, "w_c": 0.2})
            nr = norm(a.reward, r_min, r_max)
            np_ = norm(a.prestige, p_min, p_max)
            nc = norm(a.cost, c_min, c_max)
            a.score = float(w.get('w_r', 0.5)) * nr + float(w.get('w_p', 0.3)) * np_ - float(w.get('w_c', 0.2)) * nc
        return actions

    def _calc_crp(self, a: Action) -> None:
        """简化版评估公式（可用配置替换）。
        - EDU：reward 更依赖 prestige/容量；cost 随 LP_norm 上升；prestige 随 adjacency 与 LP 增益
        - IND：reward 随市场/容量（用 LP_norm 近似）；cost 随 GFA_k（用 footprint 面积近似）；prestige 略受污染扣减
        """
        fp = len(a.footprint_slots)
        lp = max(0.0, min(1.0, a.LP_norm))
        if a.agent == 'EDU':
            cap_k = {'S': 60, 'M': 120, 'L': 240}
            base_cost = {'S': 0.8, 'M': 1.2, 'L': 2.0}
            opex = {'S': 0.15, 'M': 0.25, 'L': 0.35}
            a.cost = base_cost.get(a.size, 1.0) * (0.6 + 0.4 * lp)
            a.reward = 0.004 * cap_k.get(a.size, 80) * (0.7 + 0.3 * lp) - opex.get(a.size, 0.2)
            a.prestige = {'S': 0.20, 'M': 0.35, 'L': 0.55}.get(a.size, 0.3) + 0.10 * lp + 0.02 * fp
        else:  # IND
            cap_k = {'S': 80, 'M': 200, 'L': 500}
            gfa_k = {'S': 1.0, 'M': 2.0, 'L': 4.0}
            opex = 0.12 * gfa_k.get(a.size, 1.0)
            a.cost = (0.7 + 0.8 * gfa_k.get(a.size, 1.0)) * (0.5 + 0.5 * lp)
            a.reward = (0.006 * cap_k.get(a.size, 100) * (0.8 + 0.4 * lp)) - opex
            a.prestige = {'S': 0.05, 'M': 0.02, 'L': -0.02}.get(a.size, 0.0) + 0.04 * (1.0 - lp)


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

        self.scorer = ActionScorer(self.objective, self.normalize)
        self.selector = SequenceSelector(self.length_max, self.beam_width, self.max_expansions)

    def plan(
        self,
        slots: Dict[str, SlotNode],
        candidates: Set[str],
        occupied: Set[str],
        lp_provider,
        agent_types: Optional[List[str]] = None,
        sizes: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[Action], Sequence]:
        agent_types = agent_types or ['EDU', 'IND']
        sizes = sizes or {'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L']}

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

        scored = self.scorer.score_actions(actions)
        best_seq = self.selector.choose_best_sequence(scored)
        return scored, best_seq


