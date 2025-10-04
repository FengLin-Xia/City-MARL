#!/usr/bin/env python3
"""
增强城市模拟系统 v4.0（独立版本）
- 外部槽位文件（支持浮点坐标）
- v4 动作枚举/打分/序列选择（基于 logic.v4_enumeration）
- 每月输出 actions_pool / sequences_pool / chosen_sequence
"""

import os
import json
import math
from typing import Dict, List, Tuple, Set
from collections import deque

from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.v4_enumeration import V4Planner, SlotNode, _auto_fill_neighbors_4n


def read_config(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def load_slots_from_points_file(points_file: str, map_size: List[int]) -> Dict[str, SlotNode]:
    """从简单文本点集文件加载槽位（每行含两个坐标即可，允许夹杂其他数字）。
    - 保留浮点坐标精度
    - 自动生成 4-neighbor 邻接
    """
    import re
    if not os.path.exists(points_file):
        raise FileNotFoundError(f"points_file not found: {points_file}")
    W, H = int(map_size[0]), int(map_size[1])
    nodes: Dict[str, SlotNode] = {}
    seen: Set[Tuple[float, float]] = set()
    num_total = 0
    with open(points_file, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) < 2:
                continue
            num_total += 1
            try:
                xf = float(nums[0]); yf = float(nums[1])
            except Exception:
                continue
            # 画布范围判断（保留浮点）
            if xf < 0.0 or yf < 0.0 or xf >= float(W) or yf >= float(H):
                continue
            key = (xf, yf)
            if key in seen:
                continue
            seen.add(key)
            sid = f"s_{len(nodes)}"
            nodes[sid] = SlotNode(slot_id=sid, x=int(round(xf)), y=int(round(yf)), fx=xf, fy=yf)
    # 自动补邻接（基于像素坐标的 4-neighbor）
    _auto_fill_neighbors_4n(nodes)
    return nodes


def min_dist_to_hubs(x: float, y: float, hubs: List[List[float]]) -> float:
    best = 1e9
    for hx, hy in hubs:
        d = math.hypot(x - float(hx), y - float(hy))
        if d < best:
            best = d
    return best


def load_river_coords(cfg: Dict) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    try:
        tf = cfg.get('terrain_features', {})
        rivers = tf.get('rivers', [])
        for r in rivers:
            for c in r.get('coordinates', []) or []:
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    coords.append((float(c[0]), float(c[1])))
    except Exception:
        coords = []
    # 兼容 river.txt（如存在）
    if not coords and os.path.exists('river.txt'):
        import re
        try:
            with open('river.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
                    if len(nums) >= 2:
                        coords.append((float(nums[0]), float(nums[1])))
        except Exception:
            pass
    return coords


def river_center_y_from_coords(coords: List[Tuple[float, float]]) -> float:
    if not coords:
        return 0.0
    ys = [c[1] for c in coords]
    return (min(ys) + max(ys)) / 2.0


def compute_R(month: int, hubs_cfg: Dict, use_pixel: bool = True) -> Tuple[float, float]:
    """计算 R(m-1), R(m)。
    hubs_cfg 示例：{"list":[{"R0":0,"dR":40}, ...]}；像素模式：R = R0_px + dR_px * m
    这里简单返回同一 R（对所有 hub 采用相同 R），取 hubs_cfg 中第一个的参数作为全局。
    """
    if not hubs_cfg or not isinstance(hubs_cfg.get('list', []), list) or len(hubs_cfg['list']) == 0:
        R0 = 0.0; dR = 1.0
    else:
        h0 = hubs_cfg['list'][0]
        R0 = float(h0.get('R0', 0.0))
        dR = float(h0.get('dR', 1.0))
    # 像素增长
    R_prev = R0 + dR * max(0, month - 1)
    R_curr = R0 + dR * month
    return R_prev, R_curr


def ring_candidates(nodes: Dict[str, SlotNode], hubs: List[List[float]], month: int, hubs_cfg: Dict, tol: float = 1.0) -> Set[str]:
    """返回当月候选槽位集合。

    支持两种模式（通过 hubs_cfg['candidate_mode'] 配置）：
    - 'ring'：严格环带 (R_prev, R_curr]，若为空回退至 <= R_curr
    - 'cumulative'（默认）：累计模式，直接使用 <= R_curr
    """
    mode = str(hubs_cfg.get('candidate_mode', 'cumulative')).lower()
    R_prev, R_curr = compute_R(month, hubs_cfg, True)
    cand: Set[str] = set()
    if mode == 'ring':
        for sid, n in nodes.items():
            # 使用浮点坐标 fx, fy 而不是整数坐标 x, y
            x = float(getattr(n, 'fx', n.x))
            y = float(getattr(n, 'fy', n.y))
            d = min_dist_to_hubs(x, y, hubs)
            if d <= (R_curr + tol) and d > (R_prev - tol):
                cand.add(sid)
        # 回退：若严格环带为空，允许 <= R_curr
        if not cand:
            for sid, n in nodes.items():
                x = float(getattr(n, 'fx', n.x))
                y = float(getattr(n, 'fy', n.y))
                d = min_dist_to_hubs(x, y, hubs)
                if d <= (R_curr + tol):
                    cand.add(sid)
        return cand
    # 累计模式：<= R_curr
    for sid, n in nodes.items():
        x = float(getattr(n, 'fx', n.x))
        y = float(getattr(n, 'fy', n.y))
        d = min_dist_to_hubs(x, y, hubs)
        if d <= (R_curr + tol):
            cand.add(sid)
    return cand


# --- River-based region split helpers ---
def _dist_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / ab2
    if t < 0.0:
        qx, qy = ax, ay
    elif t > 1.0:
        qx, qy = bx, by
    else:
        qx, qy = ax + t * abx, ay + t * aby
    return math.hypot(px - qx, py - qy)


def _dist_point_to_polyline(px: float, py: float, coords: List[Tuple[float, float]]) -> float:
    best = 1e9
    for i in range(len(coords) - 1):
        ax, ay = coords[i]
        bx, by = coords[i + 1]
        d = _dist_point_to_segment(px, py, ax, ay, bx, by)
        if d < best:
            best = d
    return best


def _get_river_buffer_px(cfg: Dict, default_px: float = 2.0) -> float:
    try:
        tf = cfg.get('terrain_features', {})
        rivers = tf.get('rivers', [])
        for r in rivers:
            val = r.get('buffer_distance_px')
            if val is not None:
                return float(val)
    except Exception:
        pass
    return float(default_px)


def build_river_components(map_size: List[int], river_coords: List[Tuple[float, float]], buffer_px: float) -> List[List[int]]:
    """将画布分成两个区域：把距离河流折线<=buffer_px 的格点视为障碍，
    对其余格点做 4-邻接连通分量标记。返回 comp[y][x]（-1=河流；>=0 为连通域 id）。"""
    W, H = int(map_size[0]), int(map_size[1])
    comp: List[List[int]] = [[-2 for _ in range(W)] for _ in range(H)]  # -2=未访问, -1=河流
    if not river_coords:
        # 无河流：全域同一分量0
        for y in range(H):
            for x in range(W):
                comp[y][x] = 0
        return comp
    # 标记河流缓冲
    b = float(max(0.0, buffer_px))
    for y in range(H):
        py = float(y)
        for x in range(W):
            px = float(x)
            if _dist_point_to_polyline(px, py, river_coords) <= b:
                comp[y][x] = -1  # 在河内
    # BFS 连通域
    cid = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for y in range(H):
        for x in range(W):
            if comp[y][x] != -2:
                continue
            # 新分量
            q: deque = deque()
            q.append((x, y))
            comp[y][x] = cid
            while q:
                cx, cy = q.popleft()
                for dx, dy in dirs:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H and comp[ny][nx] == -2:
                        comp[ny][nx] = cid
                        q.append((nx, ny))
            cid += 1
    # 若只有一个或0个非河域，保持即可
    return comp


# --- Optional: side by TXT lists (north/south) ---
def _load_side_points_from_txt(north_path: str, south_path: str) -> Tuple[Set[Tuple[float, float]], Set[Tuple[float, float]]]:
    import re
    def load(path: str) -> Set[Tuple[float, float]]:
        pts: Set[Tuple[float, float]] = set()
        if not os.path.exists(path):
            return pts
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                nums = re.findall(r"-?\d+(?:\.\d+)?", s)
                if len(nums) >= 2:
                    x = round(float(nums[0]), 3)
                    y = round(float(nums[1]), 3)
                    pts.add((x, y))
        return pts
    return load(north_path), load(south_path)


def build_sequences_pool(
    scored_actions: List,
    length_max: int = 5,
    beam_width: int = 50,
    max_expansions: int = 2000,
    stratified: bool = True,
    per_depth_count: int = 10,
) -> Tuple[List[Dict], Dict]:
    """基于已打分动作池生成序列池（与 SequenceSelector 类似）。
    - 默认启用分层采样：按长度1..L分别取前 per_depth_count 条并去重，合并为最终序列池（最多 L*per_depth_count）。
    - 返回调试信息：各层非冲突候选统计、beam大小、扩展计数与各层采样数。
    """
    acts = list(scored_actions)
    acts.sort(key=lambda a: a.score, reverse=True)
    BeamState = Tuple[List, Set[str], float]  # actions, used, score_sum
    beam: List[BeamState] = [([], set(), 0.0)]
    expansions = 0
    debug = { 'layers': [], 'stopped': '', 'stratified': {'enabled': bool(stratified), 'per_depth': int(per_depth_count), 'counts': {}} }
    # 收集各层候选（截断后beam中的序列）
    collected_by_depth: Dict[int, List[BeamState]] = {}
    # 序列集合键（忽略顺序）
    def _seq_key_from_actions_list(actions_list: List) -> Tuple:
        keys = []
        for a in actions_list:
            fp_key = tuple(sorted(a.footprint_slots))
            keys.append((a.agent, a.size, fp_key))
        return tuple(sorted(keys))

    for _ in range(int(max(1, length_max))):
        new_beam: List[BeamState] = []
        non_conflict_counts: List[int] = []
        for seq, used, s_sum in beam:
            ncc = 0
            for a in acts:
                if any(sid in used for sid in a.footprint_slots):
                    continue
                ncc += 1
                n_used = used | set(a.footprint_slots)
                n_seq = seq + [a]
                n_s = s_sum + float(a.score)
                new_beam.append((n_seq, n_used, n_s))
                expansions += 1
                if expansions >= max_expansions:
                    break
            non_conflict_counts.append(ncc)
            if expansions >= max_expansions:
                break
        if not new_beam:
            debug['stopped'] = debug.get('stopped') or 'no_new_beam'
            break
        # unique-before-trim：对 new_beam 以“动作集合”去重，仅保留分数最高的代表
        uniq: Dict[Tuple, BeamState] = {}
        for st in new_beam:
            k = _seq_key_from_actions_list(st[0])
            if (k not in uniq) or (st[2] > uniq[k][2]):
                uniq[k] = st
        uniq_list: List[BeamState] = list(uniq.values())
        uniq_list.sort(key=lambda st: st[2], reverse=True)
        beam = uniq_list[: int(max(1, beam_width))]
        # 收集当前深度的序列样本
        collected_by_depth.setdefault(len(beam[0][0]) if beam and beam[0][0] else _ + 1, [])  # 深度≈动作数
        collected_by_depth[len(beam[0][0]) if beam and beam[0][0] else (_ + 1)] = list(beam)
        # 记录层调试信息
        if non_conflict_counts:
            layer_info = {
                'beam_in': len(beam),
                'non_conflict_min': int(min(non_conflict_counts)),
                'non_conflict_max': int(max(non_conflict_counts)),
                'non_conflict_avg': float(sum(non_conflict_counts) / len(non_conflict_counts)),
                'new_beam_kept': len(beam),
                'expansions_used': int(expansions),
            }
        else:
            layer_info = {
                'beam_in': len(beam),
                'non_conflict_min': 0,
                'non_conflict_max': 0,
                'non_conflict_avg': 0.0,
                'new_beam_kept': len(beam),
                'expansions_used': int(expansions),
            }
        debug['layers'].append(layer_info)
        if expansions >= max_expansions:
            debug['stopped'] = 'max_expansions'
            break
    # 转换为可序列化 + 分层采样 + 去重（同集合不同顺序视为同一序列）
    def seq_key(actions_list: List) -> Tuple:
        # 以动作集合的有序键作为唯一标识，避免 {1,2} 与 {2,1} 重复
        keys = []
        for a in actions_list:
            fp_key = tuple(sorted(a.footprint_slots))
            keys.append((a.agent, a.size, fp_key))
        return tuple(sorted(keys))

    out: List[Dict] = []
    seen: Set[Tuple] = set()

    if stratified:
        for depth in range(1, int(max(1, length_max)) + 1):
            layer = collected_by_depth.get(depth, [])
            # 已按分数排序（beam截断后），取前 per_depth_count
            kept = 0
            for seq, used, s_sum in layer:
                if kept >= int(max(0, per_depth_count)):
                    break
                k = seq_key(seq)
                if k in seen:
                    continue
                seen.add(k)
                out.append({
                    'score': float(s_sum),
                    'actions': [
                        {
                            'agent': a.agent,
                            'size': a.size,
                            'footprint_slots': a.footprint_slots,
                            'score': float(a.score)
                        } for a in seq
                    ]
                })
                kept += 1
            debug['stratified']['counts'][str(depth)] = kept
        # 若不足 beam_width，总量保持不超过 beam_width
        if len(out) > int(beam_width):
            out = out[: int(beam_width)]
    else:
        for seq, used, s_sum in beam:
            k = seq_key(seq)
            if k in seen:
                continue
            seen.add(k)
            out.append({
                'score': float(s_sum),
                'actions': [
                    {
                        'agent': a.agent,
                        'size': a.size,
                        'footprint_slots': a.footprint_slots,
                        'score': float(a.score)
                    } for a in seq
                ]
            })
    return out, debug


def main():
    cfg = read_config('configs/city_config_v4_0.json')
    sim_cfg = cfg.get('simulation', {})
    total_months = int(sim_cfg.get('total_months', 15))
    city = cfg.get('city', {})
    map_size = city.get('map_size', [200, 200])
    hubs = city.get('transport_hubs', [[125, 75], [112, 121]])

    v4 = cfg.get('growth_v4_0', {})
    if not v4:
        print('[v4.0] growth_v4_0 not found in config, please add it to enable v4.0 pipeline.')
        return

    out_dir = 'enhanced_simulation_v4_0_output'
    os.makedirs(out_dir, exist_ok=True)
    dbg_dir = os.path.join(out_dir, 'v4_debug')
    os.makedirs(dbg_dir, exist_ok=True)

    # 槽位源
    slots_source = v4.get('slots', {}).get('path', 'slotpoints.txt')
    slots = load_slots_from_points_file(slots_source, map_size)

    # 地价系统
    land = GaussianLandPriceSystem(cfg)
    land.initialize_system(hubs, map_size)

    # v4 规划器 & 回合制配置
    planner = V4Planner(cfg)
    enum_cfg = cfg.get('growth_v4_0', {}).get('enumeration', {})
    turn_based = bool(enum_cfg.get('turn_based', True))  # 默认开启严格轮换
    first_agent = str(enum_cfg.get('first_agent', 'EDU')).upper()  # 第一个月起始方，默认 EDU

    # 读取河流并计算中心线，用于“同侧过滤”
    river_coords = load_river_coords(cfg)
    river_center_y = river_center_y_from_coords(river_coords)
    # 基于河流缓冲的区域分割
    buffer_px = _get_river_buffer_px(cfg, default_px=2.0)
    comp_grid = build_river_components(map_size, river_coords, buffer_px)
    def comp_of_xy(x: float, y: float) -> int:
        xi, yi = int(round(x)), int(round(y))
        if yi < 0 or yi >= int(map_size[1]) or xi < 0 or xi >= int(map_size[0]):
            return -1
        return int(comp_grid[yi][xi])
    hub1_comp = comp_of_xy(hubs[0][0], hubs[0][1]) if len(hubs) >= 1 else 0
    hub2_comp = comp_of_xy(hubs[1][0], hubs[1][1]) if len(hubs) >= 2 else hub1_comp

    # 可选：通过 demo_slots_{north,south}.txt 直接限定侧别（优先级高于连通域）
    north_txt = os.path.join(dbg_dir, 'demo_slots_north.txt')
    south_txt = os.path.join(dbg_dir, 'demo_slots_south.txt')
    north_pts, south_pts = _load_side_points_from_txt(north_txt, south_txt)
    use_txt_side = (len(north_pts) + len(south_pts)) > 0

    # 状态（仅为占用记录与类型统计）
    buildings: Dict[str, List[Dict]] = {'public': [], 'industrial': []}

    for m in range(total_months):
        # 更新地价
        land.update_land_price_field(m, buildings)

        # 候选（环带）
        cand_ids = ring_candidates(slots, hubs, m, v4.get('hubs', {}), tol=1.0)

        # 占用集合
        occupied: Set[str] = set()
        xy_to_sid: Dict[Tuple[int, int], str] = {}
        for sid, n in slots.items():
            xy_to_sid[(int(n.x), int(n.y))] = sid
        for bt in ['public', 'industrial']:
            for b in buildings.get(bt, []):
                xy = b.get('xy', [0, 0])
                sid = xy_to_sid.get((int(round(xy[0])), int(round(xy[1]))))
                if sid is not None:
                    occupied.add(sid)

        # LP 提供器
        def lp_provider(slot_id: str) -> float:
            n = slots.get(slot_id)
            if n is None:
                return 0.0
            v = float(land.get_land_price([float(n.x), float(n.y)]))
            # 归一化假设 land_price 已在 0..1 内（若需要可再 clip）
            return max(0.0, min(1.0, v))

        # 河距提供器（米）：对 footprint 中每个槽位点取到河折线的垂足距离（像素）后× meters_per_pixel
        mpp = float(cfg.get('gaussian_land_price_system', {}).get('meters_per_pixel', 2.0))
        def river_distance_provider(slot_id: str) -> float:
            n = slots.get(slot_id)
            if n is None or not river_coords:
                return 0.0
            # 用原始浮点坐标计算到折线的最近距离（像素）
            px = float(getattr(n, 'fx', n.x)); py = float(getattr(n, 'fy', n.y))
            d_px = _dist_point_to_polyline(px, py, river_coords)
            return float(d_px) * mpp

        # 计划（严格轮换默认开启）
        if turn_based:
            # 确定当月 agent：偶数/奇数月以 first_agent 为起点轮换
            # m=0 用 first_agent；m=1 另一方；再交替...
            if first_agent == 'EDU':
                active_agent = 'EDU' if (m % 2 == 0) else 'IND'
            else:
                active_agent = 'IND' if (m % 2 == 0) else 'EDU'
            active_sizes = {active_agent: ['S', 'M', 'L']}
            actions, best_seq = planner.plan(
                slots=slots,
                candidates=cand_ids,
                occupied=occupied,
                lp_provider=lp_provider,
                river_distance_provider=river_distance_provider,
                agent_types=[active_agent],
                sizes=active_sizes,
            )
        else:
            actions, best_seq = planner.plan(
                slots=slots,
                candidates=cand_ids,
                occupied=occupied,
                lp_provider=lp_provider,
                river_distance_provider=river_distance_provider,
                agent_types=['EDU', 'IND'],
                sizes={'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L']},
            )

        # 同侧过滤：IND 仅在 hub1 一侧；EDU 仅在 hub2 一侧
        def node_comp(n: SlotNode) -> int:
            # 使用 fx/fy 若有，否则用整数像素坐标
            xx = float(getattr(n, 'fx', n.x))
            yy = float(getattr(n, 'fy', n.y))
            return comp_of_xy(xx, yy)

        def action_allowed(a) -> bool:
            # TXT优先：若提供了 north/south 列表，则 EDU 只能在 north 列表、IND 只能在 south 列表
            if use_txt_side:
                for sid in a.footprint_slots:
                    n = slots.get(sid)
                    if n is None:
                        return False
                    fx = float(getattr(n, 'fx', n.x)); fy = float(getattr(n, 'fy', n.y))
                    key = (round(fx, 3), round(fy, 3))
                    if a.agent == 'EDU':
                        if key not in north_pts:
                            return False
                    elif a.agent == 'IND':
                        if key not in south_pts:
                            return False
                return True
            # 否则用连通域：IND 在 hub1_comp；EDU 在 hub2_comp
            target_comp = hub1_comp if a.agent == 'IND' else (hub2_comp if a.agent == 'EDU' else None)
            if target_comp is None:
                return True
            for sid in a.footprint_slots:
                n = slots.get(sid)
                if n is None:
                    return False
                if node_comp(n) != target_comp:
                    return False
            return True

        # 仅保留在侧别允许且完整落在当月候选集合内的动作
        filtered_actions = [
            a for a in actions
            if action_allowed(a) and all((sid in cand_ids) for sid in (a.footprint_slots or []))
        ]

        # 若原始最优序列不满足侧边约束，基于过滤后的动作池重新挑选
        if best_seq and getattr(best_seq, 'actions', None):
            ok = all(
                action_allowed(a) and all((sid in cand_ids) for sid in (a.footprint_slots or []))
                for a in best_seq.actions
            )
        else:
            ok = False
        if not ok:
            if filtered_actions:
                best_seq = planner.selector.choose_best_sequence(filtered_actions)
            else:
                # 如果过滤后为空，则当月无动作
                best_seq = None

        # 导出动作池（过滤后），并记录原始数量
        pool = []
        for a in filtered_actions:
            pool.append({
                'agent': a.agent, 'size': a.size, 'footprint_slots': a.footprint_slots,
                'lp_norm': float(a.LP_norm), 'cost': float(a.cost), 'reward': float(a.reward),
                'prestige': float(a.prestige), 'score': float(a.score)
            })
        with open(os.path.join(dbg_dir, f'actions_pool_month_{m:02d}.json'), 'w', encoding='utf-8') as f:
            out_head = {'month': m, 'count': len(pool), 'actions': pool}
            if turn_based:
                out_head['active_agent'] = active_agent
            json.dump(out_head, f, ensure_ascii=False, indent=2)

        # 导出序列池（beam 前若干高分）+ 调试统计
        seq_pool, seq_debug = build_sequences_pool(
            filtered_actions,
            length_max=planner.length_max,
            beam_width=planner.beam_width,
            max_expansions=planner.max_expansions
        )
        with open(os.path.join(dbg_dir, f'sequences_pool_month_{m:02d}.json'), 'w', encoding='utf-8') as f:
            out_seq = {'month': m, 'count': len(seq_pool), 'sequences': seq_pool, 'debug': seq_debug}
            if turn_based:
                out_seq['active_agent'] = active_agent
            json.dump(out_seq, f, ensure_ascii=False, indent=2)

        # 导出最终序列
        chosen = []
        if best_seq and getattr(best_seq, 'actions', None):
            for a in best_seq.actions:
                chosen.append({'agent': a.agent, 'size': a.size, 'footprint_slots': a.footprint_slots, 'score': float(a.score)})
        with open(os.path.join(dbg_dir, f'chosen_sequence_month_{m:02d}.json'), 'w', encoding='utf-8') as f:
            out_chosen = {'month': m, 'score': float(best_seq.score if best_seq else 0.0), 'actions': chosen}
            if turn_based:
                out_chosen['active_agent'] = active_agent
            json.dump(out_chosen, f, ensure_ascii=False, indent=2)

        # 落位（EDU→public，IND→industrial），占位按 footprint 第一格坐标代表
        if best_seq and getattr(best_seq, 'actions', None):
            for a in best_seq.actions:
                target_type = 'public' if a.agent == 'EDU' else 'industrial'
                if not a.footprint_slots:
                    continue
                rep_sid = a.footprint_slots[0]
                n = slots.get(rep_sid)
                if n is None:
                    continue
                # 输出仍使用原始浮点坐标（若存在），不影响基于整数像素的邻接
                pos = [float(n.fx) if getattr(n, 'fx', None) is not None else int(n.x),
                       float(n.fy) if getattr(n, 'fy', None) is not None else int(n.y)]
                b = {'id': f"{target_type[:3]}_{len(buildings[target_type])+1}", 'type': target_type, 'xy': pos}
                buildings[target_type].append(b)

    print(f"[v4.0] done. outputs at {out_dir}")


if __name__ == '__main__':
    main()


