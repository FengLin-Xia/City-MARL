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
            nodes[sid] = SlotNode(slot_id=sid, x=int(round(xf)), y=int(round(yf)))
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
    R_prev, R_curr = compute_R(month, hubs_cfg, True)
    cand: Set[str] = set()
    for sid, n in nodes.items():
        d = min_dist_to_hubs(n.x, n.y, hubs)
        if d <= (R_curr + tol) and d > (R_prev - tol):
            cand.add(sid)
    # 回退：若严格环带为空，允许 <= R_curr
    if not cand:
        for sid, n in nodes.items():
            d = min_dist_to_hubs(n.x, n.y, hubs)
            if d <= R_curr:
                cand.add(sid)
    return cand


def build_sequences_pool(scored_actions: List, length_max: int = 5, beam_width: int = 50, max_expansions: int = 2000) -> Tuple[List[Dict], Dict]:
    """基于已打分动作池生成序列池（与 SequenceSelector 类似），返回若干高分序列。
    同时返回调试信息：各层非冲突候选统计、beam大小与扩展计数。
    """
    acts = list(scored_actions)
    acts.sort(key=lambda a: a.score, reverse=True)
    BeamState = Tuple[List, Set[str], float]  # actions, used, score_sum
    beam: List[BeamState] = [([], set(), 0.0)]
    expansions = 0
    debug = { 'layers': [], 'stopped': '' }
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
        new_beam.sort(key=lambda st: st[2], reverse=True)
        beam = new_beam[: int(max(1, beam_width))]
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
    # 转换为可序列化
    out = []
    for seq, used, s_sum in beam:
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
    total_months = int(sim_cfg.get('total_months', 12))
    city = cfg.get('city', {})
    map_size = city.get('map_size', [200, 200])
    hubs = city.get('transport_hubs', [[125, 75], [121, 112]])

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
    hub1_y = float(hubs[0][1]) if len(hubs) >= 1 else 0.0
    hub2_y = float(hubs[1][1]) if len(hubs) >= 2 else hub1_y
    hub1_side = 'north' if hub1_y > river_center_y else 'south'
    hub2_side = 'north' if hub2_y > river_center_y else 'south'

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
                agent_types=[active_agent],
                sizes=active_sizes,
            )
        else:
            actions, best_seq = planner.plan(
                slots=slots,
                candidates=cand_ids,
                occupied=occupied,
                lp_provider=lp_provider,
                agent_types=['EDU', 'IND'],
                sizes={'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L']},
            )

        # 同侧过滤：IND 仅在 hub1 一侧；EDU 仅在 hub2 一侧
        def node_side(n: SlotNode) -> str:
            return 'north' if float(n.y) > river_center_y else 'south'

        def action_allowed(a) -> bool:
            if a.agent == 'IND':
                target_side = hub1_side
            elif a.agent == 'EDU':
                target_side = hub2_side
            else:
                return True
            for sid in a.footprint_slots:
                n = slots.get(sid)
                if n is None:
                    return False
                if node_side(n) != target_side:
                    return False
            return True

        filtered_actions = [a for a in actions if action_allowed(a)]

        # 若原始最优序列不满足侧边约束，基于过滤后的动作池重新挑选
        if best_seq and getattr(best_seq, 'actions', None):
            ok = all(action_allowed(a) for a in best_seq.actions)
        else:
            ok = False
        if not ok:
            best_seq = planner.selector.choose_best_sequence(filtered_actions)

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
                pos = [int(n.x), int(n.y)]
                b = {'id': f"{target_type[:3]}_{len(buildings[target_type])+1}", 'type': target_type, 'xy': pos}
                buildings[target_type].append(b)

    print(f"[v4.0] done. outputs at {out_dir}")


if __name__ == '__main__':
    main()


