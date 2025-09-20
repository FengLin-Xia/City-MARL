#!/usr/bin/env python3
"""
增强城市模拟系统 v3.6
依据 PRD v3.6：单池槽位 + Hub 外扩 R(m) + 分位数分类 + 锁定/重判
"""

import json
import os
import csv
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.finance_system import FinanceSystem


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
    """
    使用射线法判断点是否在多边形内部
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


@dataclass
class Slot:
    pos: List[int]
    used: bool = False
    dead: bool = False
    building_id: Optional[str] = None


class V36Config:
    def __init__(self, root: Dict):
        self.root = root
        self.map_size = root.get('city', {}).get('map_size', [110, 110])
        self.hubs: List[List[int]] = root.get('city', {}).get('transport_hubs', [[90, 55], [67, 94]])
        self.output_dir = 'enhanced_simulation_v3_6_output'
        # 河流配置
        self.rivers = root.get('terrain_features', {}).get('rivers', [])
        # 单位换算
        self.meters_per_pixel = float(root.get('gaussian_land_price_system', {}).get('meters_per_pixel', 2.0))
        # growth
        g = root.get('growth_v3_6', {})
        hubs_cfg = g.get('hubs', {})
        # Pixel 模式（优先）或米制模式倍率
        self.use_pixel_growth = bool(g.get('use_pixel_growth', True))
        self.pixel_delta_px = float(g.get('pixel_delta_px', 1.0))
        self.pixel_R0_px = float(g.get('pixel_R0_px', 0.0))
        self.global_speed_scale = float(g.get('global_speed_scale', 0.5))
        # 默认统一 ΔR 与权重
        self.hub_growth = {
            'hub2': hubs_cfg.get('hub2', { 'R0': 0, 'ΔR': 40, 'weight': 0.7 }),
            'hub3': hubs_cfg.get('hub3', { 'R0': 0, 'ΔR': 25, 'weight': 0.3 }),
        }
        placement = g.get('placement', {})
        self.monthly_quota = int(placement.get('monthly_quota', 20))
        self.use_ratio_mode = bool(placement.get('use_ratio_mode', True))
        self.max_new_per_hub = placement.get('max_new_per_hub', None)
        if isinstance(self.max_new_per_hub, (int, float)):
            try:
                self.max_new_per_hub = int(self.max_new_per_hub)
            except Exception:
                self.max_new_per_hub = None
        self.max_new_per_month = placement.get('max_new_per_month', None)
        if isinstance(self.max_new_per_month, (int, float)):
            try:
                self.max_new_per_month = int(self.max_new_per_month)
            except Exception:
                self.max_new_per_month = None
        qs = placement.get('quantiles', { 'q_high': 0.7, 'q_low': 0.3 })
        self.q_high = float(qs.get('q_high', 0.7))
        self.q_low = float(qs.get('q_low', 0.3))
        self.skip_middle_band = bool(placement.get('skip_middle_band', True))
        tr = placement.get('tail_ratio', { 'commercial': 0.3, 'residential': 0.7 })
        self.tail_share_com = float(tr.get('commercial', 0.5))
        self.tail_share_res = float(tr.get('residential', 0.5))
        locking = g.get('locking', {})
        self.lock_period = int(locking.get('lock_period', 6))
        self.hysteresis_delta = float(locking.get('hysteresis_delta', 0.0))
        # 非对称滞后（住→商 用 delta_up，商→住 用 delta_down）；未配置时退回 hysteresis_delta
        self.hysteresis_delta_up = float(locking.get('hysteresis_delta_up', self.hysteresis_delta))
        self.hysteresis_delta_down = float(locking.get('hysteresis_delta_down', self.hysteresis_delta))
        # 早期屏蔽：在该月之前不进行重判（与 lock_period 叠加）
        self.reclass_start_month = int(locking.get('reclass_start_month', self.lock_period))
        self.max_reclass_ratio = float(locking.get('max_reclass_ratio', 0.5))
        # 以“当月新建数”的比例限制可转换数量（例如 0.2 表示至多等于本月新增的 20%）
        self.max_reclass_of_new_ratio = float(locking.get('max_reclass_of_new_ratio', 0.5))
        self.reclassify_enabled = bool(locking.get('reclassify_enabled', True))
        grid_cfg = root.get('isocontour_layout', {}).get('slot_generator', {}).get('grid', {})
        self.grid_cell = int(grid_cfg.get('cell_size_px', 6))
        self.grid_origin = grid_cfg.get('origin', [0, 0])
        # 环带容差（像素）与空候选回退
        self.ring_tolerance_px = float(max(1.0, 0.75 * self.grid_cell))
        self.strict_ring_only = bool(g.get('strict_ring_only', False))


class CityV36:
    def __init__(self, config: Dict):
        self.cfg = V36Config(config)
        self.land = GaussianLandPriceSystem(config)
        self.finance = FinanceSystem()  # 财务评估系统
        self.current_month = 0
        self.state: Dict = {
            'residential': [],
            'commercial': [],
            'public': [],
            'industrial': [],
        }
        self.slots: List[Slot] = []
        self.range_cache: Dict[int, Dict] = {}
        # 延迟重判的待执行队列
        self.pending_reclassifications: List[Dict] = []
        # 重判调试信息
        self.reclass_debug: Dict[int, Dict] = {}
        # 本月新增缓存，供简化TXT导出
        self.new_buildings_by_month: Dict[int, List[Dict]] = {}
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def initialize(self):
        self.land.initialize_system(self.cfg.hubs, self.cfg.map_size)
        self._build_grid_slots()
        self._save_range_state(self.current_month)  # month 0
        # 初始化财务系统
        self.finance.initialize_building_finance(self.state)

    def _is_in_river_area(self, x: int, y: int) -> bool:
        """
        检查坐标是否在河流区域内
        """
        for river in self.cfg.rivers:
            if river.get('type') == 'obstacle':
                coordinates = river.get('coordinates', [])
                if coordinates:
                    # 检查点是否在多边形内
                    if point_in_polygon([float(x), float(y)], coordinates):
                        return True
        return False

    def _get_river_center_y(self) -> float:
        """
        获取河流中心线Y坐标
        """
        for river in self.cfg.rivers:
            if river.get('type') == 'obstacle':
                coordinates = river.get('coordinates', [])
                if coordinates:
                    y_coords = [coord[1] for coord in coordinates]
                    return (min(y_coords) + max(y_coords)) / 2
        return 0.0

    def _is_slot_on_same_side_as_hub(self, hub_idx: int, slot_x: int, slot_y: int) -> bool:
        """
        检查槽位是否与Hub在同一侧（河流的同一侧）
        """
        hub_x, hub_y = self.cfg.hubs[hub_idx]
        river_center_y = self._get_river_center_y()
        
        # 判断Hub和槽位是否在河流的同一侧
        hub_side = "north" if hub_y > river_center_y else "south"
        slot_side = "north" if slot_y > river_center_y else "south"
        
        return hub_side == slot_side

    def _build_grid_slots(self):
        w, h = int(self.cfg.map_size[0]), int(self.cfg.map_size[1])
        cell = max(1, int(self.cfg.grid_cell))
        self.slots.clear()
        # 读取 per-hub grid 配置
        slot_gen = self.cfg.root.get('isocontour_layout', {}).get('slot_generator', {}) if hasattr(self.cfg, 'root') else {}
        ph_cfg = slot_gen.get('per_hub_grid', {}) if isinstance(slot_gen, dict) else {}
        per_hub_enabled = bool(ph_cfg.get('enabled', True))
        limit_r = ph_cfg.get('limit_radius_px', None)
        disjoint = bool(ph_cfg.get('disjoint', True))
        hex_cfg = slot_gen.get('hex_grid', {}) if isinstance(slot_gen, dict) else {}
        hex_enabled = bool(hex_cfg.get('enabled', False))
        hex_size = float(hex_cfg.get('size_px', cell))
        hex_pointy = str(hex_cfg.get('orientation', 'pointy')).lower() == 'pointy'
        radial_cfg = slot_gen.get('radial', {}) if isinstance(slot_gen, dict) else {}
        radial_enabled = bool(radial_cfg.get('enabled', False))
        radial_dr = float(radial_cfg.get('delta_radius_px', cell))
        radial_dtheta = math.radians(float(radial_cfg.get('angle_step_deg', 15.0)))
        # per-hub pattern选择（可为每个Hub单独设定pattern与参数）
        per_hub_patterns = slot_gen.get('per_hub', []) if isinstance(slot_gen, dict) else []
        # 构建集合去重
        seen = set()
        def nearest_hub_idx(x: int, y: int) -> int:
            best_i, best_d = 0, 1e9
            for i, (hx, hy) in enumerate(self.cfg.hubs):
                d = ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i
        def add_slot_for_hub(gen_idx: int, x: int, y: int):
            if 0 <= x < w and 0 <= y < h:
                # 检查是否在河流区域内
                if self._is_in_river_area(x, y):
                    return
                if disjoint:
                    nh = nearest_hub_idx(x, y)
                    if nh != gen_idx:
                        return
                key = (x, y)
                if key not in seen:
                    seen.add(key)
                    self.slots.append(Slot(pos=[x, y]))
        if per_hub_enabled or hex_enabled or radial_enabled or per_hub_patterns:
            # 逐 Hub 生成使 Hub 恰好落在交点的格点：从 (hx,hy) 出发按 cell 扩展
            for idx, (hx, hy) in enumerate(self.cfg.hubs):
                # 半径限制（可选）
                max_r = float(limit_r) if isinstance(limit_r, (int, float)) and float(limit_r) > 0 else None
                # 选择本Hub的pattern
                patt_cfg = None
                if isinstance(per_hub_patterns, list) and idx < len(per_hub_patterns):
                    patt_cfg = per_hub_patterns[idx]
                pattern = None
                if isinstance(patt_cfg, dict) and 'pattern' in patt_cfg:
                    pattern = str(patt_cfg.get('pattern', 'grid')).lower()
                else:
                    # 回退：按全局开关决定
                    if per_hub_enabled:
                        pattern = 'grid'
                    elif hex_enabled:
                        pattern = 'hex'
                    elif radial_enabled:
                        pattern = 'radial'
                    else:
                        pattern = 'grid'
                if pattern == 'grid':
                    # 矩形格从hub对齐生长
                    y = int(hy)
                    while y >= 0 and (max_r is None or abs(y - hy) <= max_r):
                        x = int(hx)
                        while x < w and (max_r is None or ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5 <= max_r):
                            add_slot_for_hub(idx, x, y)
                            x += cell
                        x = int(hx) - cell
                        while x >= 0 and (max_r is None or ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5 <= max_r):
                            add_slot_for_hub(idx, x, y)
                            x -= cell
                        y -= cell
                    y = int(hy) + cell
                    while y < h and (max_r is None or abs(y - hy) <= max_r):
                        x = int(hx)
                        while x < w and (max_r is None or ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5 <= max_r):
                            add_slot_for_hub(idx, x, y)
                            x += cell
                        x = int(hx) - cell
                        while x >= 0 and (max_r is None or ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5 <= max_r):
                            add_slot_for_hub(idx, x, y)
                            x -= cell
                        y += cell
                elif pattern == 'hex':
                    # 六边形网（axial坐标）：pointy 与 flat 的步进不同
                    size = max(1.0, float((patt_cfg or {}).get('size_px', hex_size)))
                    hub_pointy = str((patt_cfg or {}).get('orientation', 'pointy')).lower() == 'pointy'
                    if hub_pointy:
                        # pointy-top axial to pixel
                        # 邻接步长
                        dq = [1, 1, 0, -1, -1, 0]
                        dr = [0, -1, -1, 0, 1, 1]
                        # 以hub为中心的环形展开
                        max_steps = int((max_r / size)) if max_r else max(w, h)
                        q0, r0 = 0, 0
                        def axial_to_pixel(q, r):
                            x = size * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
                            y = size * (3/2 * r)
                            return int(round(hx + x)), int(round(hy + y))
                        add_slot_for_hub(idx, int(hx), int(hy))
                        step = 1
                        while step <= max_steps:
                            q, r = q0 + (-step), r0 + (step)
                            for dir_idx in range(6):
                                for _ in range(step):
                                    px, py = axial_to_pixel(q, r)
                                    if max_r is None or ((px - hx)**2 + (py - hy)**2) ** 0.5 <= max_r:
                                        add_slot_for_hub(idx, px, py)
                                    q += dq[dir_idx]
                                    r += dr[dir_idx]
                            step += 1
                    else:
                        # flat-top axial
                        dq = [1, 0, -1, -1, 0, 1]
                        dr = [0, -1, -1, 0, 1, 1]
                        max_steps = int((max_r / size)) if max_r else max(w, h)
                        q0, r0 = 0, 0
                        def axial_to_pixel(q, r):
                            x = size * (3/2 * q)
                            y = size * (math.sqrt(3)/2 * q + math.sqrt(3) * r)
                            return int(round(hx + x)), int(round(hy + y))
                        add_slot_for_hub(idx, int(hx), int(hy))
                        step = 1
                        while step <= max_steps:
                            q, r = q0 + (-step), r0 + (step)
                            for dir_idx in range(6):
                                for _ in range(step):
                                    px, py = axial_to_pixel(q, r)
                                    if max_r is None or ((px - hx)**2 + (py - hy)**2) ** 0.5 <= max_r:
                                        add_slot_for_hub(idx, px, py)
                                    q += dq[dir_idx]
                                    r += dr[dir_idx]
                            step += 1
                elif pattern == 'radial':
                    # 同心环+等角射线
                    Rmax = max_r if max_r else max(w, h)
                    dr_local = float((patt_cfg or {}).get('delta_radius_px', radial_dr))
                    dtheta_local = math.radians(float((patt_cfg or {}).get('angle_step_deg', math.degrees(radial_dtheta))))
                    r = 0.0
                    while r <= Rmax:
                        theta = 0.0
                        while theta < 2 * math.pi:
                            px = int(round(hx + r * math.cos(theta)))
                            py = int(round(hy + r * math.sin(theta)))
                            add_slot_for_hub(idx, px, py)
                            theta += dtheta_local
                        r += dr_local
        else:
            # 旧的全局 origin 栅格（回退）
            ox, oy = 0, 0
            if hasattr(self.cfg, 'grid_origin'):
                ox, oy = int(self.cfg.grid_origin[0]), int(self.cfg.grid_origin[1])
            y = oy
            while y < h:
                x = ox
                while x < w:
                    # 回退模式下使用最近hub过滤可关（此处不过滤）
                    if 0 <= x < w and 0 <= y < h:
                        key = (x, y)
                        if key not in seen:
                            seen.add(key)
                            self.slots.append(Slot(pos=[x, y]))
                    x += cell
                y += cell

    def _R_for_hub(self, hub_key: str, month: int) -> float:
        # Pixel 模式：直接像素增量（1px/月默认）
        if self.cfg.use_pixel_growth:
            return float(self.cfg.pixel_R0_px) + float(self.cfg.pixel_delta_px) * month
        # 米制模式：R0 与 ΔR 以米计，需换算为像素
        g = self.cfg.hub_growth[hub_key]
        R0_m = float(g.get('R0', 0.0))
        dR_m = float(g.get('ΔR', 0.0))
        mpp = max(1e-6, self.cfg.meters_per_pixel)
        R0_px = R0_m / mpp
        dR_px = (dR_m * clamp(self.cfg.global_speed_scale, 0.0, 10.0)) / mpp
        return R0_px + dR_px * month

    def _min_dist_to_hubs(self, pos: List[int]) -> Tuple[float, int]:
        x, y = pos[0], pos[1]
        best_d, best_i = 1e9, -1
        for i, (hx, hy) in enumerate(self.cfg.hubs):
            d = ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5
            if d < best_d:
                best_d, best_i = d, i
        return best_d, best_i

    def _candidate_ring(self, month: int) -> List[Slot]:
        # 严格外扩：R(m-1) < d <= R(m)，对每个 hub 分别判断，取并集
        prev_month = max(0, month - 1)
        R_prev = [
            self._R_for_hub('hub2', prev_month),
            self._R_for_hub('hub3', prev_month),
        ]
        R_curr = [
            self._R_for_hub('hub2', month),
            self._R_for_hub('hub3', month),
        ]
        cand: List[Slot] = []
        for s in self.slots:
            if s.used or s.dead:
                continue
            d, idx = self._min_dist_to_hubs(s.pos)
            # 带容差环带：R_prev - tol < d <= R_curr + tol
            if d <= (R_curr[idx] + self.cfg.ring_tolerance_px) and d > (R_prev[idx] - self.cfg.ring_tolerance_px):
                # 添加侧边约束：只选择与Hub在同一侧的槽位
                if self._is_slot_on_same_side_as_hub(idx, s.pos[0], s.pos[1]):
                    cand.append(s)
        # 若严格环带为空且允许回退，则回退为"范围内填充"（≤ R_curr）
        if not cand and not self.cfg.strict_ring_only:
            for s in self.slots:
                if s.used or s.dead:
                    continue
                d, idx = self._min_dist_to_hubs(s.pos)
                if d <= R_curr[idx]:
                    # 添加侧边约束：只选择与Hub在同一侧的槽位
                    if self._is_slot_on_same_side_as_hub(idx, s.pos[0], s.pos[1]):
                        cand.append(s)
        return cand

    def _place_month(self, month: int):
        # 计算候选与 LP 阈值
        cand = self._candidate_ring(month)
        lp_vals = np.array([float(self.land.get_land_price(s.pos)) for s in cand]) if cand else np.array([])
        P_low = float(np.quantile(lp_vals, self.cfg.q_low)) if lp_vals.size > 0 else 0.0
        P_high = float(np.quantile(lp_vals, self.cfg.q_high)) if lp_vals.size > 0 else 0.0
        # 阈值退化保护：若区间过窄，使用均衡回退（按 LP 低端/高端各取目标数）
        eps = 1e-6
        use_fallback = (lp_vals.size == 0) or (abs(P_high - P_low) < eps)
        # 按最近 Hub 分组计算分位与配额，并独立挑选
        hubs = self.cfg.hubs
        num_hubs = len(hubs)
        # 权重（仅在配额模式使用；比例模式下仅用于可选的全局上限分摊）
        def hub_weight(idx: int) -> float:
            key = f'hub{idx+1}'
            return float(self.cfg.hub_growth.get(key, {}).get('weight', 1.0))
        weights = [hub_weight(i) for i in range(num_hubs)]
        total_w = sum(weights) if sum(weights) > 0 else 1.0
        weights = [w/total_w for w in weights]
        q_total = int(max(0, self.cfg.monthly_quota))
        per_hub_q = [0] * num_hubs
        if not self.cfg.use_ratio_mode:
            per_hub_q = [int(q_total * w) for w in weights]
            rem_q = q_total - sum(per_hub_q)
            if rem_q > 0:
                order = sorted(range(num_hubs), key=lambda i: weights[i], reverse=True)
                for i in order:
                    if rem_q <= 0:
                        break
                    per_hub_q[i] += 1
                    rem_q -= 1
        # 按最近Hub分组候选
        def nearest_idx(s: Slot) -> int:
            x, y = s.pos
            best_i, best_d = 0, 1e9
            for i, (hx, hy) in enumerate(hubs):
                d = ((x - hx)**2 + (y - hy)**2)**0.5
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i
        group: Dict[int, List[Slot]] = {i: [] for i in range(num_hubs)}
        for s in cand:
            group[nearest_idx(s)].append(s)
        picked_pairs: List[Tuple[str, Slot]] = []
        total_picked_counter = 0
        for i in range(num_hubs):
            gi = group.get(i, [])
            if not gi:
                continue
            lp_vals_i = np.array([float(self.land.get_land_price(s.pos)) for s in gi])
            P_low_i = float(np.quantile(lp_vals_i, self.cfg.q_low)) if lp_vals_i.size > 0 else 0.0
            P_high_i = float(np.quantile(lp_vals_i, self.cfg.q_high)) if lp_vals_i.size > 0 else 0.0
            eps = 1e-6
            fallback_i = (lp_vals_i.size == 0) or (abs(P_high_i - P_low_i) < eps)
            # 组内列表
            hi: List[Tuple[float, Slot]] = []
            lo: List[Tuple[float, Slot]] = []
            md: List[Tuple[float, Slot]] = []
            for s in gi:
                lp = float(self.land.get_land_price(s.pos))
                if not fallback_i and lp > P_high_i:
                    hi.append((lp, s))
                elif not fallback_i and lp < P_low_i:
                    lo.append((lp, s))
                else:
                    md.append((lp, s))
            hi.sort(key=lambda t: -t[0])
            lo.sort(key=lambda t: t[0])
            if self.cfg.use_ratio_mode:
                # 比例模式：按组内候选数决定数量
                target_total_i = len(gi)
                # 若存在每Hub上限，缩放到上限
                cap_i = None
                if isinstance(self.cfg.max_new_per_hub, int) and self.cfg.max_new_per_hub >= 0:
                    cap_i = int(self.cfg.max_new_per_hub)
                    target_total_i = min(target_total_i, cap_i)
                k_com_i = int(round(target_total_i * self.cfg.tail_share_com))
                k_res_i = max(0, target_total_i - k_com_i)
            else:
                qi = per_hub_q[i]
                if qi <= 0:
                    continue
                k_com_i = int(round(qi * self.cfg.tail_share_com))
                k_res_i = max(0, qi - k_com_i)
            if fallback_i:
                all_scored_i = [(float(self.land.get_land_price(s.pos)), s) for s in gi]
                all_scored_i.sort(key=lambda t: t[0])
                picked_res_i = [s for _, s in all_scored_i[:k_res_i]]
                remain_i = [s for s in gi if s not in picked_res_i]
                remain_high_i = [(float(self.land.get_land_price(s.pos)), s) for s in remain_i]
                remain_high_i.sort(key=lambda t: -t[0])
                picked_com_i = [s for _, s in remain_high_i[:k_com_i]]
            else:
                picked_com_i = [s for _, s in hi[:k_com_i]]
                picked_res_i = [s for _, s in lo[:k_res_i]]
                rem_i = (k_com_i + k_res_i) - (len(picked_com_i) + len(picked_res_i))
                if rem_i > 0:
                    need_res_i = max(0, k_res_i - len(picked_res_i))
                    if need_res_i > 0 and len(lo) > len(picked_res_i):
                        cap = min(need_res_i, rem_i, len(lo) - len(picked_res_i))
                        if cap > 0:
                            extra = [s for _, s in lo[len(picked_res_i): len(picked_res_i)+cap]]
                            picked_res_i.extend(extra)
                            rem_i -= len(extra)
                    need_com_i = max(0, k_com_i - len(picked_com_i))
                    if rem_i > 0 and need_com_i > 0 and len(hi) > len(picked_com_i):
                        cap = min(need_com_i, rem_i, len(hi) - len(picked_com_i))
                        if cap > 0:
                            extra = [s for _, s in hi[len(picked_com_i): len(picked_com_i)+cap]]
                            picked_com_i.extend(extra)
                            rem_i -= len(extra)
                # 中带补：优先住宅到底线
                allow_mid_i = (not self.cfg.skip_middle_band) or (k_res_i > len(picked_res_i))
                if rem_i > 0 and allow_mid_i and len(md) > 0:
                    need_res_i = max(0, k_res_i - len(picked_res_i))
                    if need_res_i > 0:
                        md.sort(key=lambda t: t[0])
                        cap = min(need_res_i, rem_i, len(md))
                        extra = [s for _, s in md[:cap]]
                        picked_res_i.extend(extra)
                        md = md[cap:]
                        rem_i -= len(extra)
                    need_com_i = max(0, k_com_i - len(picked_com_i))
                    if rem_i > 0 and need_com_i > 0 and len(md) > 0:
                        md.sort(key=lambda t: -t[0])
                        cap = min(need_com_i, rem_i, len(md))
                        extra = [s for _, s in md[:cap]]
                        picked_com_i.extend(extra)
                        md = md[cap:]
                        rem_i -= len(extra)
            picked_pairs.extend([('residential', s) for s in picked_res_i])
            picked_pairs.extend([('commercial', s) for s in picked_com_i])
            total_picked_counter += len(picked_res_i) + len(picked_com_i)
            # 全局软上限：如配置则截断
            if isinstance(self.cfg.max_new_per_month, int) and self.cfg.max_new_per_month >= 0:
                if total_picked_counter >= self.cfg.max_new_per_month:
                    break
        # 合并最终选择（先住宅后商业）
        picked = [('residential', s) for t, s in picked_pairs if t == 'residential'] + [('commercial', s) for t, s in picked_pairs if t == 'commercial']
        # 分类并落地
        new_count = 0
        new_buildings_details = []
        for btype, s in picked:
            lp = float(self.land.get_land_price(s.pos))
            b = {
                'id': f"{btype[:3]}_{len(self.state[btype]) + 1}",
                'type': btype,
                'xy': s.pos,
                'land_price_value': lp,
                'last_changed_month': month,
            }
            self.state[btype].append(b)
            s.used = True
            s.building_id = b['id']
            new_count += 1
            new_buildings_details.append({
                'id': b['id'],
                'type': btype,
                'position': b['xy'],
                'land_price_value': lp
            })
        # 重判（延迟执行）：生成本月候选并排队到 month+lock_period 执行
        reclassified = []
        if self.cfg.reclassify_enabled:
            scheduled_month = month + int(self.cfg.lock_period)
            self._schedule_reclassification(scheduled_month, P_low, P_high)
        # 保存 range_state 与增量
        self._save_range_state(month, len(cand), P_low, P_high)
        self._save_delta(month, new_buildings_details, reclassified)
        self._save_audit(month, len(cand), P_low, P_high, new_count)
        # 记录本月新增供简化TXT
        self.new_buildings_by_month[month] = list(new_buildings_details)

    def _reclassify(self, month: int, P_low: float, P_high: float, new_count_this_month: int = 0):
        # 早期屏蔽：小于启用月不重判
        if month < int(self.cfg.reclass_start_month):
            return []
        delta_up = float(self.cfg.hysteresis_delta_up)
        delta_down = float(self.cfg.hysteresis_delta_down)
        max_ratio = clamp(self.cfg.max_reclass_ratio, 0.0, 1.0)
        # 统计可重判的集合
        candidates: List[Tuple[str, Dict]] = []
        for bt in ['commercial', 'residential']:
            for b in self.state.get(bt, []):
                last = int(b.get('last_changed_month', 0))
                if (month - last) >= self.cfg.lock_period:
                    candidates.append((bt, b))
        if not candidates:
            return []
        # 允许的最大重判数量
        cap = int(max(0, round(len(candidates) * max_ratio))) if max_ratio > 0 else len(candidates)
        # 叠加“当月新建数”的比例上限
        of_new = int(max(0, round(float(new_count_this_month) * float(self.cfg.max_reclass_of_new_ratio))))
        if of_new > 0:
            cap = min(cap, of_new)
        # 分方向候选并排序：住→商按 LP 降序，商→住按 LP 升序
        up_list = []    # residential -> commercial
        down_list = []  # commercial -> residential
        for bt, b in candidates:
            lp = float(self.land.get_land_price(b['xy']))
            if bt == 'residential' and lp >= (P_high + delta_up):
                up_list.append((lp, bt, b))
            elif bt == 'commercial' and lp <= (P_low - delta_down):
                down_list.append((lp, bt, b))
        up_list.sort(key=lambda t: -t[0])
        down_list.sort(key=lambda t: t[0])
        changed = 0
        changes = []
        # 先执行商→住（有利于提高住宅占比），再执行住→商，直到 cap
        for lp, bt, b in down_list:
            if changed >= cap:
                break
            b['type'] = 'residential'
            b['last_changed_month'] = month
            self.state['residential'].append(b)
            self.state['commercial'] = [x for x in self.state['commercial'] if x['id'] != b['id']]
            changed += 1
            changes.append({'id': b['id'], 'from_type': 'commercial', 'to_type': 'residential', 'position': b['xy'], 'land_price_value': lp})
        for lp, bt, b in up_list:
            if changed >= cap:
                break
            b['type'] = 'commercial'
            b['last_changed_month'] = month
            self.state['commercial'].append(b)
            self.state['residential'] = [x for x in self.state['residential'] if x['id'] != b['id']]
            changed += 1
            changes.append({'id': b['id'], 'from_type': 'residential', 'to_type': 'commercial', 'position': b['xy'], 'land_price_value': lp})
        return changes

    # ---------- Delayed reclassification (scheduled) ----------
    def _schedule_reclassification(self, execute_month: int, P_low: float, P_high: float):
        self.pending_reclassifications.append({
            'execute_month': int(execute_month),
            'P_low': float(P_low),
            'P_high': float(P_high)
        })

    def _execute_scheduled_reclassification(self, month: int):
        # 聚合所有到期任务
        due = [task for task in self.pending_reclassifications if int(task['execute_month']) == int(month)]
        if not due:
            return
        # 以“已建建筑”的分布（可改为分Hub）计算更稳阈值；目前按到期任务平均P值进一步平滑
        all_lps = [float(self.land.get_land_price(b['xy'])) for bt in ['commercial','residential'] for b in self.state.get(bt, [])]
        if not all_lps:
            # 清理到期任务
            self.pending_reclassifications = [t for t in self.pending_reclassifications if int(t['execute_month']) != int(month)]
            return
        arr = np.array(all_lps)
        # 平滑：用到期任务平均 P_low/P_high 与“已建分布分位数”的均值
        avg_P_low = float(np.mean([t['P_low'] for t in due]))
        avg_P_high = float(np.mean([t['P_high'] for t in due]))
        built_P_low = float(np.quantile(arr, self.cfg.q_low))
        built_P_high = float(np.quantile(arr, self.cfg.q_high))
        P_low_eff = 0.5 * avg_P_low + 0.5 * built_P_low
        P_high_eff = 0.5 * avg_P_high + 0.5 * built_P_high
        # 执行重判（不传 new_count，比例上限仍然生效）
        changes = self._reclassify(month, P_low_eff, P_high_eff, 0)
        # 保存调试信息
        dbg = {
            'execute_month': int(month),
            'scheduled_tasks': len(due),
            'P_low_eff': float(P_low_eff),
            'P_high_eff': float(P_high_eff),
            'changes': changes
        }
        self.reclass_debug[month] = dbg
        try:
            with open(os.path.join(self.cfg.output_dir, f'debug_reclass_month_{month:02d}.json'), 'w', encoding='utf-8') as f:
                json.dump(dbg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # 输出仅含重判点位的简化TXT（逐月份）
        try:
            self._save_reclass_simplified_txt(month, changes)
        except Exception:
            pass
        # 清理到期任务
        self.pending_reclassifications = [t for t in self.pending_reclassifications if int(t['execute_month']) != int(month)]
        
    # ----- Simplified TXT export -----
    def _save_simplified_txt(self, month: int):
        simp_dir = os.path.join(self.cfg.output_dir, 'simplified')
        os.makedirs(simp_dir, exist_ok=True)
        type_map = {'residential': 0, 'commercial': 1, 'industrial': 2, 'public': 3}
        def fmt_entry(t: str, pos: List[int]) -> str:
            mid = type_map.get(str(t).lower(), 4)
            x = float(pos[0]) if len(pos) > 0 else 0.0
            y = float(pos[1]) if len(pos) > 1 else 0.0
            z = 0.0
            return f"{mid}({x:.3f}, {y:.3f}, {z:.0f})"
        if month == 0:
            entries = []
            for bt in ['residential', 'commercial', 'public', 'industrial']:
                for b in self.state.get(bt, []):
                    entries.append(fmt_entry(b.get('type', bt), b.get('xy', [0, 0])))
            line = ", ".join(entries)
            with open(os.path.join(simp_dir, f'simplified_buildings_{month:02d}.txt'), 'w', encoding='utf-8') as f:
                f.write(line)
            return
        # 增量：仅当月新增（不含重判）
        entries = []
        for nb in self.new_buildings_by_month.get(month, []):
            entries.append(fmt_entry(nb.get('type', 'other'), nb.get('position', [0, 0])))
        line = ", ".join(entries)
        with open(os.path.join(simp_dir, f'simplified_buildings_{month:02d}.txt'), 'w', encoding='utf-8') as f:
            f.write(line)

    def _save_reclass_simplified_txt(self, month: int, changes: List[Dict]):
        """仅输出当月重判点位（to_type, position），简化TXT格式。"""
        simp_dir = os.path.join(self.cfg.output_dir, 'simplified')
        os.makedirs(simp_dir, exist_ok=True)
        type_map = {'residential': 0, 'commercial': 1, 'industrial': 2, 'public': 3}
        def fmt_entry(t: str, pos: List[int]) -> str:
            mid = type_map.get(str(t).lower(), 4)
            x = float(pos[0]) if len(pos) > 0 else 0.0
            y = float(pos[1]) if len(pos) > 1 else 0.0
            z = 0.0
            return f"{mid}({x:.3f}, {y:.3f}, {z:.0f})"
        entries = []
        for rc in changes or []:
            entries.append(fmt_entry(rc.get('to_type', 'other'), rc.get('position', [0, 0])))
        line = ", ".join(entries)
        out_path = os.path.join(simp_dir, f'simplified_reclass_{month:02d}.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(line)

    def run(self, total_months: int):
        for m in range(total_months):
            self.current_month = m
            # 更新地价场
            self.land.update_land_price_field(m, self.state)
            # 月度放置
            self._place_month(m)
            # 执行到期的重判（用"已建建筑"分布计算阈值更稳）
            if self.cfg.reclassify_enabled:
                self._execute_scheduled_reclassification(m)
            # 财务评估计算
            try:
                self._update_finance_system(m)
            except Exception as e:
                print(f"财务系统更新失败: {e}")
            # 简化TXT导出（baseline+增量；增量含当月新增与重判）
            try:
                self._save_simplified_txt(m)
            except Exception:
                pass

    # ----- Outputs -----
    def _save_range_state(self, month: int, candidates: Optional[int] = None, P_low: Optional[float] = None, P_high: Optional[float] = None):
        prev_month = max(0, month - 1)
        R_prev = {
            'hub2': self._R_for_hub('hub2', prev_month),
            'hub3': self._R_for_hub('hub3', prev_month),
        }
        R_curr = {
            'hub2': self._R_for_hub('hub2', month),
            'hub3': self._R_for_hub('hub3', month),
        }
        obj = {
            'month': month,
            'R_prev': R_prev,
            'R_curr': R_curr,
            'candidates': int(candidates) if candidates is not None else None,
            'P_low': float(P_low) if P_low is not None else None,
            'P_high': float(P_high) if P_high is not None else None,
        }
        with open(os.path.join(self.cfg.output_dir, f'range_state_month_{month:02d}.json'), 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _post_process_building_types(self, buildings: List[Dict], month: int) -> List[Dict]:
        """后处理建筑类型，实现 Hub2 工业中心效果"""
        # Hub2 工业中心配置
        hub2_position = [90, 55]  # Hub2 位置
        hub2_radius = 30  # 影响半径
        
        processed_buildings = []
        for building in buildings:
            # 创建建筑副本
            processed_building = building.copy()
            
            # 检查是否在 Hub2 工业中心附近
            if building['type'] == 'commercial':
                x, y = building['position']
                distance = ((x - hub2_position[0])**2 + (y - hub2_position[1])**2)**0.5
                
                if distance <= hub2_radius:
                    # 转换为工业建筑类型
                    processed_building['type'] = 'industrial'
                    processed_building['original_type'] = 'commercial'
                    processed_building['hub_influence'] = 'hub2_industrial_zone'
                    processed_building['conversion_reason'] = f'Hub2工业中心影响 (距离: {distance:.1f})'
            
            processed_buildings.append(processed_building)
        
        return processed_buildings

    def _apply_industrial_conversion(self):
        """应用工业转换到建筑状态"""
        # Hub2 工业中心配置
        hub2_position = [90, 55]  # Hub2 位置
        hub2_radius = 30  # 影响半径
        
        # 处理商业建筑
        commercial_buildings = self.state.get('commercial', [])
        buildings_to_move = []
        
        for building in commercial_buildings:
            x, y = building['xy']
            distance = ((x - hub2_position[0])**2 + (y - hub2_position[1])**2)**0.5
            
            if distance <= hub2_radius:
                # 转换为工业建筑
                building['type'] = 'industrial'
                building['original_type'] = 'commercial'
                building['hub_influence'] = 'hub2_industrial_zone'
                building['conversion_reason'] = f'Hub2工业中心影响 (距离: {distance:.1f})'
                
                # 更新建筑属性
                building['capacity'] = 1200  # 工业建筑容量更大
                building['construction_cost'] = 1800  # 工业建筑建造成本更高
                building['revenue_per_person'] = 15  # 工业建筑收入更高
                
                # 标记需要移动到工业列表
                buildings_to_move.append(building)
        
        # 将转换的建筑移动到工业列表
        for building in buildings_to_move:
            self.state['commercial'].remove(building)
            if 'industrial' not in self.state:
                self.state['industrial'] = []
            self.state['industrial'].append(building)

    def _save_delta(self, month: int, new_buildings: List[Dict], reclassified: List[Dict]):
        # month 01：保存完整集；其他月度保存新增与重判
        if month == 1:
            all_buildings = []
            for bt in ['residential', 'commercial', 'public', 'industrial']:
                for b in self.state.get(bt, []):
                    all_buildings.append({
                        'id': b['id'], 'type': b['type'], 'position': b['xy'],
                        'land_price_value': b.get('land_price_value', 0.0)
                    })
            # 后处理：Hub2 工业中心建筑类型转换
            all_buildings = self._post_process_building_types(all_buildings, month)
            with open(os.path.join(self.cfg.output_dir, f'building_positions_month_{month:02d}.json'), 'w', encoding='utf-8') as f:
                json.dump({ 'buildings': all_buildings, 'timestamp': f'month_{month:02d}' }, f, ensure_ascii=False, indent=2)
        else:
            # 后处理：Hub2 工业中心建筑类型转换
            processed_new_buildings = self._post_process_building_types(new_buildings, month)
            processed_reclassified = self._post_process_building_types(reclassified, month)
            
            delta = {
                'month': month,
                'timestamp': f'month_{month:02d}',
                'new_buildings': processed_new_buildings,
                'reclassified': processed_reclassified
            }
            with open(os.path.join(self.cfg.output_dir, f'building_delta_month_{month:02d}.json'), 'w', encoding='utf-8') as f:
                json.dump(delta, f, ensure_ascii=False, indent=2)

    def _save_audit(self, month: int, candidates: int, P_low: float, P_high: float, new_count: int):
        path = os.path.join(self.cfg.output_dir, 'audits.csv')
        header = ['month', 'candidates', 'P_low', 'P_high', 'new_count', 'res_total', 'com_total']
        write_header = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow([
                month, int(candidates or 0), float(P_low or 0), float(P_high or 0), int(new_count or 0),
                len(self.state['residential']), len(self.state['commercial'])
            ])

    def _update_finance_system(self, month: int):
        """更新财务系统"""
        # 应用工业转换到建筑状态
        self._apply_industrial_conversion()
        
        # 重新初始化建筑财务数据（包含新增建筑和工业转换）
        self.finance.initialize_building_finance(self.state)
        
        # 获取地价场和热力场
        land_price_field = self.land.land_price_field
        heat_field = None  # 暂时使用None，后续可以集成热力系统
        
        # 计算月度财务数据
        monthly_finance = self.finance.calculate_monthly_finance(month, land_price_field, heat_field)
        
        # 保存建筑财务CSV
        self.finance.save_building_finance_csv(self.cfg.output_dir, month)
        
        # 每季度保存汇总数据
        if (month + 1) % 3 == 0:
            quarter = (month + 1) // 3
            self.finance.save_finance_dashboard(self.cfg.output_dir, quarter)
            self.finance.save_kpi_summary_csv(self.cfg.output_dir, quarter)


def main():
    # 读取 v3.5 的城市配置作为基础（包含地图、hubs、地价系统参数）
    try:
        with open('configs/city_config_v3_5.json', 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    sim = CityV36(cfg)
    sim.initialize()
    total_months = cfg.get('simulation', {}).get('total_months', 36)
    sim.run(total_months)
    print(f"[v3.6] done. outputs at {sim.cfg.output_dir}")


if __name__ == '__main__':
    main()


