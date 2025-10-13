"""
向量场槽位生成脚本（基于 PRD 4. Vector Field Slot Placement Module）

功能：
- 读取现有槽位点集（slotpoints.txt，浮点坐标）
- 读取 Hub 点（优先 growth_v4_0.hubs.list；否则 city.transport_hubs）
- 读取道路/曲线（优先 terrain_features.rivers[].coordinates；否则 river.txt）
- 计算向量场：F = F_hub + F_road
  - F_hub(s) = Σ w_hub * f(||s-h||) * (h-s)/||h-s||
  - F_road(s) = w_road * f(dist(s, q)) * t_hat（q 为曲线上 s 最近点，t_hat 为切向单位向量）
- 生成新槽位 s' = s + α F(s)，输出新点为 (s + s')/2
- 去重与边界处理；输出 JSON 与 TXT

使用：
  python -m scripts.generate_vector_field_slots \
      --config configs/city_config_v4_0.json \
      --input slotpoints.txt \
      --output_dir enhanced_simulation_v4_0_output/v4_vector_field \
      --w_hub 2.0 --w_road 1.2 --sigma 1.5 --d_cut 3.0 --alpha 0.5
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# 复用 v4 槽位结构（可选）
try:
    from logic.v4_enumeration import SlotNode, _auto_fill_neighbors_4n  # noqa: F401
except Exception:
    @dataclass
    class SlotNode:  # 兜底，若不可导入
        slot_id: str
        x: int
        y: int
        fx: Optional[float] = None
        fy: Optional[float] = None


def read_config(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def load_points_from_txt(points_file: str, map_size: List[int]) -> List[Tuple[float, float]]:
    if not os.path.exists(points_file):
        raise FileNotFoundError(f"points_file not found: {points_file}")
    W, H = int(map_size[0]), int(map_size[1])
    pts: List[Tuple[float, float]] = []
    seen = set()
    with open(points_file, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) < 2:
                continue
            try:
                xf = float(nums[0]); yf = float(nums[1])
            except Exception:
                continue
            if xf < 0.0 or yf < 0.0 or xf >= float(W) or yf >= float(H):
                continue
            key = (xf, yf)
            if key in seen:
                continue
            seen.add(key)
            pts.append((xf, yf))
    return pts


def load_polylines_from_config(cfg: Dict) -> List[List[Tuple[float, float]]]:
    polylines: List[List[Tuple[float, float]]] = []
    # 优先：terrain_features.rivers[].coordinates
    rivers = cfg.get('growth_v4_0', {}).get('terrain_features', {}).get('rivers', [])
    if not rivers:
        rivers = cfg.get('terrain_features', {}).get('rivers', [])
    for r in rivers or []:
        coords = r.get('coordinates', []) or []
        if isinstance(coords, list) and len(coords) >= 2 and isinstance(coords[0], list):
            poly = []
            for p in coords:
                if isinstance(p, list) and len(p) >= 2:
                    try:
                        poly.append((float(p[0]), float(p[1])))
                    except Exception:
                        pass
            if len(poly) >= 2:
                polylines.append(poly)
    # 兜底：river.txt（每行两个数）
    if not polylines and os.path.exists('river.txt'):
        pts: List[Tuple[float, float]] = []
        try:
            with open('river.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    nums = re.findall(r"-?\d+(?:\.\d+)?", line)
                    if len(nums) >= 2:
                        pts.append((float(nums[0]), float(nums[1])))
        except Exception:
            pts = []
        if len(pts) >= 2:
            polylines.append(pts)
    return polylines


def load_polylines_from_txt(path: str) -> List[List[Tuple[float, float]]]:
    """解析自定义道路文件：允许包含多个方括号 [] 区块，每个区块为一条折线。
    行内可有 2 或 3 个数字（取前两个为 x,y）。"""
    polys: List[List[Tuple[float, float]]] = []
    if not isinstance(path, str) or not os.path.exists(path):
        return polys
    current: List[Tuple[float, float]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith('[') and s.endswith(']') and len(s) == 1:
                # 形如 "[" 独立一行（兼容）→ 开始新折线
                if current:
                    if len(current) >= 2:
                        polys.append(current)
                    current = []
                continue
            if s == '[':
                if current:
                    if len(current) >= 2:
                        polys.append(current)
                    current = []
                continue
            if s == ']':
                if current:
                    if len(current) >= 2:
                        polys.append(current)
                    current = []
                continue
            # 普通数据行：提取前两个数字
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) >= 2:
                try:
                    x = float(nums[0]); y = float(nums[1])
                    current.append((x, y))
                except Exception:
                    pass
    if current and len(current) >= 2:
        polys.append(current)
    return polys


def hubs_from_config(cfg: Dict) -> List[Tuple[float, float, float]]:
    """返回 (x,y,weight)，weight 默认 1.0。优先 growth_v4_0.hubs.list.weight。"""
    hubs_cfg = cfg.get('growth_v4_0', {}).get('hubs', {})
    hubs: List[Tuple[float, float, float]] = []
    if str(hubs_cfg.get('mode', 'explicit')) == 'explicit':
        for h in hubs_cfg.get('list', []) or []:
            try:
                x = float(h.get('x'))
                y = float(h.get('y'))
                w = float(h.get('weight', 1.0))
                hubs.append((x, y, w))
            except Exception:
                continue
    if not hubs:
        # 回退 city.transport_hubs（无权重）
        for p in cfg.get('city', {}).get('transport_hubs', []) or []:
            try:
                hubs.append((float(p[0]), float(p[1]), 1.0))
            except Exception:
                continue
    return hubs


def gaussian_decay(d: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 0.0
    return math.exp(- (d * d) / (2.0 * sigma * sigma))


def inverse_square_decay(d: float, sigma: float) -> float:
    # sigma 作为软半径，避免 0
    s = max(1e-6, sigma)
    return 1.0 / (1.0 + (d / s) * (d / s))


def nearest_point_and_tangent_on_polylines(
    x: float, y: float, polylines: List[List[Tuple[float, float]]]
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], float]:
    """
    返回 (q_x, q_y), (t_hat_x, t_hat_y), dist。若无曲线返回 (None,None,inf)。
    """
    best_d2 = float('inf')
    best_q: Optional[Tuple[float, float]] = None
    best_t: Optional[Tuple[float, float]] = None
    sx, sy = float(x), float(y)
    for poly in polylines or []:
        for i in range(len(poly) - 1):
            x0, y0 = poly[i]
            x1, y1 = poly[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            seg_len2 = dx * dx + dy * dy
            if seg_len2 <= 1e-12:
                continue
            # 投影参数 t∈[0,1]
            t = ((sx - x0) * dx + (sy - y0) * dy) / seg_len2
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            qx = x0 + t * dx
            qy = y0 + t * dy
            ddx = sx - qx
            ddy = sy - qy
            d2 = ddx * ddx + ddy * ddy
            if d2 < best_d2:
                best_d2 = d2
                best_q = (qx, qy)
                seg_len = math.sqrt(seg_len2)
                best_t = (dx / seg_len, dy / seg_len)
    if best_q is None:
        return None, None, float('inf')
    return best_q, best_t, math.sqrt(best_d2)


def generate_new_points(
    points: List[Tuple[float, float]],
    hubs: List[Tuple[float, float, float]],
    polylines: List[List[Tuple[float, float]]],
    map_size: List[int],
    w_hub: float = 2.0,
    w_road: float = 1.2,
    sigma: float = 1.5,
    d_cut: float = 3.0,
    alpha: float = 0.5,
    decay: str = 'gaussian',
    min_separation_px: float = 1.0,
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, float, float, float]]]:
    """
    基于向量场生成新点列表（与原 points 去重后）。
    返回：
      - new_points: 新增点集合
      - pairs: 列表 (orig_index, dx, dy, dist)
    """
    W, H = int(map_size[0]), int(map_size[1])
    if decay.lower().startswith('gauss'):
        f_decay = lambda d: gaussian_decay(d, sigma)
    else:
        f_decay = lambda d: inverse_square_decay(d, sigma)

    existing = list(points)
    new_points: List[Tuple[float, float]] = []
    pairs: List[Tuple[int, float, float, float]] = []

    def too_close_to_any(p: Tuple[float, float], arr: List[Tuple[float, float]]) -> bool:
        px, py = p
        thr = float(min_separation_px)
        thr2 = thr * thr
        for (ax, ay) in arr:
            dx = px - ax; dy = py - ay
            if dx * dx + dy * dy <= thr2:
                return True
        return False

    for idx, (x, y) in enumerate(existing):
        # F_hub
        Fx = 0.0
        Fy = 0.0
        for (hx, hy, hw) in hubs:
            dx = hx - x
            dy = hy - y
            dist = math.hypot(dx, dy)
            if dist <= 1e-9:
                continue
            dh = f_decay(dist)
            Fx += float(w_hub) * float(hw) * dh * (dx / dist)
            Fy += float(w_hub) * float(hw) * dh * (dy / dist)

        # F_road
        if polylines:
            q, t_hat, dist_r = nearest_point_and_tangent_on_polylines(x, y, polylines)
            if q is not None and t_hat is not None and dist_r <= float(d_cut):
                fr = f_decay(dist_r)
                Fx += float(w_road) * fr * float(t_hat[0])
                Fy += float(w_road) * fr * float(t_hat[1])

        # step & midpoint
        x1 = x + float(alpha) * Fx
        y1 = y + float(alpha) * Fy
        xn = 0.5 * (x + x1)
        yn = 0.5 * (y + y1)

        # 边界：裁剪到边界内
        xn = 0.0 if xn < 0.0 else float(W - 1) if xn > float(W - 1) else xn
        yn = 0.0 if yn < 0.0 else float(H - 1) if yn > float(H - 1) else yn

        # 去重：与原集合/新增集合
        if too_close_to_any((xn, yn), existing):
            # 即便被判定过近，不追加 new_points，但仍记录位移向量（近似 0）
            pairs.append((idx, 0.0, 0.0, 0.0))
            continue
        if too_close_to_any((xn, yn), new_points):
            pairs.append((idx, 0.0, 0.0, 0.0))
            continue
        new_points.append((xn, yn))
        dx = xn - x
        dy = yn - y
        dist = math.hypot(dx, dy)
        pairs.append((idx, dx, dy, dist))

    return new_points, pairs


def save_outputs(
    original: List[Tuple[float, float]],
    new_points: List[Tuple[float, float]],
    pairs: List[Tuple[int, float, float, float]],
    output_dir: str,
    write_combined_txt: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 统计
    dists = [p[3] for p in pairs if len(p) >= 4]
    dists_sorted = sorted(dists)
    def _pct(p: float) -> float:
        if not dists_sorted:
            return 0.0
        k = max(0, min(len(dists_sorted) - 1, int(round(p * (len(dists_sorted) - 1)))))
        return float(dists_sorted[k])

    out_json = {
        'original_count': len(original),
        'new_count': len(new_points),
        'original': [{'x': float(x), 'y': float(y)} for (x, y) in original],
        'new': [{'x': float(x), 'y': float(y)} for (x, y) in new_points],
        'pairs': [
            {'orig_index': int(i), 'dx': float(dx), 'dy': float(dy), 'dist': float(dist)}
            for (i, dx, dy, dist) in pairs
        ],
        'stats': {
            'mean': float(sum(dists) / len(dists)) if dists else 0.0,
            'median': float(dists_sorted[len(dists_sorted)//2]) if dists_sorted else 0.0,
            'max': float(max(dists)) if dists else 0.0,
            'p90': _pct(0.90),
            'p95': _pct(0.95),
            'p99': _pct(0.99)
        }
    }
    with open(os.path.join(output_dir, 'vf_slots.json'), 'w', encoding='utf-8') as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    # CSV（位移向量）
    with open(os.path.join(output_dir, 'displacements.csv'), 'w', encoding='utf-8') as f:
        f.write('orig_index,orig_x,orig_y,new_x,new_y,dx,dy,dist\n')
        for (i, dx, dy, dist) in pairs:
            if i < 0 or i >= len(original):
                continue
            ox, oy = original[i]
            nx = ox + dx
            ny = oy + dy
            f.write(f"{i},{ox:.6f},{oy:.6f},{nx:.6f},{ny:.6f},{dx:.6f},{dy:.6f},{dist:.6f}\n")

    # 仅新增点 TXT
    with open(os.path.join(output_dir, 'slotpoints_new_only.txt'), 'w', encoding='utf-8') as f:
        for (x, y) in new_points:
            f.write(f"{x:.6f} {y:.6f}\n")

    # 合并点 TXT（可直接用于后续流程）
    if write_combined_txt:
        with open(os.path.join(output_dir, 'slotpoints_vector_field.txt'), 'w', encoding='utf-8') as f:
            for (x, y) in original:
                f.write(f"{x:.6f} {y:.6f}\n")
            for (x, y) in new_points:
                f.write(f"{x:.6f} {y:.6f}\n")


def main():
    import argparse
    p = argparse.ArgumentParser(description='Vector Field Slot Placement Generator')
    p.add_argument('--config', default='configs/city_config_v4_0.json')
    p.add_argument('--input', default=None, help='slotpoints.txt；若未指定则读取 config.growth_v4_0.slots.path')
    p.add_argument('--output_dir', default='enhanced_simulation_v4_0_output/v4_vector_field')
    p.add_argument('--w_hub', type=float, default=2.0)
    p.add_argument('--w_road', type=float, default=1.2)
    p.add_argument('--sigma', type=float, default=1.5)
    p.add_argument('--d_cut', type=float, default=3.0)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--decay', choices=['gaussian', 'inv_sq'], default='gaussian')
    p.add_argument('--min_sep', type=float, default=1.0, help='最小分离距离（像素）')
    p.add_argument('--roads', type=str, default=None, help='道路/河线 txt（多段折线，每段用[]包围）')
    args = p.parse_args()

    cfg = read_config(args.config)
    city = cfg.get('city', {})
    map_size = city.get('map_size', [200, 200])

    input_path = args.input or cfg.get('growth_v4_0', {}).get('slots', {}).get('path', 'slotpoints.txt')
    points = load_points_from_txt(input_path, map_size)

    hubs = hubs_from_config(cfg)
    # 道路：优先使用 --roads 文件；否则从配置/river.txt 加载
    polylines = []
    if isinstance(args.roads, str) and len(args.roads.strip()) > 0 and os.path.exists(args.roads):
        polylines = load_polylines_from_txt(args.roads)
    if not polylines:
        polylines = load_polylines_from_config(cfg)

    new_pts, pairs = generate_new_points(
        points=points,
        hubs=hubs,
        polylines=polylines,
        map_size=map_size,
        w_hub=float(args.w_hub),
        w_road=float(args.w_road),
        sigma=float(args.sigma),
        d_cut=float(args.d_cut),
        alpha=float(args.alpha),
        decay=str(args.decay),
        min_separation_px=float(args.min_sep),
    )

    save_outputs(points, new_pts, pairs, args.output_dir, write_combined_txt=True)
    # 控制台摘要
    dists = [p[3] for p in pairs]
    if dists:
        dists_sorted = sorted(dists)
        mean = sum(dists) / len(dists)
        median = dists_sorted[len(dists_sorted)//2]
        mx = max(dists)
        print(f"[vf] displacement stats: mean={mean:.4f}, median={median:.4f}, max={mx:.4f}, n={len(dists)}")
    print(f"[vf] done. original={len(points)} new={len(new_pts)} -> out={args.output_dir}")


if __name__ == '__main__':
    main()


