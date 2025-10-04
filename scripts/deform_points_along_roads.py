"""
道路带状点集变形（Push/Pull + Tangential grooming，RK2/Euler）

输入：
- points: 文本点集（每行至少两个数，取 x,y；允许第三个值，例如 south.txt）
- roads: 多段折线 txt（每段用 [] 包围，每行 2~3 个数；见 road.txt）

参数：
- r_inner, r_outer: 带状范围（内/外半径）
- w_pull: 远侧吸引（法向，d>r_outer）
- w_push: 近侧反推（法向，d<r_inner）
- w_tan: 带内切向梳理强度
- sigma: 平滑带宽（控制 smooth 过渡）
- dt: 时间步长；iters: 迭代步数；mode: rk2|euler
- clip_to_map: [W,H] 裁剪；min_sep: 迭代后去重；quantize: 输出是否量化到整格
- max_force, max_step: 限幅，防止数值发散

输出：
- deform_result.json（含 original/new/pairs/stats）与 displacements.csv
- points_deformed.txt（新点）与 points_deform_overlay 可用 visualize_displacements 渲染

用法：
  python -m scripts.deform_points_along_roads \
    --points south.txt --roads road.txt \
    --r_inner 2.0 --r_outer 6.0 --w_pull 1.2 --w_push 1.0 --w_tan 1.0 \
    --sigma 2.0 --dt 0.5 --iters 20 --mode rk2 \
    --output_dir enhanced_simulation_v4_0_output/deform_demo
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Tuple, Optional


def read_points_txt(path: str, map_size: Optional[List[int]] = None) -> List[Tuple[float, float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    W = int(map_size[0]) if (map_size and len(map_size) >= 1) else None
    H = int(map_size[1]) if (map_size and len(map_size) >= 2) else None
    pts: List[Tuple[float, float]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) < 2:
                continue
            try:
                x = float(nums[0]); y = float(nums[1])
            except Exception:
                continue
            if W is not None and H is not None:
                if x < 0.0 or y < 0.0 or x >= float(W) or y >= float(H):
                    continue
            pts.append((x, y))
    return pts


def load_polylines_from_txt(path: str) -> List[List[Tuple[float, float]]]:
    polys: List[List[Tuple[float, float]]] = []
    if not isinstance(path, str) or not os.path.exists(path):
        return polys
    current: List[Tuple[float, float]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            s = raw.strip()
            if not s:
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


def nearest_point_and_tangent_on_polylines(
    x: float, y: float, polylines: List[List[Tuple[float, float]]]
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], float]:
    best_d2 = float('inf')
    best_q = None
    best_t = None
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
            t = ((sx - x0) * dx + (sy - y0) * dy) / seg_len2
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            qx = x0 + t * dx
            qy = y0 + t * dy
            ddx = sx - qx
            ddy = sy - qy
            d2 = ddx * ddx + ddy * ddy
            if d2 < best_d2:
                best_d2 = d2
                seg_len = math.sqrt(seg_len2)
                best_q = (qx, qy)
                best_t = (dx / seg_len, dy / seg_len)
    if best_q is None:
        return None, None, float('inf')
    return best_q, best_t, math.sqrt(best_d2)


def gaussian_window(x: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 0.0
    return math.exp(- (x * x) / (2.0 * sigma * sigma))


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def polygon_centroid(poly: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not poly or len(poly) < 3:
        return None
    # Shoelace formula centroid
    A = 0.0
    Cx = 0.0
    Cy = 0.0
    n = len(poly) - 1 if is_polygon_closed(poly) else len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross
    A *= 0.5
    if abs(A) <= 1e-12:
        return None
    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    return (Cx, Cy)

def is_polygon_closed(poly: List[Tuple[float, float]], eps: float = 1e-6) -> bool:
    if not poly:
        return False
    x0, y0 = poly[0]
    x1, y1 = poly[-1]
    return (abs(x0 - x1) <= eps) and (abs(y0 - y1) <= eps)

def ensure_closed_polygon(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not poly:
        return poly
    if not is_polygon_closed(poly):
        return list(poly) + [poly[0]]
    return poly

def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    # ray casting
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n - 1):
        x0, y0 = polygon[i]
        x1, y1 = polygon[i + 1]
        if ((y0 > y) != (y1 > y)):
            t = (y - y0) / max(1e-12, (y1 - y0))
            xi = x0 + t * (x1 - x0)
            if xi > x:
                inside = not inside
    return inside

def _is_point_on_segment(px: float, py: float, x0: float, y0: float, x1: float, y1: float, eps: float = 1e-6) -> bool:
    # bounding box check
    if (px < min(x0, x1) - eps or px > max(x0, x1) + eps or py < min(y0, y1) - eps or py > max(y0, y1) + eps):
        return False
    # cross product close to 0 and colinear
    dx = x1 - x0
    dy = y1 - y0
    dxp = px - x0
    dyp = py - y0
    cross = dx * dyp - dy * dxp
    if abs(cross) > eps:
        return False
    # dot product within segment
    dot = dx * dxp + dy * dyp
    if dot < -eps:
        return False
    sq_len = dx * dx + dy * dy
    if dot - sq_len > eps:
        return False
    return True

def point_in_polygon_winding(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    n = len(polygon)
    if n < 3:
        return False
    # treat points on edges as inside
    for i in range(n - 1):
        x0, y0 = polygon[i]
        x1, y1 = polygon[i + 1]
        if _is_point_on_segment(x, y, x0, y0, x1, y1):
            return True
    wn = 0
    for i in range(n - 1):
        x0, y0 = polygon[i]
        x1, y1 = polygon[i + 1]
        if y0 <= y:
            if y1 > y:
                # upward crossing
                if (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) > 0:
                    wn += 1
        else:
            if y1 <= y:
                # downward crossing
                if (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) < 0:
                    wn -= 1
    return wn != 0


# -------- Kernel/sampling-based field (uniform influence along roads) --------
def sample_polylines(polys: List[List[Tuple[float, float]]], step: float = 1.0) -> List[Tuple[float, float, float, float]]:
    """等距采样道路折线，返回样本 (x,y,tx,ty)。"""
    step = max(1e-3, float(step))
    samples: List[Tuple[float, float, float, float]] = []
    for poly in polys or []:
        if len(poly) < 2:
            continue
        acc = 0.0
        for i in range(len(poly) - 1):
            x0, y0 = poly[i]
            x1, y1 = poly[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            seg_len = math.hypot(dx, dy)
            if seg_len <= 1e-9:
                continue
            tx = dx / seg_len
            ty = dy / seg_len
            t = 0.0
            # 包含端点
            while t <= seg_len:
                sx = x0 + tx * t
                sy = y0 + ty * t
                samples.append((sx, sy, tx, ty))
                t += step
    return samples


def weighted_dirs_to_road_samples(
    p: Tuple[float, float],
    samples: List[Tuple[float, float, float, float]],
    r_infl: float,
    sigma_k: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """对一定半径内的道路样本做核加权，返回 (n_hat_w, t_hat_w, d_eff)。
    - n_hat_w: 加权的法向单位向量（若无样本返回 (0,0)）
    - t_hat_w: 加权的切向单位向量
    - d_eff:   加权平均距离
    """
    px, py = p
    r2 = float(r_infl) * float(r_infl)
    sum_nx = 0.0
    sum_ny = 0.0
    sum_tx = 0.0
    sum_ty = 0.0
    sum_d = 0.0
    sum_w = 0.0
    for (sx, sy, tx, ty) in samples or []:
        dx = sx - px
        dy = sy - py
        d2 = dx * dx + dy * dy
        if d2 > r2:
            continue
        d = math.sqrt(d2)
        if d <= 1e-12:
            w = 1.0
            nx = 0.0
            ny = 0.0
        else:
            w = math.exp(- (d * d) / (2.0 * sigma_k * sigma_k))
            nx = dx / d
            ny = dy / d
        sum_nx += w * nx
        sum_ny += w * ny
        sum_tx += w * tx
        sum_ty += w * ty
        sum_d += w * d
        sum_w += w
    if sum_w <= 1e-12:
        return (0.0, 0.0), (0.0, 0.0), float('inf')
    # 归一化
    n_norm = math.hypot(sum_nx, sum_ny)
    t_norm = math.hypot(sum_tx, sum_ty)
    n_hat = (sum_nx / n_norm, sum_ny / n_norm) if n_norm > 1e-12 else (0.0, 0.0)
    t_hat = (sum_tx / t_norm, sum_ty / t_norm) if t_norm > 1e-12 else (0.0, 0.0)
    d_eff = sum_d / sum_w
    return n_hat, t_hat, d_eff


def compute_force(p: Tuple[float, float], polylines, r_inner: float, r_outer: float,
                  w_pull: float, w_push: float, w_tan: float, sigma: float,
                  center_k: float = 0.0,
                  max_force: float = 5.0,
                  samples: Optional[List[Tuple[float, float, float, float]]] = None,
                  r_infl: float = 8.0,
                  sigma_k: float = 3.0,
                  # strong near-road wall repulsion
                  w_push_hard: float = 0.0,
                  push_inner_ratio: float = 0.6,
                  push_eps: float = 0.25,
                  # soft near-road repulsion (independent of r_inner)
                  road_soft_rep_w: float = 0.0,
                  road_soft_rep_r: float = 0.0,
                  road_soft_sigma: float = 0.2,
                  # normal cap (limit normal component fraction)
                  normal_cap: Optional[float] = None,
                  # global/hub tangential bias toward a target point
                  hub_target: Optional[Tuple[float, float]] = None,
                  w_hub_tan: float = 0.0) -> Tuple[float, float]:
    x, y = p
    if samples:
        n_hat, t_hat, d_eff = weighted_dirs_to_road_samples((x, y), samples, r_infl=float(r_infl), sigma_k=float(sigma_k))
        # 若无采样落入范围，回退最近点
        if d_eff == float('inf'):
            q, t_hat0, d0 = nearest_point_and_tangent_on_polylines(x, y, polylines)
            if q is None or t_hat0 is None:
                return 0.0, 0.0
            t_hat = t_hat0
            d = d0
        else:
            d = d_eff
    else:
        q, t_hat, d = nearest_point_and_tangent_on_polylines(x, y, polylines)
        if q is None or t_hat is None:
            return 0.0, 0.0
        qx, qy = q
        vx = qx - x
        vy = qy - y
        if d <= 1e-9:
            n_hat = (0.0, 0.0)
        else:
            n_hat = (vx / d, vy / d)
    # 法向：pull/push
    Fx = 0.0
    Fy = 0.0
    if d > r_outer:
        # 远处吸引：随超出量增大（高斯 1-exp(-((d-r_out)/sigma)^2)）
        delta = max(0.0, d - r_outer)
        s = 1.0 - gaussian_window(delta, sigma)
        Fx += w_pull * s * n_hat[0]
        Fy += w_pull * s * n_hat[1]
    elif d < r_inner:
        # 近处反推：随越近越强（1-exp(-((r_in-d)/sigma)^2)）
        delta = max(0.0, r_inner - d)
        s = 1.0 - gaussian_window(delta, sigma)
        Fx -= w_push * s * n_hat[0]
        Fy -= w_push * s * n_hat[1]
        # 强斥力墙（更明显的内带反推），仅在 d < r_inner*push_inner_ratio
        if w_push_hard > 1e-9:
            thr = max(1e-6, push_inner_ratio * r_inner)
            if d < thr:
                # logistic-like barrier: k = 1 / (1 + exp((d-thr)/eps)) ∈ (0,1)
                k = 1.0 / (1.0 + math.exp((d - thr) / max(1e-6, push_eps)))
                Fx -= w_push_hard * k * n_hat[0]
                Fy -= w_push_hard * k * n_hat[1]
    else:
        # 中间带：可选轻微朝带中心
        if center_k != 0.0:
            mid = 0.5 * (r_inner + r_outer)
            sign = 1.0 if d < mid else -1.0
            mag = center_k * abs(d - mid) / max(1e-6, (r_outer - r_inner))
            Fx += sign * mag * n_hat[0]
            Fy += sign * mag * n_hat[1]
    # 近路软斥力（与 r_inner 无关）：靠近中心线轻推回去
    if (road_soft_rep_w > 1e-9) and (road_soft_rep_r > 1e-9) and (d < float(road_soft_rep_r)):
        g = gaussian_window(d, max(1e-6, float(road_soft_sigma)))
        # n_hat 指向中心线，因此斥力应为 -n_hat
        Fx -= float(road_soft_rep_w) * g * n_hat[0]
        Fy -= float(road_soft_rep_w) * g * n_hat[1]
    # 切向：带内最强，带外衰减为钟形
    mid = 0.5 * (r_inner + r_outer)
    halfw = 0.5 * max(1e-6, (r_outer - r_inner))
    a = gaussian_window((d - mid) / halfw, 1.0)  # 峰值 1，半宽 ~ 带宽
    Fx += w_tan * a * t_hat[0]
    Fy += w_tan * a * t_hat[1]
    # hub 切向偏置：将 p->hub 的方向投影到切向
    if hub_target is not None and (w_hub_tan > 1e-9) and t_hat is not None:
        hx, hy = hub_target
        vxh = hx - x
        vyh = hy - y
        # 投影到 t_hat
        dot = vxh * t_hat[0] + vyh * t_hat[1]
        Fx += w_hub_tan * dot * t_hat[0] / max(1.0, r_infl)
        Fy += w_hub_tan * dot * t_hat[1] / max(1.0, r_infl)
    # normal cap：限制法向分量占比
    if normal_cap is not None:
        # 当前合力的法向分量大小
        # 单位法向（与 n_hat 同方向）
        nnorm = math.hypot(n_hat[0], n_hat[1])
        if nnorm > 1e-9:
            # 分解 F 到 n_hat 和 t_hat（近似）
            FdotN = Fx * n_hat[0] + Fy * n_hat[1]
            Ftot = math.hypot(Fx, Fy)
            if Ftot > 1e-9:
                frac = abs(FdotN) / Ftot
                if frac > float(normal_cap):
                    # 缩减法向分量，使其占比不超过 normal_cap
                    target_Fn = float(normal_cap) * Ftot * (1.0 if FdotN >= 0 else -1.0)
                    scale = (target_Fn / max(1e-9, FdotN)) if abs(FdotN) > 1e-9 else 0.0
                    Fx = Fx + (scale - 1.0) * FdotN * n_hat[0]
                    Fy = Fy + (scale - 1.0) * FdotN * n_hat[1]
    # 限幅
    mag2 = Fx * Fx + Fy * Fy
    if mag2 > max_force * max_force:
        mag = math.sqrt(mag2)
        scale = max_force / max(1e-9, mag)
        Fx *= scale
        Fy *= scale
    return Fx, Fy


def integrate(points: List[Tuple[float, float]], polylines,
              iters: int, dt: float, mode: str,
              r_inner: float, r_outer: float,
              w_pull: float, w_push: float, w_tan: float, sigma: float,
              center_k: float, max_force: float, max_step: float,
              clip_size: Optional[List[int]] = None,
              use_sampling_field: bool = True,
              sample_step: float = 1.0,
              sample_influence_radius: float = 8.0,
              sample_sigma_k: float = 3.0,
              # repulsion between points
              enable_repulsion: bool = False,
              r_rep: float = 3.0,
              w_rep: float = 0.8,
              rep_sigma: float = 1.2,
              rep_max_neighbors: int = 64,
              # polygon boundary (optional)
              boundary_polygon: Optional[List[Tuple[float, float]]] = None,
              boundary_mode: str = 'clip',  # 'clip'|'project'|'reject'
              boundary_method: str = 'winding',  # 'winding' | 'ray'
              # road band barrier
              band_barrier: bool = False,
              barrier_eps: float = 0.05,
              # global trend options
              normal_cap: Optional[float] = None,
              hub_target: Optional[Tuple[float, float]] = None,
              w_hub_tan: float = 0.0) -> List[Tuple[float, float]]:
    W = int(clip_size[0]) if (clip_size and len(clip_size) >= 1) else None
    H = int(clip_size[1]) if (clip_size and len(clip_size) >= 2) else None
    pts = [(float(x), float(y)) for (x, y) in points]
    samples = sample_polylines(polylines, step=float(sample_step)) if use_sampling_field else None
    # 近邻搜索：均匀网格
    def _build_grid(pts_in: List[Tuple[float, float]], cell: float):
        g = {}
        inv = 1.0 / max(1e-6, cell)
        for idx, (px, py) in enumerate(pts_in):
            cx = int(math.floor(px * inv))
            cy = int(math.floor(py * inv))
            g.setdefault((cx, cy), []).append(idx)
        return g, inv

    def _repulsion_force(i: int, pts_in: List[Tuple[float, float]], grid, inv_cell: float, radius: float, w: float, sigma_r: float, kmax: int) -> Tuple[float, float]:
        if not enable_repulsion:
            return 0.0, 0.0
        x, y = pts_in[i]
        cs = 1.0 / max(1e-6, inv_cell)
        cx = int(math.floor(x * inv_cell))
        cy = int(math.floor(y * inv_cell))
        r2 = radius * radius
        fx = 0.0
        fy = 0.0
        seen = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (cx + dx, cy + dy)
                if cell not in grid:
                    continue
                for j in grid[cell]:
                    if j == i:
                        continue
                    xj, yj = pts_in[j]
                    ddx = x - xj
                    ddy = y - yj
                    d2 = ddx * ddx + ddy * ddy
                    if d2 <= 1e-12 or d2 > r2:
                        continue
                    d = math.sqrt(d2)
                    # 高斯核 + 距离窗（边界衰减到0）
                    k = math.exp(- (d * d) / (2.0 * sigma_r * sigma_r)) * (1.0 - (d / radius))
                    nx = ddx / d
                    ny = ddy / d
                    fx += w * k * nx
                    fy += w * k * ny
                    seen += 1
                    if seen >= kmax:
                        break
                if seen >= kmax:
                    break
            if seen >= kmax:
                break
        return fx, fy

    for _ in range(max(1, int(iters))):
        # 先过滤上轮可能产生的 None（例如落在道路上被移除）
        pts = [p for p in pts if isinstance(p, tuple)]
        # 若启用斥力，构建网格
        if enable_repulsion:
            grid, inv_cell = _build_grid(pts, cell=max(1e-6, r_rep))
        else:
            grid, inv_cell = None, 0.0
        if mode.lower() == 'rk2':
            # 计算 F1
            F1: List[Tuple[float, float]] = []
            for p in pts:
                Fx, Fy = compute_force(p, polylines, r_inner, r_outer, w_pull, w_push, w_tan, sigma, center_k, max_force, samples=samples, r_infl=sample_influence_radius, sigma_k=sample_sigma_k)
                F1.append((Fx, Fy))
            # 中点
            mid_pts: List[Tuple[float, float]] = []
            for (p, f) in zip(pts, F1):
                dx = clamp(f[0] * dt * 0.5, -max_step, max_step)
                dy = clamp(f[1] * dt * 0.5, -max_step, max_step)
                mid_pts.append((p[0] + dx, p[1] + dy))
            # 计算 F2 at mid
            F2: List[Tuple[float, float]] = []
            for idx, p in enumerate(mid_pts):
                Fx, Fy = compute_force(p, polylines, r_inner, r_outer, w_pull, w_push, w_tan, sigma, center_k, max_force, samples=samples, r_infl=sample_influence_radius, sigma_k=sample_sigma_k)
                # 加上点间斥力
                if enable_repulsion and grid is not None:
                    rx, ry = _repulsion_force(idx, mid_pts, grid, inv_cell, radius=r_rep, w=w_rep, sigma_r=rep_sigma, kmax=int(rep_max_neighbors))
                    Fx += rx
                    Fy += ry
                F2.append((Fx, Fy))
            # 更新
            new_pts: List[Tuple[float, float]] = []
            for (p, f) in zip(pts, F2):
                dx = clamp(f[0] * dt, -max_step, max_step)
                dy = clamp(f[1] * dt, -max_step, max_step)
                nx = p[0] + dx
                ny = p[1] + dy
                if W is not None and H is not None:
                    nx = 0.0 if nx < 0.0 else float(W - 1) if nx > float(W - 1) else nx
                    ny = 0.0 if ny < 0.0 else float(H - 1) if ny > float(H - 1) else ny
                # boundary polygon
                def _inside(xx: float, yy: float) -> bool:
                    if boundary_method == 'winding':
                        return point_in_polygon_winding(xx, yy, boundary_polygon)
                    return point_in_polygon(xx, yy, boundary_polygon)
                if boundary_polygon and not _inside(nx, ny):
                    if boundary_mode == 'clip':
                        # 回退到上一步位置（不越界）
                        nx, ny = p[0], p[1]
                    elif boundary_mode == 'project':
                        # 简化：回退 + 微调朝边界法向（此处先回退）
                        nx, ny = p[0], p[1]
                    else:  # 'reject'
                        nx, ny = p[0], p[1]
                # band barrier：禁止进入 [0, r_outer] 带内，投影到外缘
                if band_barrier and polylines:
                    q2, t2, d2 = nearest_point_and_tangent_on_polylines(nx, ny, polylines)
                    if q2 is not None and d2 < (r_outer + barrier_eps):
                        qx, qy = q2
                        dx2 = nx - qx
                        dy2 = ny - qy
                        dd = math.hypot(dx2, dy2)
                        if dd <= 1e-9:
                            if t2 is not None:
                                nx = qx + (r_outer + barrier_eps) * (-t2[1])
                                ny = qy + (r_outer + barrier_eps) * (t2[0])
                        else:
                            scale = (r_outer + barrier_eps) / dd
                            nx = qx + dx2 * scale
                            ny = qy + dy2 * scale
                # 若开启：落在道路中心线上则移除
                if bool(globals().get('_ARGS_REMOVE_ON_ROAD', False)):
                    q3, t3, d3 = nearest_point_and_tangent_on_polylines(nx, ny, polylines)
                    if q3 is not None and d3 <= float(globals().get('_ARGS_ON_ROAD_EPS', 0.05)):
                        # 用 None 占位，稍后过滤
                        new_pts.append(None)
                        continue
                # 若开启：落在边界轮廓线上则移除
                if boundary_polygon and bool(globals().get('_ARGS_REMOVE_ON_BOUNDARY', False)):
                    # 距离多边形边的最小距离
                    min_d = float('inf')
                    nB = len(boundary_polygon)
                    if nB >= 2:
                        for bi in range(nB - 1):
                            x0, y0 = boundary_polygon[bi]
                            x1, y1 = boundary_polygon[bi + 1]
                            # 点到线段距离
                            vx = x1 - x0; vy = y1 - y0
                            wx = nx - x0; wy = ny - y0
                            seg2 = vx*vx + vy*vy
                            if seg2 <= 1e-12:
                                dseg = math.hypot(nx - x0, ny - y0)
                            else:
                                t = max(0.0, min(1.0, (wx*vx + wy*vy) / seg2))
                                px = x0 + t * vx; py = y0 + t * vy
                                dseg = math.hypot(nx - px, ny - py)
                            if dseg < min_d:
                                min_d = dseg
                    if min_d <= float(globals().get('_ARGS_ON_BOUNDARY_EPS', 0.05)):
                        new_pts.append(None)
                        continue
                new_pts.append((nx, ny))
            # 过滤被移除的 None
            pts = [p for p in new_pts if isinstance(p, tuple)]
        else:
            new_pts: List[Tuple[float, float]] = []
            for i, p in enumerate(pts):
                fx, fy = compute_force(p, polylines, r_inner, r_outer, w_pull, w_push, w_tan, sigma, center_k, max_force, samples=samples, r_infl=sample_influence_radius, sigma_k=sample_sigma_k)
                if enable_repulsion and grid is not None:
                    rx, ry = _repulsion_force(i, pts, grid, inv_cell, radius=r_rep, w=w_rep, sigma_r=rep_sigma, kmax=int(rep_max_neighbors))
                    fx += rx
                    fy += ry
                dx = clamp(fx * dt, -max_step, max_step)
                dy = clamp(fy * dt, -max_step, max_step)
                nx = p[0] + dx
                ny = p[1] + dy
                if W is not None and H is not None:
                    nx = 0.0 if nx < 0.0 else float(W - 1) if nx > float(W - 1) else nx
                    ny = 0.0 if ny < 0.0 else float(H - 1) if ny > float(H - 1) else ny
                def _inside2(xx: float, yy: float) -> bool:
                    if boundary_method == 'winding':
                        return point_in_polygon_winding(xx, yy, boundary_polygon)
                    return point_in_polygon(xx, yy, boundary_polygon)
                if boundary_polygon and not _inside2(nx, ny):
                    if boundary_mode == 'clip':
                        nx, ny = p[0], p[1]
                    elif boundary_mode == 'project':
                        nx, ny = p[0], p[1]
                    else:
                        nx, ny = p[0], p[1]
                if band_barrier and polylines:
                    q2, t2, d2 = nearest_point_and_tangent_on_polylines(nx, ny, polylines)
                    if q2 is not None and d2 < (r_outer + barrier_eps):
                        qx, qy = q2
                        dx2 = nx - qx
                        dy2 = ny - qy
                        dd = math.hypot(dx2, dy2)
                        if dd <= 1e-9:
                            if t2 is not None:
                                nx = qx + (r_outer + barrier_eps) * (-t2[1])
                                ny = qy + (r_outer + barrier_eps) * (t2[0])
                        else:
                            scale = (r_outer + barrier_eps) / dd
                            nx = qx + dx2 * scale
                            ny = qy + dy2 * scale
                if bool(globals().get('_ARGS_REMOVE_ON_ROAD', False)):
                    q3, t3, d3 = nearest_point_and_tangent_on_polylines(nx, ny, polylines)
                    if q3 is not None and d3 <= float(globals().get('_ARGS_ON_ROAD_EPS', 0.05)):
                        new_pts.append(None)
                        continue
                if boundary_polygon and bool(globals().get('_ARGS_REMOVE_ON_BOUNDARY', False)):
                    min_d = float('inf')
                    nB = len(boundary_polygon)
                    if nB >= 2:
                        for bi in range(nB - 1):
                            x0, y0 = boundary_polygon[bi]
                            x1, y1 = boundary_polygon[bi + 1]
                            vx = x1 - x0; vy = y1 - y0
                            wx = nx - x0; wy = ny - y0
                            seg2 = vx*vx + vy*vy
                            if seg2 <= 1e-12:
                                dseg = math.hypot(nx - x0, ny - y0)
                            else:
                                t = max(0.0, min(1.0, (wx*vx + wy*vy) / seg2))
                                px = x0 + t * vx; py = y0 + t * vy
                                dseg = math.hypot(nx - px, ny - py)
                            if dseg < min_d:
                                min_d = dseg
                    if min_d <= float(globals().get('_ARGS_ON_BOUNDARY_EPS', 0.05)):
                        new_pts.append(None)
                        continue
                new_pts.append((nx, ny))
            pts = [p for p in new_pts if isinstance(p, tuple)]
    # 过滤 None（被移除的点）
    pts = [p for p in pts if isinstance(p, tuple)]
    return pts


def save_outputs(original: List[Tuple[float, float]], new_pts: List[Tuple[float, float]], out_dir: str,
                 quantize: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # pairs
    pairs: List[Dict] = []
    dists: List[float] = []
    for i, (op, np_) in enumerate(zip(original, new_pts)):
        dx = float(np_[0] - op[0])
        dy = float(np_[1] - op[1])
        dist = float(math.hypot(dx, dy))
        dists.append(dist)
        pairs.append({'orig_index': i, 'dx': dx, 'dy': dy, 'dist': dist})
    d_sorted = sorted(dists)
    def _pct(p: float) -> float:
        if not d_sorted:
            return 0.0
        k = max(0, min(len(d_sorted) - 1, int(round(p * (len(d_sorted) - 1)))))
        return float(d_sorted[k])
    out = {
        'original': [{'x': float(x), 'y': float(y)} for (x, y) in original],
        'new': [{'x': float(x), 'y': float(y)} for (x, y) in new_pts],
        'pairs': pairs,
        'stats': {
            'mean': float(sum(dists) / len(dists)) if dists else 0.0,
            'median': float(d_sorted[len(d_sorted)//2]) if d_sorted else 0.0,
            'max': float(max(dists)) if dists else 0.0,
            'p90': _pct(0.90), 'p95': _pct(0.95), 'p99': _pct(0.99)
        }
    }
    # 兼容 visualize_displacements：保存为 vf_slots.json 名称
    with open(os.path.join(out_dir, 'vf_slots.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    # CSV
    with open(os.path.join(out_dir, 'displacements.csv'), 'w', encoding='utf-8') as f:
        f.write('orig_index,orig_x,orig_y,new_x,new_y,dx,dy,dist\n')
        for i, (op, np_) in enumerate(zip(original, new_pts)):
            dx = float(np_[0] - op[0])
            dy = float(np_[1] - op[1])
            dist = float(math.hypot(dx, dy))
            f.write(f"{i},{op[0]:.6f},{op[1]:.6f},{np_[0]:.6f},{np_[1]:.6f},{dx:.6f},{dy:.6f},{dist:.6f}\n")
    # TXT
    with open(os.path.join(out_dir, 'points_deformed.txt'), 'w', encoding='utf-8') as f:
        for (x, y) in new_pts:
            if quantize:
                f.write(f"{int(round(x))} {int(round(y))}\n")
            else:
                f.write(f"{x:.6f} {y:.6f}\n")


def main():
    import argparse
    p = argparse.ArgumentParser(description='Road-band deformation for point sets (push/pull + tangential, RK2/Euler)')
    p.add_argument('--points', required=True)
    p.add_argument('--roads', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--r_inner', type=float, default=2.0)
    p.add_argument('--r_outer', type=float, default=6.0)
    p.add_argument('--w_pull', type=float, default=1.2)
    p.add_argument('--w_push', type=float, default=1.0)
    p.add_argument('--w_tan', type=float, default=1.0)
    p.add_argument('--sigma', type=float, default=2.0)
    p.add_argument('--dt', type=float, default=0.5)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--mode', choices=['rk2', 'euler'], default='rk2')
    p.add_argument('--max_force', type=float, default=5.0)
    p.add_argument('--max_step', type=float, default=2.0)
    p.add_argument('--center_k', type=float, default=0.0, help='中带居中微调力度（默认 0 不启用）')
    p.add_argument('--clip_W', type=int, default=200)
    p.add_argument('--clip_H', type=int, default=200)
    p.add_argument('--quantize', action='store_true')
    # sampling field params
    p.add_argument('--uniform_road_field', action='store_true', help='启用道路等距采样核场，使整条道路均匀施力')
    p.add_argument('--sample_step', type=float, default=1.0, help='道路采样步长（像素）')
    p.add_argument('--sample_r_infl', type=float, default=8.0, help='采样影响半径（像素）')
    p.add_argument('--sample_sigma_k', type=float, default=3.0, help='采样核带宽（像素）')
    # repulsion params
    p.add_argument('--repel', action='store_true', help='启用槽位间斥力保持网格感')
    p.add_argument('--r_rep', type=float, default=3.0, help='斥力半径（像素）')
    p.add_argument('--w_rep', type=float, default=0.8, help='斥力强度')
    p.add_argument('--rep_sigma', type=float, default=1.2, help='斥力核带宽')
    p.add_argument('--rep_max_neighbors', type=int, default=64, help='每点最大近邻数量上限')
    # boundary (closed polygon from roads)
    p.add_argument('--boundary_from_road_index', type=int, default=-1, help='从 roads 的某条折线构造闭合边界（索引）')
    p.add_argument('--boundary_mode', choices=['clip', 'project', 'reject'], default='clip', help='越界处理方式')
    p.add_argument('--boundary_from_road_indices', type=str, default=None, help='用逗号分隔的多个折线索引共同围合成边界，如 0,1')
    p.add_argument('--boundary_file', type=str, default=None, help='边界多边形 txt（每行 x y ...），优先于 road indices')
    p.add_argument('--boundary_method', choices=['winding','ray'], default='winding', help='点在多边形内判定方法')
    p.add_argument('--boundary_as_road', action='store_true', help='将边界多边形也作为一条道路参与吸引/梳理')
    # band barrier cli
    p.add_argument('--band_barrier', action='store_true', help='将道路从线转为外带屏障，禁止进入 [0,r_outer]')
    p.add_argument('--barrier_eps', type=float, default=0.05, help='带外缘微扩偏移量')
    # remove points if landing on road centerline
    p.add_argument('--remove_on_road', action='store_true', help='若点最终落在道路中心线上（距折线<=阈值）则移除')
    p.add_argument('--on_road_eps', type=float, default=0.05, help='判定“在路上”的距离阈值（像素）')
    # remove points that land on boundary contour (polygon edges)
    p.add_argument('--remove_on_boundary', action='store_true', help='若点最终落在边界轮廓线上（距边<=阈值）则移除')
    p.add_argument('--on_boundary_eps', type=float, default=0.05, help='判定“在边界轮廓上”的距离阈值（像素）')
    # global trend options
    p.add_argument('--normal_cap', type=float, default=None, help='限制法向分量占比，例如 0.35')
    p.add_argument('--hub_toward_boundary_centroid', action='store_true', help='使用边界多边形质心作为全局趋势目标')
    p.add_argument('--w_hub_tan', type=float, default=0.0, help='全局切向偏置强度')
    args = p.parse_args()

    pts = read_points_txt(args.points, map_size=[args.clip_W, args.clip_H])
    roads = load_polylines_from_txt(args.roads)
    if not roads:
        print('[deform] roads not found or empty.')
        roads = []

    # 边界构造优先级：boundary_file > boundary_from_road_indices > boundary_from_road_index
    boundary_poly = None
    if isinstance(args.boundary_file, str) and args.boundary_file and os.path.exists(args.boundary_file):
        try:
            bpts = read_points_txt(args.boundary_file, map_size=[args.clip_W, args.clip_H])
            if len(bpts) >= 3:
                boundary_poly = ensure_closed_polygon(bpts)
        except Exception:
            boundary_poly = None
    def _build_boundary_by_indices(roads_all: List[List[Tuple[float, float]]], indices_str: str) -> Optional[List[Tuple[float, float]]]:
        try:
            idx_list = [int(s.strip()) for s in indices_str.split(',') if s.strip() != '']
        except Exception:
            return None
        # 改进：按索引顺序拼接，并在段末与下一段起点之间添加直线连接，以形成闭合环
        ring: List[Tuple[float, float]] = []
        prev_end: Optional[Tuple[float, float]] = None
        for idx in idx_list:
            if idx < 0 or idx >= len(roads_all):
                continue
            seg = roads_all[idx]
            if not seg:
                continue
            if not ring:
                ring.extend(seg)
                prev_end = seg[-1]
            else:
                # 若上一段末点与本段首点不一致，添加连接段
                start = seg[0]
                if prev_end is not None and (abs(prev_end[0] - start[0]) > 1e-6 or abs(prev_end[1] - start[1]) > 1e-6):
                    ring.append(start)
                # 追加本段（避免重复首点）
                ring.extend(seg[1:])
                prev_end = seg[-1]
        if len(ring) >= 3:
            return ensure_closed_polygon(ring)
        return None

    if boundary_poly is None and isinstance(args.boundary_from_road_indices, str) and args.boundary_from_road_indices:
        boundary_poly = _build_boundary_by_indices(roads, args.boundary_from_road_indices)
    elif boundary_poly is None and isinstance(args.boundary_from_road_index, int) and args.boundary_from_road_index >= 0:
        if roads and args.boundary_from_road_index < len(roads):
            boundary_poly = ensure_closed_polygon(roads[int(args.boundary_from_road_index)])

    # 可选：将边界也作为道路加入影响
    if bool(args.boundary_as_road) and boundary_poly is not None:
        try:
            roads = list(roads) + [boundary_poly]
        except Exception:
            pass

    # hub target
    hub_target = None
    if bool(args.hub_toward_boundary_centroid) and boundary_poly is not None:
        hub_target = polygon_centroid(boundary_poly)

    # 将移除选项注入全局（简化传参给 integrate 内部）
    globals()['_ARGS_REMOVE_ON_ROAD'] = bool(args.remove_on_road)
    globals()['_ARGS_ON_ROAD_EPS'] = float(args.on_road_eps)
    globals()['_ARGS_REMOVE_ON_BOUNDARY'] = bool(args.remove_on_boundary)
    globals()['_ARGS_ON_BOUNDARY_EPS'] = float(args.on_boundary_eps)

    new_pts = integrate(
        points=pts, polylines=roads,
        iters=args.iters, dt=float(args.dt), mode=str(args.mode),
        r_inner=float(args.r_inner), r_outer=float(args.r_outer),
        w_pull=float(args.w_pull), w_push=float(args.w_push), w_tan=float(args.w_tan), sigma=float(args.sigma),
        center_k=float(args.center_k), max_force=float(args.max_force), max_step=float(args.max_step),
        clip_size=[args.clip_W, args.clip_H],
        use_sampling_field=bool(args.uniform_road_field),
        sample_step=float(args.sample_step),
        sample_influence_radius=float(args.sample_r_infl),
        sample_sigma_k=float(args.sample_sigma_k),
        enable_repulsion=bool(args.repel),
        r_rep=float(args.r_rep),
        w_rep=float(args.w_rep),
        rep_sigma=float(args.rep_sigma),
        rep_max_neighbors=int(args.rep_max_neighbors),
        boundary_polygon=boundary_poly,
        boundary_mode=str(args.boundary_mode),
        boundary_method=str(args.boundary_method),
        band_barrier=bool(args.band_barrier),
        barrier_eps=float(args.barrier_eps),
        normal_cap=args.normal_cap,
        hub_target=hub_target,
        w_hub_tan=float(args.w_hub_tan)
    )

    save_outputs(pts, new_pts, args.output_dir, quantize=bool(args.quantize))
    print(f"[deform] done. n={len(pts)} -> out={args.output_dir}")


if __name__ == '__main__':
    main()


