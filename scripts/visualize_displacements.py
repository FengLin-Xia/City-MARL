"""
位移对比可视化：读取 generate_vector_field_slots 输出目录中的 vf_slots.json，
绘制原点 → 位移点的连线覆盖图。

用法：
  python -m scripts.visualize_displacements \
      --input_dir enhanced_simulation_v4_0_output/vf_south_roads_strong \
      --output enhanced_simulation_v4_0_output/vf_south_roads_strong/disp_overlay.png \
      --limit 3000 --stride 1 --invert_y
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_json(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


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
            import re
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


def main():
    import argparse
    p = argparse.ArgumentParser(description='Visualize displacement overlay (orig -> new)')
    p.add_argument('--input_dir', required=True, help='directory containing vf_slots.json')
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=3000)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--invert_y', action='store_true')
    p.add_argument('--only_displaced', action='store_true', help='仅渲染发生位移的点')
    p.add_argument('--dist_min', type=float, default=1e-6, help='仅渲染位移超过该阈值的点')
    p.add_argument('--roads', type=str, default=None, help='道路/河线 txt（多段折线，每段用[]包围）')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    jf = os.path.join(args.input_dir, 'vf_slots.json')
    data = read_json(jf)
    originals = data.get('original', []) or []
    pairs = data.get('pairs', []) or []
    # roads
    roads: List[List[Tuple[float, float]]] = []
    if args.roads and os.path.exists(args.roads):
        roads = load_polylines_from_txt(args.roads)

    # 构造 (ox,oy)->(nx,ny) + dist
    segs: List[Tuple[float, float, float, float, float]] = []
    for pr in pairs:
        i = int(pr.get('orig_index', -1))
        if i < 0 or i >= len(originals):
            continue
        ox = float(originals[i].get('x', 0.0))
        oy = float(originals[i].get('y', 0.0))
        dx = float(pr.get('dx', 0.0))
        dy = float(pr.get('dy', 0.0))
        nx = ox + dx
        ny = oy + dy
        dist = float(pr.get('dist', (dx*dx + dy*dy) ** 0.5))
        segs.append((ox, oy, nx, ny, dist))

    # 采样
    if args.stride > 1:
        segs = segs[::max(1, args.stride)]
    if args.limit and len(segs) > args.limit:
        segs = segs[:args.limit]

    # 绘制
    fig, ax = plt.subplots(figsize=(7, 7))
    # roads overlay first
    if roads:
        for poly in roads:
            if len(poly) >= 2:
                xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                ax.plot(xs, ys, '-', color='deepskyblue', linewidth=1.4, alpha=0.9, label='roads')
    if segs:
        if args.only_displaced:
            disp = [(s[2], s[3], s[4]) for s in segs if s[4] >= float(args.dist_min)]
            if disp:
                nxs = [d[0] for d in disp]; nys = [d[1] for d in disp]
                ax.scatter(nxs, nys, s=8, c='royalblue', alpha=0.8, label=f'displaced (>= {args.dist_min})')
        else:
            oxs = [s[0] for s in segs]; oys = [s[1] for s in segs]
            nxs = [s[2] for s in segs]; nys = [s[3] for s in segs]
            ax.scatter(oxs, oys, s=6, c='gray', alpha=0.5, label='original')
            ax.scatter(nxs, nys, s=8, c='royalblue', alpha=0.7, label='displaced')
            for (ox, oy, nx, ny, _) in segs:
                ax.plot([ox, nx], [oy, ny], color='crimson', alpha=0.5, linewidth=0.8)

    # 轴设定（基于数据范围）
    if args.only_displaced:
        allx = [s[2] for s in segs if s[4] >= float(args.dist_min)]
        ally = [s[3] for s in segs if s[4] >= float(args.dist_min)]
    else:
        allx = [s for seg in segs for s in (seg[0], seg[2])]
        ally = [s for seg in segs for s in (seg[1], seg[3])]
    if allx and ally:
        xmin, xmax = min(allx), max(allx)
        ymin, ymax = min(ally), max(ally)
        dx = max(1.0, 0.02 * (xmax - xmin))
        dy = max(1.0, 0.02 * (ymax - ymin))
        ax.set_xlim(xmin - dx, xmax + dx)
        ax.set_ylim(ymin - dy, ymax + dy)

    if args.invert_y:
        ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Original vs Displaced (with segments)')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.6)
    ax.grid(True, alpha=0.2, linestyle='--')
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"[viz] saved: {args.output}")


if __name__ == '__main__':
    main()


