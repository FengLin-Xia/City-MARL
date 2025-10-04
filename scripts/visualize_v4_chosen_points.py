#!/usr/bin/env python3
"""
可视化每月 chosen 序列的落点（按 EDU/IND 着色，S/M/L 区分标记）。

输入：enhanced_simulation_v4_0_output/v4_debug/chosen_sequence_month_MM.json
      slotpoints.txt（用于 sid→(x,y) 映射，保留浮点）
      river.txt（可选，用于叠加河线）

输出：enhanced_simulation_v4_0_output/v4_chosen_pts/month_MM.png
"""

import os
import re
import json
import argparse
from typing import Dict, Tuple, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_sid_to_xy(slotpoints_path: str = 'slotpoints.txt') -> Dict[str, Tuple[float, float]]:
    sid2xy: Dict[str, Tuple[float, float]] = {}
    if not os.path.exists(slotpoints_path):
        return sid2xy
    with open(slotpoints_path, 'r', encoding='utf-8') as f:
        idx = 0
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) < 2:
                continue
            xf = float(nums[0]); yf = float(nums[1])
            sid2xy[f's_{idx}'] = (xf, yf)
            idx += 1
    return sid2xy


def load_river_coords(path: str = 'river.txt') -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    if not os.path.exists(path):
        return coords
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) >= 2:
                coords.append((float(nums[0]), float(nums[1])))
    return coords


def render_month(base_dir: str, month: int, sid2xy: Dict[str, Tuple[float, float]]) -> None:
    dbg = os.path.join(base_dir, 'v4_debug')
    path = os.path.join(dbg, f'chosen_sequence_month_{month:02d}.json')
    if not os.path.exists(path):
        return
    data = json.load(open(path, 'r', encoding='utf-8'))
    actions = data.get('actions', []) or []

    # 准备点集
    pts_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for a in actions:
        agent = str(a.get('agent', 'EDU')).upper()
        size = str(a.get('size', 'S')).upper()
        for sid in a.get('footprint_slots', []) or []:
            xy = sid2xy.get(str(sid))
            if xy is None:
                continue
            pts_by_key.setdefault((agent, size), []).append(xy)

    # 绘制
    fig, ax = plt.subplots(figsize=(6, 6))
    river = load_river_coords('river.txt')
    if len(river) >= 2:
        xr, yr = zip(*river)
        ax.plot(xr, yr, c='cyan', lw=1.2, label='river')

    color_map = {'EDU': '#d62728', 'IND': '#1f77b4'}
    marker_map = {'S': 'o', 'M': 's', 'L': '^'}
    for (agent, size), pts in pts_by_key.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=40, c=color_map.get(agent, 'gray'), marker=marker_map.get(size, 'o'), label=f'{agent}-{size}')

    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_title(f'Chosen placements • Month {month:02d}')
    ax.legend(loc='best', fontsize=8)
    os.makedirs(os.path.join(base_dir, 'v4_chosen_pts'), exist_ok=True)
    out = os.path.join(base_dir, 'v4_chosen_pts', f'month_{month:02d}.png')
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close(fig)


def render_combined(base_dir: str, months: List[int], sid2xy: Dict[str, Tuple[float, float]]) -> None:
    # 聚合所有月的 chosen 点
    pts_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    dbg = os.path.join(base_dir, 'v4_debug')
    for month in months:
        path = os.path.join(dbg, f'chosen_sequence_month_{month:02d}.json')
        if not os.path.exists(path):
            continue
        data = json.load(open(path, 'r', encoding='utf-8'))
        actions = data.get('actions', []) or []
        for a in actions:
            agent = str(a.get('agent', 'EDU')).upper()
            size = str(a.get('size', 'S')).upper()
            for sid in a.get('footprint_slots', []) or []:
                xy = sid2xy.get(str(sid))
                if xy is None:
                    continue
                pts_by_key.setdefault((agent, size), []).append(xy)

    fig, ax = plt.subplots(figsize=(7, 7))
    river = load_river_coords('river.txt')
    if len(river) >= 2:
        xr, yr = zip(*river)
        ax.plot(xr, yr, c='cyan', lw=1.2, label='river')
    color_map = {'EDU': '#d62728', 'IND': '#1f77b4'}
    marker_map = {'S': 'o', 'M': 's', 'L': '^'}
    for (agent, size), pts in pts_by_key.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=20, c=color_map.get(agent, 'gray'), marker=marker_map.get(size, 'o'), label=f'{agent}-{size}')
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_title('Chosen placements • All months')
    ax.legend(loc='best', fontsize=8)
    os.makedirs(os.path.join(base_dir, 'v4_chosen_pts'), exist_ok=True)
    out = os.path.join(base_dir, 'v4_chosen_pts', 'all_months.png')
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='enhanced_simulation_v4_0_output', help='base output dir')
    p.add_argument('--months', default='all', help='comma list like 00,01,02 or all')
    p.add_argument('--slotpoints', default='slotpoints.txt')
    p.add_argument('--combine', action='store_true', help='render a combined plot across months')
    args = p.parse_args()

    sid2xy = load_sid_to_xy(args.slotpoints)
    dbg = os.path.join(args.output_dir, 'v4_debug')
    months: List[int] = []
    if args.months == 'all':
        for fn in os.listdir(dbg):
            m = re.match(r'^chosen_sequence_month_(\d{2})\.json$', fn)
            if m:
                months.append(int(m.group(1)))
        months.sort()
    else:
        for tok in args.months.split(','):
            tok = tok.strip()
            if tok:
                months.append(int(tok))

    for m in months:
        render_month(args.output_dir, m, sid2xy)
    if args.combine and months:
        render_combined(args.output_dir, months, sid2xy)
    print(f'Exported chosen point plots to {os.path.join(args.output_dir, "v4_chosen_pts")}')


if __name__ == '__main__':
    main()


