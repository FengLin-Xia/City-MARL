#!/usr/bin/env python3
"""
可视化 demo_slots_north.txt / demo_slots_south.txt 的点位散点图（可选叠加 river.txt）。
输出：enhanced_simulation_v4_0_output/v4_debug/demo_slots_scatter.png
"""

import os
import re
import argparse
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_xy_txt(path: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if not os.path.exists(path):
        return pts
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ns = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(ns) >= 2:
                pts.append((float(ns[0]), float(ns[1])))
    return pts


def load_river(path: str = 'river.txt') -> List[Tuple[float, float]]:
    if not os.path.exists(path):
        return []
    pts: List[Tuple[float, float]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            ns = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(ns) >= 2:
                pts.append((float(ns[0]), float(ns[1])))
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='enhanced_simulation_v4_0_output', help='base output dir')
    p.add_argument('--north', default='enhanced_simulation_v4_0_output/v4_debug/demo_slots_north.txt')
    p.add_argument('--south', default='enhanced_simulation_v4_0_output/v4_debug/demo_slots_south.txt')
    args = p.parse_args()

    north = load_xy_txt(args.north)
    south = load_xy_txt(args.south)
    river = load_river('river.txt')

    fig, ax = plt.subplots(figsize=(6, 6))
    if south:
        xs, ys = zip(*south)
        ax.scatter(xs, ys, s=12, c='#1f77b4', label='south')
    if north:
        xn, yn = zip(*north)
        ax.scatter(xn, yn, s=12, c='#d62728', label='north')
    if len(river) >= 2:
        xr, yr = zip(*river)
        ax.plot(xr, yr, c='cyan', lw=1.5, label='river')
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()  # 与图像坐标一致（y向下）
    ax.legend(loc='best', fontsize=8)
    ax.set_title('North/South slots vs River')
    out_path = os.path.join(args.output_dir, 'v4_debug', 'demo_slots_scatter.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'Exported: {out_path}')


if __name__ == '__main__':
    main()




