#!/usr/bin/env python3
"""
v4.0 序列池可视化（sequences_pool）

读取 enhanced_simulation_v4_0_output/v4_debug/sequences_pool_month_XX.json，
还原槽位坐标（基于 slotpoints.txt 的同序读入与取整），将 Top-K 序列的放置位置绘制为子图网格。

用法示例：
  python scripts/visualize_v4_sequences.py --output_dir enhanced_simulation_v4_0_output --top 6
  python scripts/visualize_v4_sequences.py --month 11 --top 8
"""

import os
import json
import argparse
import math
from typing import Dict, Tuple, List, Optional

import matplotlib.pyplot as plt


def load_config(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def find_latest_month(v4_debug_dir: str, prefix: str) -> Optional[int]:
    if not os.path.exists(v4_debug_dir):
        return None
    mx = -1
    for fn in os.listdir(v4_debug_dir):
        if fn.startswith(prefix) and fn.endswith('.json'):
            try:
                m = int(fn.replace(prefix, '').replace('.json', ''))
                mx = max(mx, m)
            except Exception:
                pass
    return mx if mx >= 0 else None


def load_slots_xy(slotpoints_path: str) -> Dict[str, Tuple[int, int]]:
    """仿照 v4_0 加载顺序：逐行读取有效点，sid = s_{递增}；坐标按 round 转为 int 像素。"""
    import re
    sid2xy: Dict[str, Tuple[int, int]] = {}
    if not os.path.exists(slotpoints_path):
        raise FileNotFoundError(f'slotpoints not found: {slotpoints_path}')
    with open(slotpoints_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) < 2:
                continue
            xf = float(nums[0]); yf = float(nums[1])
            xi = int(round(xf)); yi = int(round(yf))
            sid = f's_{len(sid2xy)}'
            sid2xy[sid] = (xi, yi)
    return sid2xy


def render_sequences_grid(
    map_size: Tuple[int, int],
    sid2xy: Dict[str, Tuple[int, int]],
    sequences: List[Dict],
    save_path: str,
    title: str,
    cols: int = 3,
    point_size: int = 28,
):
    if not sequences:
        print('No sequences to render')
        return
    rows = math.ceil(len(sequences) / max(1, cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    W, H = map_size
    color_by_agent = {'EDU': '#3B82F6', 'IND': '#F59E0B'}
    size_scale = {'S': point_size, 'M': int(point_size * 1.2), 'L': int(point_size * 1.4)}

    for idx, seq in enumerate(sequences):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 以像素网格为参考，便于和栅格一致
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.set_title(f"Seq {idx+1}  score={seq.get('score', 0):.3f}")

        # 绘制动作 footprint
        for a in seq.get('actions', []) or []:
            agent = a.get('agent', 'EDU')
            size = a.get('size', 'S')
            color = color_by_agent.get(agent, '#10B981')
            ms = size_scale.get(size, point_size)
            for sid in a.get('footprint_slots', []) or []:
                xy = sid2xy.get(sid)
                if xy is None:
                    continue
                x, y = xy
                ax.scatter([x], [y], s=ms, c=color, alpha=0.85, edgecolors='k', linewidths=0.3)

    # 清理多余子图
    total = rows * cols
    for k in range(len(sequences), total):
        r = k // cols
        c = k % cols
        axes[r][c].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize v4 sequences pool placement')
    parser.add_argument('--config', default='configs/city_config_v4_0.json', help='config path')
    parser.add_argument('--output_dir', default='enhanced_simulation_v4_0_output', help='base output dir')
    parser.add_argument('--month', type=int, default=None, help='month number (default: latest)')
    parser.add_argument('--top', type=int, default=6, help='number of sequences to render')
    parser.add_argument('--cols', type=int, default=3, help='columns in grid')
    args = parser.parse_args()

    cfg = load_config(args.config)
    city = cfg.get('city', {})
    map_size = tuple(city.get('map_size', [200, 200]))
    v4 = cfg.get('growth_v4_0', {})
    slots_path = v4.get('slots', {}).get('path', 'slotpoints.txt')

    v4_debug_dir = os.path.join(args.output_dir, 'v4_debug')
    month = args.month
    if month is None:
        month = find_latest_month(v4_debug_dir, 'sequences_pool_month_')
        if month is None:
            raise RuntimeError('No sequences_pool_month_XX.json found')

    # 加载槽位映射 sid -> (x,y)
    sid2xy = load_slots_xy(slots_path)

    # 加载序列池
    seq_path = os.path.join(v4_debug_dir, f'sequences_pool_month_{month:02d}.json')
    with open(seq_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sequences = data.get('sequences', [])
    # 取 Top-N
    sequences = sequences[: max(1, args.top)]

    save_path = os.path.join(v4_debug_dir, f'sequences_pool_month_{month:02d}_grid.png')
    title = f'Sequences Pool Placement - Month {month:02d}'
    render_sequences_grid(map_size, sid2xy, sequences, save_path, title, cols=max(1, args.cols))
    print(f'Grid saved: {save_path}')


if __name__ == '__main__':
    main()


