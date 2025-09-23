#!/usr/bin/env python3
"""
v4.0 动作池可视化

功能：
- 读取 enhanced_simulation_v4_0_output/v4_debug/actions_pool_month_XX.json
- 生成动作明细表（CSV）与表格可视化（PNG）
- 支持按 score 排序、限制 Top-N、筛选 agent（EDU/IND）
"""

import os
import json
import argparse
from typing import Optional

import csv
import matplotlib.pyplot as plt


def find_latest_month(v4_debug_dir: str) -> Optional[int]:
    if not os.path.exists(v4_debug_dir):
        return None
    mx = -1
    for fn in os.listdir(v4_debug_dir):
        if fn.startswith('actions_pool_month_') and fn.endswith('.json'):
            try:
                m = int(fn.replace('actions_pool_month_', '').replace('.json', ''))
                mx = max(mx, m)
            except Exception:
                pass
    return mx if mx >= 0 else None


def load_actions(v4_debug_dir: str, month: int):
    path = os.path.join(v4_debug_dir, f'actions_pool_month_{month:02d}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    actions = data.get('actions', [])
    rows = []
    for idx, a in enumerate(actions, 1):
        rows.append({
            'idx': idx,
            'agent': a.get('agent'),
            'size': a.get('size'),
            'slots': ','.join(a.get('footprint_slots') or []),
            'lp_norm': float(a.get('lp_norm', 0.0)),
            'cost': float(a.get('cost', 0.0)),
            'reward': float(a.get('reward', 0.0)),
            'prestige': float(a.get('prestige', 0.0)),
            'score': float(a.get('score', 0.0)),
        })
    return rows


def render_table(rows, columns, save_path: str, title: str, max_rows: int = 30):
    show_rows = rows[:max_rows] if max_rows and len(rows) > max_rows else rows
    fig, ax = plt.subplots(figsize=(min(22, 0.65 * len(columns) + 6), min(0.4 * len(show_rows) + 2, 40)))
    ax.axis('off')
    cell_text = [[r.get(c, '') for c in columns] for r in show_rows]
    tbl = ax.table(cellText=cell_text,
                   colLabels=columns,
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize v4 actions pool as table')
    parser.add_argument('--output_dir', default='enhanced_simulation_v4_0_output', help='base output dir')
    parser.add_argument('--month', type=int, default=None, help='month number (e.g., 11). If None, auto-detect latest.')
    parser.add_argument('--agent', type=str, default='all', choices=['all', 'EDU', 'IND'], help='filter by agent type')
    parser.add_argument('--top', type=int, default=30, help='top N rows in table')
    parser.add_argument('--by', type=str, default='score', choices=['score', 'reward', 'prestige', 'cost', 'lp_norm'], help='sort by column')
    args = parser.parse_args()

    v4_debug_dir = os.path.join(args.output_dir, 'v4_debug')
    month = args.month
    if month is None:
        month = find_latest_month(v4_debug_dir)
        if month is None:
            raise RuntimeError(f'No actions_pool_month_XX.json found in {v4_debug_dir}')

    rows = load_actions(v4_debug_dir, month)
    if args.agent != 'all':
        rows = [r for r in rows if r.get('agent') == args.agent]
    # sort descending for benefit columns, ascending for cost
    reverse = (args.by != 'cost')
    rows.sort(key=lambda r: r.get(args.by, 0.0), reverse=reverse)

    # save CSV (full)
    csv_path = os.path.join(v4_debug_dir, f'actions_pool_month_{month:02d}_detail.csv')
    columns = ['idx','agent','size','slots','lp_norm','cost','reward','prestige','score']
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(columns)
            for r in rows:
                w.writerow([r.get(c, '') for c in columns])
    except Exception as e:
        print(f'Failed to save CSV: {e}')

    # save PNG table (top N)
    title = f'Actions Pool - Month {month:02d} ({args.agent}) sorted by {args.by}'
    png_path = os.path.join(v4_debug_dir, f'actions_pool_month_{month:02d}_table_{args.agent}_{args.by}.png')
    render_table(rows, columns, png_path, title, max_rows=max(5, args.top))

    print(f'Table saved: {png_path}')
    print(f'CSV saved  : {csv_path}')


if __name__ == '__main__':
    main()


