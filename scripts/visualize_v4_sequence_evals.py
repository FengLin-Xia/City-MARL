#!/usr/bin/env python3
"""
将 v4_eval 下的每条序列评估 CSV 渲染为 PNG 表格。

输入：enhanced_simulation_v4_0_output/v4_eval/sequences_month_MM_seq_XX_eval.csv
输出：enhanced_simulation_v4_0_output/v4_eval_png/sequences_month_MM_seq_XX_eval.png
"""

import os
import re
import csv
import argparse
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path: str) -> List[List[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [row for row in reader]


def render_table(rows: List[List[str]], title: str, out_path: str) -> None:
    # 动态高度：每行 ~0.35 英寸，最小 3 英寸，高度上限做个保护
    n_rows = max(2, len(rows))
    height = min(20.0, max(3.0, 0.35 * (n_rows + 2)))
    n_cols = max(1, len(rows[0]) if rows else 1)
    width = min(16.0, max(6.0, 1.2 * n_cols))

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis('off')
    table = ax.table(cellText=rows, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='enhanced_simulation_v4_0_output', help='base output dir')
    p.add_argument('--month', type=int, required=True, help='month to visualize')
    args = p.parse_args()

    eval_dir = os.path.join(args.output_dir, 'v4_eval')
    out_dir = os.path.join(args.output_dir, 'v4_eval_png')
    os.makedirs(out_dir, exist_ok=True)

    pattern = re.compile(rf'^sequences_month_{args.month:02d}_seq_([0-9]{{2}})_eval\.csv$')
    files = [fn for fn in os.listdir(eval_dir) if pattern.match(fn)]
    files.sort()

    if not files:
        print(f'No eval CSVs found for month {args.month:02d} under {eval_dir}')
        return

    for fn in files:
        rows = read_csv(os.path.join(eval_dir, fn))
        title = fn.replace('_', ' ')
        png = os.path.join(out_dir, fn.replace('.csv', '.png'))
        render_table(rows, title, png)
    print(f'Exported tables to {out_dir}')


if __name__ == '__main__':
    main()




