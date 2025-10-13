#!/usr/bin/env python3
"""
为每个月从 sequences_pool_month_MM.json 里取得分最高的前 N 条序列，
渲染为表格并合成为一张网格大图。

输入：enhanced_simulation_v4_0_output/v4_debug/sequences_pool_month_MM.json
      enhanced_simulation_v4_0_output/v4_debug/actions_pool_month_MM.json（补充 CRP 与 lp_norm）
输出：enhanced_simulation_v4_0_output/v4_seq_top_png/month_MM_topN_colsC.png
"""

import os
import re
import json
import math
import argparse
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_key(agent: str, size: str, footprint_slots: List[str]) -> Tuple[str, str, Tuple[str, ...]]:
    return (str(agent).upper(), str(size).upper(), tuple(sorted(str(s) for s in (footprint_slots or []))))


def load_actions_map(base_dir: str, month: int) -> Dict[Tuple[str, str, Tuple[str, ...]], Dict]:
    path = os.path.join(base_dir, 'v4_debug', f'actions_pool_month_{month:02d}.json')
    mapping: Dict[Tuple[str, str, Tuple[str, ...]], Dict] = {}
    if not os.path.exists(path):
        return mapping
    data = json.load(open(path, 'r', encoding='utf-8'))
    for a in data.get('actions', []) or []:
        mapping[make_key(a.get('agent'), a.get('size'), a.get('footprint_slots', []))] = a
    return mapping


def load_sequences(base_dir: str, month: int) -> List[Dict]:
    path = os.path.join(base_dir, 'v4_debug', f'sequences_pool_month_{month:02d}.json')
    if not os.path.exists(path):
        return []
    data = json.load(open(path, 'r', encoding='utf-8'))
    seqs = data.get('sequences', []) or []
    # 已按 score 排序存储，但保守起见再排一次
    seqs.sort(key=lambda s: float(s.get('score', 0.0)), reverse=True)
    return seqs


def fmt_float(v) -> str:
    try:
        return f"{float(v):.6f}"
    except Exception:
        return ""


def build_rows_for_sequence(seq: Dict, ap_map: Dict[Tuple[str, str, Tuple[str, ...]], Dict]) -> List[List[str]]:
    rows: List[List[str]] = []
    rows.append(['idx', 'agent', 'size', 'footprint', 'lp_norm', 'cost', 'reward', 'prestige', 'score'])
    sum_c = sum_r = sum_p = sum_s = 0.0
    for i, a in enumerate(seq.get('actions', []) or []):
        agent = str(a.get('agent'))
        size = str(a.get('size'))
        fp = a.get('footprint_slots', []) or []
        k = make_key(agent, size, fp)
        ap = ap_map.get(k)
        lp = ap.get('lp_norm') if ap else ''
        c = ap.get('cost') if ap else ''
        r = ap.get('reward') if ap else ''
        p = ap.get('prestige') if ap else ''
        s = ap.get('score') if ap else a.get('score')
        try:
            sum_c += float(c); sum_r += float(r); sum_p += float(p); sum_s += float(s)
        except Exception:
            pass
        rows.append([str(i), agent, size, '|'.join(fp), fmt_float(lp), fmt_float(c), fmt_float(r), fmt_float(p), fmt_float(s)])
    rows.append(['TOTAL', '', '', '', '', fmt_float(sum_c), fmt_float(sum_r), fmt_float(sum_p), fmt_float(sum_s)])
    return rows


def render_table(rows: List[List[str]], title: str, out_path: str) -> None:
    n_rows = max(2, len(rows))
    height = min(12.0, max(3.0, 0.35 * (n_rows + 2)))
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


def render_month_grid(base_dir: str, month: int, top: int, cols: int) -> None:
    ap_map = load_actions_map(base_dir, month)
    seqs = load_sequences(base_dir, month)
    if not seqs:
        return
    seqs = seqs[: max(1, top)]

    # 先把每条序列渲染成小图
    tmp_dir = os.path.join(base_dir, 'v4_seq_top_png', f'month_{month:02d}_items')
    os.makedirs(tmp_dir, exist_ok=True)
    pngs: List[str] = []
    for i, seq in enumerate(seqs):
        rows = build_rows_for_sequence(seq, ap_map)
        png_path = os.path.join(tmp_dir, f'seq_{i:02d}.png')
        render_table(rows, f'Month {month:02d} • Seq {i:02d} • score={fmt_float(seq.get("score", 0.0))}', png_path)
        pngs.append(png_path)

    # 合成大图
    n = len(pngs)
    cols = max(1, cols)
    rows_n = math.ceil(n / cols)
    fig_w = min(30, max(6, cols * 5))
    fig_h = min(30, max(4, rows_n * 3.5))
    fig, axes = plt.subplots(rows_n, cols, figsize=(fig_w, fig_h))
    if rows_n == 1 and cols == 1:
        axes = [[axes]]
    elif rows_n == 1:
        axes = [axes]
    for idx, path in enumerate(pngs):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(os.path.basename(path), fontsize=9)
        ax.axis('off')
    for idx in range(n, rows_n * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis('off')
    plt.tight_layout()
    out_path = os.path.join(base_dir, 'v4_seq_top_png', f'month_{month:02d}_top{n}_cols{cols}.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f'Exported: {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='enhanced_simulation_v4_0_output', help='base output dir')
    p.add_argument('--top', type=int, default=6, help='top sequences per month')
    p.add_argument('--cols', type=int, default=3, help='grid columns per month')
    p.add_argument('--months', type=str, default='all', help='comma list or "all"')
    args = p.parse_args()

    dbg = os.path.join(args.output_dir, 'v4_debug')
    months: List[int] = []
    if args.months == 'all':
        for fn in os.listdir(dbg):
            m = re.match(r'^sequences_pool_month_(\d{2})\.json$', fn)
            if m:
                months.append(int(m.group(1)))
        months.sort()
    else:
        for tok in args.months.split(','):
            tok = tok.strip()
            if not tok:
                continue
            months.append(int(tok))

    for m in months:
        render_month_grid(args.output_dir, m, args.top, args.cols)


if __name__ == '__main__':
    main()




