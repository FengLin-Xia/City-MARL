#!/usr/bin/env python3
"""
按月导出序列评估表（CSV）：
- 输入：enhanced_simulation_v4_0_output/v4_debug/actions_pool_month_MM.json
       enhanced_simulation_v4_0_output/v4_debug/sequences_pool_month_MM.json
- 输出：enhanced_simulation_v4_0_output/v4_eval/
       sequences_month_MM_seq_XX_eval.csv（每条序列一份）

评估表包含：agent,size,footprint_slots,lp_norm,cost,reward,prestige,score 与合计行。
"""

import os
import json
import argparse
from typing import Dict, Tuple, List


def make_key(agent: str, size: str, footprint_slots: List[str]) -> Tuple[str, str, Tuple[str, ...]]:
    return (str(agent).upper(), str(size).upper(), tuple(sorted(str(s) for s in (footprint_slots or []))))


def load_actions_pool(base_dir: str, month: int) -> Dict[Tuple[str, str, Tuple[str, ...]], Dict]:
    path = os.path.join(base_dir, 'v4_debug', f'actions_pool_month_{month:02d}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'not found: {path}')
    data = json.load(open(path, 'r', encoding='utf-8'))
    mapping: Dict[Tuple[str, str, Tuple[str, ...]], Dict] = {}
    for a in data.get('actions', []) or []:
        k = make_key(a.get('agent'), a.get('size'), a.get('footprint_slots', []))
        mapping[k] = a
    return mapping


def load_sequences_pool(base_dir: str, month: int) -> List[Dict]:
    path = os.path.join(base_dir, 'v4_debug', f'sequences_pool_month_{month:02d}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'not found: {path}')
    data = json.load(open(path, 'r', encoding='utf-8'))
    return data.get('sequences', []) or []


def write_csv(path: str, rows: List[List[str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(','.join(r) + '\n')


def fmt_float(v) -> str:
    try:
        return f"{float(v):.6f}"
    except Exception:
        return ""


def export_month(base_dir: str, month: int) -> None:
    ap_map = load_actions_pool(base_dir, month)
    sequences = load_sequences_pool(base_dir, month)

    out_dir = os.path.join(base_dir, 'v4_eval')
    os.makedirs(out_dir, exist_ok=True)

    for i, seq in enumerate(sequences):
        actions = seq.get('actions', []) or []
        rows: List[List[str]] = []
        # header
        rows.append(['idx', 'agent', 'size', 'footprint', 'lp_norm', 'cost', 'reward', 'prestige', 'score'])
        sum_cost = 0.0
        sum_reward = 0.0
        sum_prestige = 0.0
        sum_score = 0.0
        for j, a in enumerate(actions):
            agent = str(a.get('agent'))
            size = str(a.get('size'))
            fp = a.get('footprint_slots', []) or []
            k = make_key(agent, size, fp)
            ap = ap_map.get(k)  # 从动作池补充 crp 与 lp_norm
            lp = ap.get('lp_norm') if ap else ''
            cost = ap.get('cost') if ap else ''
            reward = ap.get('reward') if ap else ''
            prestige = ap.get('prestige') if ap else ''
            score = ap.get('score') if ap else a.get('score')
            # 累加
            try:
                sum_cost += float(cost)
                sum_reward += float(reward)
                sum_prestige += float(prestige)
                sum_score += float(score)
            except Exception:
                pass
            rows.append([
                str(j), agent, size, '|'.join(fp), fmt_float(lp), fmt_float(cost), fmt_float(reward), fmt_float(prestige), fmt_float(score)
            ])
        rows.append(['TOTAL', '', '', '', '', fmt_float(sum_cost), fmt_float(sum_reward), fmt_float(sum_prestige), fmt_float(sum_score)])
        out_path = os.path.join(out_dir, f'sequences_month_{month:02d}_seq_{i:02d}_eval.csv')
        write_csv(out_path, rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='enhanced_simulation_v4_0_output', help='base output dir')
    p.add_argument('--month', type=int, required=True, help='month to export')
    args = p.parse_args()
    export_month(args.output_dir, args.month)
    print(f'Exported to {os.path.join(args.output_dir, "v4_eval")}')


if __name__ == '__main__':
    main()




