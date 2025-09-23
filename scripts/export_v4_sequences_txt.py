#!/usr/bin/env python3
"""
v4.0 序列与最终选择 简化TXT导出

- 从 enhanced_simulation_v4_0_output/v4_debug 读取：
  - sequences_pool_month_XX.json → 为该文件中的每条序列导出一份 txt（共 count 份）
  - chosen_sequence_month_XX.json → 为该文件导出一份 txt

- 坐标来源：slotpoints.txt（按 v4_0 加载顺序映射 sid → (x,y)）

- 简化格式：
  a(x,y,z) 中 a 映射：
    EDU: S→0, M→1, L→2
    IND: S→3, M→4, L→5
  z 恒为 0

  工业（IND）特殊：
    M: {a(x,y,0), a(x,y,0)}  使用 footprint 的两个槽位
    L: {a(x,y,0), a(x,y,0), a(x,y,0), a(x,y,0)} 使用 footprint 的四个槽位

- 输出目录：enhanced_simulation_v4_0_output/v4_txt/
"""

import os
import re
import json
import argparse
from typing import Dict, Tuple, List


AGENT_SIZE_CODE: Dict[Tuple[str, str], int] = {
    ('EDU', 'S'): 0, ('EDU', 'M'): 1, ('EDU', 'L'): 2,
    ('IND', 'S'): 3, ('IND', 'M'): 4, ('IND', 'L'): 5,
}


def load_config(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def load_slots_xy(slotpoints_path: str) -> Dict[str, Tuple[float, float]]:
    """逐行读取 slotpoints.txt，sid=s_{递增}，坐标取 round 后的像素（与 v4 脚本一致）。"""
    if not os.path.exists(slotpoints_path):
        raise FileNotFoundError(f'slotpoints not found: {slotpoints_path}')
    sid2xy: Dict[str, Tuple[float, float]] = {}
    with open(slotpoints_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) < 2:
                continue
            xf = float(nums[0]); yf = float(nums[1])
            xi = round(xf); yi = round(yf)
            sid = f's_{len(sid2xy)}'
            sid2xy[sid] = (xi, yi)
    return sid2xy


def fmt_entry(agent: str, size: str, x: float, y: float) -> str:
    code = AGENT_SIZE_CODE.get((agent, size), 0)
    return f"{code}({x:.3f}, {y:.3f}, 0)"


def export_sequence_txt(seq: Dict, sid2xy: Dict[str, Tuple[float, float]]) -> str:
    parts: List[str] = []
    for a in seq.get('actions', []) or []:
        agent = str(a.get('agent', 'EDU')).upper()
        size = str(a.get('size', 'S')).upper()
        fp = a.get('footprint_slots', []) or []
        if agent == 'IND' and size in ('M', 'L'):
            # 特殊：用 {} 包住多个点
            sub = []
            for sid in fp:
                xy = sid2xy.get(sid)
                if xy is None:
                    continue
                sub.append(fmt_entry(agent, size, xy[0], xy[1]))
            parts.append('{'+', '.join(sub)+'}') if sub else None
        else:
            # 单点动作（EDU/IND S/M/L 都允许单点表示；IND M/L 会落在上面的分支）
            sid = fp[0] if fp else None
            xy = sid2xy.get(sid) if sid else None
            if xy is not None:
                parts.append(fmt_entry(agent, size, xy[0], xy[1]))
    return ', '.join(parts)


def main():
    parser = argparse.ArgumentParser(description='Export v4 sequences/chosen simplified TXT')
    parser.add_argument('--config', default='configs/city_config_v4_0.json', help='config path')
    parser.add_argument('--output_dir', default='enhanced_simulation_v4_0_output', help='base output dir')
    parser.add_argument('--month', type=int, default=None, help='specific month')
    parser.add_argument('--export_pool', action='store_true', help='export sequences_pool (all sequences)')
    parser.add_argument('--export_chosen', action='store_true', help='export chosen_sequence')
    args = parser.parse_args()

    cfg = load_config(args.config)
    slots_path = cfg.get('growth_v4_0', {}).get('slots', {}).get('path', 'slotpoints.txt')
    sid2xy = load_slots_xy(slots_path)

    v4_debug = os.path.join(args.output_dir, 'v4_debug')
    txt_dir = os.path.join(args.output_dir, 'v4_txt')
    os.makedirs(txt_dir, exist_ok=True)

    def months_from_prefix(prefix: str) -> List[int]:
        return sorted(
            int(fn.replace(prefix, '').replace('.json', ''))
            for fn in os.listdir(v4_debug)
            if fn.startswith(prefix) and fn.endswith('.json')
        )

    months = []
    if args.month is not None:
        months = [int(args.month)]
    else:
        if args.export_pool:
            months = months_from_prefix('sequences_pool_month_')
        elif args.export_chosen:
            months = months_from_prefix('chosen_sequence_month_')
        else:
            months = months_from_prefix('sequences_pool_month_')

    for m in months:
        if args.export_pool or (not args.export_pool and not args.export_chosen):
            pool_path = os.path.join(v4_debug, f'sequences_pool_month_{m:02d}.json')
            if os.path.exists(pool_path):
                data = json.load(open(pool_path, 'r', encoding='utf-8'))
                seqs = data.get('sequences', [])
                for i, seq in enumerate(seqs):
                    line = export_sequence_txt(seq, sid2xy)
                    out = os.path.join(txt_dir, f'sequences_month_{m:02d}_seq_{i:02d}.txt')
                    with open(out, 'w', encoding='utf-8') as f:
                        f.write(line)

        if args.export_chosen or (not args.export_pool and not args.export_chosen):
            chosen_path = os.path.join(v4_debug, f'chosen_sequence_month_{m:02d}.json')
            if os.path.exists(chosen_path):
                data = json.load(open(chosen_path, 'r', encoding='utf-8'))
                seq = { 'actions': data.get('actions', []) }
                line = export_sequence_txt(seq, sid2xy)
                out = os.path.join(txt_dir, f'chosen_month_{m:02d}.txt')
                with open(out, 'w', encoding='utf-8') as f:
                    f.write(line)

    print(f'Exported TXT files to: {txt_dir}')


if __name__ == '__main__':
    main()


