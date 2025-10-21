#!/usr/bin/env python3
"""
分析 outputs 目录的导出坐标，统计每个智能体使用的唯一坐标数量与Top重复点。
"""
import os
import re
import json
from collections import Counter, defaultdict

import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else 'outputs'

# 解析一行，返回 (agent, (x, y))
# 行格式示例："3(105.1,124.3,0)228.3"
line_re = re.compile(r'^(\d+)\(([-\d\.]+),\s*([-\d\.]+)')

def agent_from_action_id(aid: int) -> str:
    if 0 <= aid <= 2:
        return 'EDU'
    if 3 <= aid <= 5:
        return 'IND'
    if 6 <= aid <= 8:
        return 'COUNCIL'
    return 'UNKNOWN'

agent_to_coords = defaultdict(list)

# 遍历TXT导出
for fn in sorted(os.listdir(BASE)):
    m = re.match(r'^v4_compatible_month_(\d+)\.txt$', fn)
    if not m:
        continue
    path = os.path.join(BASE, fn)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                mm = line_re.match(s)
                if not mm:
                    continue
                aid = int(mm.group(1))
                x = round(float(mm.group(2)), 1)
                y = round(float(mm.group(3)), 1)
                ag = agent_from_action_id(aid)
                agent_to_coords[ag].append((x, y))
    except Exception as e:
        print(f'read fail {path}: {e}')

summary = {}
for ag, coords in agent_to_coords.items():
    cnt = Counter(coords)
    summary[ag] = {
        'total': len(coords),
        'unique': len(cnt),
        'top5': cnt.most_common(5),
    }

overall_unique = len(set(c for coords in agent_to_coords.values() for c in coords))

print('agents:', sorted(agent_to_coords.keys()))
print('overall_unique_coords:', overall_unique)
print(json.dumps(summary, ensure_ascii=False, indent=2))
