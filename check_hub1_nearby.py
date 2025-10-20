#!/usr/bin/env python3
"""
检查Hub1最近的槽位
"""

import re
import math

hub1_x, hub1_y = 122, 80

slots = []
with open('slots_with_angle.txt') as f:
    for line in f:
        m = re.findall(r'-?\d+(?:\.\d+)?', line.strip())
        if len(m) >= 4:
            x = float(m[0])
            y = float(m[1])
            level = int(m[3])
            dist = math.hypot(x - hub1_x, y - hub1_y)
            slots.append((x, y, level, dist))

# 按距离排序
slots_sorted = sorted(slots, key=lambda s: s[3])

print("="*80)
print("Hub1 (122, 80) 最近的槽位分析")
print("="*80)

print(f"\n距离Hub1最近的20个槽位:")
print(f"{'序号':<4} {'X':<10} {'Y':<10} {'距离':<8} {'Level':<6} {'可建类型':<15}")
print("-"*80)

for i, (x, y, level, dist) in enumerate(slots_sorted[:20], 1):
    can_build = []
    if level >= 3:
        can_build.append('S')
    if level >= 4:
        can_build.append('M')
    if level >= 5:
        can_build.append('L')
    can_str = ','.join(can_build)
    
    print(f"{i:<4} {x:<10.1f} {y:<10.1f} {dist:<8.1f} {level:<6} {can_str:<15}")

# 统计不同距离范围内的level分布
print(f"\n按距离范围统计:")
print(f"{'距离范围':<15} {'槽位数':<10} {'Level 3':<10} {'Level 4':<10} {'Level 5':<10}")
print("-"*80)

distance_ranges = [(0, 5), (5, 10), (10, 15), (15, 20)]
for d_min, d_max in distance_ranges:
    slots_in_range = [s for s in slots if d_min <= s[3] < d_max]
    count = len(slots_in_range)
    level3 = sum(1 for s in slots_in_range if s[2] == 3)
    level4 = sum(1 for s in slots_in_range if s[2] == 4)
    level5 = sum(1 for s in slots_in_range if s[2] == 5)
    
    print(f"{d_min}-{d_max}px{'':<7} {count:<10} {level3:<10} {level4:<10} {level5:<10}")
    if count > 0:
        print(f"{'':15} {'':10} ({level3/count*100:.0f}%) {'':5} ({level4/count*100:.0f}%) {'':5} ({level5/count*100:.0f}%)")

print(f"\n关键发现:")
print(f"  如果距离0-10px内大部分是level=3")
print(f"  → 前期候选池（R=6-12）中几乎没有M/L型动作")
print(f"  → RL只能选S型")

print("="*80)




