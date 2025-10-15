#!/usr/bin/env python3
"""
分析建筑类型选择
"""

import json
from collections import Counter

# 读取slot_selection_history
with open('models/v4_1_rl/slot_selection_history.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

episodes = data.get('episodes', [])
if not episodes:
    print("No episodes found")
    exit(1)

# 分析第一个episode
episode = episodes[0]
steps = episode.get('steps', [])

print("="*80)
print("建筑类型分析")
print("="*80)

# 统计IND
ind_sizes = []
ind_costs = []
for step in steps:
    if step['agent'] == 'IND':
        for action in step.get('detailed_actions', []):
            ind_sizes.append(action.get('size', 'S'))
            ind_costs.append(action.get('cost', 0))

# 统计EDU
edu_sizes = []
edu_costs = []
for step in steps:
    if step['agent'] == 'EDU':
        for action in step.get('detailed_actions', []):
            edu_sizes.append(action.get('size', 'S'))
            edu_costs.append(action.get('cost', 0))

print(f"\nIND建筑统计:")
print(f"  总数: {len(ind_sizes)}")
if ind_sizes:
    ind_counter = Counter(ind_sizes)
    for size, count in sorted(ind_counter.items()):
        print(f"  {size}型: {count}个 ({count/len(ind_sizes)*100:.1f}%)")
    print(f"  前10个: {ind_sizes[:10]}")
    print(f"  平均cost: {sum(ind_costs)/len(ind_costs) if ind_costs else 0:.0f}")

print(f"\nEDU建筑统计:")
print(f"  总数: {len(edu_sizes)}")
if edu_sizes:
    edu_counter = Counter(edu_sizes)
    for size, count in sorted(edu_counter.items()):
        print(f"  {size}型: {count}个 ({count/len(edu_sizes)*100:.1f}%)")
    print(f"  前10个: {edu_sizes[:10]}")
    print(f"  平均cost: {sum(edu_costs)/len(edu_costs) if edu_costs else 0:.0f}")

print(f"\n问题诊断:")
print("="*80)

# 检查是否全是S型
if ind_sizes and all(s == 'S' for s in ind_sizes):
    print(f"\n[PROBLEM] IND 100%选择S型！")
    print(f"\n可能原因:")
    print(f"  1. building_level限制：大部分槽位level=3（只能建S）")
    print(f"  2. RL学到了保守策略（避免高cost）")
    print(f"  3. proximity_reward不够大，无法抵消cost差异")
else:
    print(f"\n[OK] IND有多样化建筑类型")

if edu_sizes and all(s == 'S' for s in edu_sizes):
    print(f"\n[PROBLEM] EDU 100%选择S型！")
    print(f"\n可能原因同IND")
else:
    print(f"\n[OK] EDU有多样化建筑类型")

# 计算Budget变化
print(f"\nBudget分析:")
print(f"  IND: 15000 -> -1138 (总支出-总收入 = {15000-(-1138)})")
print(f"  EDU: 10000 -> +3408 (总盈余 = {3408-10000})")

# 基于cost反推建筑类型
if ind_sizes:
    expected_budget_all_s = 15000 - len(ind_sizes)*1000 + len(ind_sizes)*50
    actual_budget = -1138
    print(f"\n  如果IND全建S型，预期budget: {expected_budget_all_s:.0f}")
    print(f"  实际budget: {actual_budget}")
    if abs(expected_budget_all_s - actual_budget) < 1000:
        print(f"  -> 接近！可能确实大部分是S型")
    else:
        print(f"  -> 差异大，应该有M/L型建筑")

print("\n" + "="*80)
print("建筑等级限制检查")
print("="*80)

# 检查槽位的building_level分布
print(f"\n建议检查slots_with_angle.txt:")
print(f"  awk '{{print $4}}' slots_with_angle.txt | sort | uniq -c")
print(f"\n如果大部分槽位是level=3，那就解释了为什么只建S型")

print("="*80)




