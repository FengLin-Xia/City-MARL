#!/usr/bin/env python3
"""
检查槽位的building_level分布
"""

from collections import Counter
import json

print("="*80)
print("槽位Building Level分析")
print("="*80)

# 读取slots
with open('slots_with_angle.txt', 'r') as f:
    lines = f.readlines()

all_levels = []
hub1_levels = []  # Hub1 (122, 80) 附近，IND区域
hub2_levels = []  # Hub2 (112, 121) 附近，EDU区域

for line in lines:
    parts = line.strip().replace(',', '').split()
    if len(parts) >= 4:
        slot_id = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
            level = int(parts[3])
        except ValueError:
            continue
        
        all_levels.append(level)
        
        # Hub1附近：x在100-145, y在60-100
        if 100 < x < 145 and 60 < y < 100:
            hub1_levels.append(level)
        
        # Hub2附近：x在90-135, y在100-145
        if 90 < x < 135 and 100 < y < 145:
            hub2_levels.append(level)

print(f"\n全部槽位level分布:")
print(f"  总槽位数: {len(all_levels)}")
level_counter = Counter(all_levels)
for level, count in sorted(level_counter.items()):
    print(f"  Level {level}: {count}个 ({count/len(all_levels)*100:.1f}%)")

print(f"\nHub1附近（IND区域）level分布:")
print(f"  总槽位数: {len(hub1_levels)}")
if hub1_levels:
    hub1_counter = Counter(hub1_levels)
    for level, count in sorted(hub1_counter.items()):
        can_build = []
        if level >= 3: can_build.append('S')
        if level >= 4: can_build.append('M')
        if level >= 5: can_build.append('L')
        print(f"  Level {level}: {count}个 ({count/len(hub1_levels)*100:.1f}%) - 可建: {','.join(can_build)}")

print(f"\nHub2附近（EDU区域）level分布:")
print(f"  总槽位数: {len(hub2_levels)}")
if hub2_levels:
    hub2_counter = Counter(hub2_levels)
    for level, count in sorted(hub2_counter.items()):
        can_build = []
        if level >= 3: can_build.append('S')
        if level >= 4: can_build.append('M')  
        if level >= 5: can_build.append('L')
        print(f"  Level {level}: {count}个 ({count/len(hub2_levels)*100:.1f}%) - 可建: {','.join(can_build)}")

# 分析为什么全是S型
print(f"\n" + "="*80)
print("为什么100%选S型？")
print("="*80)

# 检查IND的building_level规则
print(f"\nIND建筑规则（logic/v4_enumeration.py）:")
print(f"  S型: level >= 3（所有槽位）")
print(f"  M型: level >= 4（部分槽位）")
print(f"  L型: level >= 5（极少槽位）")

if hub1_levels:
    level3_pct = hub1_counter.get(3, 0) / len(hub1_levels) * 100
    level4_pct = hub1_counter.get(4, 0) / len(hub1_levels) * 100
    level5_pct = hub1_counter.get(5, 0) / len(hub1_levels) * 100
    
    print(f"\nHub1（IND）的候选动作构成:")
    print(f"  可建S型的槽位: {level3_pct + level4_pct + level5_pct:.1f}% (level>=3)")
    print(f"  可建M型的槽位: {level4_pct + level5_pct:.1f}% (level>=4)")
    print(f"  可建L型的槽位: {level5_pct:.1f}% (level>=5)")
    
    if level5_pct < 5:
        print(f"\n  [PROBLEM] L型槽位太少（{level5_pct:.1f}%）")
        print(f"  -> RL几乎看不到L型动作！")
    
    if level4_pct + level5_pct < 20:
        print(f"\n  [PROBLEM] M/L型槽位太少（{level4_pct + level5_pct:.1f}%）")
        print(f"  -> 动作池中S型占绝对优势！")

print(f"\n解决方案:")
print(f"  1. 增加Hub1附近槽位的building_level")
print(f"  2. 或者：完全移除building_level限制，让所有槽位都能建M/L")
print(f"  3. 或者：调整ActionEnumerator，不基于building_level限制")

print("="*80)

