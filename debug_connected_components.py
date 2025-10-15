#!/usr/bin/env python3
import json
import re
import numpy as np
from collections import defaultdict

# 加载配置
with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

# 加载槽位
slots = {}
with open('slots_with_angle.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        nums = re.findall(r'-?\d+(?:\.\d+)?', line.strip())
        if len(nums) >= 2:
            x, y = float(nums[0]), float(nums[1])
            slots[f's_{i}'] = (x, y)

print(f"总槽位数: {len(slots)}")

# 创建环境来获取连通域信息
from envs.v4_1.city_env import CityEnvironment
env = CityEnvironment(cfg)

# 检查hub连通域
hub1 = (122, 80)
hub2 = (112, 121)

hub1_comp = env._get_component_of_xy(hub1[0], hub1[1])
hub2_comp = env._get_component_of_xy(hub2[0], hub2[1])

print(f"Hub1 (122, 80) 连通域: {hub1_comp}")
print(f"Hub2 (112, 121) 连通域: {hub2_comp}")

# 统计每个连通域的槽位数量
comp_slots = defaultdict(list)

for slot_id, (x, y) in slots.items():
    comp = env._get_component_of_xy(x, y)
    comp_slots[comp].append((slot_id, x, y))

print(f"\n连通域统计:")
for comp_id, slot_list in comp_slots.items():
    print(f"  连通域 {comp_id}: {len(slot_list)} 个槽位")

# 检查hub周围的槽位分布
print(f"\nHub1连通域 ({hub1_comp}) 的槽位:")
hub1_slots = comp_slots[hub1_comp]
for slot_id, x, y in hub1_slots[:10]:  # 显示前10个
    dist = ((x-hub1[0])**2 + (y-hub1[1])**2)**0.5
    print(f"  {slot_id}: ({x:.1f}, {y:.1f}) 距离={dist:.1f}")
if len(hub1_slots) > 10:
    print(f"  ... 还有 {len(hub1_slots)-10} 个槽位")

print(f"\nHub2连通域 ({hub2_comp}) 的槽位:")
hub2_slots = comp_slots[hub2_comp]
for slot_id, x, y in hub2_slots[:10]:  # 显示前10个
    dist = ((x-hub2[0])**2 + (y-hub2[1])**2)**0.5
    print(f"  {slot_id}: ({x:.1f}, {y:.1f}) 距离={dist:.1f}")
if len(hub2_slots) > 10:
    print(f"  ... 还有 {len(hub2_slots)-10} 个槽位")

# 检查河流缓冲影响
print(f"\n河流缓冲分析:")
river_points = []
with open('river.txt', 'r', encoding='utf-8') as f:
    for line in f:
        nums = re.findall(r'-?\d+(?:\.\d+)?', line.strip())
        if len(nums) >= 2:
            x, y = float(nums[0]), float(nums[1])
            river_points.append((x, y))

def min_dist_to_river(x, y, river_points):
    min_dist = float('inf')
    for rx, ry in river_points:
        dist = ((x - rx)**2 + (y - ry)**2)**0.5
        min_dist = min(min_dist, dist)
    return min_dist

# 检查河流缓冲对槽位的影响
buffer_px = 2.0  # 默认缓冲距离
river_blocked_count = 0
for slot_id, (x, y) in slots.items():
    dist_to_river = min_dist_to_river(x, y, river_points)
    if dist_to_river <= buffer_px:
        river_blocked_count += 1

print(f"  河流缓冲距离: {buffer_px}px")
print(f"  被河流缓冲阻挡的槽位: {river_blocked_count}")
print(f"  可用槽位: {len(slots) - river_blocked_count}")

# 检查每个连通域中距离hub不同范围的槽位数量
print(f"\nHub1连通域槽位距离分布:")
dist_ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
for min_dist, max_dist in dist_ranges:
    count = 0
    for slot_id, x, y in hub1_slots:
        dist = ((x-hub1[0])**2 + (y-hub1[1])**2)**0.5
        if min_dist <= dist < max_dist:
            count += 1
    print(f"  距离 {min_dist:2d}-{max_dist:2d}: {count:2d} 个槽位")

print(f"\nHub2连通域槽位距离分布:")
for min_dist, max_dist in dist_ranges:
    count = 0
    for slot_id, x, y in hub2_slots:
        dist = ((x-hub2[0])**2 + (y-hub2[1])**2)**0.5
        if min_dist <= dist < max_dist:
            count += 1
    print(f"  距离 {min_dist:2d}-{max_dist:2d}: {count:2d} 个槽位")





