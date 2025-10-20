#!/usr/bin/env python3
import re

# 加载槽位
slots = {}
with open('slots_with_angle.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        nums = re.findall(r'-?\d+(?:\.\d+)?', line.strip())
        if len(nums) >= 2:
            x, y = float(nums[0]), float(nums[1])
            slots[f's_{i}'] = (x, y)

# Hub位置
hub1 = (122, 80)
hub2 = (112, 121)

print('Hub1 (122, 80) 周围的槽位 (距离 <= 10):')
hub1_slots = []
for sid, (x, y) in slots.items():
    dist = ((x-hub1[0])**2 + (y-hub1[1])**2)**0.5
    if dist <= 10:
        hub1_slots.append((sid, x, y, dist))
        print(f'  {sid}: ({x:.1f}, {y:.1f}) 距离={dist:.1f}')

print(f'\nHub1周围槽位总数: {len(hub1_slots)}')

print('\nHub2 (112, 121) 周围的槽位 (距离 <= 10):')
hub2_slots = []
for sid, (x, y) in slots.items():
    dist = ((x-hub2[0])**2 + (y-hub2[1])**2)**0.5
    if dist <= 10:
        hub2_slots.append((sid, x, y, dist))
        print(f'  {sid}: ({x:.1f}, {y:.1f}) 距离={dist:.1f}')

print(f'\nHub2周围槽位总数: {len(hub2_slots)}')

# 检查河流影响
print('\n检查河流位置...')
with open('river.txt', 'r', encoding='utf-8') as f:
    river_points = []
    for line in f:
        nums = re.findall(r'-?\d+(?:\.\d+)?', line.strip())
        if len(nums) >= 2:
            x, y = float(nums[0]), float(nums[1])
            river_points.append((x, y))

# 检查hub到河流的距离
def min_dist_to_river(hub_x, hub_y, river_points):
    min_dist = float('inf')
    for rx, ry in river_points:
        dist = ((hub_x - rx)**2 + (hub_y - ry)**2)**0.5
        min_dist = min(min_dist, dist)
    return min_dist

hub1_river_dist = min_dist_to_river(hub1[0], hub1[1], river_points)
hub2_river_dist = min_dist_to_river(hub2[0], hub2[1], river_points)

print(f'Hub1到河流的最小距离: {hub1_river_dist:.1f}')
print(f'Hub2到河流的最小距离: {hub2_river_dist:.1f}')





