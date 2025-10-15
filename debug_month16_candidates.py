#!/usr/bin/env python3
import json
import re
import math
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

# Hub位置
hubs = [[122, 80], [112, 121]]

# 第16个月的半径计算
def compute_R(month, hubs_cfg):
    h0 = hubs_cfg['list'][0]
    R0 = float(h0.get('R0', 0.0))
    dR = float(h0.get('dR', 1.0))
    R_prev = R0 + dR * max(0, month - 1)
    R_curr = R0 + dR * month
    return R_prev, R_curr

hubs_cfg = cfg['growth_v4_1']['hubs']
R_prev, R_curr = compute_R(16, hubs_cfg)
tol = hubs_cfg.get('tol', 0.5)

print(f"第16个月半径计算:")
print(f"  R0 = {hubs_cfg['list'][0]['R0']}")
print(f"  dR = {hubs_cfg['list'][0]['dR']}")
print(f"  R_prev = {R_prev}")
print(f"  R_curr = {R_curr}")
print(f"  tol = {tol}")

# 计算候选槽位条件
min_dist = R_prev - tol
max_dist = R_curr + tol
print(f"  候选条件: 距离hub在 ({min_dist:.1f}, {max_dist:.1f}] 范围内")

def min_dist_to_hubs(x, y, hubs):
    best = 1e9
    for hx, hy in hubs:
        d = math.hypot(x - float(hx), y - float(hy))
        if d < best:
            best = d
    return best

# 计算每个hub的候选槽位
candidates_by_hub = {}
for hub_idx, hub in enumerate(hubs):
    hub_candidates = []
    for slot_id, (x, y) in slots.items():
        dist = min_dist_to_hubs(x, y, [hub])
        if min_dist < dist <= max_dist:
            hub_candidates.append((slot_id, x, y, dist))
    
    candidates_by_hub[hub_idx] = hub_candidates
    print(f"\nHub{hub_idx+1} ({hub[0]}, {hub[1]}) 第16个月候选槽位: {len(hub_candidates)}个")
    
    # 显示前10个候选槽位
    for i, (slot_id, x, y, dist) in enumerate(hub_candidates[:10]):
        print(f"  {slot_id}: ({x:.1f}, {y:.1f}) 距离={dist:.1f}")
    if len(hub_candidates) > 10:
        print(f"  ... 还有 {len(hub_candidates)-10} 个")

# 计算总的候选槽位（去重）
all_candidates = set()
for hub_idx, hub_candidates in candidates_by_hub.items():
    for slot_id, x, y, dist in hub_candidates:
        all_candidates.add(slot_id)

print(f"\n总候选槽位数（去重后）: {len(all_candidates)}")

# 检查河流分割影响
from envs.v4_1.city_env import CityEnvironment
env = CityEnvironment(cfg)

hub1_comp = env._get_component_of_xy(122, 80)
hub2_comp = env._get_component_of_xy(112, 121)

print(f"\n河流分割后:")
print(f"  Hub1连通域: {hub1_comp}")
print(f"  Hub2连通域: {hub2_comp}")

# 计算每个连通域的候选槽位
comp_candidates = defaultdict(set)
for slot_id in all_candidates:
    x, y = slots[slot_id]
    comp = env._get_component_of_xy(x, y)
    comp_candidates[comp].add(slot_id)

print(f"\n各连通域第16个月候选槽位数:")
for comp_id, candidates in comp_candidates.items():
    if comp_id == hub1_comp:
        hub_name = "Hub1"
    elif comp_id == hub2_comp:
        hub_name = "Hub2"
    else:
        hub_name = f"其他连通域{comp_id}"
    print(f"  {hub_name} (连通域{comp_id}): {len(candidates)}个")

# 检查前15个月的消耗情况
print(f"\n前15个月槽位消耗分析:")
monthly_consumption = 10  # 每月10个槽位（2个agent × 5个槽位/agent）
total_consumption = 15 * monthly_consumption
print(f"  总消耗: {total_consumption}个槽位")
print(f"  Hub1区域剩余: {len(comp_candidates[hub1_comp]) - total_consumption//2}个")
print(f"  Hub2区域剩余: {len(comp_candidates[hub2_comp]) - total_consumption//2}个")

# 检查距离分布
print(f"\nHub1连通域第16个月候选槽位距离分布:")
hub1_candidates = comp_candidates[hub1_comp]
dist_ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35)]
for min_dist_range, max_dist_range in dist_ranges:
    count = 0
    for slot_id in hub1_candidates:
        x, y = slots[slot_id]
        dist = min_dist_to_hubs(x, y, [hubs[0]])
        if min_dist_range <= dist < max_dist_range:
            count += 1
    print(f"  距离 {min_dist_range:2d}-{max_dist_range:2d}: {count:2d} 个槽位")

print(f"\nHub2连通域第16个月候选槽位距离分布:")
hub2_candidates = comp_candidates[hub2_comp]
for min_dist_range, max_dist_range in dist_ranges:
    count = 0
    for slot_id in hub2_candidates:
        x, y = slots[slot_id]
        dist = min_dist_to_hubs(x, y, [hubs[1]])
        if min_dist_range <= dist < max_dist_range:
            count += 1
    print(f"  距离 {min_dist_range:2d}-{max_dist_range:2d}: {count:2d} 个槽位")





