#!/usr/bin/env python3
import json
import numpy as np

# 加载配置
with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

# 模拟地价计算（简化版）
from envs.v4_1.city_env import CityEnvironment

env = CityEnvironment(cfg)

# 检查hub周围的槽位地价
hub1 = (122, 80)
hub2 = (112, 121)

print('Hub1 (122, 80) 周围槽位的地价分析:')
hub1_slots = [
    ('s_197', 120.9, 81.3, 1.8),  # 最近
    ('s_272', 121.7, 77.7, 2.3),
    ('s_257', 125.2, 78.6, 3.5),
    ('s_179', 124.3, 82.2, 3.2),
    ('s_199', 117.9, 80.2, 4.1),
    ('s_196', 120.3, 84.4, 4.7),
    ('s_275', 118.7, 76.7, 4.7),
]

print('距离 槽位ID    坐标       地价')
for sid, x, y, dist in hub1_slots:
    if sid in env.slots:
        # 获取地价
        land_price = env._get_land_price_field()
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < land_price.shape[0] and 0 <= xi < land_price.shape[1]:
            price = land_price[yi, xi]
            print(f'{dist:4.1f}  {sid:8} ({x:5.1f},{y:5.1f}) {price:6.2f}')

print('\nHub2 (112, 121) 周围槽位的地价分析:')
hub2_slots = [
    ('s_40', 112.9, 119.6, 1.7),  # 最近
    ('s_26', 111.9, 122.8, 1.8),
    ('s_11', 108.5, 122.0, 3.6),
    ('s_49', 109.3, 118.7, 3.6),
    ('s_43', 116.0, 121.0, 4.0),
    ('s_28', 114.9, 123.9, 4.1),
    ('s_41', 113.6, 116.9, 4.4),
]

print('距离 槽位ID    坐标       地价')
for sid, x, y, dist in hub2_slots:
    if sid in env.slots:
        # 获取地价
        land_price = env._get_land_price_field()
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < land_price.shape[0] and 0 <= xi < land_price.shape[1]:
            price = land_price[yi, xi]
            print(f'{dist:4.1f}  {sid:8} ({x:5.1f},{y:5.1f}) {price:6.2f}')

# 检查实际选择的建筑位置的地价
print('\n实际选择的建筑位置地价分析:')
chosen_positions = [
    # EDU (代码0) - 应该在北岸
    ('EDU', 110.060, 115.999),
    ('EDU', 113.607, 116.868),
    ('EDU', 116.719, 118.290),
    ('EDU', 116.003, 121.043),
    ('EDU', 114.879, 123.889),
    # IND (代码3) - 应该在南岸
    ('IND', 115.045, 78.791),
    ('IND', 114.839, 82.007),
    ('IND', 119.724, 73.512),
    ('IND', 122.688, 74.148),
    ('IND', 126.073, 74.937),
]

print('类型 坐标        地价  到最近hub距离')
for agent_type, x, y in chosen_positions:
    land_price = env._get_land_price_field()
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < land_price.shape[0] and 0 <= xi < land_price.shape[1]:
        price = land_price[yi, xi]
        
        # 计算到最近hub的距离
        dist1 = ((x - hub1[0])**2 + (y - hub1[1])**2)**0.5
        dist2 = ((x - hub2[0])**2 + (y - hub2[1])**2)**0.5
        min_dist = min(dist1, dist2)
        
        print(f'{agent_type:4} ({x:6.1f},{y:6.1f}) {price:6.2f} {min_dist:8.1f}')


