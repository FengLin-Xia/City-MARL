#!/usr/bin/env python3
"""
v3.5 指定月份槽位可视化（示例：Month 1 与 Month 2）
- 基于给定月份的地价场与等值线，创建层并显示槽位
- 输出: enhanced_simulation_v3_1_output/v3_5_monthXX_slots.png
"""

import json
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem
from enhanced_city_simulation_v3_5 import ProgressiveGrowthSystem


def render_month_slots(config: dict, month: int, out_dir: str = 'enhanced_simulation_v3_1_output'):
    map_size = config.get('city', {}).get('map_size', [110, 110])
    hubs = config.get('city', {}).get('transport_hubs', [[20, 55], [90, 55], [67, 94]])

    # 地价场：初始化 + 演化到指定月
    land = GaussianLandPriceSystem(config)
    land.initialize_system(hubs, map_size)
    if month > 0:
        land.update_land_price_field(month)
    land_price_field = land.get_land_price_field()

    # 等值线：指定月份
    iso = IsocontourBuildingSystem(config)
    iso.initialize_system(land_price_field, hubs, map_size, month, land)

    # 槽位层：按 v3.5 规则从等值线构建
    pgs = ProgressiveGrowthSystem(config)
    pgs.initialize_layers(iso, land_price_field)

    # 作图
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f'v3.5 Month {month} Slots')
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # 背景热力
    im = ax.imshow(
        land_price_field.T,
        origin='lower',
        extent=[0, map_size[0], 0, map_size[1]],
        cmap='Greys',
        alpha=0.35
    )

    # 槽位散点
    com_x, com_y = [], []
    for layer in pgs.layers['commercial']:
        for s in layer.slots:
            com_x.append(s.pos[0]); com_y.append(s.pos[1])
    if com_x:
        ax.scatter(com_x, com_y, s=18, c='#2196F3', marker='o', alpha=0.85, label='Commercial slots')

    res_x, res_y = [], []
    for layer in pgs.layers['residential']:
        for s in layer.slots:
            res_x.append(s.pos[0]); res_y.append(s.pos[1])
    if res_x:
        ax.scatter(res_x, res_y, s=18, c='#4CAF50', marker='^', alpha=0.85, label='Residential slots')

    # Hubs
    for i, (x, y) in enumerate(hubs):
        ax.scatter([x], [y], c='red', s=40, zorder=5)
        ax.text(x, y + 2.5, f'H{i+1}', color='red', fontsize=9, ha='center')

    ax.legend(loc='upper right')
    out_path = os.path.join(out_dir, f'v3_5_month{month:02d}_slots.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f'✅ Saved: {out_path}')


def main():
    with open('configs/city_config_v3_5.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 目标月份：1 与 2
    for m in [1, 2]:
        render_month_slots(config, m)


if __name__ == '__main__':
    main()


