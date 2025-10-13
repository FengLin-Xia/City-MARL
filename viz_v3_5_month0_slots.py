#!/usr/bin/env python3
"""
v3.5 Month 0 槽位快速可视化
- 基于 Month 0 地价场与等值线，使用 ProgressiveGrowthSystem 的规则创建层并显示槽位
- 输出: enhanced_simulation_v3_1_output/v3_5_month00_slots.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem

# 复用 v3.5 中的 ProgressiveGrowthSystem/Layer/Slot 以一致性
from enhanced_city_simulation_v3_5 import ProgressiveGrowthSystem


def main():
    # 读取 v3.5 配置
    with open('configs/city_config_v3_5.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    map_size = config.get('city', {}).get('map_size', [110, 110])
    hubs = config.get('city', {}).get('transport_hubs', [[20, 55], [90, 55], [67, 94]])

    # 初始化地价场（Month 0）
    land = GaussianLandPriceSystem(config)
    land.initialize_system(hubs, map_size)
    land_price_field = land.get_land_price_field()

    # 初始化等值线系统（Month 0）
    iso = IsocontourBuildingSystem(config)
    iso.initialize_system(land_price_field, hubs, map_size, 0, land)

    # 初始化槽位系统（按 v3.5 的规则创建层与槽位，但不运行季度激活逻辑）
    pgs = ProgressiveGrowthSystem(config)
    pgs.initialize_layers(iso, land_price_field)

    # 作图
    os.makedirs('enhanced_simulation_v3_1_output', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('v3.5 Month 0 Slots')
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # 背景：地价场
    im = ax.imshow(
        land_price_field.T,
        origin='lower',
        extent=[0, map_size[0], 0, map_size[1]],
        cmap='Greys',
        alpha=0.35
    )

    # 绘制槽位：商业(蓝)、住宅(绿)
    # 未激活状态下所有层均为 locked，此处仅展示槽位点位
    com_slots = []
    for layer in pgs.layers['commercial']:
        for s in layer.slots:
            com_slots.append((s.pos[0], s.pos[1]))
    if com_slots:
        xs, ys = zip(*com_slots)
        ax.scatter(xs, ys, s=18, c='#2196F3', marker='o', alpha=0.8, label='Commercial slots')

    res_slots = []
    for layer in pgs.layers['residential']:
        for s in layer.slots:
            res_slots.append((s.pos[0], s.pos[1]))
    if res_slots:
        xs, ys = zip(*res_slots)
        ax.scatter(xs, ys, s=18, c='#4CAF50', marker='^', alpha=0.8, label='Residential slots')

    # Hub 标注
    for i, (x, y) in enumerate(hubs):
        ax.scatter([x], [y], c='red', s=40, zorder=5)
        ax.text(x, y + 2.5, f'H{i+1}', color='red', fontsize=9, ha='center')

    ax.legend(loc='upper right')

    out_path = 'enhanced_simulation_v3_1_output/v3_5_month00_slots.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f'✅ Saved: {out_path}')
    # plt.show()


if __name__ == '__main__':
    main()
