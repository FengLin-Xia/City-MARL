#!/usr/bin/env python3
"""
v3.5 Month 0 建筑放置可视化
- 读取 enhanced_simulation_v3_1_output/building_delta_month_00.json
- 绘制建筑点位（含工业后处理结果），叠加 Hubs 与地价灰度背景
- 输出: enhanced_simulation_v3_1_output/v3_5_month00_buildings.png
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

from logic.enhanced_sdf_system import GaussianLandPriceSystem


COLOR_MAP = {
    'residential': '#4CAF50',
    'commercial': '#2196F3',
    'industrial': '#FF9800',
    'public': '#9C27B0'
}


def main():
    output_dir = 'enhanced_simulation_v3_1_output'
    delta_file = os.path.join(output_dir, 'building_delta_month_00.json')
    if not os.path.exists(delta_file):
        print(f'❌ 未找到 {delta_file}，请先运行 v3.5 模拟')
        return

    with open('configs/city_config_v3_5.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    map_size = config.get('city', {}).get('map_size', [110, 110])
    hubs = config.get('city', {}).get('transport_hubs', [[20, 55], [90, 55], [67, 94]])

    # 背景地价（Month 0）
    land = GaussianLandPriceSystem(config)
    land.initialize_system(hubs, map_size)
    land_price_field = land.get_land_price_field()

    with open(delta_file, 'r', encoding='utf-8') as f:
        delta = json.load(f)
    buildings = delta.get('new_buildings', [])

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('v3.5 Month 0 Buildings')
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # 地价灰度背景
    ax.imshow(
        land_price_field.T,
        origin='lower',
        extent=[0, map_size[0], 0, map_size[1]],
        cmap='Greys',
        alpha=0.35
    )

    # 分类型绘制
    plotted_labels = set()
    for b in buildings:
        t = str(b.get('type', 'unknown')).lower()
        x, y = b.get('position', [0.0, 0.0])
        color = COLOR_MAP.get(t, '#F44336')
        label = None if t in plotted_labels else t.capitalize()
        ax.scatter([x], [y], s=50, c=color, edgecolors='black', linewidths=0.5, alpha=0.85, label=label)
        plotted_labels.add(t)

    # Hubs
    for i, (x, y) in enumerate(hubs):
        ax.scatter([x], [y], c='red', s=40, zorder=5)
        ax.text(x, y + 2.5, f'H{i+1}', color='red', fontsize=9, ha='center')

    if plotted_labels:
        ax.legend(loc='upper right')

    out_path = os.path.join(output_dir, 'v3_5_month00_buildings.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f'✅ Saved: {out_path}')
    # plt.show()


if __name__ == '__main__':
    main()


