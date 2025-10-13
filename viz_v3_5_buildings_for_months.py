#!/usr/bin/env python3
"""
v3.5 指定月份建筑放置可视化（支持 Month 1 / 2 重建）
- 从 month_01 完整文件 + 后续增量文件重建目标月份
- 绘制建筑点位（含已后处理的工业类型）、叠加 Hubs 与地价灰度背景
- 输出: enhanced_simulation_v3_1_output/v3_5_monthXX_buildings.png
"""

import json
import os
from typing import List, Dict
import argparse
import matplotlib.pyplot as plt

from logic.enhanced_sdf_system import GaussianLandPriceSystem


COLOR_MAP = {
    'residential': '#4CAF50',
    'commercial': '#2196F3',
    'industrial': '#FF9800',
    'public': '#9C27B0'
}


def reconstruct_buildings_at_month(output_dir: str, target_month: int) -> List[Dict]:
    full_file = os.path.join(output_dir, 'building_positions_month_01.json')
    buildings: List[Dict] = []
    if not os.path.exists(full_file):
        return buildings
    with open(full_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        buildings = data.get('buildings', [])[:]
    for m in range(2, target_month + 1):
        delta_file = os.path.join(output_dir, f'building_delta_month_{m:02d}.json')
        if os.path.exists(delta_file):
            with open(delta_file, 'r', encoding='utf-8') as f:
                delta = json.load(f)
                buildings.extend(delta.get('new_buildings', []))
    return buildings


def visualize_month(month: int):
    output_dir = 'enhanced_simulation_v3_1_output'
    with open('configs/city_config_v3_5.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    map_size = config.get('city', {}).get('map_size', [110, 110])
    hubs = config.get('city', {}).get('transport_hubs', [[20, 55], [90, 55], [67, 94]])

    # 背景地价（用对应月份地价场）
    land = GaussianLandPriceSystem(config)
    land.initialize_system(hubs, map_size)
    if month > 0:
        land.update_land_price_field(month)
    land_price_field = land.get_land_price_field()

    # 重建目标月份建筑集
    buildings = reconstruct_buildings_at_month(output_dir, month)

    # 可视化
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f'v3.5 Month {month} Buildings')
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
    plotted = set()
    for b in buildings:
        t = str(b.get('type', 'unknown')).lower()
        x, y = b.get('position', [0.0, 0.0])
        color = COLOR_MAP.get(t, '#F44336')
        label = None if t in plotted else t.capitalize()
        ax.scatter([x], [y], s=50, c=color, edgecolors='black', linewidths=0.5, alpha=0.85, label=label)
        plotted.add(t)

    # Hubs
    for i, (x, y) in enumerate(hubs):
        ax.scatter([x], [y], c='red', s=40, zorder=5)
        ax.text(x, y + 2.5, f'H{i+1}', color='red', fontsize=9, ha='center')

    if plotted:
        ax.legend(loc='upper right')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'v3_5_month{month:02d}_buildings.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f'✅ Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize v3.5 buildings for specific months')
    parser.add_argument('--months', '-m', nargs='*', help='Months to visualize, e.g., 1 2 3 or 1,2,3')
    args = parser.parse_args()

    months: List[int]
    if args.months:
        # 支持空格分隔或逗号分隔
        raw = []
        for token in args.months:
            raw.extend(token.split(','))
        months = []
        for t in raw:
            t = t.strip()
            if t:
                try:
                    months.append(int(t))
                except ValueError:
                    pass
        if not months:
            months = [1, 2]
    else:
        months = [1, 2]

    for m in months:
        visualize_month(m)


if __name__ == '__main__':
    main()


