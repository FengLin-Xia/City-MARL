#!/usr/bin/env python3
"""
v3.5 Month 0 等值线快速可视化
生成 land price 热力图并叠加商业/住宅等值线与三处 Hub 位置
输出: enhanced_simulation_v3_1_output/v3_5_month00_isocontours.png
"""

import json
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem


def _to_xy_points(contour: List) -> np.ndarray:
    """兼容 [[x,y]] 或 [x,y] 两种格式，返回 Nx2 数组"""
    pts = []
    for p in contour:
        if isinstance(p, list) and len(p) == 1:
            x, y = p[0][0], p[0][1]
        else:
            x, y = p[0], p[1]
        pts.append((x, y))
    if not pts:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(pts, dtype=float)


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
    contour_data = iso.get_contour_data_for_visualization()

    # 作图
    os.makedirs('enhanced_simulation_v3_1_output', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('v3.5 Month 0 Isocontours')
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # 地价场热力图
    im = ax.imshow(
        land_price_field.T,  # 转置以使 x→列, y→行 与坐标一致
        origin='lower',
        extent=[0, map_size[0], 0, map_size[1]],
        cmap='viridis',
        alpha=0.6
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Land Price')

    # 绘制商业等值线（蓝）
    commercial_labeled = False
    for contour in contour_data.get('commercial_contours', []):
        pts = _to_xy_points(contour)
        if len(pts) > 1:
            label = 'Commercial' if not commercial_labeled else None
            ax.plot(pts[:, 0], pts[:, 1], color='#2196F3', linewidth=1.6, label=label)
            if not commercial_labeled:
                commercial_labeled = True

    # 绘制住宅等值线（绿）
    residential_labeled = False
    for contour in contour_data.get('residential_contours', []):
        pts = _to_xy_points(contour)
        if len(pts) > 1:
            label = 'Residential' if not residential_labeled else None
            ax.plot(pts[:, 0], pts[:, 1], color='#4CAF50', linewidth=1.2, linestyle='--', label=label)
            if not residential_labeled:
                residential_labeled = True

    # 绘制 Hubs（红）
    for i, (x, y) in enumerate(hubs):
        ax.scatter([x], [y], c='red', s=40, zorder=5)
        ax.text(x, y + 2.5, f'H{i+1}', color='red', fontsize=9, ha='center')

    # 图例（去重）
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l and l not in uniq:
            uniq[l] = h
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc='upper right')

    out_path = 'enhanced_simulation_v3_1_output/v3_5_month00_isocontours.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f'✅ Saved: {out_path}')
    # plt.show()


if __name__ == '__main__':
    main()


