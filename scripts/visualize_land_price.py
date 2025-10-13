#!/usr/bin/env python3
"""
可视化地价场（含河岸核/Hub/道路的混合场），支持指定月份。

输出：enhanced_simulation_v4_0_output/land_price_month_MM.png
可选：同时输出组件分图 enhanced_simulation_v4_0_output/land_price_components_month_MM.png
"""

import os
import json
import argparse
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from logic.enhanced_sdf_system import GaussianLandPriceSystem


def read_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def visualize_month(cfg_path: str, month: int, output_dir: str, show_components: bool = True) -> None:
    cfg = read_config(cfg_path)
    city = cfg.get('city', {})
    map_size: List[int] = city.get('map_size', [200, 200])
    hubs = city.get('transport_hubs', [[125, 75], [121, 112]])

    land = GaussianLandPriceSystem(cfg)
    land.initialize_system(hubs, map_size)
    land.update_land_price_field(month)

    field = land.get_land_price_field()
    os.makedirs(output_dir, exist_ok=True)

    # Combined field
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(field, cmap='viridis', origin='upper')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='LP value')
    # overlay hubs
    for hx, hy in hubs:
        ax.plot(hx, hy, 'rx', ms=8, mew=2)
    ax.set_title(f'Land Price Field • Month {month:02d}')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'land_price_month_{month:02d}.png')
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

    if show_components:
        comps = land.get_land_price_components(month)
        names = ['hub_land_price', 'road_land_price', 'river_land_price', 'combined_land_price']
        rows, cols = 2, 2
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        for i, name in enumerate(names):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            comp = comps.get(name)
            if comp is None:
                comp = np.zeros(map_size[::-1], dtype=float)
            im = ax.imshow(comp, cmap='viridis', origin='upper')
            ax.set_title(name)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out2 = os.path.join(output_dir, f'land_price_components_month_{month:02d}.png')
        plt.savefig(out2, dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/city_config_v4_0.json')
    p.add_argument('--output_dir', default='enhanced_simulation_v4_0_output')
    p.add_argument('--month', type=int, required=True)
    p.add_argument('--no_components', action='store_true')
    args = p.parse_args()
    visualize_month(args.config, args.month, args.output_dir, show_components=(not args.no_components))
    print(f'Exported land price visualization for month {args.month:02d} to {args.output_dir}')


if __name__ == '__main__':
    main()


