#!/usr/bin/env python3
"""
v3.5 高斯地价场演化可视化
- 支持从 enhanced_simulation_v3_5_output 中读取保存的月度帧
- 或按配置实时计算（不依赖已保存帧）
- 支持静态网格图与GIF动画两种模式
"""

import os
import json
import argparse
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from logic.enhanced_sdf_system import GaussianLandPriceSystem


def parse_months(arg: str, default_total: int = 36) -> List[int]:
    if not arg:
        return list(range(0, default_total + 1))
    if '-' in arg:
        a, b = arg.split('-', 1)
        start = int(a.strip())
        end = int(b.strip())
        return list(range(start, end + 1))
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    return [int(p) for p in parts]


def load_config(config_path: str = 'configs/city_config_v3_5.json') -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_frame_from_disk(output_dir: str, month: int) -> np.ndarray:
    frame_file = os.path.join(output_dir, f"land_price_frame_month_{month:02d}.json")
    if not os.path.exists(frame_file):
        return None
    with open(frame_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    field = np.array(data.get('land_price_field', []), dtype=float)
    if field.size == 0:
        return None
    return field


def compute_field(system: GaussianLandPriceSystem, hubs: List[List[int]], map_size: List[int], month: int) -> np.ndarray:
    # 首次 init 在主逻辑中执行，此处只负责更新
    system.update_land_price_field(month)
    return system.get_land_price_field()


def plot_grid(fields: List[np.ndarray], hubs: List[List[int]], months: List[int], output_dir: str, title: str = 'Gaussian Land Price Evolution (v3.5)'):
    n = len(fields)
    cols = min(6, max(1, int(np.ceil(np.sqrt(n)))))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    vmin, vmax = 0.0, 1.0
    map_h, map_w = fields[0].shape[:2]
    for idx, field in enumerate(fields):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        im = ax.imshow(field, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        for hub in hubs:
            x, y = hub[0], hub[1]
            ax.plot(x, y, 'wo', markersize=3, markeredgecolor='k', markeredgewidth=0.5)
        ax.set_title(f'Month {months[idx]:02d}')
        ax.set_xticks([])
        ax.set_yticks([])
    # 清空多余子图
    for empty_idx in range(n, rows * cols):
        r, c = divmod(empty_idx, cols)
        fig.delaxes(axes[r, c])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'gaussian_evolution_grid.png')
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"[Viz] grid saved: {png_path}")
    plt.close(fig)


def save_animation(fields: List[np.ndarray], hubs: List[List[int]], months: List[int], output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    vmin, vmax = 0.0, 1.0
    map_h, map_w = fields[0].shape[:2]
    im = ax.imshow(fields[0], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    for hub in hubs:
        x, y = hub[0], hub[1]
        ax.plot(x, y, 'wo', markersize=3, markeredgecolor='k', markeredgewidth=0.5)
    txt = ax.text(0.02, 0.98, f'Month {months[0]:02d}', transform=ax.transAxes, color='w',
                  fontsize=12, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        im.set_array(fields[i])
        txt.set_text(f'Month {months[i]:02d}')
        return im, txt

    anim = FuncAnimation(fig, update, frames=len(fields), interval=600, blit=False, repeat=True)
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, 'gaussian_evolution_v3_5.gif')
    try:
        anim.save(gif_path, writer='pillow', fps=2, dpi=120)
        print(f"[Viz] animation saved: {gif_path}")
    except Exception as e:
        print(f"[Viz] animation save failed: {e}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='v3.5 高斯地价场演化可视化')
    parser.add_argument('--mode', choices=['grid', 'anim', 'both'], default='both', help='可视化模式')
    parser.add_argument('--months', type=str, default='0-12', help='月份范围/列表，如 0-12 或 0,3,6,9,12')
    parser.add_argument('--use_saved', action='store_true', help='优先使用输出目录中的已保存帧')
    parser.add_argument('--output_dir', type=str, default='enhanced_simulation_v3_5_output', help='输出目录')
    args = parser.parse_args()

    config = load_config('configs/city_config_v3_5.json')
    city_cfg = config.get('city', {})
    hubs = city_cfg.get('transport_hubs', [[20, 55], [90, 55], [67, 94]])
    map_size = city_cfg.get('map_size', [110, 110])

    months = parse_months(args.months, default_total=config.get('simulation', {}).get('total_months', 36))

    # 初始化地价系统（用于实时计算或组件可视化）
    gp = GaussianLandPriceSystem(config)
    gp.initialize_system(hubs, map_size)

    # 收集字段
    fields = []
    for m in months:
        field = None
        if args.use_saved:
            field = load_frame_from_disk(args.output_dir, m)
        if field is None:
            field = compute_field(gp, hubs, map_size, m)
        fields.append(field)
    print(f"[Viz] collected {len(fields)} frames for months: {months}")

    # 生成可视化
    if args.mode in ('grid', 'both'):
        plot_grid(fields, hubs, months, args.output_dir)
    if args.mode in ('anim', 'both'):
        save_animation(fields, hubs, months, args.output_dir)


if __name__ == '__main__':
    main()


