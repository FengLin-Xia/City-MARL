#!/usr/bin/env python3
"""
建筑放置位置逐帧播放可视化 v3.5
- 独立读取 v3.5 输出目录 enhanced_simulation_v3_5_output
- 直接解析 building_positions_month_*.json / building_delta_month_*.json
- 简洁 UI：枢纽、建筑按类型着色，逐月播放与统计
"""

import os
import json
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle


OUTPUT_DIR = 'enhanced_simulation_v3_5_output'

TYPE_COLOR = {
    'residential': '#4CAF50',
    'commercial': '#2196F3',
    'industrial': '#FF9800',
    'public': '#9C27B0',
    'other': '#F44336'
}


def detect_max_month() -> int:
    """检测输出目录中能复原到的最大月份（支持 positions + delta）。"""
    if not os.path.isdir(OUTPUT_DIR):
        return 0
    max_m = 0
    for fname in os.listdir(OUTPUT_DIR):
        if fname.startswith('building_positions_month_') and fname.endswith('.json'):
            try:
                m = int(fname.replace('building_positions_month_', '').replace('.json', ''))
                max_m = max(max_m, m)
            except ValueError:
                pass
        if fname.startswith('building_delta_month_') and fname.endswith('.json'):
            try:
                m = int(fname.replace('building_delta_month_', '').replace('.json', ''))
                max_m = max(max_m, m)
            except ValueError:
                pass
    return max_m


def parse_building_item(b: Dict) -> Dict:
    t = str(b.get('type', 'other')).lower()
    pos = b.get('position', [0, 0])
    if len(pos) >= 2:
        return {'type': t, 'x': float(pos[0]), 'y': float(pos[1])}
    return None


def reconstruct_monthly_buildings() -> Dict[int, List[Dict]]:
    """从 month_01 完整文件 + 后续 delta 重建每月的完整建筑列表。"""
    result: Dict[int, List[Dict]] = {}
    # 起点：month 01 完整文件
    base_path = os.path.join(OUTPUT_DIR, 'building_positions_month_01.json')
    if not os.path.exists(base_path):
        return result
    with open(base_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    month_01_buildings: List[Dict] = []
    for b in data.get('buildings', []):
        pb = parse_building_item(b)
        if pb:
            month_01_buildings.append(pb)
    result[1] = month_01_buildings

    # 后续：顺序叠加 delta
    max_m = detect_max_month()
    for m in range(2, max_m + 1):
        prev = list(result.get(m - 1, []))
        delta_path = os.path.join(OUTPUT_DIR, f'building_delta_month_{m:02d}.json')
        if os.path.exists(delta_path):
            with open(delta_path, 'r', encoding='utf-8') as f:
                delta = json.load(f)
            for nb in delta.get('new_buildings', []):
                pb = parse_building_item(nb)
                if pb:
                    prev.append(pb)
        result[m] = prev
    return result


def load_config() -> Dict:
    cfg_path = 'configs/city_config_v3_5.json'
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'city': {
            'map_size': [110, 110],
            'transport_hubs': [[20, 55], [90, 55], [67, 94]]
        }
    }


def animate_play():
    cfg = load_config()
    map_size = cfg.get('city', {}).get('map_size', [110, 110])
    hubs = cfg.get('city', {}).get('transport_hubs', [])
    max_m = detect_max_month()
    if max_m == 0:
        print('❌ 未找到 v3.5 输出文件')
        return
    monthly_buildings = reconstruct_monthly_buildings()
    if not monthly_buildings:
        # 回退：如果只有 month_01，则至少播放一个帧
        month_01 = os.path.join(OUTPUT_DIR, 'building_positions_month_01.json')
        if not os.path.exists(month_01):
            print('❌ 未找到 month_01 完整文件')
            return
        with open(month_01, 'r', encoding='utf-8') as f:
            data = json.load(f)
        bs = []
        for b in data.get('buildings', []):
            pb = parse_building_item(b)
            if pb:
                bs.append(pb)
        monthly_buildings = {1: bs}
        max_m = 1

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('建筑放置位置动态演化 v3.5', fontsize=16, fontweight='bold')
    ax.set_xlim(0, map_size[0]); ax.set_ylim(0, map_size[1]); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel('X'); ax.set_ylabel('Y')

    # hubs
    for i, (x, y) in enumerate(hubs):
        ax.add_patch(Circle((x, y), 3, color='red', alpha=0.8, zorder=10))
        ax.text(x, y-6, f'H{i+1}', ha='center', va='top', fontsize=10, fontweight='bold', color='red')

    # legend
    legend_handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=8, label=label.capitalize())
                      for label, c in TYPE_COLOR.items() if label != 'other']
    ax.legend(handles=legend_handles, loc='upper right')

    scatter = ax.scatter([], [], s=50, alpha=0.8)
    month_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), va='top')

    months = list(range(1, max_m + 1))

    def frame_fn(idx):
        m = months[idx]
        blds = monthly_buildings.get(m, [])
        if blds:
            xs = [b['x'] for b in blds]
            ys = [b['y'] for b in blds]
            cs = [TYPE_COLOR.get(b['type'], TYPE_COLOR['other']) for b in blds]
            scatter.set_offsets(np.column_stack([xs, ys]))
            scatter.set_color(cs)
            scatter.set_sizes([50] * len(blds))
        else:
            scatter.set_offsets(np.empty((0,2)))
            scatter.set_color([])
            scatter.set_sizes([])
        month_text.set_text(f'第 {m} 个月')
        return scatter, month_text

    anim = animation.FuncAnimation(fig, frame_fn, frames=list(range(len(months))), interval=800, blit=False, repeat=True)
    out_gif = os.path.join(OUTPUT_DIR, 'building_placement_animation_v3_5.gif')
    print(f'💾 保存动画到 {out_gif} ...')
    anim.save(out_gif, writer='pillow', fps=1.25, dpi=100)
    print('✅ 动画已保存')
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    animate_play()


