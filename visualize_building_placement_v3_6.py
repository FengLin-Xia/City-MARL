#!/usr/bin/env python3
"""
建筑放置位置逐帧播放可视化 v3.6
- 读取 v3.6 输出目录 enhanced_simulation_v3_6_output
- 解析 building_positions_month_01.json / building_delta_month_*.json
- 叠加 Hubs 与 R(m) 环带，逐月播放
"""

import os
import json
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

OUTPUT_DIR = 'enhanced_simulation_v3_6_output'

TYPE_COLOR = {
    'residential': '#4CAF50',
    'commercial': '#2196F3',
    'industrial': '#FF9800',
    'public': '#9C27B0',
    'other': '#F44336'
}


def load_config() -> Dict:
    # 复用 v3.5 城市配置（包含 map_size 与 hubs）
    cfg_path = 'configs/city_config_v3_5.json'
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return { 
        'city': { 'map_size': [110, 110], 'transport_hubs': [[90, 55], [67, 94]] },
        'terrain_features': { 'rivers': [] }
    }


def detect_max_month() -> int:
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
    pos = b.get('position', b.get('xy', [0, 0]))
    if len(pos) >= 2:
        return {'type': t, 'x': float(pos[0]), 'y': float(pos[1])}
    return None


essential_fields = ['new_buildings', 'reclassified']


def reconstruct_monthly_buildings() -> Dict[int, List[Dict]]:
    result: Dict[int, List[Dict]] = {}
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

    max_m = detect_max_month()
    for m in range(2, max_m + 1):
        prev = list(result.get(m - 1, []))
        delta_path = os.path.join(OUTPUT_DIR, f'building_delta_month_{m:02d}.json')
        if os.path.exists(delta_path):
            with open(delta_path, 'r', encoding='utf-8') as f:
                delta = json.load(f)
            # 新增
            for nb in delta.get('new_buildings', []):
                pb = parse_building_item(nb)
                if pb:
                    prev.append(pb)
            # 重判：根据 id/type 更新上一状态（按位置不变可简化，不在此移除旧位置）
            # 可视化层仅按当月类型着色，不做几何迁移
            for rc in delta.get('reclassified', []):
                # 按位置匹配并更新类型
                pos = rc.get('position', [None, None])
                to_type = str(rc.get('to_type', 'other')).lower()
                if len(pos) >= 2:
                    for it in prev:
                        if abs(it['x'] - float(pos[0])) < 1e-6 and abs(it['y'] - float(pos[1])) < 1e-6:
                            it['type'] = to_type
            result[m] = prev
        else:
            result[m] = prev
    return result


def load_range_state(month: int) -> Dict:
    path = os.path.join(OUTPUT_DIR, f'range_state_month_{month:02d}.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_reclass(month: int) -> Dict:
    path = os.path.join(OUTPUT_DIR, f'debug_reclass_month_{month:02d}.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def animate_play():
    cfg = load_config()
    map_size = cfg.get('city', {}).get('map_size', [110, 110])
    hubs = cfg.get('city', {}).get('transport_hubs', [])
    max_m = detect_max_month()
    if max_m == 0:
        print('未找到 v3.6 输出文件')
        return
    monthly_buildings = reconstruct_monthly_buildings()
    if not monthly_buildings:
        print('未找到 month_01 基线文件')
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('建筑放置位置动态演化 v3.6', fontsize=16, fontweight='bold')
    ax.set_xlim(0, map_size[0]); ax.set_ylim(0, map_size[1]); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel('X'); ax.set_ylabel('Y')

    # 绘制河流边界
    rivers = cfg.get('terrain_features', {}).get('rivers', [])
    for river in rivers:
        if river.get('type') == 'obstacle':
            coordinates = river.get('coordinates', [])
            if coordinates:
                # 提取坐标并绘制多边形
                x_coords = [coord[0] for coord in coordinates]
                y_coords = [coord[1] for coord in coordinates]
                ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='河流')
                ax.fill(x_coords, y_coords, color='lightblue', alpha=0.3)

    # hubs
    hub_circles: List[Tuple[Circle, Circle, Circle]] = []
    for i, (x, y) in enumerate(hubs):
        ax.add_patch(Circle((x, y), 2.5, color='red', alpha=0.9, zorder=10))

    # legend
    legend_handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=8, label=label.capitalize())
                      for label, c in TYPE_COLOR.items() if label != 'other']
    # 添加河流图例
    legend_handles.append(plt.Line2D([0],[0], color='blue', linewidth=2, label='河流'))
    ax.legend(handles=legend_handles, loc='upper right')

    scatter = ax.scatter([], [], s=50, alpha=0.85)
    # 重判叠加层（用小三角/叉号标注）
    reclass_scatter = ax.scatter([], [], s=90, marker='^', edgecolor='black', facecolor='yellow', alpha=0.9, zorder=15)
    reclass_texts = []
    month_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), va='top')

    # 环带可视化：每帧根据 range_state 动态绘制
    ring_patches: List[Circle] = []

    months = list(range(1, max_m + 1))

    def frame_fn(idx):
        # 清理旧环带
        for p in ring_patches:
            p.remove()
        ring_patches.clear()

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

        # 绘制 R(m) 环带（R_prev, R_curr）
        rs = load_range_state(m)
        R_prev = rs.get('R_prev', {})
        R_curr = rs.get('R_curr', {})
        for i, (x, y) in enumerate(hubs):
            # 前一月半径
            rp = float(R_prev.get(f'hub{i+1}', 0.0))
            rc = float(R_curr.get(f'hub{i+1}', 0.0))
            if rc > 0:
                c_outer = Circle((x, y), rc, edgecolor='orange', facecolor='none', lw=1.5, alpha=0.9, zorder=5)
                ax.add_patch(c_outer); ring_patches.append(c_outer)
            if rp > 0 and rc > 0 and rc > rp:
                c_inner = Circle((x, y), rp, edgecolor='gray', facecolor='none', lw=1.0, alpha=0.5, zorder=4)
                ax.add_patch(c_inner); ring_patches.append(c_inner)

        # 重判叠加
        for t in reclass_texts:
            try:
                t.remove()
            except Exception:
                pass
        reclass_texts.clear()
        dbg = load_reclass(m)
        ch = dbg.get('changes', []) if dbg else []
        if ch:
            xs = [float(c.get('position', [0,0])[0]) for c in ch]
            ys = [float(c.get('position', [0,0])[1]) for c in ch]
            reclass_scatter.set_offsets(np.column_stack([xs, ys]))
            # 给每个变化加上简短标签（R->C 或 C->R）
            for c in ch:
                to_t = str(c.get('to_type', ''))
                x, y = c.get('position', [0,0])
                txt = 'R→C' if to_t == 'commercial' else ('C→R' if to_t == 'residential' else '')
                reclass_texts.append(ax.text(x+0.8, y+0.8, txt, fontsize=8, color='black', zorder=16))
        else:
            reclass_scatter.set_offsets(np.empty((0,2)))
        month_text.set_text(f'第 {m} 个月')
        return scatter, reclass_scatter, month_text, *ring_patches, *reclass_texts

    anim = animation.FuncAnimation(fig, frame_fn, frames=list(range(len(months))), interval=800, blit=False, repeat=True)
    out_gif = os.path.join(OUTPUT_DIR, 'building_placement_animation_v3_6.gif')
    anim.save(out_gif, writer='pillow', fps=1.25, dpi=100)
    print(f'动画已保存: {out_gif}')
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    animate_play()
