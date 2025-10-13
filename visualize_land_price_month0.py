#!/usr/bin/env python3
"""
可视化第一个月（月份0）的地价场
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Set

from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.v4_enumeration import SlotNode


def read_config(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading config: {e}")
        return {}


def load_slots_from_points_file(points_file: str, map_size: List[int]) -> Dict[str, SlotNode]:
    """从简单文本点集文件加载槽位（每行含两个坐标即可，允许夹杂其他数字）。
    - 保留浮点坐标精度
    """
    if not os.path.exists(points_file):
        raise FileNotFoundError(f"points_file not found: {points_file}")
    W, H = int(map_size[0]), int(map_size[1])
    nodes: Dict[str, SlotNode] = {}
    seen: Set[Tuple[float, float]] = set()
    with open(points_file, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) < 2:
                continue
            try:
                xf = float(nums[0]); yf = float(nums[1])
            except Exception:
                continue
            # 画布范围判断（保留浮点）
            if xf < 0.0 or yf < 0.0 or xf >= float(W) or yf >= float(H):
                continue
            key = (xf, yf)
            if key in seen:
                continue
            seen.add(key)
            sid = f"s_{len(nodes)}"
            nodes[sid] = SlotNode(slot_id=sid, x=int(round(xf)), y=int(round(yf)), fx=xf, fy=yf)
    return nodes


def get_candidate_slots(slots: Dict[str, SlotNode], hubs: List[List[float]], month: int, R0: float, dR: float) -> Set[str]:
    """获取当月候选槽位集合（累计模式：距离 <= R_curr）"""
    R_curr = R0 + dR * month
    cand: Set[str] = set()
    for sid, n in slots.items():
        x = float(getattr(n, 'fx', n.x))
        y = float(getattr(n, 'fy', n.y))
        # 计算到最近枢纽的距离
        min_dist = min([((x - float(hx))**2 + (y - float(hy))**2)**0.5 for hx, hy in hubs])
        if min_dist <= (R_curr + 1.0):  # tol=1.0
            cand.add(sid)
    return cand


def visualize_land_price_field(land_price_system, month=0, output_path='land_price_month0.png',
                               candidate_coords=None, chosen_coords=None):
    """可视化地价场"""
    
    # 获取地价场数据
    field = land_price_system.land_price_field
    
    if field is None:
        print("地价场数据为空")
        return
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：热力图
    ax1 = axes[0]
    
    # 使用自定义colormap：蓝色（低）-> 绿色 -> 黄色 -> 红色（高）
    colors = ['#2b1055', '#1e4d8b', '#2e8b57', '#ffd700', '#ff6347', '#ff0000']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('land_price', colors, N=n_bins)
    
    im1 = ax1.imshow(field, origin='lower', cmap=cmap, interpolation='bilinear')
    ax1.set_title(f'地价场热力图 - 月份 {month}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X 坐标 (像素)', fontsize=11)
    ax1.set_ylabel('Y 坐标 (像素)', fontsize=11)
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('归一化地价值', fontsize=11)
    
    # 标记候选槽位
    if candidate_coords:
        cand_xs = [c[0] for c in candidate_coords]
        cand_ys = [c[1] for c in candidate_coords]
        ax1.scatter(cand_xs, cand_ys, c='cyan', s=8, alpha=0.4, 
                   label=f'候选槽位 ({len(candidate_coords)})', zorder=5)
    
    # 标记选中的建筑位置
    if chosen_coords:
        chosen_xs = [c[0] for c in chosen_coords]
        chosen_ys = [c[1] for c in chosen_coords]
        ax1.scatter(chosen_xs, chosen_ys, c='lime', s=150, marker='s', 
                   edgecolors='white', linewidths=2, 
                   label=f'选中位置 ({len(chosen_coords)})', zorder=9)
    
    # 标记交通枢纽
    hubs = land_price_system.transport_hubs
    if hubs:
        hub_xs = [h[0] for h in hubs]
        hub_ys = [h[1] for h in hubs]
        ax1.scatter(hub_xs, hub_ys, c='magenta', s=200, marker='*', 
                   edgecolors='white', linewidths=2, label='交通枢纽', zorder=10)
        
        # 标注枢纽编号
        for i, (hx, hy) in enumerate(hubs):
            ax1.annotate(f'Hub{i+1}', xy=(hx, hy), xytext=(5, 5), 
                        textcoords='offset points', color='white', 
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='magenta', alpha=0.7))
    
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 右图：等高线图
    ax2 = axes[1]
    
    # 创建等高线
    H, W = field.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    
    # 填充等高线
    contourf = ax2.contourf(X, Y, field, levels=20, cmap=cmap, alpha=0.8)
    
    # 添加等高线
    contour = ax2.contour(X, Y, field, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    ax2.set_title(f'地价场等高线图 - 月份 {month}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X 坐标 (像素)', fontsize=11)
    ax2.set_ylabel('Y 坐标 (像素)', fontsize=11)
    
    # 添加颜色条
    cbar2 = plt.colorbar(contourf, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('归一化地价值', fontsize=11)
    
    # 标记候选槽位
    if candidate_coords:
        ax2.scatter(cand_xs, cand_ys, c='cyan', s=8, alpha=0.4, 
                   label=f'候选槽位 ({len(candidate_coords)})', zorder=5)
    
    # 标记选中的建筑位置
    if chosen_coords:
        ax2.scatter(chosen_xs, chosen_ys, c='lime', s=150, marker='s', 
                   edgecolors='white', linewidths=2, 
                   label=f'选中位置 ({len(chosen_coords)})', zorder=9)
    
    # 标记交通枢纽
    if hubs:
        ax2.scatter(hub_xs, hub_ys, c='magenta', s=200, marker='*', 
                   edgecolors='white', linewidths=2, label='交通枢纽', zorder=10)
        for i, (hx, hy) in enumerate(hubs):
            ax2.annotate(f'Hub{i+1}', xy=(hx, hy), xytext=(5, 5), 
                        textcoords='offset points', color='white', 
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='magenta', alpha=0.7))
    
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"地价场可视化已保存至: {output_path}")
    
    # 显示图片
    plt.show()
    
    # 打印统计信息
    print(f"\n地价场统计信息（月份 {month}）:")
    print(f"  最小值: {np.min(field):.4f}")
    print(f"  最大值: {np.max(field):.4f}")
    print(f"  平均值: {np.mean(field):.4f}")
    print(f"  中位数: {np.median(field):.4f}")
    print(f"  标准差: {np.std(field):.4f}")
    
    # 计算不同地价区间的面积占比
    total_pixels = field.size
    ranges = [
        (0.0, 0.2, "低价区 (0.0-0.2)"),
        (0.2, 0.4, "中低价区 (0.2-0.4)"),
        (0.4, 0.6, "中价区 (0.4-0.6)"),
        (0.6, 0.8, "中高价区 (0.6-0.8)"),
        (0.8, 1.0, "高价区 (0.8-1.0)")
    ]
    
    print(f"\n地价区间分布:")
    for low, high, label in ranges:
        count = np.sum((field >= low) & (field < high))
        percentage = (count / total_pixels) * 100
        print(f"  {label}: {percentage:.2f}%")


def main():
    # 读取配置
    cfg = read_config('configs/city_config_v4_0.json')
    
    # 获取基本参数
    city = cfg.get('city', {})
    map_size = city.get('map_size', [200, 200])
    hubs = city.get('transport_hubs', [[125, 75], [112, 121]])
    
    v4 = cfg.get('growth_v4_0', {})
    hubs_cfg = v4.get('hubs', {})
    R0 = 15.0
    dR = 2.0
    if hubs_cfg and isinstance(hubs_cfg.get('list', []), list) and len(hubs_cfg['list']) > 0:
        h0 = hubs_cfg['list'][0]
        R0 = float(h0.get('R0', 15.0))
        dR = float(h0.get('dR', 2.0))
    
    print("=" * 60)
    print("可视化第一个月（月份0）的地价场")
    print("=" * 60)
    print(f"地图大小: {map_size}")
    print(f"交通枢纽: {hubs}")
    print(f"候选范围参数: R0={R0}, dR={dR}")
    print()
    
    # 加载槽位数据
    slots_source = v4.get('slots', {}).get('path', 'slotpoints.txt')
    print(f"加载槽位文件: {slots_source}")
    slots = load_slots_from_points_file(slots_source, map_size)
    print(f"总槽位数: {len(slots)}")
    
    # 计算第0个月的候选槽位
    month = 0
    candidate_ids = get_candidate_slots(slots, hubs, month, R0, dR)
    print(f"月份 {month} 候选槽位数: {len(candidate_ids)}")
    
    # 提取候选槽位坐标
    candidate_coords = []
    for sid in candidate_ids:
        n = slots.get(sid)
        if n:
            x = float(getattr(n, 'fx', n.x))
            y = float(getattr(n, 'fy', n.y))
            candidate_coords.append((x, y))
    
    # 读取chosen_sequence
    chosen_file = f'enhanced_simulation_v4_0_output/v4_debug/chosen_sequence_month_{month:02d}.json'
    chosen_coords = []
    if os.path.exists(chosen_file):
        with open(chosen_file, 'r', encoding='utf-8') as f:
            chosen_data = json.load(f)
            actions = chosen_data.get('actions', [])
            print(f"选中的建筑数量: {len(actions)}")
            for action in actions:
                footprint_slots = action.get('footprint_slots', [])
                for sid in footprint_slots:
                    n = slots.get(sid)
                    if n:
                        x = float(getattr(n, 'fx', n.x))
                        y = float(getattr(n, 'fy', n.y))
                        chosen_coords.append((x, y))
    else:
        print(f"警告: 未找到chosen_sequence文件: {chosen_file}")
    
    print()
    
    # 初始化地价系统
    land_price_system = GaussianLandPriceSystem(cfg)
    land_price_system.initialize_system(hubs, map_size)
    
    # 更新到第0个月（初始化状态）
    buildings = {'public': [], 'industrial': []}
    land_price_system.update_land_price_field(0, buildings)
    
    # 创建输出目录
    output_dir = 'enhanced_simulation_v4_0_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化
    output_path = os.path.join(output_dir, 'land_price_month_0_visualization.png')
    visualize_land_price_field(land_price_system, month=0, output_path=output_path,
                               candidate_coords=candidate_coords, chosen_coords=chosen_coords)


if __name__ == '__main__':
    main()

