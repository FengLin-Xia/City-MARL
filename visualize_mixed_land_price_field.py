#!/usr/bin/env python3
"""
混合地价场可视化脚本
展示Hub地价核 + 河流边界地价核的混合效果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from logic.enhanced_sdf_system import GaussianLandPriceSystem

def load_config():
    """加载配置"""
    with open('configs/city_config_v3_5.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def visualize_mixed_land_price_field():
    """可视化混合地价场"""
    config = load_config()
    
    # 创建地价系统
    land_system = GaussianLandPriceSystem(config)
    
    # 初始化系统
    map_size = config.get('city', {}).get('map_size', [110, 110])
    hubs = config.get('city', {}).get('transport_hubs', [[90, 55], [67, 94]])
    
    land_system.initialize_system(hubs, map_size)
    
    # 创建地价场
    land_system._create_land_price_field()
    
    # 获取地价场数据
    land_price_field = land_system.get_land_price_field()
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('混合地价场可视化 - Hub地价核 + 河流边界地价核', fontsize=16, fontweight='bold')
    
    # 1. 原始Hub地价场
    ax1 = axes[0, 0]
    im1 = ax1.imshow(land_price_field, cmap='hot', origin='lower', aspect='equal')
    ax1.set_title('原始Hub地价场')
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    
    # 绘制Hub位置
    for i, (x, y) in enumerate(hubs):
        ax1.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
        ax1.text(x+2, y+2, f'Hub{i+1}', color='white', fontweight='bold', fontsize=10)
    
    # 绘制河流边界
    rivers = config.get('terrain_features', {}).get('rivers', [])
    for river in rivers:
        if river.get('type') == 'obstacle':
            coordinates = river.get('coordinates', [])
            if coordinates:
                x_coords = [coord[0] for coord in coordinates]
                y_coords = [coord[1] for coord in coordinates]
                ax1.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.8)
    
    plt.colorbar(im1, ax=ax1, label='地价值')
    
    # 2. 河流边界地价场
    ax2 = axes[0, 1]
    river_field = np.zeros_like(land_price_field)
    
    # 计算河流边界地价
    for y in range(map_size[1]):
        for x in range(map_size[0]):
            river_price = land_system._calculate_river_land_price(x, y)
            river_field[y, x] = river_price
    
    im2 = ax2.imshow(river_field, cmap='Blues', origin='lower', aspect='equal')
    ax2.set_title('河流边界地价场')
    ax2.set_xlabel('X坐标')
    ax2.set_ylabel('Y坐标')
    
    # 绘制河流边界
    for river in rivers:
        if river.get('type') == 'obstacle':
            coordinates = river.get('coordinates', [])
            if coordinates:
                x_coords = [coord[0] for coord in coordinates]
                y_coords = [coord[1] for coord in coordinates]
                ax2.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.8)
    
    plt.colorbar(im2, ax=ax2, label='河流地价值')
    
    # 3. 混合地价场
    ax3 = axes[1, 0]
    mixed_field = np.zeros_like(land_price_field)
    
    # 计算混合地价
    for y in range(map_size[1]):
        for x in range(map_size[0]):
            hub_price = land_price_field[y, x]
            river_price = land_system._calculate_river_land_price(x, y)
            mixed_field[y, x] = max(hub_price, river_price)
    
    im3 = ax3.imshow(mixed_field, cmap='hot', origin='lower', aspect='equal')
    ax3.set_title('混合地价场 (Hub + 河流)')
    ax3.set_xlabel('X坐标')
    ax3.set_ylabel('Y坐标')
    
    # 绘制Hub位置
    for i, (x, y) in enumerate(hubs):
        ax3.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
        ax3.text(x+2, y+2, f'Hub{i+1}', color='white', fontweight='bold', fontsize=10)
    
    # 绘制河流边界
    for river in rivers:
        if river.get('type') == 'obstacle':
            coordinates = river.get('coordinates', [])
            if coordinates:
                x_coords = [coord[0] for coord in coordinates]
                y_coords = [coord[1] for coord in coordinates]
                ax3.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.8)
    
    plt.colorbar(im3, ax=ax3, label='混合地价值')
    
    # 4. 地价场差异
    ax4 = axes[1, 1]
    diff_field = mixed_field - land_price_field
    im4 = ax4.imshow(diff_field, cmap='RdBu_r', origin='lower', aspect='equal')
    ax4.set_title('地价场差异 (混合 - 原始)')
    ax4.set_xlabel('X坐标')
    ax4.set_ylabel('Y坐标')
    
    # 绘制河流边界
    for river in rivers:
        if river.get('type') == 'obstacle':
            coordinates = river.get('coordinates', [])
            if coordinates:
                x_coords = [coord[0] for coord in coordinates]
                y_coords = [coord[1] for coord in coordinates]
                ax4.plot(x_coords, y_coords, 'k-', linewidth=2, alpha=0.8)
    
    plt.colorbar(im4, ax=ax4, label='地价值差异')
    
    plt.tight_layout()
    plt.savefig('mixed_land_price_field_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出统计信息
    print("=== 地价场统计信息 ===")
    print(f"原始Hub地价场 - 最大值: {np.max(land_price_field):.3f}, 平均值: {np.mean(land_price_field):.3f}")
    print(f"河流边界地价场 - 最大值: {np.max(river_field):.3f}, 平均值: {np.mean(river_field):.3f}")
    print(f"混合地价场 - 最大值: {np.max(mixed_field):.3f}, 平均值: {np.mean(mixed_field):.3f}")
    print(f"地价场差异 - 最大值: {np.max(diff_field):.3f}, 平均值: {np.mean(diff_field):.3f}")

if __name__ == "__main__":
    visualize_mixed_land_price_field()
