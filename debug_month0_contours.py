#!/usr/bin/env python3
"""
调试Month 0的等值线提取和槽位创建
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem

def debug_month0_contours():
    """调试Month 0的等值线和槽位"""
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 初始化地价系统
    land_price_system = GaussianLandPriceSystem(config)
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    # 初始化地价场
    land_price_system.initialize_system(transport_hubs, map_size)
    land_price_field = land_price_system.get_land_price_field()
    
    print("=== Month 0 地价场分析 ===")
    print(f"地价场范围: [{np.min(land_price_field):.3f}, {np.max(land_price_field):.3f}]")
    
    # 检查各Hub位置的地价值
    for i, hub in enumerate(transport_hubs):
        hub_x, hub_y = hub[0], hub[1]
        value = land_price_field[hub_y, hub_x]
        print(f"Hub{i+1} ({hub_x}, {hub_y}): 地价值 = {value:.3f}")
    
    # 检查组件强度
    print("\n=== 组件强度分析 ===")
    for component in ['road', 'hub1', 'hub2', 'hub3']:
        strength = land_price_system._get_component_strength(component, 0)
        print(f"{component}: {strength:.3f}")
    
    # 初始化等值线系统
    isocontour_system = IsocontourBuildingSystem(config)
    isocontour_system.initialize_system(land_price_field, transport_hubs, map_size, 0, land_price_system)
    
    # 获取等值线数据
    contour_data = isocontour_system.get_contour_data_for_visualization()
    
    print("\n=== 等值线分析 ===")
    commercial_contours = contour_data.get('commercial_contours', [])
    residential_contours = contour_data.get('residential_contours', [])
    
    print(f"商业等值线数量: {len(commercial_contours)}")
    for i, contour in enumerate(commercial_contours):
        print(f"  商业等值线 {i+1}: 长度 {len(contour)}")
        # 检查是否接近Hub1/Hub2
        for j, hub in enumerate(transport_hubs[:2]):  # 只检查Hub1和Hub2
            hub_x, hub_y = hub[0], hub[1]
            min_dist = float('inf')
            for point in contour:
                x, y = point[0], point[1]
                dist = np.sqrt((x - hub_x)**2 + (y - hub_y)**2)
                min_dist = min(min_dist, dist)
            print(f"    距离Hub{j+1}: {min_dist:.1f}像素")
    
    print(f"住宅等值线数量: {len(residential_contours)}")
    for i, contour in enumerate(residential_contours):
        print(f"  住宅等值线 {i+1}: 长度 {len(contour)}")
        # 检查是否接近Hub1/Hub2
        for j, hub in enumerate(transport_hubs[:2]):  # 只检查Hub1和Hub2
            hub_x, hub_y = hub[0], hub[1]
            min_dist = float('inf')
            for point in contour:
                x, y = point[0], point[1]
                dist = np.sqrt((x - hub_x)**2 + (y - hub_y)**2)
                min_dist = min(min_dist, dist)
            print(f"    距离Hub{j+1}: {min_dist:.1f}像素")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 地价场
    plt.subplot(1, 3, 1)
    plt.imshow(land_price_field, cmap='viridis', origin='lower')
    plt.colorbar(label='Land Price')
    plt.title('Month 0 地价场')
    for i, hub in enumerate(transport_hubs):
        plt.plot(hub[0], hub[1], 'ro', markersize=8)
        plt.text(hub[0], hub[1], f'Hub{i+1}', color='white', fontsize=10)
    
    # 商业等值线
    plt.subplot(1, 3, 2)
    plt.imshow(land_price_field, cmap='viridis', origin='lower', alpha=0.3)
    for i, contour in enumerate(commercial_contours):
        if len(contour) > 0:
            contour_array = np.array(contour)
            plt.plot(contour_array[:, 0], contour_array[:, 1], 'r-', linewidth=2, label=f'Commercial {i+1}')
    plt.title('商业等值线')
    for i, hub in enumerate(transport_hubs):
        plt.plot(hub[0], hub[1], 'ro', markersize=8)
        plt.text(hub[0], hub[1], f'Hub{i+1}', color='white', fontsize=10)
    
    # 住宅等值线
    plt.subplot(1, 3, 3)
    plt.imshow(land_price_field, cmap='viridis', origin='lower', alpha=0.3)
    for i, contour in enumerate(residential_contours):
        if len(contour) > 0:
            contour_array = np.array(contour)
            plt.plot(contour_array[:, 0], contour_array[:, 1], 'b-', linewidth=2, label=f'Residential {i+1}')
    plt.title('住宅等值线')
    for i, hub in enumerate(transport_hubs):
        plt.plot(hub[0], hub[1], 'ro', markersize=8)
        plt.text(hub[0], hub[1], f'Hub{i+1}', color='white', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('debug_month0_contours.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到 debug_month0_contours.png")

if __name__ == "__main__":
    debug_month0_contours()


