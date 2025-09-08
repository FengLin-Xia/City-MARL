#!/usr/bin/env python3
"""
检查 Hub3 的等值线生成情况
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem

def check_hub3_contours():
    """检查 Hub3 的等值线生成"""
    print("🔍 检查 Hub3 等值线生成情况...")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取交通枢纽位置
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    print(f"📍 交通枢纽位置: {transport_hubs}")
    print(f"🗺️ 地图尺寸: {map_size}")
    
    # 初始化高斯核地价场系统
    land_price_system = GaussianLandPriceSystem(config)
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 获取地价场
    land_price_field = land_price_system.get_land_price_field()
    
    # 初始化等值线系统
    isocontour_system = IsocontourBuildingSystem(config)
    isocontour_system.initialize_system(land_price_field, transport_hubs, map_size)
    
    # 检查每个 Hub 附近的地价值
    print("\n🎯 Hub 地价值分析:")
    for i, hub in enumerate(transport_hubs):
        x, y = hub[0], hub[1]
        hub_value = land_price_field[y, x]
        print(f"Hub {i+1} ({x}, {y}) 地价值: {hub_value:.3f}")
        
        # 检查周围区域的地价值
        radius = 10
        y_min, y_max = max(0, y-radius), min(map_size[1]-1, y+radius)
        x_min, x_max = max(0, x-radius), min(map_size[0]-1, x+radius)
        local_values = land_price_field[y_min:y_max+1, x_min:x_max+1]
        print(f"  周围区域地价值范围: [{np.min(local_values):.3f}, {np.max(local_values):.3f}]")
    
    # 检查等值线阈值
    print("\n📊 等值线阈值分析:")
    
    # 商业建筑等值线阈值
    commercial_levels = config['isocontour_layout']['commercial']['percentiles']
    print(f"商业建筑等值线阈值: {commercial_levels}")
    
    # 住宅建筑等值线阈值
    residential_levels = config['isocontour_layout']['residential']['percentiles']
    print(f"住宅建筑等值线阈值: {residential_levels}")
    
    # 检查 Hub3 区域的地价值是否达到阈值
    hub3 = transport_hubs[2]  # Hub3
    hub3_x, hub3_y = hub3[0], hub3[1]
    
    print(f"\n🎯 Hub3 ({hub3_x}, {hub3_y}) 详细分析:")
    
    # 检查 Hub3 周围的地价值分布
    radius = 15
    y_min, y_max = max(0, hub3_y-radius), min(map_size[1]-1, hub3_y+radius)
    x_min, x_max = max(0, hub3_x-radius), min(map_size[0]-1, hub3_x+radius)
    hub3_region = land_price_field[y_min:y_max+1, x_min:x_max+1]
    
    print(f"Hub3 区域地价值统计:")
    print(f"  最小值: {np.min(hub3_region):.3f}")
    print(f"  最大值: {np.max(hub3_region):.3f}")
    print(f"  平均值: {np.mean(hub3_region):.3f}")
    print(f"  中位数: {np.median(hub3_region):.3f}")
    
    # 检查是否达到商业建筑阈值
    max_commercial_threshold = max(commercial_levels)
    min_commercial_threshold = min(commercial_levels)
    
    print(f"\n商业建筑阈值检查:")
    print(f"  最高阈值: {max_commercial_threshold:.3f}")
    print(f"  最低阈值: {min_commercial_threshold:.3f}")
    print(f"  Hub3 最大值: {np.max(hub3_region):.3f}")
    
    if np.max(hub3_region) >= min_commercial_threshold:
        print("  ✅ Hub3 区域可以达到商业建筑最低阈值")
    else:
        print("  ❌ Hub3 区域无法达到商业建筑最低阈值")
    
    # 检查是否达到住宅建筑阈值
    max_residential_threshold = max(residential_levels)
    min_residential_threshold = min(residential_levels)
    
    print(f"\n住宅建筑阈值检查:")
    print(f"  最高阈值: {max_residential_threshold:.3f}")
    print(f"  最低阈值: {min_residential_threshold:.3f}")
    print(f"  Hub3 最大值: {np.max(hub3_region):.3f}")
    
    if np.max(hub3_region) >= min_residential_threshold:
        print("  ✅ Hub3 区域可以达到住宅建筑最低阈值")
    else:
        print("  ❌ Hub3 区域无法达到住宅建筑最低阈值")
    
    # 可视化 Hub3 区域
    plt.figure(figsize=(15, 10))
    
    # 1. 整体地价场
    plt.subplot(2, 3, 1)
    plt.imshow(land_price_field, cmap='viridis', aspect='equal')
    plt.colorbar(label='地价值')
    plt.title('整体地价场')
    plt.xlabel('X (像素)')
    plt.ylabel('Y (像素)')
    
    # 标记所有 Hub
    for i, hub in enumerate(transport_hubs):
        plt.plot(hub[0], hub[1], 'ro', markersize=12, label=f'Hub {i+1}')
    plt.legend()
    
    # 2. Hub3 区域放大
    plt.subplot(2, 3, 2)
    plt.imshow(hub3_region, cmap='viridis', aspect='equal')
    plt.colorbar(label='地价值')
    plt.title('Hub3 区域放大')
    plt.xlabel('X (像素)')
    plt.ylabel('Y (像素)')
    
    # 标记 Hub3 中心
    center_x = hub3_x - x_min
    center_y = hub3_y - y_min
    plt.plot(center_x, center_y, 'ro', markersize=12, label='Hub3')
    plt.legend()
    
    # 3. 等值线图
    plt.subplot(2, 3, 3)
    X, Y = np.meshgrid(np.arange(map_size[0]), np.arange(map_size[1]))
    
    # 绘制商业建筑等值线
    for level in commercial_levels:
        plt.contour(X, Y, land_price_field, levels=[level], colors='red', alpha=0.7, linewidths=1)
    
    # 绘制住宅建筑等值线
    for level in residential_levels:
        plt.contour(X, Y, land_price_field, levels=[level], colors='blue', alpha=0.7, linewidths=1)
    
    plt.imshow(land_price_field, cmap='viridis', aspect='equal', alpha=0.3)
    plt.title('等值线分布')
    plt.xlabel('X (像素)')
    plt.ylabel('Y (像素)')
    
    # 标记所有 Hub
    for i, hub in enumerate(transport_hubs):
        plt.plot(hub[0], hub[1], 'ro', markersize=12, label=f'Hub {i+1}')
    plt.legend()
    
    # 4. Hub3 区域等值线
    plt.subplot(2, 3, 4)
    X_hub3, Y_hub3 = np.meshgrid(np.arange(x_min, x_max+1), np.arange(y_min, y_max+1))
    
    # 绘制商业建筑等值线
    for level in commercial_levels:
        plt.contour(X_hub3, Y_hub3, hub3_region, levels=[level], colors='red', alpha=0.7, linewidths=2)
    
    # 绘制住宅建筑等值线
    for level in residential_levels:
        plt.contour(X_hub3, Y_hub3, hub3_region, levels=[level], colors='blue', alpha=0.7, linewidths=2)
    
    plt.imshow(hub3_region, cmap='viridis', aspect='equal', alpha=0.3)
    plt.title('Hub3 区域等值线')
    plt.xlabel('X (像素)')
    plt.ylabel('Y (像素)')
    
    # 标记 Hub3 中心
    plt.plot(center_x, center_y, 'ro', markersize=12, label='Hub3')
    plt.legend()
    
    # 5. 地价值分布直方图
    plt.subplot(2, 3, 5)
    plt.hist(land_price_field.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(max_commercial_threshold, color='red', linestyle='--', label='商业最高阈值')
    plt.axvline(min_commercial_threshold, color='red', linestyle='-', label='商业最低阈值')
    plt.axvline(max_residential_threshold, color='blue', linestyle='--', label='住宅最高阈值')
    plt.axvline(min_residential_threshold, color='blue', linestyle='-', label='住宅最低阈值')
    plt.title('地价值分布')
    plt.xlabel('地价值')
    plt.ylabel('频次')
    plt.legend()
    plt.yscale('log')
    
    # 6. Hub3 区域地价值分布
    plt.subplot(2, 3, 6)
    plt.hist(hub3_region.flatten(), bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(max_commercial_threshold, color='red', linestyle='--', label='商业最高阈值')
    plt.axvline(min_commercial_threshold, color='red', linestyle='-', label='商业最低阈值')
    plt.axvline(max_residential_threshold, color='blue', linestyle='--', label='住宅最高阈值')
    plt.axvline(min_residential_threshold, color='blue', linestyle='-', label='住宅最低阈值')
    plt.title('Hub3 区域地价值分布')
    plt.xlabel('地价值')
    plt.ylabel('频次')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hub3_contour_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Hub3 等值线分析完成！结果已保存到 hub3_contour_analysis.png")

if __name__ == "__main__":
    check_hub3_contours()
