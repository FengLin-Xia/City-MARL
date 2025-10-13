#!/usr/bin/env python3
"""
调试 Hub3 问题
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem

def debug_hub3():
    """调试 Hub3 问题"""
    print("🔍 调试 Hub3 问题...")
    
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
    
    # 检查 Hub3 的地价值
    hub3 = transport_hubs[2]
    hub3_x, hub3_y = hub3[0], hub3[1]
    hub3_value = land_price_field[hub3_y, hub3_x]
    
    print(f"\n🎯 Hub3 ({hub3_x}, {hub3_y}) 地价值: {hub3_value:.3f}")
    
    # 检查 Hub3 周围区域
    radius = 10
    y_min, y_max = max(0, hub3_y-radius), min(map_size[1]-1, hub3_y+radius)
    x_min, x_max = max(0, hub3_x-radius), min(map_size[0]-1, hub3_x+radius)
    hub3_region = land_price_field[y_min:y_max+1, x_min:x_max+1]
    
    print(f"Hub3 区域地价值范围: [{np.min(hub3_region):.3f}, {np.max(hub3_region):.3f}]")
    
    # 检查等值线阈值
    commercial_percentiles = config['isocontour_layout']['commercial']['percentiles']
    residential_percentiles = config['isocontour_layout']['residential']['percentiles']
    
    print(f"\n📊 等值线阈值:")
    print(f"商业建筑分位数: {commercial_percentiles}")
    print(f"住宅建筑分位数: {residential_percentiles}")
    
    # 计算实际阈值
    sdf_flat = land_price_field.flatten()
    commercial_thresholds = np.percentile(sdf_flat, commercial_percentiles)
    residential_thresholds = np.percentile(sdf_flat, residential_percentiles)
    
    print(f"商业建筑阈值: {[f'{t:.3f}' for t in commercial_thresholds]}")
    print(f"住宅建筑阈值: {[f'{t:.3f}' for t in residential_thresholds]}")
    
    # 检查 Hub3 是否达到任何阈值
    max_commercial_threshold = np.max(commercial_thresholds)
    min_commercial_threshold = np.min(commercial_thresholds)
    max_residential_threshold = np.max(residential_thresholds)
    min_residential_threshold = np.min(residential_thresholds)
    
    print(f"\n🎯 Hub3 阈值检查:")
    print(f"Hub3 最大值: {np.max(hub3_region):.3f}")
    print(f"商业阈值范围: [{min_commercial_threshold:.3f}, {max_commercial_threshold:.3f}]")
    print(f"住宅阈值范围: [{min_residential_threshold:.3f}, {max_residential_threshold:.3f}]")
    
    if np.max(hub3_region) >= min_commercial_threshold:
        print("✅ Hub3 可以达到商业建筑阈值")
    else:
        print("❌ Hub3 无法达到商业建筑阈值")
    
    if np.max(hub3_region) >= min_residential_threshold:
        print("✅ Hub3 可以达到住宅建筑阈值")
    else:
        print("❌ Hub3 无法达到住宅建筑阈值")
    
    # 测试等值线提取
    print(f"\n🔍 测试等值线提取:")
    
    # 初始化等值线系统
    isocontour_system = IsocontourBuildingSystem(config)
    isocontour_system.initialize_system(land_price_field, transport_hubs, map_size)
    
    # 测试商业建筑等值线
    commercial_contours = isocontour_system._extract_equidistant_contours(commercial_percentiles, 'commercial')
    print(f"商业建筑等值线数量: {len(commercial_contours)}")
    
    # 测试住宅建筑等值线
    residential_contours = isocontour_system._extract_equidistant_contours(residential_percentiles, 'residential')
    print(f"住宅建筑等值线数量: {len(residential_contours)}")
    
    # 检查等值线是否包含 Hub3 区域
    print(f"\n🎯 检查等值线是否包含 Hub3:")
    
    for i, contour in enumerate(commercial_contours):
        # 检查等值线是否在 Hub3 附近
        hub3_nearby = False
        for point in contour:
            dist = np.sqrt((point[0] - hub3_x)**2 + (point[1] - hub3_y)**2)
            if dist < 15:  # 15像素范围内
                hub3_nearby = True
                break
        
        if hub3_nearby:
            print(f"✅ 商业等值线 {i+1} 包含 Hub3 区域")
        else:
            print(f"❌ 商业等值线 {i+1} 不包含 Hub3 区域")
    
    for i, contour in enumerate(residential_contours):
        # 检查等值线是否在 Hub3 附近
        hub3_nearby = False
        for point in contour:
            dist = np.sqrt((point[0] - hub3_x)**2 + (point[1] - hub3_y)**2)
            if dist < 15:  # 15像素范围内
                hub3_nearby = True
                break
        
        if hub3_nearby:
            print(f"✅ 住宅等值线 {i+1} 包含 Hub3 区域")
        else:
            print(f"❌ 住宅等值线 {i+1} 不包含 Hub3 区域")
    
    # 可视化
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
    for i, threshold in enumerate(commercial_thresholds):
        plt.contour(X, Y, land_price_field, levels=[threshold], colors='red', alpha=0.7, linewidths=1)
    
    # 绘制住宅建筑等值线
    for i, threshold in enumerate(residential_thresholds):
        plt.contour(X, Y, land_price_field, levels=[threshold], colors='blue', alpha=0.7, linewidths=1)
    
    plt.imshow(land_price_field, cmap='viridis', aspect='equal', alpha=0.3)
    plt.title('等值线分布')
    plt.xlabel('X (像素)')
    plt.ylabel('Y (像素)')
    
    # 标记所有 Hub
    for i, hub in enumerate(transport_hubs):
        plt.plot(hub[0], hub[1], 'ro', markersize=12, label=f'Hub {i+1}')
    plt.legend()
    
    # 4. 商业等值线
    plt.subplot(2, 3, 4)
    plt.imshow(land_price_field, cmap='viridis', aspect='equal', alpha=0.3)
    for i, contour in enumerate(commercial_contours):
        if contour:
            contour_x = [p[0] for p in contour]
            contour_y = [p[1] for p in contour]
            plt.plot(contour_x, contour_y, 'r-', linewidth=2, label=f'商业等值线 {i+1}')
    plt.title('商业等值线')
    plt.xlabel('X (像素)')
    plt.ylabel('Y (像素)')
    
    # 标记所有 Hub
    for i, hub in enumerate(transport_hubs):
        plt.plot(hub[0], hub[1], 'ro', markersize=12, label=f'Hub {i+1}')
    plt.legend()
    
    # 5. 住宅等值线
    plt.subplot(2, 3, 5)
    plt.imshow(land_price_field, cmap='viridis', aspect='equal', alpha=0.3)
    for i, contour in enumerate(residential_contours):
        if contour:
            contour_x = [p[0] for p in contour]
            contour_y = [p[1] for p in contour]
            plt.plot(contour_x, contour_y, 'b-', linewidth=2, label=f'住宅等值线 {i+1}')
    plt.title('住宅等值线')
    plt.xlabel('X (像素)')
    plt.ylabel('Y (像素)')
    
    # 标记所有 Hub
    for i, hub in enumerate(transport_hubs):
        plt.plot(hub[0], hub[1], 'ro', markersize=12, label=f'Hub {i+1}')
    plt.legend()
    
    # 6. 地价值分布
    plt.subplot(2, 3, 6)
    plt.hist(land_price_field.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(max_commercial_threshold, color='red', linestyle='--', label='商业最高阈值')
    plt.axvline(min_commercial_threshold, color='red', linestyle='-', label='商业最低阈值')
    plt.axvline(max_residential_threshold, color='blue', linestyle='--', label='住宅最高阈值')
    plt.axvline(min_residential_threshold, color='blue', linestyle='-', label='住宅最低阈值')
    plt.axvline(np.max(hub3_region), color='orange', linestyle=':', linewidth=3, label='Hub3 最大值')
    plt.title('地价值分布')
    plt.xlabel('地价值')
    plt.ylabel('频次')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('hub3_debug_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Hub3 调试完成！结果已保存到 hub3_debug_result.png")

if __name__ == "__main__":
    debug_hub3()
