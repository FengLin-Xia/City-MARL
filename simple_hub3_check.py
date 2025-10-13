#!/usr/bin/env python3
"""
简单检查 Hub3 问题
"""

import json
import numpy as np
from logic.enhanced_sdf_system import GaussianLandPriceSystem

def simple_hub3_check():
    """简单检查 Hub3 问题"""
    print("🔍 简单检查 Hub3 问题...")
    
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
    
    # 检查每个 Hub 的地价值
    print(f"\n🎯 Hub 地价值检查:")
    for i, hub in enumerate(transport_hubs):
        x, y = hub[0], hub[1]
        hub_value = land_price_field[y, x]
        print(f"Hub {i+1} ({x}, {y}) 地价值: {hub_value:.3f}")
    
    # 检查 Hub3 周围区域
    hub3 = transport_hubs[2]
    hub3_x, hub3_y = hub3[0], hub3[1]
    
    radius = 10
    y_min, y_max = max(0, hub3_y-radius), min(map_size[1]-1, hub3_y+radius)
    x_min, x_max = max(0, hub3_x-radius), min(map_size[0]-1, hub3_x+radius)
    hub3_region = land_price_field[y_min:y_max+1, x_min:x_max+1]
    
    print(f"\n🎯 Hub3 ({hub3_x}, {hub3_y}) 详细分析:")
    print(f"Hub3 区域地价值范围: [{np.min(hub3_region):.3f}, {np.max(hub3_region):.3f}]")
    print(f"Hub3 区域地价值平均值: {np.mean(hub3_region):.3f}")
    
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
    
    print(f"商业建筑阈值范围: [{np.min(commercial_thresholds):.3f}, {np.max(commercial_thresholds):.3f}]")
    print(f"住宅建筑阈值范围: [{np.min(residential_thresholds):.3f}, {np.max(residential_thresholds):.3f}]")
    
    # 检查 Hub3 是否达到任何阈值
    hub3_max = np.max(hub3_region)
    min_commercial = np.min(commercial_thresholds)
    min_residential = np.min(residential_thresholds)
    
    print(f"\n🎯 Hub3 阈值检查:")
    print(f"Hub3 最大值: {hub3_max:.3f}")
    print(f"商业最低阈值: {min_commercial:.3f}")
    print(f"住宅最低阈值: {min_residential:.3f}")
    
    if hub3_max >= min_commercial:
        print("✅ Hub3 可以达到商业建筑阈值")
    else:
        print("❌ Hub3 无法达到商业建筑阈值")
    
    if hub3_max >= min_residential:
        print("✅ Hub3 可以达到住宅建筑阈值")
    else:
        print("❌ Hub3 无法达到住宅建筑阈值")
    
    # 检查 Hub3 是否在地图边缘
    print(f"\n🗺️ Hub3 位置检查:")
    print(f"Hub3 位置: ({hub3_x}, {hub3_y})")
    print(f"地图尺寸: {map_size}")
    print(f"距离左边缘: {hub3_x}")
    print(f"距离右边缘: {map_size[0] - hub3_x}")
    print(f"距离上边缘: {hub3_y}")
    print(f"距离下边缘: {map_size[1] - hub3_y}")
    
    if hub3_x < 20 or hub3_x > map_size[0] - 20:
        print("⚠️ Hub3 可能太靠近左右边缘")
    if hub3_y < 20 or hub3_y > map_size[1] - 20:
        print("⚠️ Hub3 可能太靠近上下边缘")

if __name__ == "__main__":
    simple_hub3_check()
