#!/usr/bin/env python3
"""
检查hub峰值和等值线阈值的关系
"""

import json
import numpy as np
from logic.enhanced_sdf_system import GaussianLandPriceSystem

def check_hub_peak_and_thresholds():
    """检查hub峰值和等值线阈值"""
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 初始化地价场系统
    land_price_system = GaussianLandPriceSystem(config)
    
    # 设置地图参数
    map_size = [110, 110]
    transport_hubs = [[20, 55], [90, 55]]
    
    # 初始化地价场
    land_price_system.initialize_system(transport_hubs, map_size)
    land_price_field = land_price_system.get_land_price_field()
    
    # 获取地价场统计
    land_price_stats = land_price_system.get_land_price_stats()
    
    print("🔍 Hub峰值和等值线阈值分析")
    print("=" * 50)
    
    print(f"📊 地价场统计:")
    print(f"   最小值: {land_price_stats['min']:.3f}")
    print(f"   最大值: {land_price_stats['max']:.3f}")
    print(f"   平均值: {land_price_stats['mean']:.3f}")
    print(f"   标准差: {land_price_stats['std']:.3f}")
    
    # 检查hub位置的地价值
    hub_values = []
    for hub in transport_hubs:
        hub_value = land_price_field[hub[1], hub[0]]
        hub_values.append(hub_value)
        print(f"   Hub {hub}: {hub_value:.3f}")
    
    print(f"\n🎯 Hub峰值: {max(hub_values):.3f}")
    
    # 获取等值线配置
    isocontour_config = config.get('isocontour_layout', {})
    commercial_percentiles = isocontour_config.get('commercial', {}).get('fallback_percentiles', [])
    residential_percentiles = isocontour_config.get('residential', {}).get('fallback_percentiles', [])
    
    print(f"\n📈 等值线百分位数:")
    print(f"   商业建筑: {commercial_percentiles}")
    print(f"   住宅建筑: {residential_percentiles}")
    
    # 计算等值线阈值
    print(f"\n🎯 等值线阈值:")
    print(f"   商业建筑:")
    for i, p in enumerate(commercial_percentiles):
        threshold = np.percentile(land_price_field, p)
        print(f"     第{i+1}圈 (P{p}): {threshold:.3f}")
    
    print(f"   住宅建筑:")
    for i, p in enumerate(residential_percentiles):
        threshold = np.percentile(land_price_field, p)
        print(f"     第{i+1}圈 (P{p}): {threshold:.3f}")
    
    # 分析阈值与hub峰值的关系
    hub_peak = max(hub_values)
    print(f"\n🔍 阈值与Hub峰值关系:")
    print(f"   Hub峰值: {hub_peak:.3f}")
    
    print(f"   商业建筑阈值与峰值比例:")
    for i, p in enumerate(commercial_percentiles):
        threshold = np.percentile(land_price_field, p)
        ratio = threshold / hub_peak
        print(f"     第{i+1}圈: {threshold:.3f} ({ratio:.1%} of peak)")
    
    print(f"   住宅建筑阈值与峰值比例:")
    for i, p in enumerate(residential_percentiles):
        threshold = np.percentile(land_price_field, p)
        ratio = threshold / hub_peak
        print(f"     第{i+1}圈: {threshold:.3f} ({ratio:.1%} of peak)")

if __name__ == "__main__":
    check_hub_peak_and_thresholds()


