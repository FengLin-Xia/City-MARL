#!/usr/bin/env python3
"""
测试地价场随时间的变化
"""

import numpy as np
import json
from logic.enhanced_sdf_system import GaussianLandPriceSystem
import matplotlib.pyplot as plt

def test_land_price_evolution():
    """测试地价场演化"""
    print("🔍 测试地价场演化...")
    
    # 加载配置
    config = json.load(open('configs/city_config_v3_1.json', encoding='utf-8'))
    system = GaussianLandPriceSystem(config)
    
    # 初始化系统
    transport_hubs = [[20, 55], [90, 55]]
    map_size = [110, 110]
    system.initialize_system(transport_hubs, map_size)
    
    # 测试不同月份的地价场
    months_to_test = [0, 6, 12, 18, 23]
    
    print(f"地价场演化分析:")
    print(f"=" * 60)
    
    for month in months_to_test:
        # 更新地价场
        system.update_land_price_field(month)
        
        # 获取地价场
        field = system.get_land_price_field()
        stats = system.get_land_price_stats()
        
        print(f"月份 {month:2d}:")
        print(f"  最小值: {stats['min']:.3f}")
        print(f"  最大值: {stats['max']:.3f}")
        print(f"  平均值: {stats['mean']:.3f}")
        print(f"  标准差: {stats['std']:.3f}")
        
        # 计算分位数
        percentiles = [50, 60, 70, 80, 85, 90, 95]
        print(f"  分位数:")
        for p in percentiles:
            value = np.percentile(field.flatten(), p)
            print(f"    {p}%: {value:.3f}")
        
        print()
    
    # 分析演化趋势
    print(f"演化趋势分析:")
    print(f"=" * 60)
    
    # 测试连续月份的变化
    all_months = list(range(24))
    min_values = []
    max_values = []
    mean_values = []
    
    for month in all_months:
        system.update_land_price_field(month)
        field = system.get_land_price_field()
        
        min_values.append(np.min(field))
        max_values.append(np.max(field))
        mean_values.append(np.mean(field))
    
    print(f"最小值变化: {min_values[0]:.3f} -> {min_values[-1]:.3f} (变化: {min_values[-1] - min_values[0]:.3f})")
    print(f"最大值变化: {max_values[0]:.3f} -> {max_values[-1]:.3f} (变化: {max_values[-1] - max_values[0]:.3f})")
    print(f"平均值变化: {mean_values[0]:.3f} -> {mean_values[-1]:.3f} (变化: {mean_values[-1] - mean_values[0]:.3f})")
    
    # 检查是否有显著变化
    min_change = abs(min_values[-1] - min_values[0])
    max_change = abs(max_values[-1] - max_values[0])
    mean_change = abs(mean_values[-1] - mean_values[0])
    
    print(f"\n变化幅度评估:")
    if min_change > 0.01:
        print(f"✅ 最小值有显著变化: {min_change:.3f}")
    else:
        print(f"❌ 最小值变化很小: {min_change:.3f}")
    
    if max_change > 0.01:
        print(f"✅ 最大值有显著变化: {max_change:.3f}")
    else:
        print(f"❌ 最大值变化很小: {max_change:.3f}")
    
    if mean_change > 0.01:
        print(f"✅ 平均值有显著变化: {mean_change:.3f}")
    else:
        print(f"❌ 平均值变化很小: {mean_change:.3f}")

if __name__ == "__main__":
    test_land_price_evolution()


