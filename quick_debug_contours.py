#!/usr/bin/env python3
"""
快速调试等值线提取问题
"""

import json
import numpy as np
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem

def quick_debug():
    """快速调试等值线提取"""
    print("🔍 快速调试等值线提取...")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 初始化系统
    land_price_system = GaussianLandPriceSystem(config)
    isocontour_system = IsocontourBuildingSystem(config)
    
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 测试Month 0
    month = 0
    print(f"\n📊 测试第 {month} 个月:")
    
    # 更新地价场
    land_price_system.update_land_price_field(month, {})
    land_price_field = land_price_system.get_land_price_field()
    
    # 初始化等值线系统
    isocontour_system.initialize_system(land_price_field, transport_hubs, map_size, month, land_price_system)
    
    # 检查活跃Hub
    isocontour_system.current_month = month
    active_hubs = isocontour_system._get_active_hubs()
    print(f"活跃Hub数量: {len(active_hubs)}")
    
    # 获取等值线数据
    contour_data = isocontour_system.get_contour_data_for_visualization()
    
    commercial_contours = contour_data.get('commercial_contours', [])
    residential_contours = contour_data.get('residential_contours', [])
    
    print(f"商业等值线数量: {len(commercial_contours)}")
    print(f"住宅等值线数量: {len(residential_contours)}")
    
    # 检查地价场统计
    max_value = np.max(land_price_field)
    mean_value = np.mean(land_price_field)
    print(f"地价场: 最大值={max_value:.3f}, 平均值={mean_value:.3f}")
    
    # 检查等值线配置
    commercial_config = config['isocontour_layout']['commercial']
    percentiles = commercial_config['percentiles']
    print(f"商业百分位数: {percentiles}")
    
    # 手动计算阈值
    for percentile in percentiles[:3]:  # 只检查前3个
        threshold = np.percentile(land_price_field, percentile)
        print(f"  {percentile}% 阈值: {threshold:.3f}")
        
        # 检查有多少像素超过阈值
        mask = (land_price_field >= threshold)
        count = np.sum(mask)
        print(f"  超过阈值的像素数: {count}")

if __name__ == "__main__":
    quick_debug()


