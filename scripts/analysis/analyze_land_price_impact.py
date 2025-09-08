#!/usr/bin/env python3
"""
分析地价场变化对建筑放置的影响
"""

import numpy as np
import json
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem
import cv2

def analyze_land_price_impact():
    """分析地价场变化对建筑放置的影响"""
    print("🔍 分析地价场变化对建筑放置的影响...")
    
    # 加载配置
    config = json.load(open('configs/city_config_v3_1.json', encoding='utf-8'))
    land_price_system = GaussianLandPriceSystem(config)
    isocontour_system = IsocontourBuildingSystem(config)
    
    # 初始化系统
    transport_hubs = [[20, 55], [90, 55]]
    map_size = [110, 110]
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 测试不同月份的地价场对等值线的影响
    months_to_test = [0, 6, 12, 18, 23]
    
    print(f"地价场变化对等值线的影响分析:")
    print(f"=" * 80)
    
    for month in months_to_test:
        # 更新地价场
        land_price_system.update_land_price_field(month)
        field = land_price_system.get_land_price_field()
        
        # 初始化等值线系统
        isocontour_system.initialize_system(field, transport_hubs, map_size)
        
        # 获取等值线数据
        contour_data = isocontour_system.get_contour_data_for_visualization()
        
        print(f"月份 {month:2d}:")
        
        # 分析商业等值线
        commercial_contours = contour_data.get('commercial_contours', [])
        commercial_percentiles = contour_data.get('commercial_percentiles', [])
        
        print(f"  商业等值线:")
        print(f"    分位数: {commercial_percentiles}")
        for i, contour in enumerate(commercial_contours):
            if len(contour) > 20:  # 只显示有效等值线
                print(f"    等值线 {i+1}: 长度 {len(contour)}")
        
        # 分析住宅等值线
        residential_contours = contour_data.get('residential_contours', [])
        residential_percentiles = contour_data.get('residential_percentiles', [])
        
        print(f"  住宅等值线:")
        print(f"    分位数: {residential_percentiles}")
        for i, contour in enumerate(residential_contours):
            if len(contour) > 20:  # 只显示有效等值线
                print(f"    等值线 {i+1}: 长度 {len(contour)}")
        
        print()
    
    # 分析建筑更新机制
    print(f"建筑更新机制分析:")
    print(f"=" * 80)
    
    print(f"1. 滞后替代系统:")
    print(f"   - 状态: {'已实现但简化' if hasattr(land_price_system, 'hysteresis_system') else '未实现'}")
    print(f"   - 功能: 住宅建筑转换为商业建筑")
    print(f"   - 触发条件: 地价变化、经济评分")
    
    print(f"\n2. 等值线重新初始化:")
    print(f"   - 频率: 每年一次")
    print(f"   - 触发: 地价场更新后")
    print(f"   - 影响: 新的等值线分布")
    
    print(f"\n3. 槽位系统:")
    print(f"   - 状态: 已实现")
    print(f"   - 特点: 冻结施工线，严格逐层满格")
    print(f"   - 限制: 一旦层被激活，槽位位置固定")
    
    # 分析影响程度
    print(f"\n影响程度评估:")
    print(f"=" * 80)
    
    # 计算地价场变化幅度
    land_price_system.update_land_price_field(0)
    field_0 = land_price_system.get_land_price_field()
    
    land_price_system.update_land_price_field(23)
    field_23 = land_price_system.get_land_price_field()
    
    # 计算变化
    mean_change = np.mean(field_23 - field_0)
    max_change = np.max(field_23 - field_0)
    std_change = np.std(field_23 - field_0)
    
    print(f"地价场变化统计:")
    print(f"  平均变化: {mean_change:.3f}")
    print(f"  最大变化: {max_change:.3f}")
    print(f"  变化标准差: {std_change:.3f}")
    
    # 评估影响
    if mean_change > 0.1:
        impact_level = "高"
    elif mean_change > 0.05:
        impact_level = "中"
    else:
        impact_level = "低"
    
    print(f"\n对建筑放置的影响评估:")
    print(f"  影响程度: {impact_level}")
    print(f"  主要影响:")
    print(f"    - 等值线位置变化: {'显著' if mean_change > 0.05 else '轻微'}")
    print(f"    - 新建筑选址: {'受影响' if mean_change > 0.05 else '基本不受影响'}")
    print(f"    - 现有建筑: {'可能转换' if hasattr(land_price_system, 'hysteresis_system') else '位置固定'}")
    
    # 分析问题
    print(f"\n当前问题分析:")
    print(f"=" * 80)
    
    print(f"1. 槽位系统限制:")
    print(f"   - 问题: 槽位位置在层激活时固定，不随地价场变化")
    print(f"   - 影响: 地价场变化对已激活层的建筑位置无影响")
    
    print(f"\n2. 滞后替代简化:")
    print(f"   - 问题: v3.1中的滞后替代系统被简化")
    print(f"   - 影响: 缺乏住宅到商业的转换机制")
    
    print(f"\n3. 等值线重新初始化:")
    print(f"   - 问题: 每年重新初始化等值线，但槽位系统不更新")
    print(f"   - 影响: 新等值线无法影响现有建筑分布")

if __name__ == "__main__":
    analyze_land_price_impact()


