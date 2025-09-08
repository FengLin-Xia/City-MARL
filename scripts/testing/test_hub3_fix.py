#!/usr/bin/env python3
"""
测试Hub3修复效果
验证修复后的等值线提取是否能正确处理Hub3
"""

import json
import numpy as np
from logic.isocontour_building_system import IsocontourBuildingSystem

def test_hub3_fix():
    """测试Hub3修复效果"""
    
    print("=== 测试Hub3修复效果 ===")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建等值线系统
    isocontour_system = IsocontourBuildingSystem(config)
    
    # 读取地价场数据
    with open('enhanced_simulation_v3_1_output/land_price_frame_month_02.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    land_price_field = np.array(data['land_price_field'])
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    # 初始化等值线系统
    isocontour_system.initialize_system(land_price_field, transport_hubs, map_size)
    
    # 获取等值线数据
    contour_data = isocontour_system.get_contour_data_for_visualization()
    
    print("商业等值线:")
    commercial_contours = contour_data.get('commercial_contours', [])
    print(f"  等值线数量: {len(commercial_contours)}")
    
    for i, contour in enumerate(commercial_contours):
        print(f"  等值线 {i+1}: 长度 {len(contour)}")
        
        # 检查Hub3是否在等值线附近
        hub3_x, hub3_y = 67, 94
        min_distance = float('inf')
        for point in contour:
            x, y = point[0], point[1] if isinstance(point, (list, tuple)) and len(point) >= 2 else (0, 0)
            distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
            min_distance = min(min_distance, distance)
        
        print(f"    到Hub3最小距离: {min_distance:.1f}")
        if min_distance < 30:
            print(f"    ✅ Hub3附近有等值线")
        else:
            print(f"    ❌ Hub3附近没有等值线")
    
    print("\n住宅等值线:")
    residential_contours = contour_data.get('residential_contours', [])
    print(f"  等值线数量: {len(residential_contours)}")
    
    for i, contour in enumerate(residential_contours):
        print(f"  等值线 {i+1}: 长度 {len(contour)}")
        
        # 检查Hub3是否在等值线附近
        hub3_x, hub3_y = 67, 94
        min_distance = float('inf')
        for point in contour:
            x, y = point[0], point[1] if isinstance(point, (list, tuple)) and len(point) >= 2 else (0, 0)
            distance = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
            min_distance = min(min_distance, distance)
        
        print(f"    到Hub3最小距离: {min_distance:.1f}")
        if min_distance < 30:
            print(f"    ✅ Hub3附近有等值线")
        else:
            print(f"    ❌ Hub3附近没有等值线")

if __name__ == "__main__":
    test_hub3_fix()
