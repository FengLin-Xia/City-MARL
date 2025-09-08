#!/usr/bin/env python3
"""
分析地价场分布，了解等值线生成问题
"""

import numpy as np
import json
from logic.enhanced_sdf_system import GaussianLandPriceSystem
import cv2

def analyze_land_price_distribution():
    """分析地价场分布"""
    print("🔍 分析地价场分布...")
    
    # 加载配置
    config = json.load(open('configs/city_config_v3_1.json', encoding='utf-8'))
    system = GaussianLandPriceSystem(config)
    
    # 初始化系统
    transport_hubs = [[20, 55], [90, 55]]
    map_size = [110, 110]
    system.initialize_system(transport_hubs, map_size)
    
    # 获取地价场
    field = system.get_land_price_field()
    
    print(f"地价场统计:")
    print(f"  最小值: {np.min(field):.3f}")
    print(f"  最大值: {np.max(field):.3f}")
    print(f"  平均值: {np.mean(field):.3f}")
    print(f"  标准差: {np.std(field):.3f}")
    
    print(f"\n分位数分析:")
    for p in [50, 60, 70, 80, 85, 90, 95]:
        value = np.percentile(field.flatten(), p)
        print(f"  {p}%: {value:.3f}")
    
    # 分析等值线生成
    print(f"\n等值线分析:")
    
    # 商业建筑分位数
    commercial_percentiles = [95, 90, 85]
    commercial_thresholds = [np.percentile(field.flatten(), p) for p in commercial_percentiles]
    
    print(f"商业建筑分位数: {commercial_percentiles}")
    print(f"商业建筑阈值: {[f'{t:.3f}' for t in commercial_thresholds]}")
    
    for i, threshold in enumerate(commercial_thresholds):
        # 创建二值图像
        binary = (field >= threshold).astype(np.uint8) * 255
        
        # 使用OpenCV的findContours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            print(f"  阈值 {threshold:.3f}: {len(contours)} 个轮廓，最大轮廓面积 {area:.1f}，长度 {len(largest_contour)}")
        else:
            print(f"  阈值 {threshold:.3f}: 无轮廓")
    
    # 住宅建筑分位数
    residential_percentiles = [80, 70, 60, 50]
    residential_thresholds = [np.percentile(field.flatten(), p) for p in residential_percentiles]
    
    print(f"\n住宅建筑分位数: {residential_percentiles}")
    print(f"住宅建筑阈值: {[f'{t:.3f}' for t in residential_thresholds]}")
    
    for i, threshold in enumerate(residential_thresholds):
        # 创建二值图像
        binary = (field >= threshold).astype(np.uint8) * 255
        
        # 使用OpenCV的findContours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            print(f"  阈值 {threshold:.3f}: {len(contours)} 个轮廓，最大轮廓面积 {area:.1f}，长度 {len(largest_contour)}")
        else:
            print(f"  阈值 {threshold:.3f}: 无轮廓")

if __name__ == "__main__":
    analyze_land_price_distribution()
