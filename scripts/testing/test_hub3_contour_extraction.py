#!/usr/bin/env python3
"""
测试Hub3等值线提取问题
验证Hub3的等值线是否被正确提取
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def test_hub3_contour_extraction():
    """测试Hub3等值线提取"""
    
    # Hub3位置
    hub3_x, hub3_y = 67, 94
    hub3_pos = (hub3_x, hub3_y)
    
    print("=== Hub3 等值线提取测试 ===")
    print(f"Hub3 位置: {hub3_pos}")
    
    # 读取地价场数据
    try:
        with open('enhanced_simulation_v3_1_output/land_price_frame_month_02.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        land_price_field = np.array(data['land_price_field'])
        print(f"地价场形状: {land_price_field.shape}")
        
        # 计算百分位数阈值
        percentiles = [99, 95, 90, 80]
        percentile_values = {}
        for p in percentiles:
            percentile_values[p] = np.percentile(land_price_field, p)
        
        print("百分位数阈值:")
        for p in percentiles:
            print(f"  {p}%: {percentile_values[p]:.4f}")
        
        # 测试每个百分位数的等值线提取
        for percentile in percentiles:
            threshold = percentile_values[percentile]
            print(f"\n--- 测试 {percentile}% 等值线 (阈值: {threshold:.4f}) ---")
            
            # 创建二值化图像
            binary = (land_price_field >= threshold).astype(np.uint8) * 255
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"找到轮廓数量: {len(contours)}")
            
            if contours:
                # 分析每个轮廓
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    # 检查Hub3是否在轮廓内
                    inside = cv2.pointPolygonTest(contour, hub3_pos, False)
                    
                    # 计算到Hub3的最小距离
                    min_dist = float('inf')
                    for point in contour:
                        x, y = point[0][0], point[0][1]
                        dist = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
                        min_dist = min(min_dist, dist)
                    
                    print(f"  轮廓 {i}: 面积={area:.1f}, 周长={perimeter:.1f}")
                    print(f"    Hub3在轮廓内: {'是' if inside >= 0 else '否'}")
                    print(f"    到Hub3最小距离: {min_dist:.1f}")
                    
                    # 检查是否会被包含在合并逻辑中
                    contains_hub = False
                    if inside >= 0:
                        contains_hub = True
                        print(f"    ✅ 会被包含（Hub3在轮廓内）")
                    elif min_dist < 20:
                        contains_hub = True
                        print(f"    ✅ 会被包含（距离<20像素）")
                    else:
                        print(f"    ❌ 不会被包含（距离≥20像素）")
                
                # 模拟轮廓合并逻辑
                all_contour_points = []
                for contour in contours:
                    contains_hub = False
                    inside = cv2.pointPolygonTest(contour, hub3_pos, False)
                    if inside >= 0:
                        contains_hub = True
                    else:
                        min_dist = float('inf')
                        for point in contour:
                            x, y = point[0][0], point[0][1]
                            dist = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
                            min_dist = min(min_dist, dist)
                        if min_dist < 20:
                            contains_hub = True
                    
                    if contains_hub:
                        for point in contour:
                            x, y = point[0][0], point[0][1]
                            all_contour_points.append((x, y))
                
                print(f"合并后的轮廓点数: {len(all_contour_points)}")
                
                if len(all_contour_points) > 20:
                    print(f"✅ 合并后的轮廓足够长，可以生成槽位")
                else:
                    print(f"❌ 合并后的轮廓太短，无法生成槽位")
            else:
                print("❌ 没有找到轮廓")
        
    except Exception as e:
        print(f"❌ 测试出错: {e}")

if __name__ == "__main__":
    test_hub3_contour_extraction()