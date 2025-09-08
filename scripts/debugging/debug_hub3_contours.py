#!/usr/bin/env python3
"""
调试Hub3等值线生成
检查Hub3是否在提取的等值线轮廓内
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def analyze_hub3_contours():
    """分析Hub3的等值线生成情况"""
    
    # Hub3位置
    hub3_x, hub3_y = 67, 94
    hub3_pos = (hub3_x, hub3_y)
    
    # 检查几个关键月份
    months_to_check = [2, 10, 20, 30]
    
    print("=== Hub3 等值线生成分析 ===")
    print(f"Hub3 位置: {hub3_pos}")
    print()
    
    for month in months_to_check:
        try:
            # 读取地价场数据
            filename = f"enhanced_simulation_v3_1_output/land_price_frame_month_{month:02d}.json"
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            land_price_field = np.array(data['land_price_field'])
            
            # 计算百分位数阈值
            percentiles = [99, 95, 90, 80, 70, 60]
            percentile_values = {}
            for p in percentiles:
                percentile_values[p] = np.percentile(land_price_field, p)
            
            print(f"--- Month {month} ---")
            print("百分位数阈值:")
            for p in percentiles:
                print(f"  {p}%: {percentile_values[p]:.4f}")
            
            # 检查每个百分位数的等值线
            for percentile in [99, 95, 90, 80]:
                threshold = percentile_values[percentile]
                
                # 创建二值化图像
                binary_image = (land_price_field >= threshold).astype(np.uint8) * 255
                
                # 查找轮廓
                contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                print(f"\n{percentile}% 等值线 (阈值: {threshold:.4f}):")
                print(f"  轮廓数量: {len(contours)}")
                
                if len(contours) > 0:
                    # 检查Hub3是否在任何轮廓内
                    hub3_in_contour = False
                    hub3_contour_index = -1
                    
                    for i, contour in enumerate(contours):
                        # 检查Hub3是否在轮廓内
                        result = cv2.pointPolygonTest(contour, hub3_pos, False)
                        if result >= 0:  # 在轮廓内或边界上
                            hub3_in_contour = True
                            hub3_contour_index = i
                            break
                    
                    if hub3_in_contour:
                        print(f"  ✅ Hub3 在轮廓 {hub3_contour_index} 内")
                        
                        # 分析轮廓信息
                        contour = contours[hub3_contour_index]
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        
                        print(f"  轮廓面积: {area:.1f}")
                        print(f"  轮廓周长: {perimeter:.1f}")
                        
                        # 检查轮廓是否足够大
                        if area < 100:  # 面积太小
                            print(f"  ⚠️  轮廓面积过小，可能无法生成有效槽位")
                        else:
                            print(f"  ✅ 轮廓面积足够，可以生成槽位")
                            
                    else:
                        print(f"  ❌ Hub3 不在任何轮廓内")
                        
                        # 检查Hub3周围是否有轮廓
                        nearby_contours = []
                        for i, contour in enumerate(contours):
                            # 计算轮廓中心
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                # 计算到Hub3的距离
                                distance = np.sqrt((cx - hub3_x)**2 + (cy - hub3_y)**2)
                                if distance < 50:  # 50像素范围内
                                    nearby_contours.append((i, distance, cx, cy))
                        
                        if nearby_contours:
                            print(f"  Hub3 附近有 {len(nearby_contours)} 个轮廓:")
                            for i, dist, cx, cy in sorted(nearby_contours, key=lambda x: x[1])[:3]:
                                print(f"    轮廓 {i}: 中心({cx}, {cy}), 距离: {dist:.1f}")
                        else:
                            print(f"  Hub3 附近没有轮廓")
                
                else:
                    print(f"  ❌ 没有找到轮廓")
            
            print()
            
        except FileNotFoundError:
            print(f"❌ 文件不存在: {filename}")
        except Exception as e:
            print(f"❌ 读取文件出错: {e}")
    
    # 检查等值线提取逻辑
    print("=== 等值线提取逻辑分析 ===")
    print("检查 IsocontourBuildingSystem 的轮廓提取逻辑...")
    
    # 读取等值线建筑系统代码
    try:
        with open('logic/isocontour_building_system.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 查找关键方法
        if '_extract_contour_at_level_cv2' in code:
            print("✅ 找到 _extract_contour_at_level_cv2 方法")
            
            # 检查是否有多个轮廓的处理逻辑
            if 'len(contours) > 1' in code:
                print("✅ 有多轮廓处理逻辑")
            else:
                print("⚠️  可能缺少多轮廓处理逻辑")
                
            # 检查是否有Hub检测逻辑
            if 'transport_hub' in code or 'pointPolygonTest' in code:
                print("✅ 有Hub检测逻辑")
            else:
                print("⚠️  可能缺少Hub检测逻辑")
        else:
            print("❌ 未找到 _extract_contour_at_level_cv2 方法")
            
    except Exception as e:
        print(f"❌ 读取代码文件出错: {e}")

if __name__ == "__main__":
    analyze_hub3_contours()
