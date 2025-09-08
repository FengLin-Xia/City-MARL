#!/usr/bin/env python3
"""
测试等值线提取和建筑放置逻辑
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

def test_isocontour_extraction():
    """测试等值线提取逻辑"""
    
    print("🧪 测试等值线提取和建筑放置逻辑")
    print("=" * 50)
    
    # 枢纽位置
    hubs = [[40, 128], [216, 128]]
    
    # 加载SDF场数据
    try:
        with open('enhanced_simulation_v2_3_output/sdf_field_month_21.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        sdf_field = np.array(data['sdf_field'])
        print(f"✅ SDF场加载成功，形状: {sdf_field.shape}")
    except Exception as e:
        print(f"❌ 无法加载SDF场数据: {e}")
        return
    
    # 测试等值线提取
    print(f"\n🔍 测试等值线提取:")
    
    # 商业建筑等值线阈值
    commercial_percentiles = [95, 90, 85]
    sdf_flat = sdf_field.flatten()
    commercial_thresholds = np.percentile(sdf_flat, commercial_percentiles)
    
    print(f"  商业建筑等值线阈值: {[f'{t:.3f}' for t in commercial_thresholds]}")
    
    # 住宅建筑等值线阈值
    residential_percentiles = [80, 70, 60, 50]
    residential_thresholds = np.percentile(sdf_flat, residential_percentiles)
    
    print(f"  住宅建筑等值线阈值: {[f'{t:.3f}' for t in residential_thresholds]}")
    
    # 测试每个阈值的等值线提取
    print(f"\n📊 商业建筑等值线测试:")
    
    commercial_contours = []
    for i, threshold in enumerate(commercial_thresholds):
        contour = extract_contour_at_level_cv2(sdf_field, threshold)
        if len(contour) > 20:
            commercial_contours.append(contour)
            area_ratio = calculate_contour_area_ratio(sdf_field, threshold)
            print(f"  - 等值线 {i+1}: 阈值 {threshold:.3f}, 长度 {len(contour)}, 覆盖面积 {area_ratio:.1f}%")
            
            # 分析等值线覆盖的枢纽区域
            analyze_contour_coverage(contour, threshold, hubs, f"商业等值线 {i+1}")
        else:
            print(f"  - 等值线 {i+1}: 阈值 {threshold:.3f}, 长度 {len(contour)} (跳过)")
    
    print(f"\n📊 住宅建筑等值线测试:")
    
    residential_contours = []
    for i, threshold in enumerate(residential_thresholds):
        contour = extract_contour_at_level_cv2(sdf_field, threshold)
        if len(contour) > 20:
            residential_contours.append(contour)
            area_ratio = calculate_contour_area_ratio(sdf_field, threshold)
            print(f"  - 等值线 {i+1}: 阈值 {threshold:.3f}, 长度 {len(contour)}, 覆盖面积 {area_ratio:.1f}%")
            
            # 分析等值线覆盖的枢纽区域
            analyze_contour_coverage(contour, threshold, hubs, f"住宅等值线 {i+1}")
        else:
            print(f"  - 等值线 {i+1}: 阈值 {threshold:.3f}, 长度 {len(contour)} (跳过)")
    
    # 测试建筑放置逻辑
    print(f"\n🏗️ 测试建筑放置逻辑:")
    
    # 测试商业建筑放置
    if commercial_contours:
        print(f"  商业建筑等值线数量: {len(commercial_contours)}")
        test_building_placement(commercial_contours, 'commercial', hubs, 20)
    
    # 测试住宅建筑放置
    if residential_contours:
        print(f"  住宅建筑等值线数量: {len(residential_contours)}")
        test_building_placement(residential_contours, 'residential', hubs, 20)
    
    # 创建可视化
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Isocontour Extraction and Building Placement Test', fontsize=16)
        
        # 左上图：SDF场 + 枢纽位置
        im1 = ax1.imshow(sdf_field, cmap='viridis', origin='lower', 
                          extent=[0, 256, 0, 256], alpha=0.8)
        
        # 绘制枢纽
        for i, hub in enumerate(hubs):
            ax1.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
        
        ax1.set_title('SDF Field + Transport Hubs')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.legend()
        
        # 添加颜色条
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('SDF Value')
        
        # 右上图：商业等值线
        im2 = ax2.imshow(sdf_field, cmap='viridis', origin='lower', 
                          extent=[0, 256, 0, 256], alpha=0.6)
        
        # 绘制枢纽
        for i, hub in enumerate(hubs):
            ax2.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
        
        # 绘制商业等值线
        for i, contour in enumerate(commercial_contours):
            if contour:
                x_coords = [p[0] for p in contour]
                y_coords = [p[1] for p in contour]
                ax2.plot(x_coords, y_coords, color='orange', linewidth=2, 
                        alpha=0.8, label=f'Commercial {i+1}')
        
        ax2.set_title('Commercial Isocontours')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.legend()
        
        # 添加颜色条
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('SDF Value')
        
        # 左下图：住宅等值线
        im3 = ax3.imshow(sdf_field, cmap='viridis', origin='lower', 
                          extent=[0, 256, 0, 256], alpha=0.6)
        
        # 绘制枢纽
        for i, hub in enumerate(hubs):
            ax3.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
        
        # 绘制住宅等值线
        for i, contour in enumerate(residential_contours):
            if contour:
                x_coords = [p[0] for p in contour]
                y_coords = [p[1] for p in contour]
                ax3.plot(x_coords, y_coords, color='blue', linewidth=2, 
                        alpha=0.8, label=f'Residential {i+1}')
        
        ax3.set_title('Residential Isocontours')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.legend()
        
        # 添加颜色条
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
        cbar3.set_label('SDF Value')
        
        # 右下图：等值线覆盖面积对比
        ax4.clear()
        
        # 商业等值线覆盖面积
        commercial_areas = []
        for threshold in commercial_thresholds:
            area = calculate_contour_area_ratio(sdf_field, threshold)
            commercial_areas.append(area)
        
        # 住宅等值线覆盖面积
        residential_areas = []
        for threshold in residential_thresholds:
            area = calculate_contour_area_ratio(sdf_field, threshold)
            residential_areas.append(area)
        
        x1 = np.arange(len(commercial_areas))
        x2 = np.arange(len(residential_areas))
        
        ax4.bar(x1 - 0.2, commercial_areas, 0.4, label='Commercial', color='orange', alpha=0.7)
        ax4.bar(x2 + 0.2, residential_areas, 0.4, label='Residential', color='blue', alpha=0.7)
        
        ax4.set_xlabel('Isocontour Index')
        ax4.set_ylabel('Coverage Area (%)')
        ax4.set_title('Isocontour Coverage Area Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n🎨 可视化完成！")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

def extract_contour_at_level_cv2(sdf_field: np.ndarray, level: float) -> list:
    """使用OpenCV在指定SDF值水平提取等值线"""
    # 创建二值图像
    binary = (sdf_field >= level).astype(np.uint8) * 255
    
    # 使用OpenCV的findContours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 转换为点列表
    contour_points = []
    for point in largest_contour:
        x, y = point[0][0], point[0][1]
        contour_points.append((x, y))
    
    return contour_points

def calculate_contour_area_ratio(sdf_field: np.ndarray, threshold: float) -> float:
    """计算等值线覆盖面积比例"""
    # 计算大于等于阈值的像素数量
    area_pixels = np.sum(sdf_field >= threshold)
    total_pixels = sdf_field.size
    
    return (area_pixels / total_pixels) * 100

def analyze_contour_coverage(contour: list, threshold: float, hubs: list, contour_name: str):
    """分析等值线覆盖的枢纽区域"""
    if not contour:
        return
    
    # 计算等值线覆盖的区域
    x_coords = [p[0] for p in contour]
    y_coords = [p[1] for p in contour]
    
    contour_x_min, contour_x_max = min(x_coords), max(x_coords)
    contour_y_min, contour_y_max = min(y_coords), max(y_coords)
    
    print(f"    {contour_name}覆盖区域: X[{contour_x_min}, {contour_x_max}], Y[{contour_y_min}, {contour_y_max}]")
    
    # 检查是否覆盖两个枢纽
    hub1_covered = contour_x_min <= hubs[0][0] <= contour_x_max and contour_y_min <= hubs[0][1] <= contour_y_max
    hub2_covered = contour_x_min <= hubs[1][0] <= contour_x_max and contour_y_min <= hubs[1][1] <= contour_y_max
    
    print(f"    覆盖Hub 1: {'✅' if hub1_covered else '❌'}")
    print(f"    覆盖Hub 2: {'✅' if hub2_covered else '❌'}")
    
    if not hub2_covered:
        print(f"    ⚠️ 警告: {contour_name}没有覆盖Hub 2！")

def test_building_placement(contours: list, building_type: str, hubs: list, target_count: int):
    """测试建筑放置逻辑"""
    print(f"  {building_type.title()}建筑放置测试:")
    
    positions = []
    
    for i, contour in enumerate(contours):
        if len(contour) < 10:
            continue
        
        # 简单的均匀采样
        contour_length = len(contour)
        spacing = max(1, contour_length // target_count)
        
        for j in range(0, contour_length, spacing):
            if len(positions) >= target_count:
                break
            
            pos = contour[j]
            positions.append([pos[0], pos[1]])
    
    print(f"    生成位置数量: {len(positions)}")
    
    if positions:
        # 分析位置分布
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        
        print(f"    平均位置: ({x_mean:.1f}, {y_mean:.1f})")
        
        # 检查是否集中在某个枢纽
        hub1_distances = []
        hub2_distances = []
        
        for pos in positions:
            dist1 = np.sqrt((pos[0] - hubs[0][0])**2 + (pos[1] - hubs[0][1])**2)
            dist2 = np.sqrt((pos[0] - hubs[1][0])**2 + (pos[1] - hubs[1][1])**2)
            hub1_distances.append(dist1)
            hub2_distances.append(dist2)
        
        avg_dist1 = np.mean(hub1_distances)
        avg_dist2 = np.mean(hub2_distances)
        
        print(f"    到Hub 1平均距离: {avg_dist1:.1f}")
        print(f"    到Hub 2平均距离: {avg_dist2:.1f}")
        
        if avg_dist1 < avg_dist2:
            print(f"    ⚠️ 建筑位置偏向Hub 1")
        elif avg_dist2 < avg_dist1:
            print(f"    ⚠️ 建筑位置偏向Hub 2")
        else:
            print(f"    ✅ 建筑位置分布均衡")

if __name__ == "__main__":
    test_isocontour_extraction()


