#!/usr/bin/env python3
"""
修复等值线提取问题，确保等值线能够跨越两个枢纽区域
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def fix_isocontour_extraction():
    """修复等值线提取问题"""
    
    print("🔧 修复等值线提取问题")
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
    
    # 测试修复后的等值线提取
    print(f"\n🧪 测试修复后的等值线提取:")
    
    # 商业建筑等值线阈值
    commercial_percentiles = [95, 90, 85]
    sdf_flat = sdf_field.flatten()
    commercial_thresholds = np.percentile(sdf_flat, commercial_percentiles)
    
    print(f"  商业建筑等值线阈值: {[f'{t:.3f}' for t in commercial_thresholds]}")
    
    # 住宅建筑等值线阈值
    residential_percentiles = [80, 70, 60, 50]
    residential_thresholds = np.percentile(sdf_flat, residential_percentiles)
    
    print(f"  住宅建筑等值线阈值: {[f'{t:.3f}' for t in residential_thresholds]}")
    
    # 测试修复后的商业建筑等值线提取
    print(f"\n🏢 测试修复后的商业建筑等值线提取:")
    
    commercial_contours_fixed = []
    for i, threshold in enumerate(commercial_thresholds):
        contour = extract_contour_fixed(sdf_field, threshold, hubs, 'commercial')
        if len(contour) > 20:
            commercial_contours_fixed.append(contour)
            print(f"  - 等值线 {i+1}: 阈值 {threshold:.3f}, 长度 {len(contour)}")
            
            # 分析等值线覆盖的枢纽区域
            analyze_contour_coverage_fixed(contour, threshold, hubs, f"修复后商业等值线 {i+1}")
            
            # 测试建筑放置
            test_positions = test_contour_building_placement(contour, 'commercial', 10, hubs)
            if test_positions:
                analyze_building_distribution(test_positions, hubs, f"修复后商业等值线 {i+1}")
    
    # 测试修复后的住宅建筑等值线提取
    print(f"\n🏠 测试修复后的住宅建筑等值线提取:")
    
    residential_contours_fixed = []
    for i, threshold in enumerate(residential_thresholds):
        contour = extract_contour_fixed(sdf_field, threshold, hubs, 'residential')
        if len(contour) > 20:
            residential_contours_fixed.append(contour)
            print(f"  - 等值线 {i+1}: 阈值 {threshold:.3f}, 长度 {len(contour)}")
            
            # 分析等值线覆盖的枢纽区域
            analyze_contour_coverage_fixed(contour, threshold, hubs, f"修复后住宅等值线 {i+1}")
            
            # 测试建筑放置
            test_positions = test_contour_building_placement(contour, 'residential', 10, hubs)
            if test_positions:
                analyze_building_distribution(test_positions, hubs, f"修复后住宅等值线 {i+1}")
    
    # 对比修复前后的效果
    print(f"\n📊 修复前后对比:")
    
    # 原始方法
    print(f"  原始方法:")
    commercial_contours_original = []
    for i, threshold in enumerate(commercial_thresholds):
        contour = extract_contour_original(sdf_field, threshold)
        if len(contour) > 20:
            commercial_contours_original.append(contour)
            x_coords = [p[0] for p in contour]
            x_range = f"[{min(x_coords)}, {max(x_coords)}]"
            print(f"    商业等值线 {i+1}: X范围 {x_range}")
    
    residential_contours_original = []
    for i, threshold in enumerate(residential_thresholds):
        contour = extract_contour_original(sdf_field, threshold)
        if len(contour) > 20:
            residential_contours_original.append(contour)
            x_coords = [p[0] for p in contour]
            x_range = f"[{min(x_coords)}, {max(x_coords)}]"
            print(f"    住宅等值线 {i+1}: X范围 {x_range}")
    
    # 修复后方法
    print(f"  修复后方法:")
    for i, contour in enumerate(commercial_contours_fixed):
        x_coords = [p[0] for p in contour]
        x_range = f"[{min(x_coords)}, {max(x_coords)}]"
        print(f"    商业等值线 {i+1}: X范围 {x_range}")
    
    for i, contour in enumerate(residential_contours_fixed):
        x_coords = [p[0] for p in contour]
        x_range = f"[{min(x_coords)}, {max(x_coords)}]"
        print(f"    住宅等值线 {i+1}: X范围 {x_range}")
    
    # 创建可视化
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Isocontour Extraction Fix Comparison', fontsize=16)
        
        # 左上图：原始商业等值线
        im1 = ax1.imshow(sdf_field, cmap='viridis', origin='lower', 
                          extent=[0, 256, 0, 256], alpha=0.6)
        
        # 绘制枢纽
        for i, hub in enumerate(hubs):
            ax1.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
        
        # 绘制原始商业等值线
        for i, contour in enumerate(commercial_contours_original):
            if contour:
                x_coords = [p[0] for p in contour]
                y_coords = [p[1] for p in contour]
                ax1.plot(x_coords, y_coords, color='orange', linewidth=2, 
                        alpha=0.8, label=f'Original Commercial {i+1}')
        
        ax1.set_title('Original Commercial Isocontours')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.legend()
        
        # 添加颜色条
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('SDF Value')
        
        # 右上图：修复后商业等值线
        im2 = ax2.imshow(sdf_field, cmap='viridis', origin='lower', 
                          extent=[0, 256, 0, 256], alpha=0.6)
        
        # 绘制枢纽
        for i, hub in enumerate(hubs):
            ax2.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
        
        # 绘制修复后商业等值线
        for i, contour in enumerate(commercial_contours_fixed):
            if contour:
                x_coords = [p[0] for p in contour]
                y_coords = [p[1] for p in contour]
                ax2.plot(x_coords, y_coords, color='red', linewidth=2, 
                        alpha=0.8, label=f'Fixed Commercial {i+1}')
        
        ax2.set_title('Fixed Commercial Isocontours')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.legend()
        
        # 添加颜色条
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('SDF Value')
        
        # 左下图：原始住宅等值线
        im3 = ax3.imshow(sdf_field, cmap='viridis', origin='lower', 
                          extent=[0, 256, 0, 256], alpha=0.6)
        
        # 绘制枢纽
        for i, hub in enumerate(hubs):
            ax3.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
        
        # 绘制原始住宅等值线
        for i, contour in enumerate(residential_contours_original):
            if contour:
                x_coords = [p[0] for p in contour]
                y_coords = [p[1] for p in contour]
                ax3.plot(x_coords, y_coords, color='blue', linewidth=2, 
                        alpha=0.8, label=f'Original Residential {i+1}')
        
        ax3.set_title('Original Residential Isocontours')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.legend()
        
        # 添加颜色条
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
        cbar3.set_label('SDF Value')
        
        # 右下图：修复后住宅等值线
        im4 = ax4.imshow(sdf_field, cmap='viridis', origin='lower', 
                          extent=[0, 256, 0, 256], alpha=0.6)
        
        # 绘制枢纽
        for i, hub in enumerate(hubs):
            ax4.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
        
        # 绘制修复后住宅等值线
        for i, contour in enumerate(residential_contours_fixed):
            if contour:
                x_coords = [p[0] for p in contour]
                y_coords = [p[1] for p in contour]
                ax4.plot(x_coords, y_coords, color='green', linewidth=2, 
                        alpha=0.8, label=f'Fixed Residential {i+1}')
        
        ax4.set_title('Fixed Residential Isocontours')
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')
        ax4.legend()
        
        # 添加颜色条
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
        cbar4.set_label('SDF Value')
        
        plt.tight_layout()
        plt.show()
        
        print("\n🎨 可视化完成！")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

def extract_contour_original(sdf_field: np.ndarray, level: float) -> list:
    """原始等值线提取方法（只选择最大轮廓）"""
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

def extract_contour_fixed(sdf_field: np.ndarray, level: float, hubs: list, building_type: str) -> list:
    """修复后的等值线提取方法（强制跨越两个枢纽区域）"""
    # 创建二值图像
    binary = (sdf_field >= level).astype(np.uint8) * 255
    
    # 使用OpenCV的findContours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # 如果只有一个轮廓，直接返回
    if len(contours) == 1:
        contour = contours[0]
        contour_points = []
        for point in contour:
            x, y = point[0][0], point[0][1]
            contour_points.append((x, y))
        return contour_points
    
    # 如果有多个轮廓，尝试合并或选择最佳的一个
    best_contour = select_best_contour(contours, hubs, building_type)
    
    # 转换为点列表
    contour_points = []
    for point in best_contour:
        x, y = point[0][0], point[0][1]
        contour_points.append((x, y))
    
    return contour_points

def select_best_contour(contours: list, hubs: list, building_type: str) -> list:
    """选择最佳轮廓（优先选择跨越两个枢纽区域的）"""
    best_contour = None
    best_score = -1
    
    for contour in contours:
        # 计算轮廓的边界框
        x_coords = [point[0][0] for point in contour]
        y_coords = [point[0][1] for point in contour]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 计算轮廓覆盖的枢纽数量
        covered_hubs = 0
        for hub in hubs:
            if x_min <= hub[0] <= x_max and y_min <= hub[1] <= y_max:
                covered_hubs += 1
        
        # 计算轮廓跨越的区域宽度
        span_width = x_max - x_min
        
        # 计算综合评分
        score = covered_hubs * 100 + span_width * 0.1 + len(contour) * 0.01
        
        if score > best_score:
            best_score = score
            best_contour = contour
    
    return best_contour

def analyze_contour_coverage_fixed(contour: list, threshold: float, hubs: list, contour_name: str):
    """分析修复后等值线覆盖的枢纽区域"""
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
    
    if hub1_covered and hub2_covered:
        print(f"    ✅ 等值线成功跨越两个枢纽区域！")
    elif not hub1_covered and not hub2_covered:
        print(f"    ❌ 等值线没有覆盖任何枢纽")
    else:
        print(f"    ⚠️ 等值线只覆盖了一个枢纽")

def test_contour_building_placement(contour: list, building_type: str, target_count: int, hubs: list) -> list:
    """测试在等值线上放置建筑"""
    if not contour or len(contour) < 10:
        return []
    
    positions = []
    contour_length = len(contour)
    
    # 简单的均匀采样
    spacing = max(1, contour_length // target_count)
    
    for i in range(0, contour_length, spacing):
        if len(positions) >= target_count:
            break
        
        pos = contour[i]
        positions.append([pos[0], pos[1]])
    
    return positions

def analyze_building_distribution(positions: list, hubs: list, contour_name: str):
    """分析建筑位置分布"""
    if not positions:
        return
    
    print(f"    {contour_name}建筑位置分析:")
    
    # 分析位置分布
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_coords)
    
    print(f"      平均位置: ({x_mean:.1f}, {y_mean:.1f})")
    
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
    
    print(f"      到Hub 1平均距离: {avg_dist1:.1f}")
    print(f"      到Hub 2平均距离: {avg_dist2:.1f}")
    
    if avg_dist1 < avg_dist2:
        print(f"      ⚠️ 建筑位置偏向Hub 1")
    elif avg_dist2 < avg_dist1:
        print(f"      ⚠️ 建筑位置偏向Hub 2")
    else:
        print(f"      ✅ 建筑位置分布均衡")

if __name__ == "__main__":
    fix_isocontour_extraction()


