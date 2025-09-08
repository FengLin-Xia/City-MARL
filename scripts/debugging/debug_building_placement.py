#!/usr/bin/env python3
"""
调试建筑放置逻辑，找出为什么建筑都集中在Hub 1
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def debug_building_placement():
    """调试建筑放置逻辑"""
    
    print("🐛 调试建筑放置逻辑")
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
    
    # 加载建筑位置数据
    try:
        with open('enhanced_simulation_v2_3_output/building_positions_month_21.json', 'r', encoding='utf-8') as f:
            building_data = json.load(f)
        buildings = building_data['buildings']
        print(f"✅ 建筑数据加载成功，数量: {len(buildings)}")
    except Exception as e:
        print(f"❌ 无法加载建筑数据: {e}")
        buildings = []
    
    # 分析现有建筑分布
    print(f"\n🏗️ 现有建筑分布分析:")
    
    building_counts = {'residential': 0, 'commercial': 0, 'public': 0}
    building_positions = {'residential': [], 'commercial': [], 'public': []}
    
    for building in buildings:
        building_type = building.get('type', 'unknown')
        if building_type in building_counts:
            building_counts[building_type] += 1
            building_positions[building_type].append(building['position'])
    
    print(f"  建筑总数: {sum(building_counts.values())}")
    print(f"  商业建筑: {building_counts['commercial']}")
    print(f"  住宅建筑: {building_counts['residential']}")
    print(f"  公共建筑: {building_counts['public']}")
    
    # 分析每个枢纽周围的建筑分布
    print(f"\n📍 枢纽周围建筑分布分析:")
    
    for i, hub in enumerate(hubs):
        hub_x, hub_y = hub[0], hub[1]
        print(f"\n  Hub {i+1} ({hub_x}, {hub_y}):")
        
        # 统计枢纽周围的建筑
        nearby_buildings = {'residential': 0, 'commercial': 0, 'public': 0}
        nearby_positions = {'residential': [], 'commercial': [], 'public': []}
        
        for building_type, positions in building_positions.items():
            for pos in positions:
                distance = np.sqrt((pos[0] - hub_x)**2 + (pos[1] - hub_y)**2)
                if distance <= 100:  # 100像素范围内
                    nearby_buildings[building_type] += 1
                    nearby_positions[building_type].append(pos)
        
        print(f"    100像素范围内建筑:")
        print(f"      商业: {nearby_buildings['commercial']}")
        print(f"      住宅: {nearby_buildings['residential']}")
        print(f"      公共: {nearby_buildings['public']}")
        
        # 分析建筑位置的平均坐标
        for building_type, positions in nearby_positions.items():
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                x_mean = np.mean(x_coords)
                y_mean = np.mean(y_coords)
                print(f"      {building_type}平均位置: ({x_mean:.1f}, {y_mean:.1f})")
    
    # 测试等值线提取和建筑放置
    print(f"\n🧪 测试等值线提取和建筑放置:")
    
    # 商业建筑等值线阈值
    commercial_percentiles = [95, 90, 85]
    sdf_flat = sdf_field.flatten()
    commercial_thresholds = np.percentile(sdf_flat, commercial_percentiles)
    
    print(f"  商业建筑等值线阈值: {[f'{t:.3f}' for t in commercial_thresholds]}")
    
    # 住宅建筑等值线阈值
    residential_percentiles = [80, 70, 60, 50]
    residential_thresholds = np.percentile(sdf_flat, residential_percentiles)
    
    print(f"  住宅建筑等值线阈值: {[f'{t:.3f}' for t in residential_thresholds]}")
    
    # 测试商业建筑放置
    print(f"\n🏢 测试商业建筑放置:")
    
    commercial_contours = []
    for i, threshold in enumerate(commercial_thresholds):
        contour = extract_contour_at_level_cv2(sdf_field, threshold)
        if len(contour) > 20:
            commercial_contours.append(contour)
            print(f"  - 等值线 {i+1}: 阈值 {threshold:.3f}, 长度 {len(contour)}")
            
            # 测试在这个等值线上放置建筑
            test_positions = test_contour_building_placement(contour, 'commercial', 10, hubs)
            if test_positions:
                analyze_building_distribution(test_positions, hubs, f"商业等值线 {i+1}")
    
    # 测试住宅建筑放置
    print(f"\n🏠 测试住宅建筑放置:")
    
    residential_contours = []
    for i, threshold in enumerate(residential_thresholds):
        contour = extract_contour_at_level_cv2(sdf_field, threshold)
        if len(contour) > 20:
            residential_contours.append(contour)
            print(f"  - 等值线 {i+1}: 阈值 {threshold:.3f}, 长度 {len(contour)}")
            
            # 测试在这个等值线上放置建筑
            test_positions = test_contour_building_placement(contour, 'residential', 10, hubs)
            if test_positions:
                analyze_building_distribution(test_positions, hubs, f"住宅等值线 {i+1}")
    
    # 创建可视化
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Building Placement Debug Analysis', fontsize=16)
        
        # 左上图：SDF场 + 现有建筑
        im1 = ax1.imshow(sdf_field, cmap='viridis', origin='lower', 
                          extent=[0, 256, 0, 256], alpha=0.8)
        
        # 绘制枢纽
        for i, hub in enumerate(hubs):
            ax1.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
        
        # 绘制现有建筑
        colors = {'residential': '#F6C344', 'commercial': '#FD7E14', 'public': '#22A6B3'}
        for building_type, positions in building_positions.items():
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                ax1.scatter(x_coords, y_coords, c=colors[building_type], s=50, 
                           alpha=0.7, label=f'{building_type.title()} ({len(positions)})')
        
        ax1.set_title('SDF Field + Existing Buildings')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.legend()
        
        # 添加颜色条
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('SDF Value')
        
        # 右上图：商业等值线 + 测试建筑位置
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
        
        # 左下图：住宅等值线 + 测试建筑位置
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
        
        # 右下图：建筑分布热力图
        ax4.clear()
        
        # 创建建筑密度热力图
        density_map = np.zeros((256, 256))
        
        for building_type, positions in building_positions.items():
            for pos in positions:
                x, y = pos[0], pos[1]
                if 0 <= x < 256 and 0 <= y < 256:
                    density_map[y, x] += 1
        
        im4 = ax4.imshow(density_map, cmap='hot', origin='lower', 
                          extent=[0, 256, 0, 256], alpha=0.8)
        
        # 绘制枢纽
        for i, hub in enumerate(hubs):
            ax4.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
        
        ax4.set_title('Building Density Heatmap')
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')
        ax4.legend()
        
        # 添加颜色条
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
        cbar4.set_label('Building Count')
        
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
    debug_building_placement()


