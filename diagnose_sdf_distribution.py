#!/usr/bin/env python3
"""
诊断SDF场分布 - 理解为什么Hub 2没有进入建筑生长逻辑
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def diagnose_sdf_distribution():
    """诊断SDF场分布"""
    
    print("🔍 SDF场分布诊断")
    print("=" * 50)
    
    # 枢纽位置
    hubs = [[40, 128], [216, 128]]
    
    # 加载SDF场数据
    try:
        with open('enhanced_simulation_v2_3_output/sdf_field_month_21.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sdf_field = np.array(data['sdf_field'])
        print(f"✅ SDF场加载成功，形状: {sdf_field.shape}")
        
        # 分析SDF场统计
        print(f"\n📊 SDF场统计:")
        print(f"   最小值: {np.min(sdf_field):.6f}")
        print(f"   最大值: {np.max(sdf_field):.6f}")
        print(f"   平均值: {np.mean(sdf_field):.6f}")
        print(f"   标准差: {np.std(sdf_field):.6f}")
        
        # 分析两个枢纽附近的SDF值
        print(f"\n🎯 枢纽附近SDF值分析:")
        
        for i, hub in enumerate(hubs):
            hub_x, hub_y = hub[0], hub[1]
            print(f"\n   Hub {i+1} ({hub_x}, {hub_y}):")
            
            # 枢纽本身的SDF值
            hub_sdf = sdf_field[hub_y, hub_x]
            print(f"     枢纽位置SDF值: {hub_sdf:.6f}")
            
            # 枢纽周围区域的SDF值
            radius = 20  # 20像素半径
            y_min = max(0, hub_y - radius)
            y_max = min(sdf_field.shape[0], hub_y + radius + 1)
            x_min = max(0, hub_x - radius)
            x_max = min(sdf_field.shape[1], hub_x + radius + 1)
            
            hub_region = sdf_field[y_min:y_max, x_min:x_max]
            print(f"     周围区域({radius}px): 最小={np.min(hub_region):.6f}, 最大={np.max(hub_region):.6f}, 平均={np.mean(hub_region):.6f}")
            
            # 检查是否达到建筑生成阈值
            commercial_threshold = 0.85
            residential_threshold = 0.55
            
            above_commercial = np.sum(hub_region >= commercial_threshold)
            above_residential = np.sum(hub_region >= residential_threshold)
            total_pixels = hub_region.size
            
            print(f"     达到商业阈值(≥{commercial_threshold}): {above_commercial}/{total_pixels} ({above_commercial/total_pixels*100:.1f}%)")
            print(f"     达到住宅阈值(≥{residential_threshold}): {above_residential}/{total_pixels} ({above_residential/total_pixels*100:.1f}%)")
        
        # 分析主干道沿线的SDF值
        print(f"\n🛣️ 主干道沿线SDF值分析:")
        trunk_road = [[40, 128], [216, 128]]
        
        # 沿主干道采样点
        num_samples = 20
        road_sdf_values = []
        road_positions = []
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = int(trunk_road[0][0] + t * (trunk_road[1][0] - trunk_road[0][0]))
            y = int(trunk_road[0][1] + t * (trunk_road[1][1] - trunk_road[0][1]))
            
            if 0 <= x < sdf_field.shape[1] and 0 <= y < sdf_field.shape[0]:
                road_sdf_values.append(sdf_field[y, x])
                road_positions.append((x, y))
        
        if road_sdf_values:
            print(f"   主干道沿线SDF值:")
            print(f"     最小值: {np.min(road_sdf_values):.6f}")
            print(f"     最大值: {np.max(road_sdf_values):.6f}")
            print(f"     平均值: {np.mean(road_sdf_values):.6f}")
            
            # 检查主干道是否达到建筑生成阈值
            above_commercial_road = sum(1 for v in road_sdf_values if v >= commercial_threshold)
            above_residential_road = sum(1 for v in road_sdf_values if v >= residential_threshold)
            
            print(f"     达到商业阈值: {above_commercial_road}/{len(road_sdf_values)} ({above_commercial_road/len(road_sdf_values)*100:.1f}%)")
            print(f"     达到住宅阈值: {above_residential_road}/{len(road_sdf_values)} ({above_residential_road/len(road_sdf_values)*100:.1f}%)")
        
        # 可视化SDF场分布
        visualize_sdf_distribution(sdf_field, hubs, trunk_road)
        
    except Exception as e:
        print(f"❌ 无法加载SDF场数据: {e}")

def visualize_sdf_distribution(sdf_field, hubs, trunk_road):
    """可视化SDF场分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：SDF场热力图
    im1 = ax1.imshow(sdf_field, cmap='viridis', origin='lower', 
                      extent=[0, 256, 0, 256], alpha=0.8)
    
    # 绘制枢纽
    for i, hub in enumerate(hubs):
        ax1.scatter(hub[0], hub[1], c='red', s=200, marker='*', 
                   edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
    
    # 绘制主干道
    ax1.plot([trunk_road[0][0], trunk_road[1][0]], 
             [trunk_road[0][1], trunk_road[1][1]], 
             color='red', linewidth=3, alpha=0.8, label='Trunk Road')
    
    # 绘制等值线
    levels = [0.55, 0.85]  # 住宅和商业阈值
    contours = ax1.contour(sdf_field, levels=levels, colors='white', 
                           linewidths=2, alpha=0.9)
    ax1.clabel(contours, inline=True, fontsize=12, fmt='%.2f')
    
    ax1.set_title('SDF Field Distribution')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('SDF Value')
    
    # 右图：SDF值直方图
    ax2.hist(sdf_field.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    # 标记阈值线
    ax2.axvline(x=0.55, color='orange', linestyle='--', linewidth=2, label='Residential Threshold (0.55)')
    ax2.axvline(x=0.85, color='red', linestyle='--', linewidth=2, label='Commercial Threshold (0.85)')
    
    ax2.set_title('SDF Value Distribution')
    ax2.set_xlabel('SDF Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    diagnose_sdf_distribution()


