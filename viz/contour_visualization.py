#!/usr/bin/env python3
"""
等值线和建筑分布可视化 - 展示修正后的效果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_contours_and_buildings():
    """可视化等值线和建筑分布"""
    
    print("🎨 等值线和建筑分布可视化")
    print("=" * 50)
    
    # 枢纽位置
    hubs = [[40, 128], [216, 128]]
    trunk_road = [[40, 128], [216, 128]]
    
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
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Contour and Building Distribution Analysis (Fixed Version)', fontsize=16)
    
    # 左上图：SDF场热力图
    im1 = ax1.imshow(sdf_field, cmap='viridis', origin='lower', 
                      extent=[0, 256, 0, 256], alpha=0.8)
    
    # 绘制枢纽和主干道
    for i, hub in enumerate(hubs):
        ax1.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                   edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
    
    ax1.plot([trunk_road[0][0], trunk_road[1][0]], 
             [trunk_road[0][1], trunk_road[1][1]], 
             color='red', linewidth=3, alpha=0.8, label='Trunk Road')
    
    ax1.set_title('SDF Field Distribution')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('SDF Value')
    
    # 右上图：密集等值线
    im2 = ax2.imshow(sdf_field, cmap='viridis', origin='lower', 
                      extent=[0, 256, 0, 256], alpha=0.6)
    
    # 绘制枢纽和主干道
    for i, hub in enumerate(hubs):
        ax2.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                   edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
    
    ax2.plot([trunk_road[0][0], trunk_road[1][0]], 
             [trunk_road[0][1], trunk_road[1][1]], 
             color='red', linewidth=3, alpha=0.8, label='Trunk Road')
    
    # 生成密集等值线
    sdf_min, sdf_max = np.min(sdf_field), np.max(sdf_field)
    
    # 商业建筑等值线
    commercial_levels = np.linspace(0.85, sdf_min + 0.1, 8)
    commercial_contours = ax2.contour(sdf_field, levels=commercial_levels, 
                                     colors='orange', linewidths=1.5, alpha=0.8)
    ax2.clabel(commercial_contours, inline=True, fontsize=8, fmt='%.2f')
    
    # 住宅建筑等值线
    residential_levels = np.linspace(0.55, sdf_min + 0.1, 10)
    residential_contours = ax2.contour(sdf_field, levels=residential_levels, 
                                      colors='blue', linewidths=1, alpha=0.6)
    ax2.clabel(residential_contours, inline=True, fontsize=6, fmt='%.2f')
    
    ax2.set_title('Dense Isocontours (Fixed Version)')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.legend()
    
    # 添加颜色条
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('SDF Value')
    
    # 左下图：建筑分布
    ax3.clear()
    
    # 绘制主干道
    ax3.plot([trunk_road[0][0], trunk_road[1][0]], 
             [trunk_road[0][1], trunk_road[1][1]], 
             color='gray', linewidth=3, label='Trunk Road')
    
    # 绘制交通枢纽
    for i, hub in enumerate(hubs):
        ax3.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                   edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
    
    # 绘制建筑
    building_counts = {'residential': 0, 'commercial': 0, 'public': 0}
    
    for building in buildings:
        building_type = building.get('type', 'unknown')
        if building_type in building_counts:
            building_counts[building_type] += 1
            
            pos = building['position']
            color = {'residential': '#F6C344', 'commercial': '#FD7E14', 'public': '#0B5ED7'}.get(building_type, '#999999')
            
            # 根据建筑类型绘制不同形状
            if building_type == 'residential':
                rect = patches.Rectangle((pos[0]-2, pos[1]-2), 4, 4, 
                                       facecolor=color, edgecolor='black', linewidth=0.5)
                ax3.add_patch(rect)
            elif building_type == 'commercial':
                circle = patches.Circle((pos[0], pos[1]), 3, 
                                      facecolor=color, edgecolor='black', linewidth=0.5)
                ax3.add_patch(circle)
            elif building_type == 'public':
                hexagon = patches.RegularPolygon((pos[0], pos[1]), 6, radius=3,
                                               facecolor=color, edgecolor='black', linewidth=0.5)
                ax3.add_patch(hexagon)
    
    # 设置图形属性
    ax3.set_xlim(0, 256)
    ax3.set_ylim(0, 256)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    ax3.set_title('Building Distribution')
    
    # 添加图例
    legend_elements = [
        patches.Patch(color='#F6C344', label=f'Residential ({building_counts["residential"]})'),
        patches.Patch(color='#FD7E14', label=f'Commercial ({building_counts["commercial"]})'),
        patches.Patch(color='#0B5ED7', label=f'Public ({building_counts["public"]})')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    # 右下图：SDF值分布直方图
    ax4.hist(sdf_field.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    # 标记阈值线
    ax4.axvline(x=0.55, color='orange', linestyle='--', linewidth=2, label='Residential Threshold (0.55)')
    ax4.axvline(x=0.85, color='red', linestyle='--', linewidth=2, label='Commercial Threshold (0.85)')
    
    # 标记等值线值
    for level in commercial_levels:
        ax4.axvline(x=level, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    
    for level in residential_levels:
        ax4.axvline(x=level, color='blue', linestyle=':', linewidth=1, alpha=0.5)
    
    ax4.set_title('SDF Value Distribution with Isocontour Levels')
    ax4.set_xlabel('SDF Value')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"\n📊 可视化统计:")
    print(f"   SDF场范围: [{sdf_min:.3f}, {sdf_max:.3f}]")
    print(f"   商业等值线数量: {len(commercial_levels)}")
    print(f"   住宅等值线数量: {len(residential_levels)}")
    print(f"   建筑总数: {sum(building_counts.values())}")
    print(f"   住宅建筑: {building_counts['residential']}")
    print(f"   商业建筑: {building_counts['commercial']}")
    print(f"   公共建筑: {building_counts['public']}")

if __name__ == "__main__":
    visualize_contours_and_buildings()


