#!/usr/bin/env python3
"""
分析v3.3系统中两个hub的集聚效应
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from enhanced_city_simulation_v3_3 import GaussianLandPriceSystemV3_3, ContourExtractionSystemV3_3

def analyze_hub_agglomeration():
    """分析hub集聚效应"""
    print("🔍 分析两个hub的集聚效应...")
    
    # 创建配置
    config = {
        'city': {'meters_per_pixel': 2.0},
        'gaussian_land_price_system': {
            'w_r': 0.6, 'w_c': 0.5, 'w_i': 0.5, 'w_cor': 0.2, 'bias': 0.0,
            'hub_sigma_base_m': 40, 'road_sigma_base_m': 20,
            'hub_growth_rate': 0.03, 'road_growth_rate': 0.02,
            'max_hub_multiplier': 2.0, 'max_road_multiplier': 2.5,
            'normalize': True, 'smoothstep_tau': 0.0
        },
        'isocontour_layout': {
            'commercial': {'levels': [0.85, 0.78, 0.71], 'arc_spacing_m': [25, 35]},
            'industrial': {'levels': [0.60, 0.70, 0.80], 'arc_spacing_m': [35, 55]},
            'residential': {'band': [0.45, 0.65], 'arc_spacing_m': [35, 55]},
            'normal_offset_m': 1.0, 'jitter_m': 0.5
        }
    }
    
    # 初始化系统
    land_price_system = GaussianLandPriceSystemV3_3(config)
    contour_system = ContourExtractionSystemV3_3(config)
    
    # 创建测试地价场
    map_size = [110, 110]
    transport_hubs = [[37, 55], [73, 55]]  # 商业枢纽和工业枢纽
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 获取地价场
    land_price_field = land_price_system.get_land_price_field()
    
    # 分析不同时间点的集聚效应
    time_points = [0, 6, 12, 18, 23]  # 不同月份
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, month in enumerate(time_points):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # 更新地价场到指定月份
        land_price_system.current_month = month
        land_price_system._generate_land_price_field()
        current_land_price_field = land_price_system.get_land_price_field()
        
        # 绘制地价场
        im = ax.imshow(current_land_price_field, cmap='YlOrRd', alpha=0.8)
        
        # 绘制枢纽位置
        ax.scatter(37, 55, c='red', marker='*', s=200, alpha=0.9, 
                  edgecolors='black', linewidth=2, label='Commercial Hub')
        ax.scatter(73, 55, c='blue', marker='*', s=200, alpha=0.9, 
                  edgecolors='black', linewidth=2, label='Industrial Hub')
        
        # 绘制主干道
        ax.axhline(y=55, color='black', linewidth=3, alpha=0.8, label='Main Road')
        
        # 绘制等值线
        levels = [0.2, 0.4, 0.6, 0.8]
        contours = ax.contour(current_land_price_field, levels=levels, 
                             colors=['white', 'yellow', 'orange', 'red'], 
                             linewidths=1, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        
        # 绘制分区半径
        hub_com_radius_px = 350 / 2.0  # 350米转换为像素
        hub_ind_radius_px = 450 / 2.0  # 450米转换为像素
        
        # 商业枢纽分区圆
        circle_com = plt.Circle((37, 55), hub_com_radius_px, fill=False, 
                               color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.add_patch(circle_com)
        
        # 工业枢纽分区圆
        circle_ind = plt.Circle((73, 55), hub_ind_radius_px, fill=False, 
                               color='blue', linestyle='--', alpha=0.7, linewidth=2)
        ax.add_patch(circle_ind)
        
        ax.set_title(f'Month {month} - Hub Agglomeration')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for i in range(len(time_points), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('hub_agglomeration_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 分析集聚强度
    analyze_agglomeration_intensity(land_price_system, map_size, transport_hubs)

def analyze_agglomeration_intensity(land_price_system, map_size, transport_hubs):
    """分析集聚强度"""
    print("\n📊 分析集聚强度...")
    
    # 分析不同月份的地价场强度
    months = range(0, 24, 3)  # 每3个月分析一次
    
    hub_com_pos = transport_hubs[0]
    hub_ind_pos = transport_hubs[1]
    
    # 计算枢纽周围的地价强度
    com_intensities = []
    ind_intensities = []
    
    for month in months:
        land_price_system.current_month = month
        land_price_system._generate_land_price_field()
        current_land_price_field = land_price_system.get_land_price_field()
        
        # 计算商业枢纽周围的地价强度
        com_radius = 20  # 20像素半径
        com_intensity = 0
        com_count = 0
        
        for y in range(max(0, hub_com_pos[1] - com_radius), 
                      min(map_size[1], hub_com_pos[1] + com_radius + 1)):
            for x in range(max(0, hub_com_pos[0] - com_radius), 
                          min(map_size[0], hub_com_pos[0] + com_radius + 1)):
                distance = np.sqrt((x - hub_com_pos[0])**2 + (y - hub_com_pos[1])**2)
                if distance <= com_radius:
                    com_intensity += current_land_price_field[y, x]
                    com_count += 1
        
        com_intensities.append(com_intensity / com_count if com_count > 0 else 0)
        
        # 计算工业枢纽周围的地价强度
        ind_radius = 20  # 20像素半径
        ind_intensity = 0
        ind_count = 0
        
        for y in range(max(0, hub_ind_pos[1] - ind_radius), 
                      min(map_size[1], hub_ind_pos[1] + ind_radius + 1)):
            for x in range(max(0, hub_ind_pos[0] - ind_radius), 
                          min(map_size[0], hub_ind_pos[0] + ind_radius + 1)):
                distance = np.sqrt((x - hub_ind_pos[0])**2 + (y - hub_ind_pos[1])**2)
                if distance <= ind_radius:
                    ind_intensity += current_land_price_field[y, x]
                    ind_count += 1
        
        ind_intensities.append(ind_intensity / ind_count if ind_count > 0 else 0)
    
    # 绘制集聚强度变化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：集聚强度变化
    ax1.plot(months, com_intensities, 'r-o', label='Commercial Hub', linewidth=2, markersize=6)
    ax1.plot(months, ind_intensities, 'b-s', label='Industrial Hub', linewidth=2, markersize=6)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Land Price Intensity')
    ax1.set_title('Hub Agglomeration Intensity Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：集聚强度对比
    x = np.arange(len(months))
    width = 0.35
    
    ax2.bar(x - width/2, com_intensities, width, label='Commercial Hub', color='red', alpha=0.7)
    ax2.bar(x + width/2, ind_intensities, width, label='Industrial Hub', color='blue', alpha=0.7)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Average Land Price Intensity')
    ax2.set_title('Hub Agglomeration Intensity Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(months)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hub_agglomeration_intensity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"  商业枢纽集聚强度: {com_intensities[0]:.3f} → {com_intensities[-1]:.3f}")
    print(f"  工业枢纽集聚强度: {ind_intensities[0]:.3f} → {ind_intensities[-1]:.3f}")

def analyze_building_distribution():
    """分析建筑分布"""
    print("\n🏗️ 分析建筑分布...")
    
    # 检查是否有模拟输出数据
    output_dir = 'enhanced_simulation_v3_3_output'
    if not os.path.exists(output_dir):
        print("  未找到模拟输出数据，跳过建筑分布分析")
        return
    
    # 加载建筑数据
    building_files = [f for f in os.listdir(output_dir) if f.startswith('building_positions_month_')]
    if not building_files:
        print("  未找到建筑位置数据")
        return
    
    # 分析最后一个月的数据
    latest_file = max(building_files)
    with open(os.path.join(output_dir, latest_file), 'r') as f:
        data = json.load(f)
    
    buildings = data['buildings']
    
    # 计算建筑到枢纽的距离分布
    hub_com_pos = [37, 55]
    hub_ind_pos = [73, 55]
    
    com_distances = []
    ind_distances = []
    
    for building_type, building_list in buildings.items():
        if building_type == 'public':
            continue
            
        for building in building_list:
            pos = building['xy']
            
            # 计算到商业枢纽的距离
            dist_to_com = np.sqrt((pos[0] - hub_com_pos[0])**2 + (pos[1] - hub_com_pos[1])**2)
            com_distances.append(dist_to_com)
            
            # 计算到工业枢纽的距离
            dist_to_ind = np.sqrt((pos[0] - hub_ind_pos[0])**2 + (pos[1] - hub_ind_pos[1])**2)
            ind_distances.append(dist_to_ind)
    
    # 绘制距离分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：到商业枢纽的距离分布
    ax1.hist(com_distances, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax1.axvline(350/2.0, color='red', linestyle='--', linewidth=2, label='Zoning Radius (350m)')
    ax1.set_xlabel('Distance to Commercial Hub (pixels)')
    ax1.set_ylabel('Number of Buildings')
    ax1.set_title('Building Distribution vs Commercial Hub')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：到工业枢纽的距离分布
    ax2.hist(ind_distances, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(450/2.0, color='blue', linestyle='--', linewidth=2, label='Zoning Radius (450m)')
    ax2.set_xlabel('Distance to Industrial Hub (pixels)')
    ax2.set_ylabel('Number of Buildings')
    ax2.set_title('Building Distribution vs Industrial Hub')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('building_distribution_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"  总建筑数: {len(com_distances)}")
    print(f"  商业枢纽附近建筑数: {sum(1 for d in com_distances if d <= 350/2.0)}")
    print(f"  工业枢纽附近建筑数: {sum(1 for d in ind_distances if d <= 450/2.0)}")

def main():
    """主函数"""
    print("🔍 v3.3系统hub集聚效应分析")
    
    # 分析hub集聚效应
    analyze_hub_agglomeration()
    
    # 分析建筑分布
    analyze_building_distribution()
    
    print("\n✅ 分析完成！")
    print("  生成的文件:")
    print("  - hub_agglomeration_analysis.png: hub集聚效应分析")
    print("  - hub_agglomeration_intensity.png: 集聚强度变化")
    print("  - building_distribution_analysis.png: 建筑分布分析")

if __name__ == "__main__":
    main()
