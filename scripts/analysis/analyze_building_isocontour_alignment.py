#!/usr/bin/env python3
"""
分析建筑位置和等值线对应关系的脚本
检查为什么建筑放置和等值线对应不上
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def analyze_building_isocontour_alignment():
    """分析建筑位置和等值线的对应关系"""
    output_dir = 'enhanced_simulation_v2_3_output'
    
    print("🔍 建筑位置与等值线对应关系分析")
    print("=" * 60)
    
    # 加载数据
    building_data = {}
    sdf_data = {}
    
    # 加载建筑数据
    building_files = glob.glob(f'{output_dir}/building_positions_month_*.json')
    for file_path in building_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                month_str = data['timestamp']
                month_num = int(month_str.split('_')[1])
                building_data[month_num] = data
        except Exception as e:
            print(f"Failed to load building data {file_path}: {e}")
    
    # 加载SDF数据
    sdf_files = glob.glob(f'{output_dir}/sdf_field_month_*.json')
    for file_path in sdf_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                month = data['month']
                sdf_data[month] = data
        except Exception as e:
            print(f"Failed to load SDF data {file_path}: {e}")
    
    # 分析每个月份
    for month in sorted(building_data.keys()):
        if month not in sdf_data:
            continue
            
        print(f"\n📊 月份 {month} 分析:")
        print("-" * 40)
        
        buildings = building_data[month]['buildings']
        sdf_field = np.array(sdf_data[month]['sdf_field'])
        
        # 分析建筑类型分布
        residential_buildings = [b for b in buildings if b['type'] == 'residential']
        commercial_buildings = [b for b in buildings if b['type'] == 'commercial']
        public_buildings = [b for b in buildings if b['type'] == 'public']
        
        print(f"  建筑总数: {len(buildings)}")
        print(f"  住宅建筑: {len(residential_buildings)}")
        print(f"  商业建筑: {len(commercial_buildings)}")
        print(f"  公共建筑: {len(public_buildings)}")
        
        # 分析SDF值分布
        sdf_values = [b.get('sdf_value', 0.0) for b in buildings]
        if sdf_values:
            print(f"  SDF值范围: {min(sdf_values):.3f} - {max(sdf_values):.3f}")
            print(f"  平均SDF值: {np.mean(sdf_values):.3f}")
        
        # 检查建筑位置是否在对应的等值线上
        print(f"\n  🎯 等值线对应检查:")
        
        # 商业建筑应该在商业等值线上
        commercial_levels = [0.85, 0.70, 0.55]
        for level in commercial_levels:
            level_buildings = [b for b in commercial_buildings if abs(b.get('sdf_value', 0.0) - level) < 0.05]
            print(f"    商业等值线 {level}: {len(level_buildings)} 个建筑")
        
        # 住宅建筑应该在住宅等值线上
        residential_levels = [0.55, 0.40, 0.25]
        for level in residential_levels:
            level_buildings = [b for b in residential_buildings if abs(b.get('sdf_value', 0.0) - level) < 0.05]
            print(f"    住宅等值线 {level}: {len(level_buildings)} 个建筑")
        
        # 创建可视化
        create_alignment_visualization(month, buildings, sdf_field, residential_levels, commercial_levels)

def create_alignment_visualization(month, buildings, sdf_field, residential_levels, commercial_levels):
    """创建对齐可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左侧：建筑位置和等值线
    ax1.set_xlim(0, 256)
    ax1.set_ylim(0, 256)
    ax1.set_aspect('equal')
    
    # 绘制主干道
    trunk_road = [[40, 128], [216, 128]]
    x_coords = [trunk_road[0][0], trunk_road[1][0]]
    y_coords = [trunk_road[0][1], trunk_road[1][1]]
    ax1.plot(x_coords, y_coords, 'k-', linewidth=3, alpha=0.7, label='Trunk Road')
    
    # 绘制交通枢纽
    for i, hub in enumerate(trunk_road):
        ax1.plot(hub[0], hub[1], 'o', markersize=10, color='blue', 
                markeredgecolor='black', markeredgewidth=2, label=f'Hub {chr(65+i)}' if i == 0 else "")
    
    # 绘制等值线
    y_coords = np.arange(sdf_field.shape[0])
    x_coords = np.arange(sdf_field.shape[1])
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # 商业等值线
    for level in commercial_levels:
        try:
            if np.min(sdf_field) <= level <= np.max(sdf_field):
                contour = ax1.contour(X, Y, sdf_field, levels=[level], colors='red', 
                                    linestyles='dashed', alpha=0.6, linewidths=2)
        except:
            pass
    
    # 住宅等值线
    for level in residential_levels:
        try:
            if np.min(sdf_field) <= level <= np.max(sdf_field):
                contour = ax1.contour(X, Y, sdf_field, levels=[level], colors='blue', 
                                    linestyles='dashed', alpha=0.6, linewidths=2)
        except:
            pass
    
    # 绘制建筑
    for building in buildings:
        pos = building['position']
        building_type = building['type']
        sdf_value = building.get('sdf_value', 0.0)
        
        if building_type == 'residential':
            rect = patches.Rectangle((pos[0]-2, pos[1]-2), 4, 4, 
                                   facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1)
            ax1.add_patch(rect)
            # 添加SDF值标签
            ax1.text(pos[0]+3, pos[1], f'{sdf_value:.2f}', fontsize=6, color='black')
            
        elif building_type == 'commercial':
            circle = patches.Circle((pos[0], pos[1]), radius=3, 
                                  facecolor='orange', alpha=0.8, edgecolor='black', linewidth=1)
            ax1.add_patch(circle)
            # 添加SDF值标签
            ax1.text(pos[0]+3, pos[1], f'{sdf_value:.2f}', fontsize=6, color='black')
            
        elif building_type == 'public':
            triangle = patches.RegularPolygon((pos[0], pos[1]), numVertices=3, radius=4,
                                            facecolor='cyan', alpha=0.8, edgecolor='black', linewidth=1)
            ax1.add_patch(triangle)
            # 添加SDF值标签
            ax1.text(pos[0]+3, pos[1], f'{sdf_value:.2f}', fontsize=6, color='black')
    
    ax1.set_title(f'Month {month:02d} - Buildings and Isocontours', fontsize=14)
    ax1.legend()
    
    # 右侧：SDF值分布直方图
    sdf_values = [b.get('sdf_value', 0.0) for b in buildings]
    if sdf_values:
        ax2.hist(sdf_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 标记等值线位置
        for level in commercial_levels:
            ax2.axvline(x=level, color='red', linestyle='--', alpha=0.7, label=f'Commercial {level}')
        for level in residential_levels:
            ax2.axvline(x=level, color='blue', linestyle='--', alpha=0.7, label=f'Residential {level}')
        
        ax2.set_xlabel('SDF Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Month {month:02d} - SDF Value Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def check_isocontour_generation_logic():
    """检查等值线生成逻辑"""
    print("\n🔧 等值线生成逻辑检查:")
    print("=" * 40)
    
    # 检查配置文件
    try:
        with open('configs/city_config_v2_3.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        isocontour_config = config.get('isocontour_layout', {})
        print(f"  等值线配置:")
        print(f"    商业起始值: {isocontour_config.get('commercial', {}).get('start_P', 'N/A')}")
        print(f"    住宅起始值: {isocontour_config.get('residential', {}).get('start_P', 'N/A')}")
        print(f"    回退分位数: {isocontour_config.get('fallback_percentiles', 'N/A')}")
        
    except Exception as e:
        print(f"  无法读取配置文件: {e}")
    
    # 检查几何等值线系统
    print(f"\n  📐 几何等值线系统:")
    print(f"    应该使用 marching squares 算法")
    print(f"    应该使用等弧长采样")
    print(f"    应该应用法向偏移和切向抖动")

if __name__ == "__main__":
    analyze_building_isocontour_alignment()
    check_isocontour_generation_logic()


