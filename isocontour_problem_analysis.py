#!/usr/bin/env python3
"""
等值线建筑对应问题分析
详细分析为什么建筑位置和等值线对应不上
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def analyze_isocontour_problem():
    """分析等值线建筑对应问题"""
    print("🔍 等值线建筑对应问题分析")
    print("=" * 60)
    
    # 加载数据
    building_data = {}
    sdf_data = {}
    
    # 加载建筑数据
    building_files = glob.glob('enhanced_simulation_v2_3_output/building_positions_month_*.json')
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
    sdf_files = glob.glob('enhanced_simulation_v2_3_output/sdf_field_month_*.json')
    for file_path in sdf_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                month = data['month']
                sdf_data[month] = data
        except Exception as e:
            print(f"Failed to load SDF data {file_path}: {e}")
    
    # 分析问题
    print("\n🎯 问题分析:")
    print("=" * 40)
    
    # 1. 检查等值线生成逻辑
    print("\n1️⃣ 等值线生成逻辑问题:")
    print("   - 几何等距等值线使用公式: P_k = P₀ · exp(-k·Δd/λ⊥)")
    print("   - 商业建筑: start_P = 0.85, depth_m = 20, gap_m = 10")
    print("   - 住宅建筑: start_P = 0.55, depth_m = 14, gap_m = 26")
    print("   - 但实际建筑SDF值分布与理论等值线不匹配")
    
    # 2. 检查建筑放置逻辑
    print("\n2️⃣ 建筑放置逻辑问题:")
    print("   - 建筑应该放置在等值线上")
    print("   - 使用等弧长采样 + 法向偏移 + 切向抖动")
    print("   - 但实际建筑位置与等值线有偏差")
    
    # 3. 检查SDF值记录
    print("\n3️⃣ SDF值记录问题:")
    print("   - 建筑数据中的sdf_value是建筑位置处的SDF值")
    print("   - 但建筑可能不在等值线上，导致SDF值不匹配")
    
    # 分析具体数据
    for month in [0, 21]:  # 分析开始和结束月份
        if month not in building_data or month not in sdf_data:
            continue
            
        print(f"\n📊 月份 {month} 详细分析:")
        print("-" * 40)
        
        buildings = building_data[month]['buildings']
        sdf_field = np.array(sdf_data[month]['sdf_field'])
        
        # 分析商业建筑
        commercial_buildings = [b for b in buildings if b['type'] == 'commercial']
        if commercial_buildings:
            print(f"  商业建筑分析:")
            sdf_values = [b.get('sdf_value', 0.0) for b in commercial_buildings]
            print(f"    SDF值范围: {min(sdf_values):.3f} - {max(sdf_values):.3f}")
            print(f"    理论等值线: [0.85, 0.70, 0.55]")
            
            # 检查是否在理论等值线上
            on_contour_count = 0
            for building in commercial_buildings:
                sdf_val = building.get('sdf_value', 0.0)
                for level in [0.85, 0.70, 0.55]:
                    if abs(sdf_val - level) < 0.05:
                        on_contour_count += 1
                        break
            
            print(f"    在理论等值线上的建筑: {on_contour_count}/{len(commercial_buildings)}")
        
        # 分析住宅建筑
        residential_buildings = [b for b in buildings if b['type'] == 'residential']
        if residential_buildings:
            print(f"  住宅建筑分析:")
            sdf_values = [b.get('sdf_value', 0.0) for b in residential_buildings]
            print(f"    SDF值范围: {min(sdf_values):.3f} - {max(sdf_values):.3f}")
            print(f"    理论等值线: [0.55, 0.40, 0.25]")
            
            # 检查是否在理论等值线上
            on_contour_count = 0
            for building in residential_buildings:
                sdf_val = building.get('sdf_value', 0.0)
                for level in [0.55, 0.40, 0.25]:
                    if abs(sdf_val - level) < 0.05:
                        on_contour_count += 1
                        break
            
            print(f"    在理论等值线上的建筑: {on_contour_count}/{len(residential_buildings)}")
    
    # 4. 根本原因分析
    print("\n4️⃣ 根本原因分析:")
    print("   a) 等值线生成问题:")
    print("      - 几何等距等值线可能没有正确生成")
    print("      - 可能使用了分位数回退机制")
    print("      - marching squares算法可能有问题")
    
    print("   b) 建筑放置问题:")
    print("      - 等弧长采样可能不正确")
    print("      - 法向偏移可能过大")
    print("      - 位置验证可能过于宽松")
    
    print("   c) 数据记录问题:")
    print("      - 建筑位置处的SDF值可能与等值线值不同")
    print("      - 可能存在坐标转换问题")
    
    # 5. 解决方案
    print("\n5️⃣ 解决方案:")
    print("   a) 修复等值线生成:")
    print("      - 确保几何等距等值线正确生成")
    print("      - 验证marching squares算法")
    print("      - 检查分位数回退逻辑")
    
    print("   b) 修复建筑放置:")
    print("      - 确保建筑严格放置在等值线上")
    print("      - 减少法向偏移距离")
    print("      - 加强位置验证")
    
    print("   c) 修复数据记录:")
    print("      - 记录等值线值而不是位置SDF值")
    print("      - 确保坐标系统一致")

def create_problem_visualization():
    """创建问题可视化"""
    print("\n🎨 创建问题可视化...")
    
    # 加载数据
    month = 21  # 使用最后一个月的数据
    
    try:
        with open(f'enhanced_simulation_v2_3_output/building_positions_month_{month:02d}.json', 'r') as f:
            building_data = json.load(f)
        
        with open(f'enhanced_simulation_v2_3_output/sdf_field_month_{month}.json', 'r') as f:
            sdf_data = json.load(f)
        
        buildings = building_data['buildings']
        sdf_field = np.array(sdf_data['sdf_field'])
        
        # 创建可视化
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
        commercial_levels = [0.85, 0.70, 0.55]
        for level in commercial_levels:
            try:
                if np.min(sdf_field) <= level <= np.max(sdf_field):
                    contour = ax1.contour(X, Y, sdf_field, levels=[level], colors='red', 
                                        linestyles='dashed', alpha=0.6, linewidths=2)
            except:
                pass
        
        # 住宅等值线
        residential_levels = [0.55, 0.40, 0.25]
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
        
        ax1.set_title(f'Month {month} - Buildings vs Isocontours (Problem)', fontsize=14)
        ax1.legend()
        
        # 右侧：SDF值分布
        sdf_values = [b.get('sdf_value', 0.0) for b in buildings]
        if sdf_values:
            ax2.hist(sdf_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            
            # 标记理论等值线位置
            for level in commercial_levels:
                ax2.axvline(x=level, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Commercial {level}')
            for level in residential_levels:
                ax2.axvline(x=level, color='blue', linestyle='--', alpha=0.7, linewidth=2, label=f'Residential {level}')
            
            ax2.set_xlabel('SDF Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Month {month} - SDF Value Distribution vs Theory')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"可视化创建失败: {e}")

if __name__ == "__main__":
    analyze_isocontour_problem()
    create_problem_visualization()


