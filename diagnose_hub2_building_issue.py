#!/usr/bin/env python3
"""
诊断Hub 2没有建筑生成的问题
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def diagnose_hub2_issue():
    """诊断Hub 2建筑生成问题"""
    
    print("🔍 诊断Hub 2建筑生成问题")
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
    
    # 分析SDF场在枢纽周围的分布
    print(f"\n📊 SDF场枢纽周围分析:")
    
    for i, hub in enumerate(hubs):
        hub_x, hub_y = hub[0], hub[1]
        print(f"\n  Hub {i+1} ({hub_x}, {hub_y}):")
        
        # 枢纽位置的SDF值
        hub_sdf = sdf_field[hub_y, hub_x]
        print(f"    枢纽SDF值: {hub_sdf:.3f}")
        
        # 枢纽周围区域的SDF值范围
        x_min, x_max = max(0, hub_x - 50), min(256, hub_x + 50)
        y_min, y_max = max(0, hub_y - 50), min(256, hub_y + 50)
        
        hub_region = sdf_field[y_min:y_max, x_min:x_max]
        region_min, region_max = np.min(hub_region), np.max(hub_region)
        region_mean = np.mean(hub_region)
        
        print(f"    周围区域SDF范围: [{region_min:.3f}, {region_max:.3f}]")
        print(f"    周围区域SDF均值: {region_mean:.3f}")
        
        # 分析等值线阈值
        sdf_flat = sdf_field.flatten()
        commercial_95 = np.percentile(sdf_flat, 95)
        commercial_90 = np.percentile(sdf_flat, 90)
        commercial_85 = np.percentile(sdf_flat, 85)
        residential_80 = np.percentile(sdf_flat, 80)
        residential_70 = np.percentile(sdf_flat, 70)
        
        print(f"    商业等值线阈值: 95%={commercial_95:.3f}, 90%={commercial_90:.3f}, 85%={commercial_85:.3f}")
        print(f"    住宅等值线阈值: 80%={residential_80:.3f}, 70%={residential_70:.3f}")
        
        # 检查枢纽周围是否有足够的SDF值
        high_sdf_pixels = np.sum(hub_region >= commercial_85)
        total_pixels = hub_region.size
        high_sdf_ratio = (high_sdf_pixels / total_pixels) * 100
        
        print(f"    高SDF值像素比例: {high_sdf_ratio:.1f}%")
        
        if high_sdf_ratio < 5:
            print(f"    ⚠️ 警告: Hub {i+1}周围高SDF值区域过少")
    
    # 分析建筑分布
    print(f"\n🏗️ 建筑分布分析:")
    
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
    
    # 分析建筑到枢纽的距离分布
    print(f"\n📏 建筑到枢纽距离分析:")
    
    for building_type, positions in building_positions.items():
        if not positions:
            continue
            
        print(f"\n  {building_type.title()}建筑:")
        
        for i, hub in enumerate(hubs):
            distances = []
            for pos in positions:
                dist = np.sqrt((pos[0] - hub[0])**2 + (pos[1] - hub[1])**2)
                distances.append(dist)
            
            min_dist = min(distances)
            max_dist = max(distances)
            mean_dist = np.mean(distances)
            
            print(f"    到Hub {i+1}距离: 最小={min_dist:.1f}, 最大={max_dist:.1f}, 平均={mean_dist:.1f}")
            
            # 检查是否有建筑在分带范围内
            if building_type == 'residential':
                in_residential_zone = sum(1 for d in distances if 60 <= d <= 300)
                print(f"    在住宅分带内: {in_residential_zone}/{len(positions)}")
    
    # 分析SDF场整体分布
    print(f"\n📊 SDF场整体分布分析:")
    
    sdf_min, sdf_max = np.min(sdf_field), np.max(sdf_field)
    sdf_mean = np.mean(sdf_field)
    sdf_std = np.std(sdf_field)
    
    print(f"  SDF场范围: [{sdf_min:.3f}, {sdf_max:.3f}]")
    print(f"  SDF场均值: {sdf_mean:.3f}")
    print(f"  SDF场标准差: {sdf_std:.3f}")
    
    # 分析分位数分布
    percentiles = [50, 60, 70, 80, 85, 90, 95]
    sdf_percentiles = np.percentile(sdf_field.flatten(), percentiles)
    
    print(f"  SDF分位数分布:")
    for p, val in zip(percentiles, sdf_percentiles):
        print(f"    {p}%: {val:.3f}")
    
    # 创建可视化
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Hub 2 Building Generation Issue Diagnosis', fontsize=16)
        
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
        
        # 右上图：建筑分布
        ax2.clear()
        
        # 绘制主干道
        ax2.plot([hubs[0][0], hubs[1][0]], [hubs[0][1], hubs[1][1]], 
                 color='gray', linewidth=3, label='Trunk Road')
        
        # 绘制枢纽
        for i, hub in enumerate(hubs):
            ax2.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
        
        # 绘制建筑
        colors = {'residential': '#F6C344', 'commercial': '#FD7E14', 'public': '#22A6B3'}
        
        for building_type, positions in building_positions.items():
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                ax2.scatter(x_coords, y_coords, c=colors[building_type], s=50, 
                           alpha=0.7, label=f'{building_type.title()} ({len(positions)})')
        
        ax2.set_xlim(0, 256)
        ax2.set_ylim(0, 256)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('Building Distribution')
        ax2.legend()
        
        # 左下图：SDF值分布直方图
        ax3.hist(sdf_field.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        # 标记阈值线
        ax3.axvline(x=sdf_percentiles[4], color='orange', linestyle='--', linewidth=2, label='Commercial 85%')
        ax3.axvline(x=sdf_percentiles[2], color='red', linestyle='--', linewidth=2, label='Residential 70%')
        
        ax3.set_title('SDF Value Distribution with Thresholds')
        ax3.set_xlabel('SDF Value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 右下图：枢纽周围SDF值对比
        ax4.clear()
        
        hub_labels = ['Hub 1', 'Hub 2']
        hub_region_means = []
        
        for hub in hubs:
            x_min, x_max = max(0, hub[0] - 50), min(256, hub[0] + 50)
            y_min, y_max = max(0, hub[1] - 50), min(256, hub[1] + 50)
            hub_region = sdf_field[y_min:y_max, x_min:x_max]
            hub_region_means.append(np.mean(hub_region))
        
        bars = ax4.bar(hub_labels, hub_region_means, color=['red', 'blue'], alpha=0.7)
        ax4.set_title('Average SDF Values Around Hubs')
        ax4.set_ylabel('Average SDF Value')
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, hub_region_means):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print("\n🎨 可视化完成！")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
    
    # 总结问题
    print(f"\n🔍 问题诊断总结:")
    
    if building_counts['residential'] == 0:
        print(f"  ❌ 主要问题: 住宅建筑数量为0")
        print(f"    可能原因: 分带检查失败或等值线提取失败")
    
    if building_counts['commercial'] > 0:
        # 检查商业建筑是否集中在Hub 1
        if building_positions['commercial']:
            com_x = [pos[0] for pos in building_positions['commercial']]
            com_mean_x = np.mean(com_x)
            
            if com_mean_x < 128:  # 如果平均X坐标小于地图中心
                print(f"  ⚠️ 商业建筑集中在Hub 1周围")
                print(f"    平均X坐标: {com_mean_x:.1f} (地图中心: 128)")
                print(f"    可能原因: Hub 2周围SDF值不足或等值线生成失败")
    
    print(f"\n💡 建议解决方案:")
    print(f"  1. 检查Hub 2周围的SDF值分布")
    print(f"  2. 验证等值线提取算法是否正常工作")
    print(f"  3. 检查分带逻辑是否过于严格")
    print(f"  4. 考虑调整等值线阈值或分带参数")

if __name__ == "__main__":
    diagnose_hub2_issue()


