#!/usr/bin/env python3
"""
分析 Hub3 生长逻辑问题
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def analyze_hub3_growth():
    """分析 Hub3 生长逻辑问题"""
    print("🔍 分析 Hub3 生长逻辑问题...")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Hub3 位置
    hub3 = [67, 94]
    hub3_x, hub3_y = hub3[0], hub3[1]
    
    print(f"📍 Hub3 位置: ({hub3_x}, {hub3_y})")
    
    # 分析建筑数据
    building_files = sorted(glob.glob("enhanced_simulation_v3_1_output/building_positions_month_*.json"))
    
    hub3_buildings_by_month = {}
    hub3_building_counts = []
    months = []
    
    print(f"\n📊 分析 Hub3 区域建筑变化:")
    
    for file_path in building_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 从文件名提取月份
            filename = os.path.basename(file_path)
            if 'month_' in filename:
                month_str = filename.split('month_')[1].split('.')[0]
                month = int(month_str)
            else:
                month = data.get('month', 0)
            
            months.append(month)
            
            # 分析 Hub3 附近的建筑
            buildings = data.get('buildings', [])
            hub3_buildings = []
            
            for building in buildings:
                pos = building.get('position', [0, 0])
                x, y = pos[0], pos[1]
                
                # 检查是否在 Hub3 附近（30像素范围内）
                dist = np.sqrt((x - hub3_x)**2 + (y - hub3_y)**2)
                if dist <= 30:
                    hub3_buildings.append({
                        'type': building.get('type', 'unknown'),
                        'position': pos,
                        'distance': dist
                    })
            
            hub3_buildings_by_month[month] = hub3_buildings
            hub3_building_counts.append(len(hub3_buildings))
            
            print(f"第 {month:2d} 个月: {len(hub3_buildings)} 个建筑")
            
            # 显示建筑详情
            if hub3_buildings:
                for i, building in enumerate(hub3_buildings):
                    print(f"  - {building['type']}: ({building['position'][0]:.1f}, {building['position'][1]:.1f}), 距离: {building['distance']:.1f}")
            
        except Exception as e:
            print(f"⚠️ 加载文件失败: {file_path}, 错误: {e}")
    
    # 分析地价场变化
    print(f"\n📊 分析 Hub3 地价场变化:")
    
    land_price_files = sorted(glob.glob("enhanced_simulation_v3_1_output/land_price_frame_month_*.json"))
    
    hub3_land_values = []
    land_price_months = []
    
    for file_path in land_price_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            month = data.get('month', 0)
            land_price_field = np.array(data.get('land_price_field', []))
            
            if land_price_field.size > 0:
                # 获取 Hub3 的地价值
                hub3_value = land_price_field[hub3_y, hub3_x]
                hub3_land_values.append(hub3_value)
                land_price_months.append(month)
                
                print(f"第 {month:2d} 个月: Hub3 地价值 = {hub3_value:.3f}")
            
        except Exception as e:
            print(f"⚠️ 加载地价场文件失败: {file_path}, 错误: {e}")
    
    # 分析层状态
    print(f"\n📊 分析层状态变化:")
    
    layer_files = sorted(glob.glob("enhanced_simulation_v3_1_output/layer_state_month_*.json"))
    
    for file_path in layer_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            month = data.get('month', 0)
            layers = data.get('layers', {})
            
            print(f"\n第 {month:2d} 个月层状态:")
            
            # 商业建筑层
            if 'commercial' in layers:
                commercial_layers = layers['commercial']
                print(f"  商业建筑层:")
                for i, layer in enumerate(commercial_layers):
                    status = layer.get('status', 'unknown')
                    density = layer.get('density', 0)
                    placed = layer.get('placed', 0)
                    capacity = layer.get('capacity_effective', 0)
                    print(f"    P{i}: {status}, 密度: {density:.1%}, 已放置: {placed}/{capacity}")
            
            # 住宅建筑层
            if 'residential' in layers:
                residential_layers = layers['residential']
                print(f"  住宅建筑层:")
                for i, layer in enumerate(residential_layers):
                    status = layer.get('status', 'unknown')
                    density = layer.get('density', 0)
                    placed = layer.get('placed', 0)
                    capacity = layer.get('capacity_effective', 0)
                    print(f"    P{i}: {status}, 密度: {density:.1%}, 已放置: {placed}/{capacity}")
            
        except Exception as e:
            print(f"⚠️ 加载层状态文件失败: {file_path}, 错误: {e}")
    
    # 可视化分析结果
    plt.figure(figsize=(15, 10))
    
    # 1. Hub3 建筑数量变化
    plt.subplot(2, 3, 1)
    plt.plot(months, hub3_building_counts, 'bo-', linewidth=2, markersize=6)
    plt.title('Hub3 区域建筑数量变化')
    plt.xlabel('月份')
    plt.ylabel('建筑数量')
    plt.grid(True, alpha=0.3)
    
    # 2. Hub3 地价值变化
    plt.subplot(2, 3, 2)
    plt.plot(land_price_months, hub3_land_values, 'ro-', linewidth=2, markersize=6)
    plt.title('Hub3 地价值变化')
    plt.xlabel('月份')
    plt.ylabel('地价值')
    plt.grid(True, alpha=0.3)
    
    # 3. 建筑数量 vs 地价值
    plt.subplot(2, 3, 3)
    # 对齐月份数据
    aligned_months = []
    aligned_buildings = []
    aligned_land_values = []
    
    for month in months:
        if month in hub3_buildings_by_month and month in land_price_months:
            aligned_months.append(month)
            aligned_buildings.append(len(hub3_buildings_by_month[month]))
            # 找到对应的地价值
            land_idx = land_price_months.index(month)
            aligned_land_values.append(hub3_land_values[land_idx])
    
    if aligned_months:
        plt.scatter(aligned_land_values, aligned_buildings, c=aligned_months, cmap='viridis', s=100)
        plt.colorbar(label='月份')
        plt.title('Hub3 建筑数量 vs 地价值')
        plt.xlabel('地价值')
        plt.ylabel('建筑数量')
        plt.grid(True, alpha=0.3)
    
    # 4. 建筑类型分布
    plt.subplot(2, 3, 4)
    all_building_types = {}
    for month, buildings in hub3_buildings_by_month.items():
        for building in buildings:
            btype = building['type']
            all_building_types[btype] = all_building_types.get(btype, 0) + 1
    
    if all_building_types:
        labels = list(all_building_types.keys())
        values = list(all_building_types.values())
        colors = ['#F6C344', '#FD7E14', '#22A6B3']
        
        plt.pie(values, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%')
        plt.title('Hub3 建筑类型分布')
    
    # 5. 建筑距离分布
    plt.subplot(2, 3, 5)
    all_distances = []
    for month, buildings in hub3_buildings_by_month.items():
        for building in buildings:
            all_distances.append(building['distance'])
    
    if all_distances:
        plt.hist(all_distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Hub3 建筑距离分布')
        plt.xlabel('距离 Hub3 中心 (像素)')
        plt.ylabel('建筑数量')
        plt.grid(True, alpha=0.3)
    
    # 6. 时间线分析
    plt.subplot(2, 3, 6)
    # 标记有建筑和无建筑的月份
    has_buildings = [1 if count > 0 else 0 for count in hub3_building_counts]
    no_buildings = [1 if count == 0 else 0 for count in hub3_building_counts]
    
    plt.bar(months, has_buildings, color='green', alpha=0.7, label='有建筑')
    plt.bar(months, no_buildings, color='red', alpha=0.7, label='无建筑')
    plt.title('Hub3 建筑存在时间线')
    plt.xlabel('月份')
    plt.ylabel('建筑状态')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hub3_growth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 总结分析
    print(f"\n📋 Hub3 生长分析总结:")
    print(f"总月份数: {len(months)}")
    print(f"有建筑的月份: {sum(1 for count in hub3_building_counts if count > 0)}")
    print(f"无建筑的月份: {sum(1 for count in hub3_building_counts if count == 0)}")
    
    if hub3_building_counts:
        print(f"最大建筑数量: {max(hub3_building_counts)}")
        print(f"平均建筑数量: {np.mean(hub3_building_counts):.1f}")
    
    if hub3_land_values:
        print(f"地价值范围: [{min(hub3_land_values):.3f}, {max(hub3_land_values):.3f}]")
    
    # 找出建筑消失的月份
    print(f"\n🔍 建筑消失分析:")
    for i, count in enumerate(hub3_building_counts):
        if i > 0 and hub3_building_counts[i-1] > 0 and count == 0:
            print(f"第 {months[i]} 个月: 建筑从 {hub3_building_counts[i-1]} 个减少到 0 个")
    
    print("\n✅ Hub3 生长分析完成！结果已保存到 hub3_growth_analysis.png")

if __name__ == "__main__":
    analyze_hub3_growth()
