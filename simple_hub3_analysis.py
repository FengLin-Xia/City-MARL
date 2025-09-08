#!/usr/bin/env python3
"""
简化版 Hub3 生长分析（无图形显示）
"""

import json
import numpy as np
import glob
import os

def simple_hub3_analysis():
    """简化版 Hub3 生长分析"""
    print("🔍 简化版 Hub3 生长分析...")
    
    # Hub3 位置
    hub3 = [67, 94]
    hub3_x, hub3_y = hub3[0], hub3[1]
    
    print(f"📍 Hub3 位置: ({hub3_x}, {hub3_y})")
    
    # 分析建筑数据
    building_files = sorted(glob.glob("enhanced_simulation_v3_1_output/building_positions_month_*.json"))
    
    hub3_building_counts = []
    months = []
    hub3_buildings_by_month = {}
    
    print(f"\n📊 Hub3 区域建筑变化:")
    
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
            count = len(hub3_buildings)
            hub3_building_counts.append(count)
            
            print(f"第 {month:2d} 个月: {count} 个建筑")
            
            # 显示建筑详情（仅前几个月份）
            if hub3_buildings and month <= 5:
                for i, building in enumerate(hub3_buildings):
                    print(f"  - {building['type']}: ({building['position'][0]:.1f}, {building['position'][1]:.1f}), 距离: {building['distance']:.1f}")
            
        except Exception as e:
            print(f"⚠️ 加载文件失败: {file_path}, 错误: {e}")
    
    # 分析地价场变化
    print(f"\n📊 Hub3 地价场变化:")
    
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
                
                if month <= 10:  # 只显示前10个月
                    print(f"第 {month:2d} 个月: Hub3 地价值 = {hub3_value:.3f}")
            
        except Exception as e:
            print(f"⚠️ 加载地价场文件失败: {file_path}, 错误: {e}")
    
    # 分析层状态（仅前几个月份）
    print(f"\n📊 层状态变化（前5个月）:")
    
    layer_files = sorted(glob.glob("enhanced_simulation_v3_1_output/layer_state_month_*.json"))
    
    for file_path in layer_files[:5]:  # 只分析前5个月
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
                for i, layer in enumerate(commercial_layers[:3]):  # 只显示前3层
                    status = layer.get('status', 'unknown')
                    density = layer.get('density', 0)
                    placed = layer.get('placed', 0)
                    capacity = layer.get('capacity_effective', 0)
                    print(f"    P{i}: {status}, 密度: {density:.1%}, 已放置: {placed}/{capacity}")
            
            # 住宅建筑层
            if 'residential' in layers:
                residential_layers = layers['residential']
                print(f"  住宅建筑层:")
                for i, layer in enumerate(residential_layers[:3]):  # 只显示前3层
                    status = layer.get('status', 'unknown')
                    density = layer.get('density', 0)
                    placed = layer.get('placed', 0)
                    capacity = layer.get('capacity_effective', 0)
                    print(f"    P{i}: {status}, 密度: {density:.1%}, 已放置: {placed}/{capacity}")
            
        except Exception as e:
            print(f"⚠️ 加载层状态文件失败: {file_path}, 错误: {e}")
    
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
    
    # 找出建筑重新出现的月份
    print(f"\n🔍 建筑重新出现分析:")
    for i, count in enumerate(hub3_building_counts):
        if i > 0 and hub3_building_counts[i-1] == 0 and count > 0:
            print(f"第 {months[i]} 个月: 建筑从 0 个增加到 {count} 个")
    
    # 分析建筑类型变化
    print(f"\n🔍 建筑类型分析:")
    all_building_types = {}
    for month, buildings in hub3_buildings_by_month.items():
        for building in buildings:
            btype = building['type']
            all_building_types[btype] = all_building_types.get(btype, 0) + 1
    
    if all_building_types:
        print("Hub3 区域建筑类型统计:")
        for btype, count in all_building_types.items():
            print(f"  {btype}: {count} 个")
    
    print("\n✅ Hub3 生长分析完成！")

if __name__ == "__main__":
    simple_hub3_analysis()
