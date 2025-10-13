#!/usr/bin/env python3
"""
测试改进后的哑铃状配置
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from enhanced_city_simulation_v3_3 import EnhancedCitySimulationV3_3

def test_dumbbell_config():
    """测试哑铃状配置"""
    print("🧪 测试改进后的哑铃状配置...")
    
    # 加载改进的配置
    with open('configs/city_config_v3_3_dumbbell.json', 'r') as f:
        config = json.load(f)
    
    # 创建模拟系统
    simulation = EnhancedCitySimulationV3_3(config)
    simulation.initialize_simulation()
    
    # 运行短期模拟（6个月）
    print("  运行6个月模拟...")
    simulation.run_simulation(total_months=6)
    
    # 分析结果
    analyze_dumbbell_results()

def analyze_dumbbell_results():
    """分析哑铃状结果"""
    print("\n📊 分析哑铃状结果...")
    
    # 加载建筑数据
    import os
    output_dir = 'enhanced_simulation_v3_3_output'
    
    if not os.path.exists(output_dir):
        print("  未找到输出数据")
        return
    
    # 重建完整的建筑状态
    buildings = rebuild_building_state(output_dir, 6)  # 重建到第6个月
    
    # 枢纽位置
    hub_com = [28, 55]  # 改进后的位置
    hub_ind = [82, 55]
    
    # 分析建筑分布
    print(f"  枢纽位置: 商业{hub_com}, 工业{hub_ind}")
    print(f"  枢纽距离: {np.sqrt((hub_ind[0]-hub_com[0])**2 + (hub_ind[1]-hub_com[1])**2)*2:.1f}米")
    
    # 统计各类型建筑数量
    total_buildings = 0
    for building_type, building_list in buildings.items():
        if building_type != 'public':
            count = len(building_list)
            total_buildings += count
            print(f"  {building_type}: {count}个")
    
    print(f"  总建筑数: {total_buildings}")
    
    # 分析建筑到枢纽的距离分布
    com_distances = []
    ind_distances = []
    
    for building_type, building_list in buildings.items():
        if building_type == 'public':
            continue
            
        for building in building_list:
            pos = building['xy']
            
            # 计算到商业枢纽的距离
            dist_to_com = np.sqrt((pos[0] - hub_com[0])**2 + (pos[1] - hub_com[1])**2) * 2.0
            com_distances.append(dist_to_com)
            
            # 计算到工业枢纽的距离
            dist_to_ind = np.sqrt((pos[0] - hub_ind[0])**2 + (pos[1] - hub_ind[1])**2) * 2.0
            ind_distances.append(dist_to_ind)
    
    # 分析集聚效果
    com_nearby = sum(1 for d in com_distances if d <= 200)  # 200米内
    ind_nearby = sum(1 for d in ind_distances if d <= 250)  # 250米内
    
    print(f"\n  集聚效果分析:")
    print(f"  商业枢纽200m内建筑: {com_nearby}个 ({com_nearby/len(com_distances)*100:.1f}%)")
    print(f"  工业枢纽250m内建筑: {ind_nearby}个 ({ind_nearby/len(ind_distances)*100:.1f}%)")
    
    # 检查是否有重叠区域
    overlap_count = 0
    for i, (d_com, d_ind) in enumerate(zip(com_distances, ind_distances)):
        if d_com <= 200 and d_ind <= 250:
            overlap_count += 1
    
    print(f"  重叠区域建筑: {overlap_count}个 ({overlap_count/len(com_distances)*100:.1f}%)")
    
    if overlap_count < len(com_distances) * 0.1:  # 重叠少于10%
        print("  ✅ 成功形成哑铃状分布！")
    else:
        print("  ❌ 仍有较多重叠，需要进一步调整")

def rebuild_building_state(output_dir, target_month):
    """重建完整的建筑状态"""
    import os
    # 加载基础状态
    with open(os.path.join(output_dir, 'building_positions_month_00.json'), 'r') as f:
        base_data = json.load(f)
    
    buildings = base_data['buildings'].copy()
    
    # 应用增量更新
    for month in range(1, target_month + 1):
        delta_file = os.path.join(output_dir, f'building_delta_month_{month:02d}.json')
        if os.path.exists(delta_file):
            with open(delta_file, 'r') as f:
                delta_data = json.load(f)
            
            for building in delta_data.get('new_buildings', []):
                building_type = building['building_type']
                buildings[building_type].append(building)
    
    return buildings

def visualize_dumbbell_results():
    """可视化哑铃状结果"""
    print("\n📈 可视化哑铃状结果...")
    
    # 加载建筑数据
    import os
    output_dir = 'enhanced_simulation_v3_3_output'
    
    if not os.path.exists(output_dir):
        print("  未找到输出数据")
        return
    
    # 重建完整的建筑状态
    buildings = rebuild_building_state(output_dir, 6)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：建筑分布
    hub_com = [28, 55]
    hub_ind = [82, 55]
    
    # 绘制建筑
    colors = {'residential': 'green', 'commercial': 'red', 'industrial': 'blue'}
    for building_type, building_list in buildings.items():
        if building_type == 'public':
            continue
            
        for building in building_list:
            pos = building['xy']
            ax1.scatter(pos[0], pos[1], c=colors[building_type], 
                       s=30, alpha=0.7, label=building_type if building == building_list[0] else "")
    
    # 绘制枢纽
    ax1.scatter(hub_com[0], hub_com[1], c='red', marker='*', s=200, 
               edgecolors='black', linewidth=2, label='Commercial Hub')
    ax1.scatter(hub_ind[0], hub_ind[1], c='blue', marker='*', s=200, 
               edgecolors='black', linewidth=2, label='Industrial Hub')
    
    # 绘制主干道
    ax1.axhline(y=55, color='black', linewidth=3, alpha=0.8, label='Main Road')
    
    # 绘制影响范围
    circle_com = plt.Circle((hub_com[0], hub_com[1]), 200/2.0, 
                           fill=False, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.add_patch(circle_com)
    
    circle_ind = plt.Circle((hub_ind[0], hub_ind[1]), 250/2.0, 
                           fill=False, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    ax1.add_patch(circle_ind)
    
    ax1.set_title('Dumbbell Building Distribution')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 右图：距离分布
    com_distances = []
    ind_distances = []
    
    for building_type, building_list in buildings.items():
        if building_type == 'public':
            continue
            
        for building in building_list:
            pos = building['xy']
            
            dist_to_com = np.sqrt((pos[0] - hub_com[0])**2 + (pos[1] - hub_com[1])**2) * 2.0
            dist_to_ind = np.sqrt((pos[0] - hub_ind[0])**2 + (pos[1] - hub_ind[1])**2) * 2.0
            
            com_distances.append(dist_to_com)
            ind_distances.append(dist_to_ind)
    
    # 绘制距离分布
    ax2.hist(com_distances, bins=15, alpha=0.7, color='red', 
             label='Distance to Commercial Hub', edgecolor='black')
    ax2.hist(ind_distances, bins=15, alpha=0.7, color='blue', 
             label='Distance to Industrial Hub', edgecolor='black')
    
    # 绘制影响范围线
    ax2.axvline(200, color='red', linestyle='--', linewidth=2, label='Commercial Zone (200m)')
    ax2.axvline(250, color='blue', linestyle='--', linewidth=2, label='Industrial Zone (250m)')
    
    ax2.set_xlabel('Distance (meters)')
    ax2.set_ylabel('Number of Buildings')
    ax2.set_title('Building Distance Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dumbbell_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("🧪 测试改进后的哑铃状配置")
    
    # 测试配置
    test_dumbbell_config()
    
    # 可视化结果
    visualize_dumbbell_results()
    
    print("\n✅ 测试完成！")
    print("  生成的文件:")
    print("  - dumbbell_test_results.png: 哑铃状测试结果")

if __name__ == "__main__":
    main()
