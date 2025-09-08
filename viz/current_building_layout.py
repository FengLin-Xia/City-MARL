#!/usr/bin/env python3
"""
当前建筑布局可视化
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_current_layout():
    """可视化当前建筑布局"""
    
    print("🏗️ 当前建筑布局可视化")
    print("=" * 40)
    
    # 枢纽位置
    hubs = [[40, 128], [216, 128]]
    trunk_road = [[40, 128], [216, 128]]
    
    # 加载建筑位置数据
    try:
        with open('enhanced_simulation_v2_3_output/building_positions_month_21.json', 'r', encoding='utf-8') as f:
            building_data = json.load(f)
        buildings = building_data['buildings']
        print(f"✅ 建筑数据加载成功，数量: {len(buildings)}")
    except Exception as e:
        print(f"❌ 无法加载建筑数据: {e}")
        return
    
    # 加载SDF场数据
    try:
        with open('enhanced_simulation_v2_3_output/sdf_field_month_21.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        sdf_field = np.array(data['sdf_field'])
        print(f"✅ SDF场加载成功，形状: {sdf_field.shape}")
    except Exception as e:
        print(f"❌ 无法加载SDF场数据: {e}")
        sdf_field = None
    
    # 统计建筑类型
    building_counts = {'residential': 0, 'commercial': 0, 'public': 0}
    building_positions = {'residential': [], 'commercial': [], 'public': []}
    
    for building in buildings:
        building_type = building.get('type', 'unknown')
        if building_type in building_counts:
            building_counts[building_type] += 1
            building_positions[building_type].append(building['position'])
    
    print(f"\n📊 建筑统计:")
    print(f"  住宅建筑: {building_counts['residential']}")
    print(f"  商业建筑: {building_counts['commercial']}")
    print(f"  公共建筑: {building_counts['public']}")
    print(f"  总计: {sum(building_counts.values())}")
    
    # 创建可视化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Current Building Layout Analysis', fontsize=16)
    
    # 左上图：建筑分布总览
    ax1.clear()
    
    # 绘制主干道
    ax1.plot([trunk_road[0][0], trunk_road[1][0]], 
             [trunk_road[0][1], trunk_road[1][1]], 
             color='gray', linewidth=3, label='Trunk Road')
    
    # 绘制交通枢纽
    for i, hub in enumerate(hubs):
        ax1.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                   edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
    
    # 绘制建筑
    colors = {'residential': '#F6C344', 'commercial': '#FD7E14', 'public': '#22A6B3'}
    
    for building_type, positions in building_positions.items():
        if positions:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            ax1.scatter(x_coords, y_coords, c=colors[building_type], s=50, 
                       alpha=0.7, label=f'{building_type.title()} ({len(positions)})')
    
    ax1.set_xlim(0, 256)
    ax1.set_ylim(0, 256)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Building Distribution Overview')
    ax1.legend()
    
    # 右上图：SDF场 + 建筑位置
    if sdf_field is not None:
        im2 = ax2.imshow(sdf_field, cmap='viridis', origin='lower', 
                          extent=[0, 256, 0, 256], alpha=0.6)
        
        # 绘制主干道和枢纽
        ax2.plot([trunk_road[0][0], trunk_road[1][0]], 
                 [trunk_road[0][1], trunk_road[1][1]], 
                 color='red', linewidth=3, alpha=0.8, label='Trunk Road')
        
        for i, hub in enumerate(hubs):
            ax2.scatter(hub[0], hub[1], c='red', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
        
        # 绘制建筑位置
        for building_type, positions in building_positions.items():
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                ax2.scatter(x_coords, y_coords, c=colors[building_type], s=30, 
                           alpha=0.8, label=f'{building_type.title()}')
        
        ax2.set_title('SDF Field + Building Positions')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.legend()
        
        # 添加颜色条
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('SDF Value')
    
    # 左下图：建筑密度热力图
    ax3.clear()
    
    # 创建建筑密度图
    density_map = np.zeros((256, 256))
    
    for building_type, positions in building_positions.items():
        for pos in positions:
            x, y = pos[0], pos[1]
            if 0 <= x < 256 and 0 <= y < 256:
                # 在建筑周围增加密度
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 256 and 0 <= ny < 256:
                            density_map[ny, nx] += 1
    
    im3 = ax3.imshow(density_map, cmap='hot', origin='lower', 
                      extent=[0, 256, 0, 256], alpha=0.8)
    
    # 绘制主干道和枢纽
    ax3.plot([trunk_road[0][0], trunk_road[1][0]], 
             [trunk_road[0][1], trunk_road[1][1]], 
             color='white', linewidth=3, alpha=0.9, label='Trunk Road')
    
    for i, hub in enumerate(hubs):
        ax3.scatter(hub[0], hub[1], c='white', s=300, marker='*', 
                   edgecolors='black', linewidth=2, label=f'Hub {i+1}', zorder=10)
    
    ax3.set_title('Building Density Heatmap')
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    ax3.legend()
    
    # 添加颜色条
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Density')
    
        # 右下图：建筑类型分布饼图
    ax4.clear()
    
    if sum(building_counts.values()) > 0:
        labels = [f'{k.title()}\n({v})' for k, v in building_counts.items() if v > 0]
        sizes = [v for v in building_counts.values() if v > 0]
        colors_list = [colors[k] for k, v in building_counts.items() if v > 0]
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_list, 
                                          autopct='%1.1f%%', startangle=90)
        
        # 设置文本样式
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax4.set_title('Building Type Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # 分析建筑分布特征
    print(f"\n🔍 建筑分布特征分析:")
    
    # 分析住宅建筑分布
    if building_positions['residential']:
        res_x = [pos[0] for pos in building_positions['residential']]
        res_y = [pos[1] for pos in building_positions['residential']]
        print(f"  住宅建筑:")
        print(f"    X范围: [{min(res_x)}, {max(res_x)}]")
        print(f"    Y范围: [{min(res_y)}, {max(res_y)}]")
        print(f"    中心位置: ({np.mean(res_x):.1f}, {np.mean(res_y):.1f})")
    
    # 分析商业建筑分布
    if building_positions['commercial']:
        com_x = [pos[0] for pos in building_positions['commercial']]
        com_y = [pos[1] for pos in building_positions['commercial']]
        print(f"  商业建筑:")
        print(f"    X范围: [{min(com_x)}, {max(com_x)}]")
        print(f"    Y范围: [{min(com_y)}, {max(com_y)}]")
        print(f"    中心位置: ({np.mean(com_x):.1f}, {np.mean(com_y):.1f})")
    
    # 分析公共建筑分布
    if building_positions['public']:
        pub_x = [pos[0] for pos in building_positions['public']]
        pub_y = [pos[1] for pos in building_positions['public']]
        print(f"  公共建筑:")
        print(f"    X范围: [{min(pub_x)}, {max(pub_x)}]")
        print(f"    Y范围: [{min(pub_y)}, {max(pub_y)}]")
        print(f"    中心位置: ({np.mean(pub_x):.1f}, {np.mean(pub_y):.1f})")

if __name__ == "__main__":
    visualize_current_layout()
