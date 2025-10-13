#!/usr/bin/env python3
"""
可视化最终等值线状态
显示所有Hub的等值线分布和建筑位置
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os

def load_final_data():
    """加载最终数据"""
    print("加载最终数据...")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 加载最终地价场
    with open('enhanced_simulation_v3_1_output/land_price_frame_month_23.json', 'r', encoding='utf-8') as f:
        land_price_data = json.load(f)
    
    # 加载最终建筑位置
    with open('enhanced_simulation_v3_1_output/building_positions_month_23.json', 'r', encoding='utf-8') as f:
        buildings_data = json.load(f)
    
    return config, land_price_data, buildings_data

def create_isocontour_visualization():
    """创建等值线可视化"""
    print("创建等值线可视化...")
    
    # 加载数据
    config, land_price_data, buildings_data = load_final_data()
    
    # 提取数据
    land_price_field = np.array(land_price_data['land_price_field'])
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 左图：地价场和等值线
    ax1.set_title('第23月地价场与等值线分布', fontsize=16, fontweight='bold')
    
    # 显示地价场
    im1 = ax1.imshow(land_price_field, cmap='viridis', origin='lower', alpha=0.7)
    plt.colorbar(im1, ax=ax1, label='地价强度')
    
    # 标记Hub位置
    hub_colors = ['red', 'blue', 'green']
    hub_labels = ['Hub1', 'Hub2', 'Hub3']
    for i, (hub_x, hub_y) in enumerate(transport_hubs):
        ax1.scatter(hub_x, hub_y, c=hub_colors[i], s=200, marker='*', 
                   edgecolors='white', linewidth=2, label=hub_labels[i], zorder=10)
        ax1.annotate(hub_labels[i], (hub_x, hub_y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor=hub_colors[i], alpha=0.8))
    
    # 绘制等值线
    x = np.arange(map_size[0])
    y = np.arange(map_size[1])
    X, Y = np.meshgrid(x, y)
    
    # 计算等值线阈值
    min_price = np.min(land_price_field)
    max_price = np.max(land_price_field)
    
    # 生成等值线阈值
    thresholds = np.linspace(min_price, max_price, 8)
    
    # 绘制等值线
    contours = ax1.contour(X, Y, land_price_field, levels=thresholds, colors='white', alpha=0.6, linewidths=1)
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%.3f')
    
    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：建筑分布
    ax2.set_title('第23月建筑分布', fontsize=16, fontweight='bold')
    
    # 显示地价场背景
    im2 = ax2.imshow(land_price_field, cmap='viridis', origin='lower', alpha=0.3)
    
    # 标记Hub位置
    for i, (hub_x, hub_y) in enumerate(transport_hubs):
        ax2.scatter(hub_x, hub_y, c=hub_colors[i], s=200, marker='*', 
                   edgecolors='white', linewidth=2, label=hub_labels[i], zorder=10)
        ax2.annotate(hub_labels[i], (hub_x, hub_y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor=hub_colors[i], alpha=0.8))
    
    # 绘制建筑
    building_colors = {'residential': 'lightblue', 'commercial': 'orange', 'industrial': 'red', 'public': 'green'}
    building_markers = {'residential': 'o', 'commercial': 's', 'industrial': '^', 'public': 'D'}
    
    for building_type in building_colors.keys():
        buildings = [b for b in buildings_data['buildings'] if b['type'] == building_type]
        if buildings:
            x_coords = [b['position'][0] for b in buildings]
            y_coords = [b['position'][1] for b in buildings]
            ax2.scatter(x_coords, y_coords, c=building_colors[building_type], 
                       marker=building_markers[building_type], s=30, alpha=0.8, 
                       label=f'{building_type} ({len(buildings)})', edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('X 坐标')
    ax2.set_ylabel('Y 坐标')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = 'visualization_output/month_23_isocontours_visualization.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {output_path}")
    
    # 显示图片
    plt.show()
    
    # 打印统计信息
    print("\n=== 第23月状态统计 ===")
    print(f"地图大小: {map_size[0]} x {map_size[1]}")
    print(f"Hub数量: {len(transport_hubs)}")
    print(f"Hub位置: {transport_hubs}")
    
    print(f"\n建筑统计:")
    for building_type in building_colors.keys():
        buildings = [b for b in buildings_data['buildings'] if b['type'] == building_type]
        print(f"  {building_type}: {len(buildings)} 个")
    
    print(f"\n地价场统计:")
    print(f"  最小值: {min_price:.4f}")
    print(f"  最大值: {max_price:.4f}")
    print(f"  平均值: {np.mean(land_price_field):.4f}")
    
    # 分析每个Hub的影响范围
    print(f"\nHub影响范围分析:")
    for i, (hub_x, hub_y) in enumerate(transport_hubs):
        hub_price = land_price_field[hub_y, hub_x]
        print(f"  {hub_labels[i]} ({hub_x}, {hub_y}): 地价强度 {hub_price:.4f}")
        
        # 计算该Hub附近的建筑数量
        nearby_buildings = 0
        for building in buildings_data['buildings']:
            bx, by = building['position']
            distance = np.sqrt((bx - hub_x)**2 + (by - hub_y)**2)
            if distance <= 30:  # 30格范围内
                nearby_buildings += 1
        
        print(f"    30格范围内建筑数: {nearby_buildings}")

def main():
    """主函数"""
    try:
        create_isocontour_visualization()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
