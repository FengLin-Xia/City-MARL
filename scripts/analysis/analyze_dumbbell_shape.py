#!/usr/bin/env python3
"""
分析当前hub配置问题并提出哑铃状分布改进方案
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from enhanced_city_simulation_v3_3 import GaussianLandPriceSystemV3_3

def analyze_current_hub_config():
    """分析当前hub配置的问题"""
    print("🔍 分析当前hub配置问题...")
    
    # 当前配置
    map_size = [110, 110]
    current_hub_com = [map_size[0] // 3, map_size[1] // 2]  # [37, 55]
    current_hub_ind = [2 * map_size[0] // 3, map_size[1] // 2]  # [73, 55]
    
    # 计算当前距离
    current_distance = math.sqrt((current_hub_ind[0] - current_hub_com[0])**2 + 
                                (current_hub_ind[1] - current_hub_com[1])**2)
    current_distance_m = current_distance * 2.0  # 2米/像素
    
    print(f"  当前商业枢纽位置: {current_hub_com}")
    print(f"  当前工业枢纽位置: {current_hub_ind}")
    print(f"  当前枢纽距离: {current_distance:.1f}像素 ({current_distance_m:.1f}米)")
    print(f"  当前影响范围: 商业350m, 工业450m")
    
    # 问题分析
    print(f"\n❌ 问题分析:")
    print(f"  1. 枢纽距离太近: {current_distance_m:.1f}米 < 理想距离(800-1000米)")
    print(f"  2. 影响范围重叠: 350m + 450m = 800m > 枢纽距离{current_distance_m:.1f}m")
    print(f"  3. 无法形成哑铃状分布")
    
    return current_hub_com, current_hub_ind, current_distance_m

def propose_dumbbell_config():
    """提出哑铃状配置方案"""
    print("\n💡 提出哑铃状配置方案...")
    
    map_size = [110, 110]
    
    # 方案1: 扩大枢纽距离，减小影响范围
    hub_com_x = int(map_size[0] * 0.25)  # 25%位置
    hub_ind_x = int(map_size[0] * 0.75)  # 75%位置
    hub_y = map_size[1] // 2  # 中间位置
    
    new_hub_com = [hub_com_x, hub_y]
    new_hub_ind = [hub_ind_x, hub_y]
    
    new_distance = math.sqrt((new_hub_ind[0] - new_hub_com[0])**2 + 
                            (new_hub_ind[1] - new_hub_com[1])**2)
    new_distance_m = new_distance * 2.0
    
    print(f"  方案1 - 扩大距离:")
    print(f"    商业枢纽: {new_hub_com}")
    print(f"    工业枢纽: {new_hub_ind}")
    print(f"    枢纽距离: {new_distance:.1f}像素 ({new_distance_m:.1f}米)")
    print(f"    建议影响范围: 商业200m, 工业250m")
    
    # 方案2: 更极端的哑铃状
    hub_com_x2 = int(map_size[0] * 0.2)   # 20%位置
    hub_ind_x2 = int(map_size[0] * 0.8)   # 80%位置
    
    new_hub_com2 = [hub_com_x2, hub_y]
    new_hub_ind2 = [hub_ind_x2, hub_y]
    
    new_distance2 = math.sqrt((new_hub_ind2[0] - new_hub_com2[0])**2 + 
                             (new_hub_ind2[1] - new_hub_com2[1])**2)
    new_distance2_m = new_distance2 * 2.0
    
    print(f"\n  方案2 - 极端哑铃状:")
    print(f"    商业枢纽: {new_hub_com2}")
    print(f"    工业枢纽: {new_hub_ind2}")
    print(f"    枢纽距离: {new_distance2:.1f}像素 ({new_distance2_m:.1f}米)")
    print(f"    建议影响范围: 商业150m, 工业200m")
    
    return (new_hub_com, new_hub_ind, new_distance_m), (new_hub_com2, new_hub_ind2, new_distance2_m)

def visualize_dumbbell_comparison():
    """可视化哑铃状对比"""
    print("\n📊 可视化哑铃状对比...")
    
    # 创建配置
    config = {
        'city': {'meters_per_pixel': 2.0},
        'gaussian_land_price_system': {
            'w_r': 0.6, 'w_c': 0.5, 'w_i': 0.5, 'w_cor': 0.2, 'bias': 0.0,
            'hub_sigma_base_m': 40, 'road_sigma_base_m': 20,
            'hub_growth_rate': 0.03, 'road_growth_rate': 0.02,
            'max_hub_multiplier': 2.0, 'max_road_multiplier': 2.5,
            'normalize': True, 'smoothstep_tau': 0.0
        }
    }
    
    map_size = [110, 110]
    
    # 三种配置
    configs = [
        {
            'name': 'Current (问题配置)',
            'hubs': [[37, 55], [73, 55]],
            'radius': [350, 450],
            'sigma': [40, 40],
            'color': 'red'
        },
        {
            'name': '方案1 (扩大距离)',
            'hubs': [[28, 55], [82, 55]],
            'radius': [200, 250],
            'sigma': [25, 30],
            'color': 'blue'
        },
        {
            'name': '方案2 (极端哑铃)',
            'hubs': [[22, 55], [88, 55]],
            'radius': [150, 200],
            'sigma': [20, 25],
            'color': 'green'
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, config_data in enumerate(configs):
        ax = axes[i]
        
        # 创建地价场系统
        land_price_system = GaussianLandPriceSystemV3_3(config)
        land_price_system.initialize_system(config_data['hubs'], map_size)
        land_price_field = land_price_system.get_land_price_field()
        
        # 绘制地价场
        im = ax.imshow(land_price_field, cmap='YlOrRd', alpha=0.8)
        
        # 绘制枢纽位置
        hub_com, hub_ind = config_data['hubs']
        ax.scatter(hub_com[0], hub_com[1], c='red', marker='*', s=200, alpha=0.9, 
                  edgecolors='black', linewidth=2, label='Commercial Hub')
        ax.scatter(hub_ind[0], hub_ind[1], c='blue', marker='*', s=200, alpha=0.9, 
                  edgecolors='black', linewidth=2, label='Industrial Hub')
        
        # 绘制主干道
        ax.axhline(y=55, color='black', linewidth=3, alpha=0.8, label='Main Road')
        
        # 绘制影响范围
        radius_com_px = config_data['radius'][0] / 2.0
        radius_ind_px = config_data['radius'][1] / 2.0
        
        circle_com = plt.Circle((hub_com[0], hub_com[1]), radius_com_px, 
                               fill=False, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.add_patch(circle_com)
        
        circle_ind = plt.Circle((hub_ind[0], hub_ind[1]), radius_ind_px, 
                               fill=False, color='blue', linestyle='--', alpha=0.7, linewidth=2)
        ax.add_patch(circle_ind)
        
        # 绘制等值线
        levels = [0.2, 0.4, 0.6, 0.8]
        contours = ax.contour(land_price_field, levels=levels, 
                             colors=['white', 'yellow', 'orange', 'red'], 
                             linewidths=1, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        
        ax.set_title(f'{config_data["name"]}\n距离: {math.sqrt((hub_ind[0]-hub_com[0])**2 + (hub_ind[1]-hub_com[1])**2)*2:.0f}m')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('dumbbell_shape_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_improved_config():
    """创建改进的配置文件"""
    print("\n⚙️ 创建改进的配置文件...")
    
    # 方案1配置
    config_v3_3_improved = {
        'city': {
            'map_size': [110, 110],
            'meters_per_pixel': 2.0,
            'trunk_road': [[20, 55], [90, 55]],
            'transport_hubs': [[28, 55], [82, 55]]  # 扩大距离
        },
        'government_backbone': {
            'road_corridor': {
                'sigma_perp_m': 40,
                'setback_m': {'commercial': 8, 'residential': 10, 'industrial': 14}
            },
            'hubs': {
                'commercial': {'sigma_perp_m': 25, 'sigma_parallel_m': 75},  # 减小影响范围
                'industrial': {'sigma_perp_m': 30, 'sigma_parallel_m': 90}
            },
            'zoning': {
                'hub_com_radius_m': 200,  # 减小分区半径
                'hub_ind_radius_m': 250,
                'mid_corridor_residential': True
            },
            'quotas_per_quarter': {
                'residential': [10, 20, 15, 25],
                'commercial': [5, 12, 8, 15],
                'industrial': [4, 10, 6, 12]
            },
            'strict_layering': True,
            'dead_slots_ratio_max': 0.05
        },
        'gaussian_land_price_system': {
            'w_r': 0.6, 'w_c': 0.5, 'w_i': 0.5, 'w_cor': 0.2, 'bias': 0.0,
            'hub_sigma_base_m': 30,  # 减小基础σ
            'road_sigma_base_m': 20,
            'hub_growth_rate': 0.03, 'road_growth_rate': 0.02,
            'max_hub_multiplier': 1.5,  # 减小最大倍数
            'max_road_multiplier': 2.0,
            'normalize': True, 'smoothstep_tau': 0.0
        },
        'scoring_weights': {
            'commercial': {
                'f_price': 0.35, 'f_hub_com': 0.25, 'f_road': 0.20,
                'f_heat': 0.15, 'f_access': 0.05,
                'crowding': -0.03, 'junction_penalty': -0.02
            },
            'industrial': {
                'f_price': -0.20, 'f_hub_ind': 0.45, 'f_road': 0.25,
                'f_access': 0.05, 'crowding': -0.10, 'junction_penalty': -0.05
            },
            'residential': {
                'f_price': 0.10, 'f_road': 0.45, 'f_access': 0.15,
                'f_hub_com': -0.15, 'f_hub_ind': -0.10, 'crowding': -0.05
            }
        },
        'isocontour_layout': {
            'commercial': {'levels': [0.85, 0.78, 0.71], 'arc_spacing_m': [25, 35]},
            'industrial': {'levels': [0.60, 0.70, 0.80], 'arc_spacing_m': [35, 55]},
            'residential': {'band': [0.45, 0.65], 'arc_spacing_m': [35, 55]},
            'normal_offset_m': 1.0, 'jitter_m': 0.5
        }
    }
    
    # 保存配置文件
    import json
    with open('configs/city_config_v3_3_dumbbell.json', 'w') as f:
        json.dump(config_v3_3_improved, f, indent=2)
    
    print("✅ 已创建改进配置文件: configs/city_config_v3_3_dumbbell.json")
    
    # 显示关键改进
    print("\n🔧 关键改进:")
    print(f"  1. 枢纽距离: 36像素 → 54像素 (72m → 108m)")
    print(f"  2. 商业枢纽影响范围: 350m → 200m")
    print(f"  3. 工业枢纽影响范围: 450m → 250m")
    print(f"  4. 基础σ: 40m → 30m")
    print(f"  5. 最大倍数: 2.0 → 1.5")
    
    return config_v3_3_improved

def main():
    """主函数"""
    print("🔍 v3.3系统哑铃状分布分析")
    
    # 分析当前配置问题
    current_hub_com, current_hub_ind, current_distance = analyze_current_hub_config()
    
    # 提出改进方案
    scheme1, scheme2 = propose_dumbbell_config()
    
    # 可视化对比
    visualize_dumbbell_comparison()
    
    # 创建改进配置
    improved_config = create_improved_config()
    
    print("\n✅ 分析完成！")
    print("  生成的文件:")
    print("  - dumbbell_shape_comparison.png: 哑铃状对比图")
    print("  - configs/city_config_v3_3_dumbbell.json: 改进配置文件")
    
    print("\n💡 建议:")
    print("  1. 使用方案1配置，扩大枢纽距离到108米")
    print("  2. 减小影响范围，避免重叠")
    print("  3. 调整σ参数，形成更清晰的哑铃状分布")
    print("  4. 可以考虑在中间区域增加住宅密度")

if __name__ == "__main__":
    main()
