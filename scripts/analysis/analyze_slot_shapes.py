#!/usr/bin/env python3
"""
分析v3.3系统中槽位的形状和随机性
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os
from enhanced_city_simulation_v3_3 import ContourExtractionSystemV3_3, GaussianLandPriceSystemV3_3

def analyze_slot_generation():
    """分析槽位生成过程"""
    print("🔍 分析v3.3槽位生成过程...")
    
    # 创建配置
    config = {
        'city': {'meters_per_pixel': 2.0},
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
    transport_hubs = [[37, 55], [73, 55]]
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 获取地价场
    land_price_field = land_price_system.get_land_price_field()
    
    # 分析不同建筑类型的等值线
    building_types = ['commercial', 'industrial', 'residential']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, building_type in enumerate(building_types):
        # 提取等值线
        contours = contour_system.extract_contours_from_land_price(
            land_price_field, building_type, map_size
        )
        
        # 在等值线上采样槽位
        slots = contour_system.sample_slots_on_contours(
            contours, building_type, map_size
        )
        
        # 绘制地价场和等值线
        ax1 = axes[0, i]
        im1 = ax1.imshow(land_price_field, cmap='YlOrRd', alpha=0.7)
        ax1.set_title(f'{building_type.title()} - 地价场和等值线')
        
        # 绘制等值线
        for contour in contours:
            contour_array = np.array(contour)
            ax1.plot(contour_array[:, 0], contour_array[:, 1], 'b-', linewidth=2, alpha=0.8)
        
        # 绘制槽位
        ax2 = axes[1, i]
        im2 = ax2.imshow(land_price_field, cmap='YlOrRd', alpha=0.7)
        ax2.set_title(f'{building_type.title()} - 槽位分布')
        
        # 绘制槽位点
        slot_positions = [slot.pos for slot in slots]
        if slot_positions:
            x_coords = [pos[0] for pos in slot_positions]
            y_coords = [pos[1] for pos in slot_positions]
            ax2.scatter(x_coords, y_coords, c='red', s=20, alpha=0.8, edgecolors='black')
        
        # 添加颜色条
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        print(f"  {building_type}: 提取了 {len(contours)} 条等值线，生成了 {len(slots)} 个槽位")
    
    plt.tight_layout()
    plt.savefig('slot_shape_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 分析随机性
    analyze_randomness(contour_system, land_price_field, map_size)

def analyze_randomness(contour_system, land_price_field, map_size):
    """分析槽位生成的随机性"""
    print("\n🎲 分析槽位生成随机性...")
    
    # 多次生成槽位，观察随机性
    num_trials = 5
    building_type = 'commercial'
    
    fig, axes = plt.subplots(1, num_trials, figsize=(20, 4))
    
    all_slot_positions = []
    
    for trial in range(num_trials):
        # 提取等值线
        contours = contour_system.extract_contours_from_land_price(
            land_price_field, building_type, map_size
        )
        
        # 在等值线上采样槽位
        slots = contour_system.sample_slots_on_contours(
            contours, building_type, map_size
        )
        
        # 记录槽位位置
        slot_positions = [slot.pos for slot in slots]
        all_slot_positions.append(slot_positions)
        
        # 绘制
        ax = axes[trial]
        im = ax.imshow(land_price_field, cmap='YlOrRd', alpha=0.7)
        
        if slot_positions:
            x_coords = [pos[0] for pos in slot_positions]
            y_coords = [pos[1] for pos in slot_positions]
            ax.scatter(x_coords, y_coords, c='red', s=15, alpha=0.8, edgecolors='black')
        
        ax.set_title(f'试验 {trial + 1} ({len(slots)} 个槽位)')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('slot_randomness_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 分析槽位位置的变化
    analyze_slot_variation(all_slot_positions)

def analyze_slot_variation(all_slot_positions):
    """分析槽位位置的变化"""
    print("\n📊 分析槽位位置变化...")
    
    if len(all_slot_positions) < 2:
        print("  需要至少2次试验才能分析变化")
        return
    
    # 计算槽位数量的变化
    slot_counts = [len(positions) for positions in all_slot_positions]
    print(f"  槽位数量变化: {slot_counts}")
    print(f"  平均槽位数: {np.mean(slot_counts):.1f}")
    print(f"  槽位数标准差: {np.std(slot_counts):.1f}")
    
    # 分析槽位位置的变化
    if len(all_slot_positions[0]) > 0:
        # 计算第一个试验的槽位位置
        first_trial = all_slot_positions[0]
        
        # 计算其他试验与第一个试验的差异
        for i in range(1, len(all_slot_positions)):
            current_trial = all_slot_positions[i]
            
            # 计算位置差异
            if len(current_trial) == len(first_trial):
                differences = []
                for j in range(len(first_trial)):
                    pos1 = first_trial[j]
                    pos2 = current_trial[j]
                    diff = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    differences.append(diff)
                
                avg_diff = np.mean(differences)
                max_diff = np.max(differences)
                print(f"  试验 {i+1} vs 试验 1: 平均位置差异 {avg_diff:.2f} 像素, 最大差异 {max_diff:.2f} 像素")

def analyze_contour_shapes():
    """分析等值线的形状特征"""
    print("\n🔍 分析等值线形状特征...")
    
    # 创建配置
    config = {
        'city': {'meters_per_pixel': 2.0},
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
    transport_hubs = [[37, 55], [73, 55]]
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 获取地价场
    land_price_field = land_price_system.get_land_price_field()
    
    # 分析不同等值线级别的形状
    levels_to_analyze = [0.85, 0.78, 0.71, 0.60, 0.70, 0.80, 0.55]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, level in enumerate(levels_to_analyze):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # 创建二值图像
        binary_image = (land_price_field >= level).astype(np.uint8) * 255
        
        # 查找轮廓
        contours_found, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制地价场
        im = ax.imshow(land_price_field, cmap='YlOrRd', alpha=0.7)
        
        # 绘制轮廓
        for contour in contours_found:
            # 简化轮廓
            epsilon = 0.02 * cv2.arcLength(contour, True)
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # 绘制原始轮廓
            contour_array = np.array(contour).reshape(-1, 2)
            ax.plot(contour_array[:, 0], contour_array[:, 1], 'b-', linewidth=1, alpha=0.6, label='原始轮廓')
            
            # 绘制简化轮廓
            simplified_array = np.array(simplified_contour).reshape(-1, 2)
            ax.plot(simplified_array[:, 0], simplified_array[:, 1], 'r-', linewidth=2, alpha=0.8, label='简化轮廓')
        
        ax.set_title(f'等值线级别 {level}')
        ax.set_aspect('equal')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for i in range(len(levels_to_analyze), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('contour_shape_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("🔍 v3.3槽位形状和随机性分析")
    
    # 分析槽位生成
    analyze_slot_generation()
    
    # 分析等值线形状
    analyze_contour_shapes()
    
    print("\n✅ 分析完成！")
    print("  生成的文件:")
    print("  - slot_shape_analysis.png: 槽位形状分析")
    print("  - slot_randomness_analysis.png: 槽位随机性分析")
    print("  - contour_shape_analysis.png: 等值线形状分析")

if __name__ == "__main__":
    main()
