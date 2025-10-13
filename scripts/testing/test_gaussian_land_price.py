#!/usr/bin/env python3
"""
测试高斯核地价场系统
验证连续地价分布和时间演化效果
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from logic.enhanced_sdf_system import GaussianLandPriceSystem

def test_gaussian_land_price_system():
    """测试高斯核地价场系统"""
    
    # 加载配置
    config_path = "configs/city_config_v2_3.json"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 初始化高斯核地价场系统
    land_price_system = GaussianLandPriceSystem(config)
    
    # 设置交通枢纽和地图尺寸
    transport_hubs = [[64, 64], [240, 64]]  # 两个Hub
    map_size = [256, 256]
    
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 测试不同月份的地价场
    test_months = [0, 6, 12, 18, 24]
    
    # 创建可视化
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('高斯核地价场时间演化测试', fontsize=16)
    
    for i, month in enumerate(test_months):
        print(f"\n🔄 测试月份: {month}")
        
        # 更新地价场
        land_price_system.update_land_price_field(month)
        
        # 获取地价场组件
        components = land_price_system.get_land_price_components(month)
        hub_land_price = components['hub_land_price']
        road_land_price = components['road_land_price']
        combined_land_price = components['combined_land_price']
        
        # 第一行：Hub地价场
        im1 = axes[0, i].imshow(hub_land_price, cmap='Reds', vmin=0, vmax=1)
        axes[0, i].set_title(f'Hub地价场 (月{month})')
        axes[0, i].set_xlabel('X (像素)')
        axes[0, i].set_ylabel('Y (像素)')
        
        # 标记Hub位置
        for hub in transport_hubs:
            axes[0, i].plot(hub[0], hub[1], 'ko', markersize=8, markeredgecolor='white')
        
        # 第二行：组合地价场
        im2 = axes[1, i].imshow(combined_land_price, cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'组合地价场 (月{month})')
        axes[1, i].set_xlabel('X (像素)')
        axes[1, i].set_ylabel('Y (像素)')
        
        # 标记Hub位置
        for hub in transport_hubs:
            axes[1, i].plot(hub[0], hub[1], 'ko', markersize=8, markeredgecolor='white')
        
        # 添加等值线
        levels = np.linspace(0.1, 0.9, 9)
        contours = axes[1, i].contour(combined_land_price, levels=levels, colors='white', alpha=0.7, linewidths=0.5)
        
        print(f"   地价范围: [{np.min(combined_land_price):.3f}, {np.max(combined_land_price):.3f}]")
        print(f"   平均地价: {np.mean(combined_land_price):.3f}")
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_file = 'gaussian_land_price_test.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 测试结果保存: {output_file}")
    
    # 显示图片
    plt.show()
    
    # 分析演化效果
    analyze_evolution_effect(land_price_system, test_months)

def analyze_evolution_effect(land_price_system, months):
    """分析演化效果"""
    print(f"\n📊 演化效果分析")
    print("=" * 50)
    
    # 选择月份18进行详细分析
    test_month = 18
    land_price_system.update_land_price_field(test_month)
    components = land_price_system.get_land_price_components(test_month)
    
    hub_land_price = components['hub_land_price']
    road_land_price = components['road_land_price']
    combined_land_price = components['combined_land_price']
    
    # 分析Hub高值区域
    hub_high_mask = hub_land_price >= 0.8
    hub_high_count = np.sum(hub_high_mask)
    
    # 分析道路高值区域
    road_high_mask = road_land_price >= 0.5
    road_high_count = np.sum(road_high_mask)
    
    # 分析组合地价场
    combined_high_mask = combined_land_price >= 0.7
    combined_high_count = np.sum(combined_high_mask)
    
    print(f"月份 {test_month} 分析结果:")
    print(f"  Hub高值区域 (≥0.8): {hub_high_count} 像素")
    print(f"  道路高值区域 (≥0.5): {road_high_count} 像素")
    print(f"  组合高值区域 (≥0.7): {combined_high_count} 像素")
    print(f"  地价场覆盖率: {np.sum(combined_land_price > 0.1) / combined_land_price.size:.1%}")
    
    # 创建演化分析图
    create_evolution_analysis_plot(hub_land_price, road_land_price, combined_land_price, test_month)

def create_evolution_analysis_plot(hub_land_price, road_land_price, combined_land_price, month):
    """创建演化分析图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'高斯核地价场演化分析 (月份 {month})', fontsize=16)
    
    # 1. Hub地价场热图
    im1 = axes[0, 0].imshow(hub_land_price, cmap='Reds', vmin=0, vmax=1)
    axes[0, 0].set_title('Hub地价场')
    axes[0, 0].set_xlabel('X (像素)')
    axes[0, 0].set_ylabel('Y (像素)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 道路地价场热图
    im2 = axes[0, 1].imshow(road_land_price, cmap='Blues', vmin=0, vmax=1)
    axes[0, 1].set_title('道路地价场')
    axes[0, 1].set_xlabel('X (像素)')
    axes[0, 1].set_ylabel('Y (像素)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. 组合地价场热图
    im3 = axes[0, 2].imshow(combined_land_price, cmap='viridis', vmin=0, vmax=1)
    axes[0, 2].set_title('组合地价场')
    axes[0, 2].set_xlabel('X (像素)')
    axes[0, 2].set_ylabel('Y (像素)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 4. Hub vs 道路地价散点图（采样）
    sample_size = 1000
    indices = np.random.choice(hub_land_price.size, sample_size, replace=False)
    hub_samples = hub_land_price.ravel()[indices]
    road_samples = road_land_price.ravel()[indices]
    
    axes[1, 0].scatter(hub_samples, road_samples, alpha=0.6, s=20)
    axes[1, 0].set_xlabel('Hub地价')
    axes[1, 0].set_ylabel('道路地价')
    axes[1, 0].set_title('Hub vs 道路地价关系')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 组合地价 vs Hub地价散点图
    combined_samples = combined_land_price.ravel()[indices]
    axes[1, 1].scatter(hub_samples, combined_samples, alpha=0.6, s=20, color='green')
    axes[1, 1].set_xlabel('Hub地价')
    axes[1, 1].set_ylabel('组合地价')
    axes[1, 1].set_title('组合地价 vs Hub地价关系')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 组合地价 vs 道路地价散点图
    axes[1, 2].scatter(road_samples, combined_samples, alpha=0.6, s=20, color='orange')
    axes[1, 2].set_xlabel('道路地价')
    axes[1, 2].set_ylabel('组合地价')
    axes[1, 2].set_title('组合地价 vs 道路地价关系')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存分析图
    analysis_file = f'gaussian_land_price_analysis_month_{month}.png'
    plt.savefig(analysis_file, dpi=150, bbox_inches='tight')
    print(f"💾 演化分析图保存: {analysis_file}")
    
    plt.show()

if __name__ == "__main__":
    test_gaussian_land_price_system()


