#!/usr/bin/env python3
"""
测试修复后的SDF系统
验证线SDF矩形分布和组合优先关系
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from logic.enhanced_sdf_system import EnhancedSDFSystem

def test_fixed_sdf_system():
    """测试修复后的SDF系统"""
    
    print("🧪 测试修复后的SDF系统")
    print("=" * 50)
    
    # 加载配置
    config_file = 'configs/city_config_v2_3.json'
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建增强版SDF系统
    sdf_system = EnhancedSDFSystem(config)
    
    # 设置交通枢纽位置（模拟数据）
    transport_hubs = [[64, 64], [240, 64]]  # 两个hub，距离176px
    map_size = [256, 256]
    
    # 初始化系统
    sdf_system.initialize_system(transport_hubs, map_size)
    
    # 测试不同月份的SDF
    test_months = [0, 6, 12, 18, 24]
    
    # 创建可视化
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle('修复后的SDF系统测试 - 矩形分布和组合优先关系', fontsize=16, fontweight='bold')
    
    # 设置颜色映射
    hub_cmap = plt.cm.Reds
    road_cmap = plt.cm.Blues
    combined_cmap = plt.cm.RdYlBu_r
    
    for i, month in enumerate(test_months):
        # 更新SDF场
        sdf_system.update_sdf_field(month)
        
        # 获取SDF组成部分
        sdf_components = sdf_system.get_sdf_components(month)
        hub_sdf = sdf_components['hub_sdf']
        road_sdf = sdf_components['road_sdf']
        combined_sdf = sdf_components['combined_sdf']
        
        # 获取演化阶段信息
        evolution_stage = sdf_system._get_evolution_stage(month)
        road_multiplier = evolution_stage.get('road_multiplier', 1.0)
        
        # 绘制Hub SDF
        ax1 = axes[0, i]
        im1 = ax1.imshow(hub_sdf, cmap=hub_cmap, vmin=0, vmax=1)
        ax1.set_title(f'Hub SDF - {month}月', fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # 标记交通枢纽位置
        for hub in transport_hubs:
            ax1.plot(hub[0], hub[1], 'ko', markersize=8, markeredgecolor='white', markeredgewidth=2)
        
        # 绘制道路SDF
        ax2 = axes[1, i]
        im2 = ax2.imshow(road_sdf, cmap=road_cmap, vmin=0, vmax=1)
        ax2.set_title(f'道路SDF - {month}月\n扩展倍数: {road_multiplier:.1f}', fontsize=10)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # 标记交通枢纽位置
        for hub in transport_hubs:
            ax2.plot(hub[0], hub[1], 'ko', markersize=8, markeredgecolor='white', markeredgewidth=2)
        
        # 绘制组合SDF
        ax3 = axes[2, i]
        im3 = ax3.imshow(combined_sdf, cmap=combined_cmap, vmin=0, vmax=1)
        ax3.set_title(f'组合SDF - {month}月', fontsize=10)
        ax3.set_xticks([])
        ax3.set_yticks([])
        
        # 标记交通枢纽位置
        for hub in transport_hubs:
            ax3.plot(hub[0], hub[1], 'ko', markersize=8, markeredgecolor='white', markeredgewidth=2)
        
        # 添加颜色条
        if i == 4:  # 只在最后一列添加颜色条
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Hub SDF值', fontsize=9)
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('道路SDF值', fontsize=9)
            cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
            cbar3.set_label('组合SDF值', fontsize=9)
        
        print(f"📊 月份 {month}: Hub SDF [{np.min(hub_sdf):.3f}, {np.max(hub_sdf):.3f}], "
              f"道路SDF [{np.min(road_sdf):.3f}, {np.max(road_sdf):.3f}], "
              f"组合SDF [{np.min(combined_sdf):.3f}, {np.max(combined_sdf):.3f}], "
              f"扩展倍数 {road_multiplier:.1f}")
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_file = 'fixed_sdf_system_test.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 测试结果保存: {output_file}")
    
    # 显示图像
    plt.show()
    
    # 分析组合优先关系
    analyze_combination_priority(sdf_system, test_months)

def analyze_combination_priority(sdf_system, months):
    """分析组合优先关系"""
    
    print(f"\n🔍 分析组合优先关系")
    print("=" * 50)
    
    # 选择一个月进行分析
    test_month = 18  # 成熟阶段
    sdf_system.update_sdf_field(test_month)
    
    sdf_components = sdf_system.get_sdf_components(test_month)
    hub_sdf = sdf_components['hub_sdf']
    road_sdf = sdf_components['road_sdf']
    combined_sdf = sdf_components['combined_sdf']
    
    # 分析不同区域的组合效果
    print(f"📊 月份 {test_month} 的组合分析:")
    
    # 1. Hub高值区域
    hub_high_mask = hub_sdf >= 0.7
    hub_high_count = np.sum(hub_high_mask)
    if hub_high_count > 0:
        hub_high_avg = np.mean(combined_sdf[hub_high_mask])
        hub_high_road_influence = np.mean(road_sdf[hub_high_mask])
        print(f"   Hub高值区域 (≥0.7): {hub_high_count} 像素")
        print(f"     组合SDF均值: {hub_high_avg:.3f}")
        print(f"     道路SDF影响: {hub_high_road_influence:.3f}")
    
    # 2. Hub中值区域
    hub_mid_mask = (hub_sdf >= 0.3) & (hub_sdf < 0.7)
    hub_mid_count = np.sum(hub_mid_mask)
    if hub_mid_count > 0:
        hub_mid_avg = np.mean(combined_sdf[hub_mid_mask])
        hub_mid_road_influence = np.mean(road_sdf[hub_mid_mask])
        print(f"   Hub中值区域 (0.3-0.7): {hub_mid_count} 像素")
        print(f"     组合SDF均值: {hub_mid_avg:.3f}")
        print(f"     道路SDF影响: {hub_mid_road_influence:.3f}")
    
    # 3. Hub低值区域
    hub_low_mask = hub_sdf < 0.3
    hub_low_count = np.sum(hub_low_mask)
    if hub_low_count > 0:
        hub_low_avg = np.mean(combined_sdf[hub_low_mask])
        hub_low_road_influence = np.mean(road_sdf[hub_low_mask])
        print(f"   Hub低值区域 (<0.3): {hub_low_count} 像素")
        print(f"     组合SDF均值: {hub_low_avg:.3f}")
        print(f"     道路SDF影响: {hub_low_road_influence:.3f}")
    
    # 4. 道路高值区域
    road_high_mask = road_sdf >= 0.5
    road_high_count = np.sum(road_high_mask)
    if road_high_count > 0:
        road_high_avg = np.mean(combined_sdf[road_high_mask])
        road_high_hub_influence = np.mean(hub_sdf[road_high_mask])
        print(f"   道路高值区域 (≥0.5): {road_high_count} 像素")
        print(f"     组合SDF均值: {road_high_avg:.3f}")
        print(f"     Hub SDF影响: {road_high_hub_influence:.3f}")
    
    # 创建组合分析可视化
    create_combination_analysis_plot(hub_sdf, road_sdf, combined_sdf, test_month)

def create_combination_analysis_plot(hub_sdf, road_sdf, combined_sdf, month):
    """创建组合分析可视化"""
    
    print(f"\n🎨 创建组合分析可视化")
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'SDF组合分析 - {month}月', fontsize=16, fontweight='bold')
    
    # 1. Hub SDF
    ax1 = axes[0, 0]
    im1 = ax1.imshow(hub_sdf, cmap='Reds', vmin=0, vmax=1)
    ax1.set_title('Hub SDF', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 2. 道路SDF
    ax2 = axes[0, 1]
    im2 = ax2.imshow(road_sdf, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('道路SDF', fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 3. 组合SDF
    ax3 = axes[0, 2]
    im3 = ax3.imshow(combined_sdf, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax3.set_title('组合SDF', fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # 4. Hub SDF vs 道路SDF 散点图
    ax4 = axes[1, 0]
    # 随机采样点以避免图像过于密集
    sample_size = 1000
    indices = np.random.choice(hub_sdf.size, sample_size, replace=False)
    hub_samples = hub_sdf.flat[indices]
    road_samples = road_sdf.flat[indices]
    
    ax4.scatter(hub_samples, road_samples, alpha=0.6, s=20)
    ax4.set_xlabel('Hub SDF值')
    ax4.set_ylabel('道路SDF值')
    ax4.set_title('Hub SDF vs 道路SDF')
    ax4.grid(True, alpha=0.3)
    
    # 5. 组合SDF vs Hub SDF 散点图
    ax5 = axes[1, 1]
    combined_samples = combined_sdf.flat[indices]
    ax5.scatter(hub_samples, combined_samples, alpha=0.6, s=20, color='red')
    ax5.set_xlabel('Hub SDF值')
    ax5.set_ylabel('组合SDF值')
    ax5.set_title('组合SDF vs Hub SDF')
    ax5.grid(True, alpha=0.3)
    
    # 6. 组合SDF vs 道路SDF 散点图
    ax6 = axes[1, 2]
    ax6.scatter(road_samples, combined_samples, alpha=0.6, s=20, color='blue')
    ax6.set_xlabel('道路SDF值')
    ax6.set_ylabel('组合SDF值')
    ax6.set_title('组合SDF vs 道路SDF')
    ax6.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_file = f'sdf_combination_analysis_month_{month}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 组合分析图保存: {output_file}")
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    test_fixed_sdf_system()


