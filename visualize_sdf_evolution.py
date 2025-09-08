#!/usr/bin/env python3
"""
SDF演化可视化脚本
展示SDF随时间的变化，包括Hub SDF、道路SDF和组合SDF
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import json
import os
from logic.enhanced_sdf_system import EnhancedSDFSystem

def create_sdf_evolution_visualization():
    """创建SDF演化可视化"""
    
    print("🎨 创建SDF演化可视化")
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
    
    # 模拟24个月的演化
    months = list(range(0, 25, 3))  # 每3个月一个时间点
    print(f"📅 模拟时间范围: {months[0]} - {months[-1]} 个月")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('SDF演化可视化 - 渐进式城市发展', fontsize=16, fontweight='bold')
    
    # 设置颜色映射
    hub_cmap = plt.cm.Reds
    road_cmap = plt.cm.Blues
    combined_cmap = plt.cm.RdYlBu_r
    
    # 选择关键时间点进行可视化
    key_months = [0, 6, 12, 18, 24]
    key_descriptions = ['初始阶段', '早期增长', '中期增长', '成熟阶段', '完全发展']
    
    for i, month in enumerate(key_months):
            
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
        ax1.set_title(f'Hub SDF - {month}月\n{key_descriptions[i]}', fontsize=10)
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
        
        # 添加颜色条
        if i == 4:  # 只在最后一列添加颜色条
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Hub SDF值', fontsize=9)
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('道路SDF值', fontsize=9)
        
        print(f"📊 月份 {month}: Hub SDF [{np.min(hub_sdf):.3f}, {np.max(hub_sdf):.3f}], "
              f"道路SDF [{np.min(road_sdf):.3f}, {np.max(road_sdf):.3f}], "
              f"扩展倍数 {road_multiplier:.1f}")
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_file = 'sdf_evolution_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 可视化图像保存: {output_file}")
    
    # 显示图像
    plt.show()
    
    # 创建动画版本
    create_sdf_animation(sdf_system, months, transport_hubs)

def create_sdf_animation(sdf_system, months, transport_hubs):
    """创建SDF演化动画"""
    
    print(f"\n🎬 创建SDF演化动画")
    
    # 创建图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('SDF演化动画 - 渐进式城市发展', fontsize=16, fontweight='bold')
    
    # 设置颜色映射
    hub_cmap = plt.cm.Reds
    road_cmap = plt.cm.Blues
    combined_cmap = plt.cm.RdYlBu_r
    
    # 初始化图像
    im1 = ax1.imshow(np.zeros((256, 256)), cmap=hub_cmap, vmin=0, vmax=1)
    im2 = ax2.imshow(np.zeros((256, 256)), cmap=road_cmap, vmin=0, vmax=1)
    im3 = ax3.imshow(np.zeros((256, 256)), cmap=combined_cmap, vmin=0, vmax=1)
    
    # 设置标题
    ax1.set_title('Hub SDF', fontsize=12)
    ax2.set_title('道路SDF', fontsize=12)
    ax3.set_title('组合SDF', fontsize=12)
    
    # 标记交通枢纽位置
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
        for hub in transport_hubs:
            ax.plot(hub[0], hub[1], 'ko', markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Hub SDF值')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('道路SDF值')
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('组合SDF值')
    
    # 添加时间信息文本
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        """动画更新函数"""
        month = months[frame]
        
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
        description = evolution_stage.get('description', '')
        
        # 更新图像数据
        im1.set_array(hub_sdf)
        im2.set_array(road_sdf)
        im3.set_array(combined_sdf)
        
        # 更新标题
        ax1.set_title(f'Hub SDF - {month}月', fontsize=12)
        ax2.set_title(f'道路SDF - {month}月\n扩展倍数: {road_multiplier:.1f}', fontsize=12)
        ax3.set_title(f'组合SDF - {month}月', fontsize=12)
        
        # 更新时间信息
        time_text.set_text(f'月份: {month}\n阶段: {description}')
        
        return im1, im2, im3, time_text
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=len(months), interval=1000, blit=False, repeat=True)
    
    # 保存动画
    output_file = 'sdf_evolution_animation.gif'
    anim.save(output_file, writer='pillow', fps=1, dpi=100)
    print(f"💾 动画保存: {output_file}")
    
    # 显示动画
    plt.tight_layout()
    plt.show()
    
    return anim

def analyze_sdf_evolution(sdf_system, months):
    """分析SDF演化统计信息"""
    
    print(f"\n📊 分析SDF演化统计信息")
    print("=" * 50)
    
    # 收集统计数据
    evolution_stats = []
    
    for month in months:
        sdf_system.update_sdf_field(month)
        sdf_components = sdf_system.get_sdf_components(month)
        
        stats = {
            'month': month,
            'hub_sdf': {
                'min': float(np.min(sdf_components['hub_sdf'])),
                'max': float(np.max(sdf_components['hub_sdf'])),
                'mean': float(np.mean(sdf_components['hub_sdf'])),
                'std': float(np.std(sdf_components['hub_sdf']))
            },
            'road_sdf': {
                'min': float(np.min(sdf_components['road_sdf'])),
                'max': float(np.max(sdf_components['road_sdf'])),
                'mean': float(np.mean(sdf_components['road_sdf'])),
                'std': float(np.std(sdf_components['road_sdf']))
            },
            'combined_sdf': {
                'min': float(np.min(sdf_components['combined_sdf'])),
                'max': float(np.max(sdf_components['combined_sdf'])),
                'mean': float(np.mean(sdf_components['combined_sdf'])),
                'std': float(np.std(sdf_components['combined_sdf']))
            }
        }
        
        evolution_stats.append(stats)
        
        # 获取演化阶段信息
        evolution_stage = sdf_system._get_evolution_stage(month)
        road_multiplier = evolution_stage.get('road_multiplier', 1.0)
        
        print(f"月份 {month:2d}: Hub SDF均值 {stats['hub_sdf']['mean']:.3f}, "
              f"道路SDF均值 {stats['road_sdf']['mean']:.3f}, "
              f"组合SDF均值 {stats['combined_sdf']['mean']:.3f}, "
              f"扩展倍数 {road_multiplier:.1f}")
    
    # 创建统计图表
    create_evolution_statistics(evolution_stats)

def create_evolution_statistics(evolution_stats):
    """创建演化统计图表"""
    
    months = [stats['month'] for stats in evolution_stats]
    
    # 创建统计图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SDF演化统计信息', fontsize=16, fontweight='bold')
    
    # 1. SDF均值变化
    ax1 = axes[0, 0]
    hub_means = [stats['hub_sdf']['mean'] for stats in evolution_stats]
    road_means = [stats['road_sdf']['mean'] for stats in evolution_stats]
    combined_means = [stats['combined_sdf']['mean'] for stats in evolution_stats]
    
    ax1.plot(months, hub_means, 'r-', linewidth=2, label='Hub SDF', marker='o')
    ax1.plot(months, road_means, 'b-', linewidth=2, label='道路SDF', marker='s')
    ax1.plot(months, combined_means, 'g-', linewidth=2, label='组合SDF', marker='^')
    ax1.set_xlabel('月份')
    ax1.set_ylabel('SDF均值')
    ax1.set_title('SDF均值随时间变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. SDF标准差变化
    ax2 = axes[0, 1]
    hub_stds = [stats['hub_sdf']['std'] for stats in evolution_stats]
    road_stds = [stats['road_sdf']['std'] for stats in evolution_stats]
    combined_stds = [stats['combined_sdf']['std'] for stats in evolution_stats]
    
    ax2.plot(months, hub_stds, 'r-', linewidth=2, label='Hub SDF', marker='o')
    ax2.plot(months, road_stds, 'b-', linewidth=2, label='道路SDF', marker='s')
    ax2.plot(months, combined_stds, 'g-', linewidth=2, label='组合SDF', marker='^')
    ax2.set_xlabel('月份')
    ax2.set_ylabel('SDF标准差')
    ax2.set_title('SDF标准差随时间变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. SDF最大值变化
    ax3 = axes[1, 0]
    hub_maxs = [stats['hub_sdf']['max'] for stats in evolution_stats]
    road_maxs = [stats['road_sdf']['max'] for stats in evolution_stats]
    combined_maxs = [stats['combined_sdf']['max'] for stats in evolution_stats]
    
    ax3.plot(months, hub_maxs, 'r-', linewidth=2, label='Hub SDF', marker='o')
    ax3.plot(months, road_maxs, 'b-', linewidth=2, label='道路SDF', marker='s')
    ax3.plot(months, combined_maxs, 'g-', linewidth=2, label='组合SDF', marker='^')
    ax3.set_xlabel('月份')
    ax3.set_ylabel('SDF最大值')
    ax3.set_title('SDF最大值随时间变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 道路SDF与Hub SDF比值
    ax4 = axes[1, 1]
    road_hub_ratios = [stats['road_sdf']['mean'] / max(stats['hub_sdf']['mean'], 1e-6) for stats in evolution_stats]
    
    ax4.plot(months, road_hub_ratios, 'purple', linewidth=2, marker='o')
    ax4.set_xlabel('月份')
    ax4.set_ylabel('道路SDF/Hub SDF比值')
    ax4.set_title('道路SDF相对强度变化')
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存统计图表
    output_file = 'sdf_evolution_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 统计图表保存: {output_file}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    # 创建SDF演化可视化
    create_sdf_evolution_visualization()
    
    # 重新加载配置和系统进行统计分析
    with open('configs/city_config_v2_3.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    sdf_system = EnhancedSDFSystem(config)
    transport_hubs = [[64, 64], [240, 64]]
    map_size = [256, 256]
    sdf_system.initialize_system(transport_hubs, map_size)
    
    months = list(range(0, 25, 3))
    analyze_sdf_evolution(sdf_system, months)
