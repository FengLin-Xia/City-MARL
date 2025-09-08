#!/usr/bin/env python3
"""
测试渐进式地价场演化机制
验证道路优先发展 → Hub渐进增长 → 完整地价场的演化过程
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from logic.enhanced_sdf_system import GaussianLandPriceSystem

def test_land_price_evolution():
    """测试地价场演化机制"""
    print("🧪 测试渐进式地价场演化机制")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建地价场系统
    land_price_system = GaussianLandPriceSystem(config)
    
    # 初始化系统
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    land_price_system.initialize_system(transport_hubs, map_size)
    
    print(f"📍 交通枢纽位置: {transport_hubs}")
    print(f"🗺️ 地图尺寸: {map_size}")
    
    # 测试不同月份的演化状态
    test_months = [0, 3, 6, 9, 12, 15, 18, 24, 30, 36]
    
    print(f"\n📊 演化阶段测试:")
    print("-" * 60)
    
    for month in test_months:
        # 更新地价场
        land_price_system.update_land_price_field(month)
        
        # 获取演化阶段信息
        evolution_stage = land_price_system._get_evolution_stage(month)
        component_strengths = evolution_stage.get('component_strengths', {})
        
        # 获取地价场统计
        land_price_stats = land_price_system.get_land_price_stats()
        
        print(f"📅 第 {month:2d} 个月:")
        print(f"   阶段: {evolution_stage['description']} ({evolution_stage['name']})")
        print(f"   组件强度: 道路={component_strengths.get('road', 0):.1f}, Hub1={component_strengths.get('hub1', 0):.1f}, Hub2={component_strengths.get('hub2', 0):.1f}, Hub3={component_strengths.get('hub3', 0):.1f}")
        print(f"   地价统计: 最小值={land_price_stats['min']:.3f}, 最大值={land_price_stats['max']:.3f}, 平均值={land_price_stats['mean']:.3f}")
        print()
    
    # 创建可视化
    create_evolution_visualization(land_price_system, test_months)
    
    print("✅ 渐进式地价场演化机制测试完成")

def create_evolution_visualization(land_price_system, test_months):
    """创建演化可视化"""
    print("📊 创建演化可视化...")
    
    # 选择关键月份进行可视化
    key_months = [0, 6, 9, 12, 18, 24]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('渐进式地价场演化过程', fontsize=16, fontweight='bold')
    
    for i, month in enumerate(key_months):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 更新地价场
        land_price_system.update_land_price_field(month)
        land_price_field = land_price_system.get_land_price_field()
        
        # 获取演化阶段信息
        evolution_stage = land_price_system._get_evolution_stage(month)
        component_strengths = evolution_stage.get('component_strengths', {})
        
        # 绘制地价场
        im = ax.imshow(land_price_field, cmap='viridis', aspect='equal')
        ax.set_title(f'第 {month} 个月 - {evolution_stage["description"]}', fontsize=12, fontweight='bold')
        
        # 添加交通枢纽标记
        transport_hubs = land_price_system.transport_hubs
        for j, hub in enumerate(transport_hubs):
            strength = component_strengths.get(f'hub{j+1}', 0) if j < 3 else component_strengths.get('hub3', 0)
            if strength > 0:
                ax.plot(hub[0], hub[1], 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
                ax.text(hub[0], hub[1]-5, f'Hub{j+1}\n{strength:.1f}', ha='center', va='top', 
                       color='white', fontsize=8, fontweight='bold')
        
        # 添加道路标记（如果激活）
        road_strength = component_strengths.get('road', 0)
        if road_strength > 0 and len(transport_hubs) >= 2:
            ax.plot([transport_hubs[0][0], transport_hubs[1][0]], 
                   [transport_hubs[0][1], transport_hubs[1][1]], 
                   'w-', linewidth=3, alpha=0.8)
            ax.text((transport_hubs[0][0] + transport_hubs[1][0])/2, 
                   (transport_hubs[0][1] + transport_hubs[1][1])/2, 
                   f'道路\n{road_strength:.1f}', ha='center', va='center', 
                   color='white', fontsize=8, fontweight='bold')
        
        ax.set_xlim(0, 110)
        ax.set_ylim(0, 110)
        ax.invert_yaxis()  # 翻转Y轴以匹配图像坐标
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='地价值')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'visualization_output/land_price_evolution_test.png'
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 演化可视化已保存到: {output_path}")
    
    plt.show()

def test_component_strength_calculation():
    """测试组件强度计算"""
    print("\n🔍 测试组件强度计算:")
    print("-" * 40)
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建地价场系统
    land_price_system = GaussianLandPriceSystem(config)
    
    # 测试不同月份的组件强度
    test_months = [0, 3, 6, 7, 9, 12, 13, 15, 18, 24]
    
    for month in test_months:
        road_strength = land_price_system._get_component_strength('road', month)
        hub1_strength = land_price_system._get_component_strength('hub1', month)
        hub2_strength = land_price_system._get_component_strength('hub2', month)
        hub3_strength = land_price_system._get_component_strength('hub3', month)
        
        print(f"第 {month:2d} 个月: 道路={road_strength:.1f}, Hub1={hub1_strength:.1f}, Hub2={hub2_strength:.1f}, Hub3={hub3_strength:.1f}")

def main():
    """主函数"""
    print("🧪 渐进式地价场演化机制测试")
    print("=" * 60)
    
    # 测试组件强度计算
    test_component_strength_calculation()
    
    # 测试完整演化过程
    test_land_price_evolution()
    
    print("\n✅ 所有测试完成")

if __name__ == "__main__":
    main()
