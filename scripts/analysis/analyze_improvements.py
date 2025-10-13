#!/usr/bin/env python3
"""
分析改进前后的城市模拟差异
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_population_growth():
    """分析人口增长模式"""
    print("📊 人口增长模式分析")
    print("="*50)
    
    # 原始模拟数据
    try:
        with open('test_long_term_output/poi_evolution.json', 'r') as f:
            original_data = json.load(f)
        
        # 计算原始模拟的人口增长
        original_population = []
        for i, day_data in enumerate(original_data):
            # 估算人口：基于住宅数量 * 平均容量
            residential_count = day_data['residential']
            estimated_population = residential_count * 200 * 0.6  # 假设60%入住率
            original_population.append({
                'day': i,
                'population': estimated_population,
                'residential': residential_count
            })
        
        print("🏠 原始模拟（每天30人）：")
        print(f"   初始人口: {original_population[0]['population']:.0f}")
        print(f"   最终人口: {original_population[-1]['population']:.0f}")
        print(f"   增长率: {(original_population[-1]['population'] - original_population[0]['population']) / original_population[0]['population'] * 100:.1f}%")
        print(f"   住宅设施: {original_population[0]['residential']} → {original_population[-1]['residential']}")
        
    except FileNotFoundError:
        print("❌ 未找到原始模拟数据")
        return
    
    # 改进模拟数据（如果存在）
    try:
        with open('improved_simulation_output/population_history.json', 'r') as f:
            improved_data = json.load(f)
        
        print("\n🏙️ 改进模拟（每月5%增长率）：")
        print(f"   初始人口: {improved_data[0]['population']}")
        print(f"   最终人口: {improved_data[-1]['population']}")
        print(f"   增长率: {(improved_data[-1]['population'] - improved_data[0]['population']) / improved_data[0]['population'] * 100:.1f}%")
        
        # 绘制对比图
        plot_population_comparison(original_population, improved_data)
        
    except FileNotFoundError:
        print("❌ 未找到改进模拟数据，请先运行 improved_city_simulation.py")
        return

def plot_population_comparison(original_data, improved_data):
    """绘制人口增长对比图"""
    plt.figure(figsize=(12, 8))
    
    # 原始数据
    original_days = [d['day'] for d in original_data]
    original_pop = [d['population'] for d in original_data]
    plt.plot(original_days, original_pop, 'r-', linewidth=2, label='原始模拟（每天30人）')
    
    # 改进数据
    improved_days = [d['day'] for d in improved_data]
    improved_pop = [d['population'] for d in improved_data]
    plt.plot(improved_days, improved_pop, 'b-', linewidth=2, label='改进模拟（每月5%增长率）')
    
    plt.xlabel('天数')
    plt.ylabel('人口数量')
    plt.title('人口增长模式对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plt.savefig('population_growth_comparison.png', dpi=300, bbox_inches='tight')
    print("📈 人口增长对比图已保存: population_growth_comparison.png")
    plt.show()

def analyze_residential_management():
    """分析住宅管理改进"""
    print("\n🏠 住宅管理改进分析")
    print("="*50)
    
    print("原始模拟问题：")
    print("❌ 每天固定增加30人，不考虑住宅容量")
    print("❌ 居民和住宅关系模糊，随机分配")
    print("❌ 无限制增长，可能导致过度拥挤")
    print("❌ 缺乏现实的人口增长模式")
    
    print("\n改进模拟解决方案：")
    print("✅ 每月5%增长率，更符合现实")
    print("✅ 智能住宅分配，考虑容量和距离")
    print("✅ 最大人口密度限制（80%）")
    print("✅ 明确的居民-住宅关系管理")
    print("✅ 基于评分的住宅选择算法")

def analyze_business_logic():
    """分析商业逻辑改进"""
    print("\n🏢 商业逻辑改进分析")
    print("="*50)
    
    print("住宅分配算法：")
    print("1. 容量检查：确保住宅有可用空间")
    print("2. 距离评分：偏好靠近主干道的住宅")
    print("3. 综合评分：空间优先，距离次之")
    print("4. 关系管理：维护居民-住宅映射")
    
    print("\n人口增长控制：")
    print("1. 月度增长：每30天计算一次增长")
    print("2. 容量限制：基于总住宅容量计算上限")
    print("3. 密度控制：最大80%入住率")
    print("4. 动态调整：根据可用容量调整增长")

def show_improvement_summary():
    """显示改进总结"""
    print("\n🎯 改进总结")
    print("="*50)
    
    improvements = [
        "📈 更现实的人口增长模式（月度而非每日）",
        "🏠 智能住宅分配和管理",
        "👥 明确的居民-住宅关系",
        "⚖️ 容量和密度控制",
        "🎯 基于评分的选址算法",
        "📊 更准确的统计计算",
        "🔄 动态增长调整",
        "💾 完整的数据跟踪"
    ]
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    print("\n🚀 建议运行改进模拟：")
    print("  python improved_city_simulation.py")

def main():
    """主函数"""
    print("🔍 城市模拟改进分析")
    print("="*60)
    
    analyze_population_growth()
    analyze_residential_management()
    analyze_business_logic()
    show_improvement_summary()

if __name__ == "__main__":
    main()
