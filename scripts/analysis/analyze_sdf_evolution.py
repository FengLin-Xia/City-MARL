#!/usr/bin/env python3
"""
分析SDF演化过程，检查线SDF是否真正起作用
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_sdf_evolution():
    """分析SDF演化过程"""
    
    # 读取不同月份的SDF数据
    months = [0, 6, 12, 18, 21]
    sdf_data = {}
    
    for month in months:
        filename = f'enhanced_simulation_v2_3_output/sdf_field_month_{month:02d}.json'
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                sdf_data[month] = {
                    'evolution_stage': data.get('evolution_stage', 0.0),
                    'sdf_field': np.array(data.get('sdf_field', [])),
                    'sdf_stats': data.get('sdf_stats', {})
                }
                print(f"📊 月份 {month}: 演化阶段 {data.get('evolution_stage', 0.0):.2f}")
        except Exception as e:
            print(f"❌ 无法读取 {filename}: {e}")
    
    # 分析SDF场的变化
    if len(sdf_data) > 1:
        print("\n🔍 SDF演化分析:")
        
        # 检查SDF值范围变化
        for month in months:
            if month in sdf_data:
                sdf_field = sdf_data[month]['sdf_field']
                evolution_stage = sdf_data[month]['evolution_stage']
                
                print(f"  月份 {month:2d} (阶段 {evolution_stage:.2f}):")
                print(f"    SDF范围: [{np.min(sdf_field):.3f}, {np.max(sdf_field):.3f}]")
                print(f"    SDF均值: {np.mean(sdf_field):.3f}")
                print(f"    SDF标准差: {np.std(sdf_field):.3f}")
                
                # 检查是否有明显的线状分布
                if evolution_stage > 0.5:
                    # 分析主干道沿线的SDF值
                    trunk_line_sdf = sdf_field[128, :]  # y=128是主干道位置
                    print(f"    主干道沿线SDF: 均值={np.mean(trunk_line_sdf):.3f}, 标准差={np.std(trunk_line_sdf):.3f}")
                    
                    # 检查是否有沿线的峰值
                    peaks = []
                    for i in range(1, len(trunk_line_sdf)-1):
                        if trunk_line_sdf[i] > trunk_line_sdf[i-1] and trunk_line_sdf[i] > trunk_line_sdf[i+1]:
                            peaks.append((i, trunk_line_sdf[i]))
                    
                    print(f"    主干道沿线峰值数量: {len(peaks)}")
                    if peaks:
                        print(f"    峰值位置: {[p[0] for p in peaks[:5]]}...")
        
        # 可视化SDF演化
        visualize_sdf_evolution(sdf_data, months)

def visualize_sdf_evolution(sdf_data, months):
    """可视化SDF演化过程"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, month in enumerate(months):
        if month in sdf_data:
            ax = axes[i]
            sdf_field = sdf_data[month]['sdf_field']
            evolution_stage = sdf_data[month]['evolution_stage']
            
            im = ax.imshow(sdf_field, cmap='hot', interpolation='nearest')
            ax.set_title(f'Month {month} (Stage {evolution_stage:.2f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # 标记交通枢纽
            ax.plot(40, 128, 'bo', markersize=10, label='Hub A')
            ax.plot(216, 128, 'bo', markersize=10, label='Hub B')
            
            # 标记主干道
            ax.plot([40, 216], [128, 128], 'w--', linewidth=2, alpha=0.7, label='Trunk Road')
            
            if i == 0:
                ax.legend()
    
    # 移除多余的子图
    for i in range(len(months), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('sdf_evolution_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 SDF演化可视化已保存为 sdf_evolution_analysis.png")

if __name__ == "__main__":
    analyze_sdf_evolution()


