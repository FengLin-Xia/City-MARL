#!/usr/bin/env python3
"""
检查槽位系统状态
分析槽位数量、分布、使用情况等
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def check_slot_system_status():
    """检查槽位系统状态"""
    print("=== 槽位系统状态检查 ===")
    
    # 检查层状态文件
    layer_files = sorted(glob.glob("enhanced_simulation_v3_1_output/layer_state_month_*.json"))
    
    if not layer_files:
        print("❌ 没有找到层状态文件")
        return
    
    print(f"📁 找到 {len(layer_files)} 个层状态文件")
    
    # 分析每个月的槽位状态
    slot_analysis = []
    
    for file_path in layer_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            month = data.get('month', 0)
            layers = data.get('layers', {})
            
            month_analysis = {
                'month': month,
                'commercial': {'total_slots': 0, 'used_slots': 0, 'dead_slots': 0, 'active_layers': 0},
                'residential': {'total_slots': 0, 'used_slots': 0, 'dead_slots': 0, 'active_layers': 0}
            }
            
            # 分析商业建筑层
            if 'commercial' in layers:
                commercial_layers = layers['commercial']
                for layer in commercial_layers:
                    month_analysis['commercial']['total_slots'] += layer.get('capacity', 0)
                    month_analysis['commercial']['used_slots'] += layer.get('placed', 0)
                    month_analysis['commercial']['dead_slots'] += layer.get('dead_slots', 0)
                    if layer.get('status') == 'active':
                        month_analysis['commercial']['active_layers'] += 1
            
            # 分析住宅建筑层
            if 'residential' in layers:
                residential_layers = layers['residential']
                for layer in residential_layers:
                    month_analysis['residential']['total_slots'] += layer.get('capacity', 0)
                    month_analysis['residential']['used_slots'] += layer.get('placed', 0)
                    month_analysis['residential']['dead_slots'] += layer.get('dead_slots', 0)
                    if layer.get('status') == 'active':
                        month_analysis['residential']['active_layers'] += 1
            
            slot_analysis.append(month_analysis)
            
        except Exception as e:
            print(f"⚠️ 加载文件失败: {file_path}, 错误: {e}")
    
    # 打印分析结果
    print("\n📊 槽位系统状态分析:")
    print("=" * 80)
    
    for analysis in slot_analysis:
        month = analysis['month']
        commercial = analysis['commercial']
        residential = analysis['residential']
        
        print(f"\n📅 第 {month} 个月:")
        print(f"  🏢 商业建筑:")
        print(f"    总槽位: {commercial['total_slots']}")
        print(f"    已使用: {commercial['used_slots']}")
        print(f"    死槽: {commercial['dead_slots']}")
        print(f"    激活层数: {commercial['active_layers']}")
        print(f"    使用率: {commercial['used_slots']/commercial['total_slots']*100:.1f}%" if commercial['total_slots'] > 0 else "    使用率: 0%")
        
        print(f"  🏠 住宅建筑:")
        print(f"    总槽位: {residential['total_slots']}")
        print(f"    已使用: {residential['used_slots']}")
        print(f"    死槽: {residential['dead_slots']}")
        print(f"    激活层数: {residential['active_layers']}")
        print(f"    使用率: {residential['used_slots']/residential['total_slots']*100:.1f}%" if residential['total_slots'] > 0 else "    使用率: 0%")
    
    # 创建可视化图表
    create_slot_visualization(slot_analysis)
    
    return slot_analysis

def create_slot_visualization(slot_analysis: List[Dict]):
    """创建槽位系统可视化图表"""
    if not slot_analysis:
        return
    
    # 提取数据
    months = [a['month'] for a in slot_analysis]
    commercial_total = [a['commercial']['total_slots'] for a in slot_analysis]
    commercial_used = [a['commercial']['used_slots'] for a in slot_analysis]
    residential_total = [a['residential']['total_slots'] for a in slot_analysis]
    residential_used = [a['residential']['used_slots'] for a in slot_analysis]
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('槽位系统状态分析', fontsize=16, fontweight='bold')
    
    # 1. 总槽位数量变化
    ax1.plot(months, commercial_total, 'o-', label='商业建筑', color='orange', linewidth=2)
    ax1.plot(months, residential_total, 's-', label='住宅建筑', color='blue', linewidth=2)
    ax1.set_title('总槽位数量变化')
    ax1.set_xlabel('月份')
    ax1.set_ylabel('槽位数量')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 已使用槽位数量变化
    ax2.plot(months, commercial_used, 'o-', label='商业建筑', color='orange', linewidth=2)
    ax2.plot(months, residential_used, 's-', label='住宅建筑', color='blue', linewidth=2)
    ax2.set_title('已使用槽位数量变化')
    ax2.set_xlabel('月份')
    ax2.set_ylabel('已使用槽位数量')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 槽位使用率
    commercial_usage = [used/total*100 if total > 0 else 0 for used, total in zip(commercial_used, commercial_total)]
    residential_usage = [used/total*100 if total > 0 else 0 for used, total in zip(residential_used, residential_total)]
    
    ax3.plot(months, commercial_usage, 'o-', label='商业建筑', color='orange', linewidth=2)
    ax3.plot(months, residential_usage, 's-', label='住宅建筑', color='blue', linewidth=2)
    ax3.set_title('槽位使用率变化')
    ax3.set_xlabel('月份')
    ax3.set_ylabel('使用率 (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # 4. 最终状态饼图
    final_analysis = slot_analysis[-1]
    commercial_final = final_analysis['commercial']
    residential_final = final_analysis['residential']
    
    # 商业建筑槽位状态
    commercial_labels = ['已使用', '可用', '死槽']
    commercial_values = [
        commercial_final['used_slots'],
        commercial_final['total_slots'] - commercial_final['used_slots'] - commercial_final['dead_slots'],
        commercial_final['dead_slots']
    ]
    commercial_colors = ['#FF6B6B', '#4ECDC4', '#95A5A6']
    
    wedges1, texts1, autotexts1 = ax4.pie(commercial_values, labels=commercial_labels, colors=commercial_colors,
                                          autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'商业建筑槽位状态 (第{final_analysis["month"]}月)')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = 'visualization_output/slot_system_analysis.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 槽位系统分析图表已保存到: {output_path}")
    
    plt.show()

def check_slot_distribution():
    """检查槽位分布情况"""
    print("\n=== 槽位分布检查 ===")
    
    # 检查最新的层状态文件
    layer_files = sorted(glob.glob("enhanced_simulation_v3_1_output/layer_state_month_*.json"))
    if not layer_files:
        print("❌ 没有找到层状态文件")
        return
    
    latest_file = layer_files[-1]
    print(f"📁 分析最新文件: {latest_file}")
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        month = data.get('month', 0)
        layers = data.get('layers', {})
        
        print(f"\n📅 第 {month} 个月槽位分布:")
        
        # 分析商业建筑层
        if 'commercial' in layers:
            commercial_layers = layers['commercial']
            print(f"\n🏢 商业建筑层 ({len(commercial_layers)} 层):")
            
            for i, layer in enumerate(commercial_layers):
                status = layer.get('status', 'unknown')
                capacity = layer.get('capacity', 0)
                placed = layer.get('placed', 0)
                dead_slots = layer.get('dead_slots', 0)
                density = layer.get('density', 0)
                
                status_icon = {'locked': '🔒', 'active': '🟢', 'complete': '✅'}.get(status, '❓')
                
                print(f"  {status_icon} 第{i}层: {status}")
                print(f"    容量: {placed}/{capacity} (死槽: {dead_slots})")
                print(f"    密度: {density:.1%}")
        
        # 分析住宅建筑层
        if 'residential' in layers:
            residential_layers = layers['residential']
            print(f"\n🏠 住宅建筑层 ({len(residential_layers)} 层):")
            
            for i, layer in enumerate(residential_layers):
                status = layer.get('status', 'unknown')
                capacity = layer.get('capacity', 0)
                placed = layer.get('placed', 0)
                dead_slots = layer.get('dead_slots', 0)
                density = layer.get('density', 0)
                
                status_icon = {'locked': '🔒', 'active': '🟢', 'complete': '✅'}.get(status, '❓')
                
                print(f"  {status_icon} 第{i}层: {status}")
                print(f"    容量: {placed}/{capacity} (死槽: {dead_slots})")
                print(f"    密度: {density:.1%}")
    
    except Exception as e:
        print(f"⚠️ 分析失败: {e}")

def main():
    """主函数"""
    print("🔍 槽位系统状态检查工具")
    print("=" * 50)
    
    # 检查槽位系统状态
    slot_analysis = check_slot_system_status()
    
    # 检查槽位分布
    check_slot_distribution()
    
    print("\n✅ 槽位系统检查完成")

if __name__ == "__main__":
    main()
