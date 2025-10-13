#!/usr/bin/env python3
"""
分析槽位系统控制机制
检查槽位系统的控制位置、数量、分布等
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def analyze_slot_control_system():
    """分析槽位系统控制机制"""
    print("=== 槽位系统控制机制分析 ===")
    
    # 1. 检查配置文件中的槽位相关设置
    print("\n1. 配置文件分析:")
    config_file = "configs/city_config_v3_1.json"
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查渐进式增长配置
        progressive_config = config.get('progressive_growth', {})
        print(f"  严格满格要求: {progressive_config.get('strict_fill_required', True)}")
        print(f"  死槽容忍率: {progressive_config.get('allow_dead_slots_ratio', 0.05):.1%}")
        print(f"  配额结转: {progressive_config.get('carry_over_quota', True)}")
        print(f"  冻结施工线: {progressive_config.get('freeze_contour_on_activation', True)}")
        
        # 检查等值线配置
        isocontour_config = config.get('isocontour_layout', {})
        print(f"  等值线间隔: {isocontour_config.get('spacing_pixels', 8)} 像素")
        print(f"  最小距离: {isocontour_config.get('min_distance', 8)} 像素")
    
    # 2. 分析槽位创建逻辑
    print("\n2. 槽位创建逻辑分析:")
    analyze_slot_creation_logic()
    
    # 3. 分析槽位激活机制
    print("\n3. 槽位激活机制分析:")
    analyze_slot_activation_mechanism()
    
    # 4. 分析槽位使用情况
    print("\n4. 槽位使用情况分析:")
    analyze_slot_usage()
    
    # 5. 分析槽位分布
    print("\n5. 槽位分布分析:")
    analyze_slot_distribution()

def analyze_slot_creation_logic():
    """分析槽位创建逻辑"""
    print("  槽位创建流程:")
    print("    1. 从等值线提取轮廓点")
    print("    2. 计算轮廓总长度")
    print("    3. 按固定间隔(8像素)创建槽位")
    print("    4. 检查槽位距离(最小8像素)")
    print("    5. 确保至少有一个槽位")
    
    # 检查实际的槽位创建结果
    layer_files = sorted(glob.glob("enhanced_simulation_v3_1_output/layer_state_month_*.json"))
    if layer_files:
        latest_file = layer_files[-1]
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        layers = data.get('layers', {})
        
        # 分析商业建筑槽位
        if 'commercial' in layers:
            commercial_layers = layers['commercial']
            total_commercial_slots = sum(layer.get('capacity', 0) for layer in commercial_layers)
            print(f"    商业建筑总槽位: {total_commercial_slots}")
            
            # 分析每层的槽位数量
            for i, layer in enumerate(commercial_layers):
                capacity = layer.get('capacity', 0)
                print(f"      第{i}层: {capacity} 个槽位")
        
        # 分析住宅建筑槽位
        if 'residential' in layers:
            residential_layers = layers['residential']
            total_residential_slots = sum(layer.get('capacity', 0) for layer in residential_layers)
            print(f"    住宅建筑总槽位: {total_residential_slots}")
            
            # 分析每层的槽位数量
            for i, layer in enumerate(residential_layers):
                capacity = layer.get('capacity', 0)
                print(f"      第{i}层: {capacity} 个槽位")

def analyze_slot_activation_mechanism():
    """分析槽位激活机制"""
    print("  槽位激活流程:")
    print("    1. 检查当前层是否完成")
    print("    2. 计算死槽率")
    print("    3. 死槽率 <= 5% 时激活下一层")
    print("    4. 更新层状态为 'active'")
    
    # 检查实际的激活情况
    layer_files = sorted(glob.glob("enhanced_simulation_v3_1_output/layer_state_month_*.json"))
    if layer_files:
        latest_file = layer_files[-1]
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        layers = data.get('layers', {})
        
        # 分析商业建筑层状态
        if 'commercial' in layers:
            commercial_layers = layers['commercial']
            print(f"    商业建筑层状态:")
            for i, layer in enumerate(commercial_layers):
                status = layer.get('status', 'unknown')
                dead_slots = layer.get('dead_slots', 0)
                capacity = layer.get('capacity', 0)
                dead_ratio = dead_slots / capacity if capacity > 0 else 0
                
                status_icon = {'locked': '🔒', 'active': '🟢', 'complete': '✅'}.get(status, '❓')
                print(f"      第{i}层: {status_icon} {status} (死槽率: {dead_ratio:.1%})")
        
        # 分析住宅建筑层状态
        if 'residential' in layers:
            residential_layers = layers['residential']
            print(f"    住宅建筑层状态:")
            for i, layer in enumerate(residential_layers):
                status = layer.get('status', 'unknown')
                dead_slots = layer.get('dead_slots', 0)
                capacity = layer.get('capacity', 0)
                dead_ratio = dead_slots / capacity if capacity > 0 else 0
                
                status_icon = {'locked': '🔒', 'active': '🟢', 'complete': '✅'}.get(status, '❓')
                print(f"      第{i}层: {status_icon} {status} (死槽率: {dead_ratio:.1%})")

def analyze_slot_usage():
    """分析槽位使用情况"""
    print("  槽位使用统计:")
    
    # 检查所有月份的使用情况
    layer_files = sorted(glob.glob("enhanced_simulation_v3_1_output/layer_state_month_*.json"))
    
    if layer_files:
        # 分析最后几个月的使用情况
        recent_files = layer_files[-5:]  # 最近5个月
        
        for file_path in recent_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            month = data.get('month', 0)
            layers = data.get('layers', {})
            
            print(f"    第{month}个月:")
            
            # 商业建筑使用情况
            if 'commercial' in layers:
                commercial_layers = layers['commercial']
                total_capacity = sum(layer.get('capacity', 0) for layer in commercial_layers)
                total_placed = sum(layer.get('placed', 0) for layer in commercial_layers)
                total_dead = sum(layer.get('dead_slots', 0) for layer in commercial_layers)
                usage_rate = total_placed / total_capacity if total_capacity > 0 else 0
                
                print(f"      商业: {total_placed}/{total_capacity} ({usage_rate:.1%}) 死槽: {total_dead}")
            
            # 住宅建筑使用情况
            if 'residential' in layers:
                residential_layers = layers['residential']
                total_capacity = sum(layer.get('capacity', 0) for layer in residential_layers)
                total_placed = sum(layer.get('placed', 0) for layer in residential_layers)
                total_dead = sum(layer.get('dead_slots', 0) for layer in residential_layers)
                usage_rate = total_placed / total_capacity if total_capacity > 0 else 0
                
                print(f"      住宅: {total_placed}/{total_capacity} ({usage_rate:.1%}) 死槽: {total_dead}")

def analyze_slot_distribution():
    """分析槽位分布"""
    print("  槽位分布分析:")
    
    # 检查等值线数据
    contour_files = sorted(glob.glob("enhanced_simulation_v3_1_output/land_price_frame_month_*.json"))
    if contour_files:
        latest_file = contour_files[-1]
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        month = data.get('month', 0)
        print(f"    第{month}个月等值线分布:")
        
        # 检查等值线数据
        if 'contour_data' in data:
            contour_data = data['contour_data']
            
            # 商业等值线
            if 'commercial_contours' in contour_data:
                commercial_contours = contour_data['commercial_contours']
                print(f"      商业等值线: {len(commercial_contours)} 条")
                for i, contour in enumerate(commercial_contours):
                    if isinstance(contour, list) and len(contour) > 0:
                        print(f"        第{i}条: {len(contour)} 个点")
            
            # 住宅等值线
            if 'residential_contours' in contour_data:
                residential_contours = contour_data['residential_contours']
                print(f"      住宅等值线: {len(residential_contours)} 条")
                for i, contour in enumerate(residential_contours):
                    if isinstance(contour, list) and len(contour) > 0:
                        print(f"        第{i}条: {len(contour)} 个点")

def create_slot_control_visualization():
    """创建槽位控制可视化"""
    print("\n6. 创建槽位控制可视化:")
    
    # 加载数据
    layer_files = sorted(glob.glob("enhanced_simulation_v3_1_output/layer_state_month_*.json"))
    if not layer_files:
        print("  没有找到层状态文件")
        return
    
    # 分析数据
    months = []
    commercial_capacity = []
    commercial_placed = []
    residential_capacity = []
    residential_placed = []
    
    for file_path in layer_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        month = data.get('month', 0)
        layers = data.get('layers', {})
        
        months.append(month)
        
        # 商业建筑
        if 'commercial' in layers:
            commercial_layers = layers['commercial']
            total_capacity = sum(layer.get('capacity', 0) for layer in commercial_layers)
            total_placed = sum(layer.get('placed', 0) for layer in commercial_layers)
            commercial_capacity.append(total_capacity)
            commercial_placed.append(total_placed)
        else:
            commercial_capacity.append(0)
            commercial_placed.append(0)
        
        # 住宅建筑
        if 'residential' in layers:
            residential_layers = layers['residential']
            total_capacity = sum(layer.get('capacity', 0) for layer in residential_layers)
            total_placed = sum(layer.get('placed', 0) for layer in residential_layers)
            residential_capacity.append(total_capacity)
            residential_placed.append(total_placed)
        else:
            residential_capacity.append(0)
            residential_placed.append(0)
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Slot Control System Analysis', fontsize=16, fontweight='bold')
    
    # 1. 槽位容量变化
    ax1.plot(months, commercial_capacity, 'o-', label='Commercial', color='orange', linewidth=2)
    ax1.plot(months, residential_capacity, 's-', label='Residential', color='blue', linewidth=2)
    ax1.set_title('Slot Capacity Over Time')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Slots')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 已使用槽位变化
    ax2.plot(months, commercial_placed, 'o-', label='Commercial', color='orange', linewidth=2)
    ax2.plot(months, residential_placed, 's-', label='Residential', color='blue', linewidth=2)
    ax2.set_title('Used Slots Over Time')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Used Slots')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 使用率变化
    commercial_usage = [placed/capacity*100 if capacity > 0 else 0 for placed, capacity in zip(commercial_placed, commercial_capacity)]
    residential_usage = [placed/capacity*100 if capacity > 0 else 0 for placed, capacity in zip(residential_placed, residential_capacity)]
    
    ax3.plot(months, commercial_usage, 'o-', label='Commercial', color='orange', linewidth=2)
    ax3.plot(months, residential_usage, 's-', label='Residential', color='blue', linewidth=2)
    ax3.set_title('Slot Usage Rate Over Time')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Usage Rate (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # 4. 最终状态对比
    final_commercial = commercial_placed[-1] if commercial_placed else 0
    final_residential = residential_placed[-1] if residential_placed else 0
    
    categories = ['Commercial', 'Residential']
    values = [final_commercial, final_residential]
    colors = ['orange', 'blue']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7)
    ax4.set_title(f'Final Slot Usage (Month {months[-1]})')
    ax4.set_ylabel('Used Slots')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'visualization_output/slot_control_analysis.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  槽位控制分析图表已保存到: {output_path}")
    
    plt.show()

def main():
    """主函数"""
    print("🔍 槽位系统控制机制分析工具")
    print("=" * 60)
    
    # 分析槽位系统控制机制
    analyze_slot_control_system()
    
    # 创建可视化
    create_slot_control_visualization()
    
    print("\n✅ 槽位系统控制机制分析完成")

if __name__ == "__main__":
    main()
