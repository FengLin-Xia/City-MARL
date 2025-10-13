#!/usr/bin/env python3
"""
测试 Enhanced City Simulation v3.1.1 可视化效果
"""

import os
import sys
from visualize_v3_1_1_evolution import V3_1_1EvolutionPlayback

def test_hub2_industrial_effect():
    """测试 Hub2 工业中心效果"""
    print("=== 测试 Hub2 工业中心颜色伪装效果 ===")
    
    # 创建可视化系统
    visualizer = V3_1_1EvolutionPlayback()
    
    if not visualizer.months:
        print("错误: 没有找到可用的数据文件")
        return
    
    print(f"找到 {len(visualizer.months)} 个月份的数据")
    print(f"月份范围: {min(visualizer.months)} - {max(visualizer.months)}")
    print()
    
    # 分析 Hub2 工业中心效果
    visualizer.analyze_hub2_industrial_effect()
    
    # 创建最后几个月的静态图像
    print("创建最后3个月的静态图像...")
    last_months = visualizer.months[-3:]
    visualizer.create_static_plots(last_months)
    
    print("测试完成！")

def test_color_scheme():
    """测试颜色方案"""
    print("=== 测试颜色方案 ===")
    
    visualizer = V3_1_1EvolutionPlayback()
    
    print("Hub 颜色方案:")
    for hub_type, color in visualizer.colors['hubs'].items():
        print(f"  {hub_type}: {color}")
    
    print("\n建筑颜色方案:")
    for building_type, color in visualizer.colors['buildings'].items():
        print(f"  {building_type}: {color}")
    
    print(f"\nHub2 工业中心影响半径: {visualizer.hub2_industrial_radius} 像素")
    
    # 测试距离计算
    test_points = [
        (90, 55),   # Hub2 中心
        (100, 55),  # Hub2 附近
        (120, 55),  # Hub2 远处
        (67, 94),   # Hub3 位置
    ]
    
    print("\n距离测试:")
    for x, y in test_points:
        is_near = visualizer._is_near_hub2(x, y)
        print(f"  点 ({x}, {y}): {'在 Hub2 附近' if is_near else '远离 Hub2'}")

if __name__ == "__main__":
    print("Enhanced City Simulation v3.1.1 测试脚本")
    print("=" * 50)
    
    # 测试颜色方案
    test_color_scheme()
    print()
    
    # 测试 Hub2 工业中心效果
    test_hub2_industrial_effect()
