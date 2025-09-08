#!/usr/bin/env python3
"""
调试坐标系统问题
"""

import json
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_coordinates():
    """调试坐标系统"""
    print("🔍 调试坐标系统问题...")
    
    # 读取最新的模拟数据
    try:
        with open('enhanced_simulation_output/city_state_output.json', 'r', encoding='utf-8') as f:
            city_data = json.load(f)
        
        with open('enhanced_simulation_output/trajectory_data.json', 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e}")
        return
    
    print("\n📍 建筑位置信息：")
    
    # 显示住宅建筑位置
    residential = city_data.get('buildings', {}).get('residential', [])
    print(f"住宅建筑 ({len(residential)} 个):")
    for building in residential[:3]:  # 只显示前3个
        print(f"  {building['id']}: 位置 {building['xy']}")
    
    # 显示商业建筑位置
    commercial = city_data.get('buildings', {}).get('commercial', [])
    print(f"商业建筑 ({len(commercial)} 个):")
    for building in commercial[:3]:  # 只显示前3个
        print(f"  {building['id']}: 位置 {building['xy']}")
    
    # 显示居民信息
    residents = city_data.get('residents', [])
    print(f"\n👥 居民信息 ({len(residents)} 个):")
    for resident in residents[:3]:  # 只显示前3个
        home = resident.get('home', 'None')
        workplace = resident.get('workplace', 'None')
        print(f"  {resident['id']}: 住宅={home}, 工作={workplace}")
    
    # 检查热力图数据
    heatmap_data = trajectory_data.get('heatmap_data', {})
    commute_heatmap = np.array(heatmap_data.get('commute_heatmap', []))
    commercial_heatmap = np.array(heatmap_data.get('commercial_heatmap', []))
    
    print(f"\n🔥 热力图信息:")
    print(f"  通勤热力图形状: {commute_heatmap.shape}")
    print(f"  通勤热力图最大值: {commute_heatmap.max():.2f}")
    print(f"  通勤热力图总和: {commute_heatmap.sum():.2f}")
    print(f"  商业热力图形状: {commercial_heatmap.shape}")
    print(f"  商业热力图最大值: {commercial_heatmap.max():.2f}")
    print(f"  商业热力图总和: {commercial_heatmap.sum():.2f}")
    
    # 检查热力图中的热点位置
    if commute_heatmap.size > 0:
        print(f"\n🔥 通勤热力图热点:")
        # 找到热力值大于0的位置
        hot_spots = np.where(commute_heatmap > 0)
        if len(hot_spots[0]) > 0:
            for i in range(min(5, len(hot_spots[0]))):  # 显示前5个热点
                y, x = hot_spots[0][i], hot_spots[1][i]
                intensity = commute_heatmap[y, x]
                print(f"  位置 ({x}, {y}): 强度 {intensity:.2f}")
        else:
            print("  没有发现热点")
    
    # 检查坐标系转换问题
    print(f"\n🔍 坐标系分析:")
    print(f"  地图大小: [256, 256]")
    print(f"  热力图矩阵: {commute_heatmap.shape} (高度×宽度)")
    print(f"  建筑坐标格式: [x, y]")
    print(f"  热力图索引格式: [y, x]")
    
    # 模拟一个轨迹，检查坐标转换
    if residential and commercial:
        home_pos = residential[0]['xy']  # [x, y]
        work_pos = commercial[0]['xy']   # [x, y]
        
        print(f"\n🛤️ 轨迹测试:")
        print(f"  住宅位置: {home_pos}")
        print(f"  工作位置: {work_pos}")
        
        # 检查这些位置在热力图中的索引
        print(f"  住宅在热力图中的索引: [{home_pos[1]}, {home_pos[0]}]")
        print(f"  工作在热力图中的索引: [{work_pos[1]}, {work_pos[0]}]")
        
        # 检查这些位置是否在热力图范围内
        valid_home = 0 <= home_pos[0] < 256 and 0 <= home_pos[1] < 256
        valid_work = 0 <= work_pos[0] < 256 and 0 <= work_pos[1] < 256
        print(f"  住宅位置有效: {valid_home}")
        print(f"  工作位置有效: {valid_work}")

if __name__ == "__main__":
    debug_coordinates()


