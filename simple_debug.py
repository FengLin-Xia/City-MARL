#!/usr/bin/env python3
"""
简单调试脚本
"""

import json
import os

def simple_test():
    """简单测试"""
    print("=== 简单调试测试 ===")
    
    # 检查文件
    data_file = "enhanced_simulation_v3_1_output/building_positions_month_23.json"
    if not os.path.exists(data_file):
        print("文件不存在")
        return
    
    # 加载数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    buildings = data.get('buildings', [])
    print(f"总建筑数: {len(buildings)}")
    
    # 统计类型
    types = {}
    for building in buildings:
        t = building['type']
        types[t] = types.get(t, 0) + 1
    
    print("建筑类型分布:")
    for t, count in types.items():
        print(f"  {t}: {count}")
    
    # 检查商业建筑位置
    commercial_buildings = [b for b in buildings if b['type'] == 'commercial']
    print(f"\n商业建筑数: {len(commercial_buildings)}")
    
    if commercial_buildings:
        print("前5个商业建筑位置:")
        for i, building in enumerate(commercial_buildings[:5]):
            pos = building['position']
            print(f"  {building['id']}: {pos}")
            
            # 计算到 Hub2 的距离
            hub2 = [90, 55]
            distance = ((pos[0] - hub2[0])**2 + (pos[1] - hub2[1])**2)**0.5
            print(f"    到 Hub2 距离: {distance:.1f}")

if __name__ == "__main__":
    simple_test()
