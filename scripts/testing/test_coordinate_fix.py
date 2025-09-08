#!/usr/bin/env python3
"""
测试坐标系修复效果
"""

import json
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_city_simulation import EnhancedCitySimulation

def test_coordinate_fix():
    """测试坐标系修复"""
    print("🧪 测试坐标系修复效果...")
    
    # 创建模拟实例
    simulation = EnhancedCitySimulation()
    simulation.initialize_simulation()
    
    # 运行前2个月，检查轨迹对应关系
    for month in range(2):
        simulation.current_month = month
        
        print(f"\n📅 第 {month} 个月：")
        
        # 显示建筑位置
        residential = simulation.city_state.get('residential', [])
        commercial = simulation.city_state.get('commercial', [])
        
        print(f"  住宅建筑: {len(residential)} 个")
        if residential:
            print(f"    示例住宅位置: {residential[0]['xy']}")
        
        print(f"  商业建筑: {len(commercial)} 个")
        if commercial:
            print(f"    示例商业位置: {commercial[0]['xy']}")
        
        # 显示居民信息
        residents = simulation.city_state.get('residents', [])
        working_residents = [r for r in residents if r.get('home') and r.get('workplace')]
        print(f"  有工作的居民: {len(working_residents)} 个")
        
        # 执行每月更新
        simulation._monthly_update()
        
        # 获取热力图数据
        heatmap_data = simulation.trajectory_system.get_heatmap_data()
        commute_heatmap = heatmap_data['commute_heatmap']
        commercial_heatmap = heatmap_data['commercial_heatmap']
        
        # 分析热力图
        print(f"  通勤热力图:")
        print(f"    最大值: {commute_heatmap.max():.2f}")
        print(f"    总和: {commute_heatmap.sum():.2f}")
        
        # 找到热点位置
        hot_spots = np.where(commute_heatmap > 0)
        if len(hot_spots[0]) > 0:
            # 找到最热的点
            max_idx = np.argmax(commute_heatmap)
            max_y, max_x = np.unravel_index(max_idx, commute_heatmap.shape)
            max_intensity = commute_heatmap[max_y, max_x]
            print(f"    最热点位置: ({max_x}, {max_y}), 强度: {max_intensity:.2f}")
            
            # 检查这个位置是否靠近建筑
            if residential and commercial:
                home_pos = residential[0]['xy']
                work_pos = commercial[0]['xy']
                
                # 计算距离
                dist_to_home = ((max_x - home_pos[0])**2 + (max_y - home_pos[1])**2)**0.5
                dist_to_work = ((max_x - work_pos[0])**2 + (max_y - work_pos[1])**2)**0.5
                
                print(f"    到住宅的距离: {dist_to_home:.1f}")
                print(f"    到工作地点的距离: {dist_to_work:.1f}")
                
                # 判断轨迹是否合理
                if dist_to_home < 50 or dist_to_work < 50:
                    print(f"    ✅ 热点位置合理（靠近建筑）")
                else:
                    print(f"    ❌ 热点位置异常（远离建筑）")
        else:
            print(f"    没有发现热点")
    
    print("\n🎯 测试完成！")

if __name__ == "__main__":
    test_coordinate_fix()


