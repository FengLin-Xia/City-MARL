#!/usr/bin/env python3
"""
调试实际模拟中的商业建筑建设问题
"""

import json
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_city_simulation import EnhancedCitySimulation

def debug_simulation_issue():
    """调试实际模拟中的问题"""
    print("🔍 调试实际模拟中的商业建筑建设问题...")
    
    # 创建模拟实例
    simulation = EnhancedCitySimulation()
    simulation.initialize_simulation()
    
    # 运行前3个月，详细记录每一步
    for month in range(3):
        print(f"\n📅 第 {month} 个月开始")
        
        # 记录当前状态
        total_buildings = len(simulation.city_state['public']) + len(simulation.city_state['residential']) + len(simulation.city_state['commercial'])
        print(f"   当前状态: 人口 {len(simulation.city_state['residents'])}, 建筑 {total_buildings}")
        print(f"   住宅建筑: {len(simulation.city_state['residential'])}")
        print(f"   商业建筑: {len(simulation.city_state['commercial'])}")
        
        # 检查商业扩张需求
        needs_commercial = simulation.business_agent._needs_commercial_expansion(simulation.city_state)
        print(f"   需要商业扩张: {needs_commercial}")
        
        if needs_commercial:
            # 尝试建设商业建筑
            land_price_matrix = simulation.land_price_system.get_land_price_matrix()
            heatmap_data = simulation.trajectory_system.get_heatmap_data()
            
            new_commercial = simulation.business_agent._decide_commercial_development_enhanced(
                simulation.city_state, simulation.land_price_system, land_price_matrix, heatmap_data
            )
            
            print(f"   商业建筑建设结果: {len(new_commercial)} 个")
            if new_commercial:
                print(f"   新建筑位置: {new_commercial[0]['xy']}")
        
        # 检查Logistic增长计算
        monthly_new_buildings = simulation._calculate_monthly_new_buildings(month)
        print(f"   Logistic增长计算: 第{month}个月应新增 {monthly_new_buildings} 个建筑")
        
        # 执行每月更新
        simulation.current_month = month
        
        # 手动调用智能体决策来调试
        print(f"   🔍 手动调用智能体决策...")
        
        # 检查建筑分布
        monthly_new_buildings = simulation._calculate_monthly_new_buildings(month)
        if monthly_new_buildings > 0:
            building_distribution = simulation._get_building_type_distribution(monthly_new_buildings)
            print(f"   建筑分布: {building_distribution}")
            
            # 手动调用建筑建设
            simulation._build_buildings_in_batches(building_distribution)
        else:
            print(f"   本月不需要新增建筑")
        
        simulation._agent_decisions()
        
        # 执行其他更新
        simulation._update_trajectories()
        simulation.trajectory_system.apply_decay()
        simulation.land_price_system.update_land_prices(simulation.city_state)
        simulation.city_state['land_price_stats'] = simulation.land_price_system.get_land_price_stats()
        simulation._spawn_new_residents()
        simulation._update_building_usage()
        simulation._calculate_monthly_stats()
        
        # 记录更新后状态
        total_buildings_after = len(simulation.city_state['public']) + len(simulation.city_state['residential']) + len(simulation.city_state['commercial'])
        print(f"   更新后状态: 人口 {len(simulation.city_state['residents'])}, 建筑 {total_buildings_after}")
        print(f"   住宅建筑: {len(simulation.city_state['residential'])}")
        print(f"   商业建筑: {len(simulation.city_state['commercial'])}")
        
        # 检查是否有新建筑
        if total_buildings_after > total_buildings:
            print(f"   ✅ 新增了 {total_buildings_after - total_buildings} 个建筑")
        else:
            print(f"   ❌ 没有新增建筑")

if __name__ == "__main__":
    debug_simulation_issue()
