#!/usr/bin/env python3
"""
测试商业建筑建设修复
"""

import json
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_city_simulation import EnhancedCitySimulation

def test_commercial_building():
    """测试商业建筑建设"""
    print("🧪 测试商业建筑建设修复...")
    
    # 创建模拟实例
    simulation = EnhancedCitySimulation()
    simulation.initialize_simulation()
    
    # 运行前6个月
    simulation_months = 6
    render_every_month = 1
    
    print(f"🚀 开始运行 {simulation_months} 个月测试...")
    
    for month in range(simulation_months):
        simulation.current_month = month
        
        # 每月更新
        simulation._monthly_update()
        
        # 定期渲染
        if month % render_every_month == 0:
            simulation._render_frame(month)
        
        # 显示建筑统计
        total_buildings = len(simulation.city_state['public']) + len(simulation.city_state['residential']) + len(simulation.city_state['commercial'])
        target_total = simulation._calculate_logistic_growth(month)
        
        print(f"📅 第 {month} 个月：")
        print(f"   人口: {len(simulation.city_state['residents'])}")
        print(f"   公共建筑: {len(simulation.city_state['public'])}")
        print(f"   住宅建筑: {len(simulation.city_state['residential'])}")
        print(f"   商业建筑: {len(simulation.city_state['commercial'])}")
        print(f"   总建筑: {total_buildings}/{target_total} (目标)")
        print(f"   有工作居民: {sum(1 for r in simulation.city_state['residents'] if r.get('workplace'))}")
        print()
    
    print("✅ 测试完成！")
    
    # 分析结果
    final_commercial = len(simulation.city_state['commercial'])
    final_residential = len(simulation.city_state['residential'])
    final_public = len(simulation.city_state['public'])
    
    print("📊 最终建筑分布:")
    print(f"   公共建筑: {final_public}")
    print(f"   住宅建筑: {final_residential}")
    print(f"   商业建筑: {final_commercial}")
    
    if final_commercial > 0:
        print("✅ 商业建筑建设成功！")
    else:
        print("❌ 商业建筑建设失败，需要进一步调试")

if __name__ == "__main__":
    test_commercial_building()
