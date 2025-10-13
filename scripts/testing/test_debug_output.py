#!/usr/bin/env python3
"""
测试调试输出
运行一个简短的模拟来查看调试信息
"""

from enhanced_city_simulation_v3_1 import EnhancedCitySimulationV3_1

def test_debug_output():
    """测试调试输出"""
    
    print("=== 测试调试输出 ===")
    
    # 创建模拟系统
    simulation = EnhancedCitySimulationV3_1()
    simulation.initialize_simulation()
    
    # 只运行前3个月来查看调试输出
    print("运行前3个月...")
    
    for month in range(3):
        simulation.current_month = month
        simulation.current_quarter = month // 3
        simulation.current_year = month // 12
        
        print(f"\n--- Month {month} ---")
        
        # 每月更新
        simulation._monthly_update()
        
        # 季度更新
        if month % 3 == 0:
            simulation._quarterly_update()
        
        # 年度更新
        if month % 12 == 0:
            simulation._yearly_update()
    
    print("\n=== 调试输出完成 ===")

if __name__ == "__main__":
    test_debug_output()
