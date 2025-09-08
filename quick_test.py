#!/usr/bin/env python3
"""
快速测试版本 - 用于调试和快速验证
"""

import sys
sys.path.append('logic')
sys.path.append('viz')

from main_demo import CitySimulation

def quick_test():
    """快速测试仿真"""
    print("开始快速测试...")
    
    # 创建仿真实例
    simulation = CitySimulation('data/poi_example.json')
    
    # 修改为快速测试参数
    simulation.days = 2
    simulation.steps_per_day = 48  # 减少步数
    
    # 运行仿真
    simulation.run_simulation()
    
    print("快速测试完成！")

if __name__ == "__main__":
    quick_test()



