#!/usr/bin/env python3
"""
调试商业建筑建设问题
"""

import json
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.enhanced_agents import BusinessAgent
from logic.land_price_system import LandPriceSystem

def debug_commercial_issue():
    """调试商业建筑建设问题"""
    print("🔍 调试商业建筑建设问题...")
    
    # 加载配置
    with open('configs/building_config.json', 'r', encoding='utf-8') as f:
        building_config = json.load(f)
    
    with open('configs/agent_config.json', 'r', encoding='utf-8') as f:
        agent_config = json.load(f)
    
    # 合并配置
    business_config = agent_config['business_agent'].copy()
    if 'building_growth' in building_config:
        business_config.update(building_config['building_growth'])
    
    # 创建智能体
    business_agent = BusinessAgent(business_config)
    land_price_system = LandPriceSystem(building_config)
    
    # 初始化地价系统
    land_price_system.initialize_land_prices([256, 256], [[40, 128], [216, 128]])
    
    # 模拟不同阶段的城市状态
    test_cases = [
        {
            'name': '初始状态',
            'residents': 100,
            'residential_buildings': 5,
            'commercial_buildings': 0
        },
        {
            'name': '人口增长后',
            'residents': 150,
            'residential_buildings': 8,
            'commercial_buildings': 0
        },
        {
            'name': '更多人口',
            'residents': 200,
            'residential_buildings': 10,
            'commercial_buildings': 0
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📊 测试案例 {i+1}: {test_case['name']}")
        
        # 创建模拟城市状态
        city_state = {
            'residents': [{'id': f'agent_{j}'} for j in range(test_case['residents'])],
            'residential': [{'id': f'res_{j}', 'xy': [100 + j*20, 128]} for j in range(test_case['residential_buildings'])],
            'commercial': [{'id': f'com_{j}', 'xy': [150 + j*20, 128]} for j in range(test_case['commercial_buildings'])]
        }
        
        # 检查商业扩张需求
        needs_expansion = business_agent._needs_commercial_expansion(city_state)
        
        print(f"   人口: {test_case['residents']}")
        print(f"   住宅建筑: {test_case['residential_buildings']}")
        print(f"   商业建筑: {test_case['commercial_buildings']}")
        print(f"   需要商业扩张: {needs_expansion}")
        
        if needs_expansion:
            # 尝试建设商业建筑
            land_price_matrix = land_price_system.get_land_price_matrix()
            heatmap_data = {
                'combined_heatmap': land_price_matrix * 0.1  # 模拟热力图
            }
            
            new_commercial = business_agent._decide_commercial_development_enhanced(
                city_state, land_price_system, land_price_matrix, heatmap_data
            )
            
            print(f"   建设结果: {len(new_commercial)} 个新商业建筑")
            if new_commercial:
                print(f"   新建筑位置: {new_commercial[0]['xy']}")
        else:
            print("   不满足建设条件")
            
            # 分析为什么不满足条件
            residents = city_state.get('residents', [])
            commercial_buildings = city_state.get('commercial', [])
            residential_buildings = city_state.get('residential', [])
            
            print(f"   基础条件检查:")
            print(f"     人口 >= 30: {len(residents) >= 30}")
            print(f"     住宅建筑 >= 3: {len(residential_buildings) >= 3}")
            
            if len(residents) >= 30 and len(residential_buildings) >= 3:
                target_commercial = len(residents) // 50
                current_commercial = len(commercial_buildings)
                print(f"     目标商业建筑: {target_commercial} (每50人1个)")
                print(f"     当前商业建筑: {current_commercial}")
                print(f"     需要建设: {current_commercial < target_commercial}")

if __name__ == "__main__":
    debug_commercial_issue()
