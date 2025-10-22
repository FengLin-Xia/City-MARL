#!/usr/bin/env python3
"""
测试候选集使用情况

检查候选集的使用和建筑分布
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def test_candidate_usage():
    """测试候选集使用情况"""
    print("=" * 80)
    print("测试候选集使用情况")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 检查Hub配置
        print(f"\n   Hub配置:")
        hubs_config = env.config.get("hubs", {})
        hub_list = hubs_config.get("list", [])
        for hub in hub_list:
            print(f"   - {hub['id']}: ({hub['x']}, {hub['y']}), R0={hub['R0']}, dR={hub['dR']}")
        
        # 检查河流配置
        print(f"\n   河流配置:")
        river_config = env.config.get("env", {}).get("river_restrictions", {})
        print(f"   - 河流限制: {river_config.get('enabled')}")
        print(f"   - 影响智能体: {river_config.get('affects_agents')}")
        print(f"   - Council绕过: {river_config.get('council_bypass')}")
        
        # 检查候选范围配置
        print(f"\n   候选范围配置:")
        print(f"   - 模式: {hubs_config.get('mode')}")
        print(f"   - 候选模式: {hubs_config.get('candidate_mode')}")
        print(f"   - 容差: {hubs_config.get('tol')}")
        
        # 分析不同月份的候选分布
        print(f"\n   不同月份的候选分布分析:")
        
        for month in [0, 5, 10, 15, 20, 25, 29]:
            env.current_month = month
            print(f"\n   月份 {month}:")
            
            # 分析IND候选
            ind_candidates = env.get_action_candidates("IND")
            if ind_candidates:
                ind_slots = []
                for candidate in ind_candidates:
                    slots = candidate.meta.get("slots", [])
                    if slots:
                        ind_slots.extend(slots)
                
                if ind_slots:
                    # 获取槽位坐标
                    ind_positions = []
                    for slot_id in ind_slots:
                        if slot_id in env.slots:
                            slot = env.slots[slot_id]
                            ind_positions.append((slot['x'], slot['y']))
                    
                    if ind_positions:
                        # 分析坐标分布
                        x_coords = [pos[0] for pos in ind_positions]
                        y_coords = [pos[1] for pos in ind_positions]
                        
                        print(f"     - IND候选: {len(ind_candidates)} 个")
                        print(f"     - IND槽位: {len(ind_positions)} 个")
                        print(f"     - X坐标范围: {min(x_coords):.1f} - {max(x_coords):.1f}")
                        print(f"     - Y坐标范围: {min(y_coords):.1f} - {max(y_coords):.1f}")
                        
                        # 分析相对于Hub的分布
                        hub1_pos = (122, 80)
                        hub2_pos = (112, 121)
                        
                        # 计算到Hub1的距离
                        distances_to_hub1 = [np.sqrt((x - hub1_pos[0])**2 + (y - hub1_pos[1])**2) 
                                           for x, y in ind_positions]
                        distances_to_hub2 = [np.sqrt((x - hub2_pos[0])**2 + (y - hub2_pos[1])**2) 
                                           for x, y in ind_positions]
                        
                        print(f"     - 到Hub1距离: {min(distances_to_hub1):.1f} - {max(distances_to_hub1):.1f}")
                        print(f"     - 到Hub2距离: {min(distances_to_hub2):.1f} - {max(distances_to_hub2):.1f}")
                        
                        # 分析河流侧别分布
                        river_x = 100  # 假设河流在x=100处
                        left_side = [pos for pos in ind_positions if pos[0] < river_x]
                        right_side = [pos for pos in ind_positions if pos[0] >= river_x]
                        
                        print(f"     - 河流左侧: {len(left_side)} 个槽位")
                        print(f"     - 河流右侧: {len(right_side)} 个槽位")
                        
                        if len(left_side) > 0 and len(right_side) > 0:
                            print(f"     - [PASS] 候选分布在河流两侧")
                        elif len(left_side) > 0:
                            print(f"     - [WARNING] 候选只在河流左侧")
                        elif len(right_side) > 0:
                            print(f"     - [WARNING] 候选只在河流右侧")
                        else:
                            print(f"     - [FAIL] 无候选槽位")
        
        # 检查河流限制中间件
        print(f"\n   河流限制中间件:")
        action_mw = env.config.get("action_mw", [])
        print(f"   - 中间件列表: {action_mw}")
        
        if "river_restriction" in action_mw:
            print(f"   - [PASS] 河流限制中间件已启用")
        else:
            print(f"   - [FAIL] 河流限制中间件未启用")
        
        # 检查候选范围中间件
        if "candidate_range" in action_mw:
            print(f"   - [PASS] 候选范围中间件已启用")
        else:
            print(f"   - [FAIL] 候选范围中间件未启用")
        
        # 分析实际建筑分布
        print(f"\n   实际建筑分布分析:")
        
        # 模拟一些建筑
        for i in range(5):
            # 获取当前智能体
            current_agent = env.current_agent
            candidates = env.get_action_candidates(current_agent)
            
            if candidates:
                # 选择第一个候选
                selected_candidate = candidates[0]
                selected_slots = selected_candidate.meta.get("slots", [])
                
                if selected_slots:
                    slot_id = selected_slots[0]
                    if slot_id in env.slots:
                        slot = env.slots[slot_id]
                        print(f"   - 建筑 {i+1}: {current_agent} 在 ({slot['x']:.1f}, {slot['y']:.1f})")
                        
                        # 分析河流侧别
                        river_x = 100
                        if slot['x'] < river_x:
                            side = "左侧"
                        else:
                            side = "右侧"
                        print(f"     - 河流{side}")
                
                # 执行动作
                sequence = Sequence(
                    agent=current_agent,
                    actions=[selected_candidate.id]
                )
                
                next_state, reward, done, info = env.step(current_agent, sequence)
            else:
                print(f"   - 建筑 {i+1}: 无候选动作")
        
        # 检查最终建筑分布
        print(f"\n   最终建筑分布:")
        if env.buildings:
            building_positions = []
            for building in env.buildings:
                # 从建筑信息中获取位置（这里需要根据实际建筑数据结构调整）
                print(f"   - {building.get('agent', 'Unknown')}: {building}")
        else:
            print(f"   - 无建筑记录")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_candidate_usage()

