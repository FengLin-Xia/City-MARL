#!/usr/bin/env python3
"""
测试候选范围逐渐扩大功能

验证候选范围是否随时间逐渐扩大
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def test_candidate_range_expansion():
    """测试候选范围逐渐扩大功能"""
    print("=" * 80)
    print("测试候选范围逐渐扩大功能")
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
        print(f"   - 模式: {hubs_config.get('mode')}")
        print(f"   - 候选模式: {hubs_config.get('candidate_mode')}")
        print(f"   - 容差: {hubs_config.get('tol')}")
        
        hub_list = hubs_config.get("list", [])
        for i, hub in enumerate(hub_list):
            print(f"   - Hub {i+1}: {hub['id']} at ({hub['x']}, {hub['y']})")
            print(f"     * R0: {hub['R0']}, dR: {hub['dR']}, weight: {hub['weight']}")
        
        # 检查候选范围中间件
        print(f"\n   候选范围中间件:")
        action_mw = env.config.get("action_mw", [])
        print(f"   - 中间件列表: {action_mw}")
        if "candidate_range" in action_mw:
            print(f"   - [PASS] candidate_range中间件已启用")
        else:
            print(f"   - [FAIL] candidate_range中间件未启用")
        
        # 测试不同月份的候选范围
        print(f"\n   测试不同月份的候选范围:")
        
        test_months = [0, 5, 10, 15, 20, 25, 29]
        
        for month in test_months:
            # 设置环境到指定月份
            env.current_month = month
            env.current_step = month
            
            # 获取候选动作
            candidates = env.get_action_candidates("IND")
            
            if candidates:
                # 分析候选动作的槽位分布
                slot_positions = []
                for candidate in candidates[:50]:  # 只分析前50个候选
                    slots = candidate.meta.get('slots', [])
                    if slots:
                        slot_id = slots[0]
                        if slot_id in env.slots:
                            slot = env.slots[slot_id]
                            slot_positions.append((slot['x'], slot['y']))
                
                if slot_positions:
                    # 计算距离Hub的距离
                    hub1_pos = (122, 80)
                    hub2_pos = (112, 121)
                    
                    distances_to_hub1 = [np.sqrt((x - hub1_pos[0])**2 + (y - hub1_pos[1])**2) 
                                       for x, y in slot_positions]
                    distances_to_hub2 = [np.sqrt((x - hub2_pos[0])**2 + (y - hub2_pos[1])**2) 
                                       for x, y in slot_positions]
                    
                    min_dist_hub1 = min(distances_to_hub1) if distances_to_hub1 else 0
                    max_dist_hub1 = max(distances_to_hub1) if distances_to_hub1 else 0
                    min_dist_hub2 = min(distances_to_hub2) if distances_to_hub2 else 0
                    max_dist_hub2 = max(distances_to_hub2) if distances_to_hub2 else 0
                    
                    print(f"   - 月份 {month}: {len(candidates)} 个候选")
                    print(f"     * Hub1距离: {min_dist_hub1:.1f} - {max_dist_hub1:.1f}")
                    print(f"     * Hub2距离: {min_dist_hub2:.1f} - {max_dist_hub2:.1f}")
                    
                    # 计算理论半径
                    R0 = 5  # hub1的R0
                    dR = 1.5  # hub1的dR
                    theoretical_radius = R0 + month * dR
                    print(f"     * 理论半径: {theoretical_radius:.1f}")
                else:
                    print(f"   - 月份 {month}: 无有效槽位位置")
            else:
                print(f"   - 月份 {month}: 无候选动作")
        
        # 测试候选范围中间件是否被调用
        print(f"\n   测试候选范围中间件调用:")
        
        # 创建测试序列
        from contracts import ActionCandidate
        test_candidate = ActionCandidate(
            id=3,
            features=np.array([1.0, 2.0, 3.0]),
            meta={
                "agent": "IND",
                "action_id": 3,
                "cost": 900,
                "reward": 150,
                "prestige": 0.2,
                "slots": ["slot_0"],
                "zone": "default",
                "lp_norm": 0.5
            }
        )
        
        test_sequence = Sequence(
            agent="IND",
            actions=[3]
        )
        
        # 检查中间件是否被调用
        print(f"   - 测试序列: {test_sequence}")
        print(f"   - 测试候选: {test_candidate}")
        
        # 模拟中间件调用
        try:
            from action_mw.candidate_range import CandidateRangeMiddleware
            middleware = CandidateRangeMiddleware(env.config)
            filtered_sequence = middleware.apply(test_sequence, state)
            print(f"   - [PASS] 候选范围中间件调用成功")
            print(f"   - 过滤后序列: {filtered_sequence}")
        except Exception as e:
            print(f"   - [FAIL] 候选范围中间件调用失败: {e}")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_candidate_range_expansion()

