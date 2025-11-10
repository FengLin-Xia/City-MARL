#!/usr/bin/env python3
"""
测试统一使用V5RewardCalculator的效果
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(__file__))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import ActionCandidate


def test_reward_calculator_unification():
    """测试奖励计算器统一的效果"""
    print("=" * 60)
    print("测试统一使用V5RewardCalculator")
    print("=" * 60)
    
    # 配置文件路径
    config_path = "configs/city_config_v5_0.json"
    
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在: {config_path}")
        return
    
    try:
        # 初始化环境
        print("初始化V5城市环境...")
        env = V5CityEnvironment(config_path)
        
        # 重置环境
        print("重置环境...")
        env.reset()
        
        # 创建测试动作候选（添加必需的features参数）
        test_candidates = [
            ActionCandidate(
                id=0,  # EDU_S
                features=np.array([0.8, 0.5, 0.3, 0.2, 0.1]),  # 示例特征
                meta={
                    'agent': 'EDU',
                    'action_id': 0,
                    'point_id': 'slot_1',
                    'slots': ['slot_1'],
                    'zone': 'near',
                    'land_price_norm': 0.8,
                    'river_dist_m': 50.0,
                    'adj': 1
                }
            ),
            ActionCandidate(
                id=3,  # IND_S
                features=np.array([0.6, 0.4, 0.2, 0.1, 0.05]),  # 示例特征
                meta={
                    'agent': 'IND',
                    'action_id': 3,
                    'point_id': 'slot_2',
                    'slots': ['slot_2'],
                    'zone': 'mid',
                    'land_price_norm': 0.6,
                    'river_dist_m': 100.0,
                    'adj': 0
                }
            ),
            ActionCandidate(
                id=6,  # COUNCIL_A
                features=np.array([0.4, 0.3, 0.1, 0.05, 0.02]),  # 示例特征
                meta={
                    'agent': 'COUNCIL',
                    'action_id': 6,
                    'point_id': 'slot_3',
                    'slots': ['slot_3'],
                    'zone': 'far',
                    'land_price_norm': 0.4,
                    'river_dist_m': 200.0,
                    'adj': 1
                }
            )
        ]
        
        print(f"\n测试奖励计算...")
        print(f"当前月份: {env.current_month}")
        print(f"当前步数: {env.current_step}")
        
        # 测试每个动作候选的奖励计算
        for i, candidate in enumerate(test_candidates):
            print(f"\n--- 测试动作 {i+1}: {candidate.meta['agent']} 动作 {candidate.id} ---")
            
            # 计算奖励
            total_reward, reward_dict = env._compute_reward(candidate.meta['agent'], candidate)
            
            print(f"总奖励: {total_reward:.2f}")
            print(f"奖励明细:")
            for key, value in reward_dict.items():
                if isinstance(value, (int, float)):
                    print(f"  - {key}: {value:.2f}")
                else:
                    print(f"  - {key}: {value}")
        
        # 测试奖励计算器的一致性
        print(f"\n--- 测试奖励计算一致性 ---")
        
        # 直接使用V5RewardCalculator
        current_state = env._get_current_environment_state()
        direct_reward = env.reward_calculator.calculate_reward(test_candidates[0], current_state)
        
        # 通过环境计算奖励
        env_reward, env_dict = env._compute_reward(test_candidates[0].meta['agent'], test_candidates[0])
        
        # 计算直接奖励的总和
        direct_total = direct_reward.revenue + direct_reward.cost + direct_reward.prestige + direct_reward.proximity + direct_reward.diversity
        if direct_reward.other:
            numeric_values = [v for v in direct_reward.other.values() if isinstance(v, (int, float))]
            direct_total += sum(numeric_values)
        
        print(f"直接计算奖励: {direct_total:.2f}")
        print(f"环境计算奖励: {env_reward:.2f}")
        print(f"一致性检查: {'一致' if abs(direct_total - env_reward) < 0.01 else '不一致'}")
        
        # 显示奖励计算器的详细信息
        print(f"\n--- V5RewardCalculator详细信息 ---")
        print(f"启用状态: {env.reward_calculator.enabled}")
        print(f"调试模式: {env.reward_calculator.debug_mode}")
        print(f"精度模式: {env.reward_calculator.precision}")
        print(f"舍入模式: {env.reward_calculator.rounding_mode}")
        print(f"启用组件: {env.reward_calculator.components}")
        
        print(f"\n奖励计算器统一测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_reward_calculator_unification()
