#!/usr/bin/env python3
"""
测试奖励计算修复效果
验证 action_reward 是否回到正常范围
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from contracts import ActionCandidate, EnvironmentState, BuildingRegistry
from logic.v5_reward_calculator import V5RewardCalculator
from envs.v5_0.city_env import V5CityEnvironment
import json

def test_reward_calculation_fix():
    """测试奖励计算修复效果"""
    print("=== 测试奖励计算修复效果 ===")
    
    # 加载配置
    with open("configs/city_config_v5_0.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 创建环境
    env = V5CityEnvironment("configs/city_config_v5_0.json")
    
    # 创建测试动作候选
    test_action = ActionCandidate(
        id=3,  # IND_S
        features=np.array([0.0, 0.0, 0.0]),
        meta={
            "action_id": 3,
            "point_id": "slot_test",
            "slots": ["slot_test"],
            "zone": "far",
            "land_price_norm": 0.0,
            "river_dist_m": 1e9,  # 10亿米，这是导致问题的值
            "adj": 0
        }
    )
    
    # 创建测试状态
    test_state = EnvironmentState(
        month=1,
        land_prices=np.array([]),
        buildings=[],
        budgets={},
        slots=[],
        building_registry=BuildingRegistry()
    )
    
    # 测试修复后的奖励计算
    print("\n--- 测试修复后的奖励计算 ---")
    action_reward, reward_dict = env._compute_reward("IND", test_action)
    
    print(f"action_reward: {action_reward:.2f}")
    print(f"reward_dict keys: {list(reward_dict.keys())}")
    
    # 检查关键值
    print(f"\n关键值检查:")
    print(f"  action_reward: {action_reward:.2f}")
    print(f"  delta_reward: {reward_dict.get('delta_reward', 0):.2f}")
    print(f"  shaping_reward: {reward_dict.get('shaping_reward', 0):.2f}")
    print(f"  monthly_total: {reward_dict.get('monthly_total', 0):.2f}")
    
    # 验证修复效果
    if abs(action_reward) < 10000:  # 应该在合理范围内
        print(f"\n[SUCCESS] 修复成功！action_reward 回到正常范围: {action_reward:.2f}")
        return True
    else:
        print(f"\n[FAILED] 修复失败！action_reward 仍然异常: {action_reward:.2f}")
        return False

def test_multiple_actions():
    """测试多个动作的奖励计算"""
    print("\n=== 测试多个动作的奖励计算 ===")
    
    # 加载配置
    with open("configs/city_config_v5_0.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 创建环境
    env = V5CityEnvironment("configs/city_config_v5_0.json")
    
    # 测试不同动作
    test_actions = [
        {"id": 3, "agent": "IND", "desc": "IND_S"},
        {"id": 4, "agent": "IND", "desc": "IND_M"},
        {"id": 5, "agent": "IND", "desc": "IND_L"},
        {"id": 0, "agent": "EDU", "desc": "EDU_S"},
        {"id": 8, "agent": "COUNCIL", "desc": "COUNCIL_C"}
    ]
    
    for action_info in test_actions:
        test_action = ActionCandidate(
            id=action_info["id"],
            features=np.array([0.0, 0.0, 0.0]),
            meta={
                "action_id": action_info["id"],
                "point_id": "slot_test",
                "slots": ["slot_test"],
                "zone": "far",
                "land_price_norm": 0.0,
                "river_dist_m": 1e9,  # 10亿米
                "adj": 0
            }
        )
        
        test_state = EnvironmentState(
            month=1,
            land_prices=np.array([]),
            buildings=[],
            budgets={},
            slots=[],
            building_registry=BuildingRegistry()
        )
        
        action_reward, reward_dict = env._compute_reward(action_info["agent"], test_action)
        
        print(f"{action_info['desc']} (ID={action_info['id']}): action_reward={action_reward:.2f}")
        
        if abs(action_reward) > 10000:
            print(f"  [WARNING] {action_info['desc']} 的奖励仍然异常！")
            return False
    
    print("\n[SUCCESS] 所有动作的奖励计算都正常！")
    return True

if __name__ == "__main__":
    try:
        # 测试基本修复效果
        success1 = test_reward_calculation_fix()
        
        # 测试多个动作
        success2 = test_multiple_actions()
        
        if success1 and success2:
            print("\n[SUCCESS] 所有测试通过！奖励计算修复成功！")
            print("现在可以重新训练，预期效果：")
            print("- action_reward 回到正常范围")
            print("- entropy_sum > 0，恢复探索能力")
            print("- IND 开始选择动作4/5，利用size bonus")
        else:
            print("\n[FAILED] 测试失败！需要进一步检查修复。")
            
    except Exception as e:
        print(f"\n[ERROR] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
