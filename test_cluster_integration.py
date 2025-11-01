#!/usr/bin/env python3
"""
测试工业集群协同奖励在V5RewardCalculator中的集成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from contracts import ActionCandidate, EnvironmentState, BuildingRegistry
from logic.v5_reward_calculator import V5RewardCalculator

def test_cluster_integration():
    """测试工业集群奖励集成"""
    print("=== 测试工业集群协同奖励集成 ===")
    
    # 加载配置
    with open("configs/city_config_v5_0.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 创建奖励计算器
    calculator = V5RewardCalculator(config)
    
    # 测试1: 没有高级工业建筑时的传统工业动作
    print("\n--- 测试1: 没有高级工业建筑 ---")
    
    # 创建动作候选（动作3 - 传统工业）
    action = ActionCandidate(
        id=3,
        features=np.array([1.0, 2.0, 3.0]),
        meta={
            "zone": "mid",
            "land_price_norm": 0.5,
            "river_dist_m": 1000.0,
            "adj": 0
        }
    )
    
    # 创建环境状态（没有高级工业建筑）
    state = EnvironmentState(
        month=20,
        land_prices=np.array([1.0, 2.0, 3.0]),
        buildings=[],
        budgets={"IND": 50000.0},
        slots=[],
        building_registry=BuildingRegistry()
    )
    
    # 计算奖励
    reward_terms = calculator.calculate_reward(action, state)
    print(f"动作3奖励: {reward_terms.revenue:.2f}")
    print(f"协同奖励: {reward_terms.other.get('cluster_bonus', 0):.2f}")
    
    # 测试2: 有高级工业建筑时的传统工业动作
    print("\n--- 测试2: 有高级工业建筑 ---")
    
    # 创建环境状态（有高级工业建筑）
    state_with_cluster = EnvironmentState(
        month=20,
        land_prices=np.array([1.0, 2.0, 3.0]),
        buildings=[],
        budgets={"IND": 50000.0},
        slots=[],
        building_registry=BuildingRegistry(
            action_counts={9: 1, 10: 0, 11: 0}  # 建造了1个动作9
        )
    )
    
    # 计算奖励
    reward_terms_with_cluster = calculator.calculate_reward(action, state_with_cluster)
    print(f"动作3奖励（有协同）: {reward_terms_with_cluster.revenue:.2f}")
    print(f"协同奖励: {reward_terms_with_cluster.other.get('cluster_bonus', 0):.2f}")
    
    # 测试3: 高级工业动作本身不获得协同奖励
    print("\n--- 测试3: 高级工业动作 ---")
    
    # 创建动作候选（动作9 - 高级工业）
    action_advanced = ActionCandidate(
        id=9,
        features=np.array([1.0, 2.0, 3.0]),
        meta={
            "zone": "mid",
            "land_price_norm": 0.5,
            "river_dist_m": 1000.0,
            "adj": 0
        }
    )
    
    # 计算奖励
    reward_terms_advanced = calculator.calculate_reward(action_advanced, state_with_cluster)
    print(f"动作9奖励: {reward_terms_advanced.revenue:.2f}")
    print(f"协同奖励: {reward_terms_advanced.other.get('cluster_bonus', 0):.2f}")
    
    # 验证结果
    print("\n--- 验证结果 ---")
    
    # 检查协同奖励是否正确计算
    base_reward = config["action_params"]["3"]["base_reward"]
    bonus_rate = config["reward_mechanisms"]["industrial_cluster_bonus"]["bonus_rate"]
    expected_cluster_bonus = base_reward * bonus_rate
    
    actual_cluster_bonus = reward_terms_with_cluster.other.get('cluster_bonus', 0)
    
    print(f"预期协同奖励: {expected_cluster_bonus:.2f}")
    print(f"实际协同奖励: {actual_cluster_bonus:.2f}")
    print(f"奖励差异: {reward_terms_with_cluster.revenue - reward_terms.revenue:.2f}")
    
    if abs(actual_cluster_bonus - expected_cluster_bonus) < 0.01:
        print("SUCCESS: 协同奖励计算正确！")
    else:
        print("ERROR: 协同奖励计算错误！")
    
    if reward_terms_advanced.other.get('cluster_bonus', 0) == 0:
        print("SUCCESS: 高级工业动作不获得协同奖励！")
    else:
        print("ERROR: 高级工业动作错误地获得了协同奖励！")

if __name__ == "__main__":
    test_cluster_integration()
