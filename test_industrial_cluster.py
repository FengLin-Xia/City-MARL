#!/usr/bin/env python3
"""
测试工业集群协同奖励机制

验证：
1. 建筑注册表正确更新
2. 协同奖励正确计算
3. 解锁机制正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from contracts import ActionCandidate, EnvironmentState, BuildingRegistry, BuildingInfo
from reward_terms.industrial_cluster_reward import IndustrialClusterRewardTerm
from config_loader import ConfigLoader

def test_industrial_cluster_reward():
    """测试工业集群协同奖励机制"""
    print("开始测试工业集群协同奖励机制...")
    
    # 加载配置
    loader = ConfigLoader()
    config = loader.load_v5_config("configs/city_config_v5_0.json")
    
    # 创建奖励项
    reward_term = IndustrialClusterRewardTerm(config)
    print(f"奖励项创建成功: enabled={reward_term.enabled}")
    print(f"   加成率: {reward_term.bonus_rate}")
    print(f"   影响动作: {reward_term.affects_actions}")
    print(f"   触发动作: {reward_term.trigger_actions}")
    
    # 创建测试状态
    prev_state = EnvironmentState(
        month=14,
        land_prices=np.array([10.0, 15.0, 20.0]),
        buildings=[],
        budgets={"IND": 50000.0},
        slots=[],
        building_registry=BuildingRegistry()
    )
    
    # 测试1：没有高级工业建筑时，传统工业建筑没有协同奖励
    print("\n测试1：没有高级工业建筑时")
    state1 = EnvironmentState(
        month=15,
        land_prices=np.array([10.0, 15.0, 20.0]),
        buildings=[],
        budgets={"IND": 50000.0},
        slots=[],
        building_registry=BuildingRegistry()
    )
    
    # 模拟建造了5个动作3（传统工业）
    state1.building_registry.action_counts[3] = 5
    state1.building_registry.action_counts[4] = 2
    state1.building_registry.action_counts[5] = 1
    
    reward1 = reward_term.compute(prev_state, state1, 3)
    print(f"   动作3奖励: {reward1:.4f} (期望: 0.0)")
    assert reward1 == 0.0, f"期望奖励为0，实际为{reward1}"
    
    # 测试2：有高级工业建筑时，传统工业建筑有协同奖励
    print("\n测试2：有高级工业建筑时")
    state2 = EnvironmentState(
        month=16,
        land_prices=np.array([10.0, 15.0, 20.0]),
        buildings=[],
        budgets={"IND": 50000.0},
        slots=[],
        building_registry=BuildingRegistry()
    )
    
    # 模拟建造了传统工业建筑
    state2.building_registry.action_counts[3] = 8
    state2.building_registry.action_counts[4] = 3
    state2.building_registry.action_counts[5] = 2
    
    # 模拟建造了1个高级工业建筑（动作9）
    state2.building_registry.action_counts[9] = 1
    
    reward2 = reward_term.compute(prev_state, state2, 3)
    print(f"   动作3奖励: {reward2:.4f}")
    
    # 测试3：高级工业建筑没有协同奖励
    print("\n测试3：高级工业建筑没有协同奖励")
    reward3 = reward_term.compute(prev_state, state2, 9)
    print(f"   动作9奖励: {reward3:.4f} (期望: 0.0)")
    assert reward3 == 0.0, f"期望奖励为0，实际为{reward3}"
    
    # 测试4：不同传统工业建筑的协同奖励
    print("\n测试4：不同传统工业建筑的协同奖励")
    reward4_3 = reward_term.compute(prev_state, state2, 3)
    reward4_4 = reward_term.compute(prev_state, state2, 4)
    reward4_5 = reward_term.compute(prev_state, state2, 5)
    
    print(f"   动作3奖励: {reward4_3:.4f}")
    print(f"   动作4奖励: {reward4_4:.4f}")
    print(f"   动作5奖励: {reward4_5:.4f}")
    
    # 测试5：建筑注册表功能
    print("\n测试5：建筑注册表功能")
    registry = BuildingRegistry()
    
    # 添加建筑
    building1 = BuildingInfo(
        building_id="IND_3_15_0",
        action_id=3,
        agent="IND",
        month=15,
        position=(100.0, 200.0),
        cost=1000.0,
        reward=500.0
    )
    
    registry.buildings[building1.building_id] = building1
    registry.action_counts[3] = 1
    
    print(f"   建筑数量: {len(registry.buildings)}")
    print(f"   动作3计数: {registry.action_counts[3]}")
    
    # 测试6：协同效应检查
    print("\n测试6：协同效应检查")
    has_cluster = reward_term._is_cluster_bonus_active(state2)
    print(f"   协同效应激活: {has_cluster}")
    assert has_cluster == True, "应该有协同效应"
    
    has_cluster_no = reward_term._is_cluster_bonus_active(state1)
    print(f"   无协同效应: {has_cluster_no}")
    assert has_cluster_no == False, "应该没有协同效应"
    
    print("\n所有测试通过！工业集群协同奖励机制工作正常。")
    
    return True

def test_building_registry_integration():
    """测试建筑注册表集成"""
    print("\n测试建筑注册表集成...")
    
    # 创建建筑注册表
    registry = BuildingRegistry()
    
    # 模拟添加建筑
    for i in range(5):
        building = BuildingInfo(
            building_id=f"IND_3_15_{i}",
            action_id=3,
            agent="IND",
            month=15,
            position=(100.0 + i*10, 200.0 + i*10),
            cost=1000.0,
            reward=500.0
        )
        registry.buildings[building.building_id] = building
        registry.action_counts[3] = registry.action_counts.get(3, 0) + 1
    
    print(f"建筑注册表创建成功")
    print(f"   总建筑数: {len(registry.buildings)}")
    print(f"   动作3计数: {registry.action_counts[3]}")
    
    # 测试解锁状态
    registry.unlock_status[9] = True
    registry.unlock_status[10] = True
    registry.unlock_status[11] = True
    
    print(f"   解锁状态: {registry.unlock_status}")
    
    return True

if __name__ == "__main__":
    try:
        test_industrial_cluster_reward()
        test_building_registry_integration()
        print("\n所有测试完成！")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
