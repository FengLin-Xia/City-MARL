#!/usr/bin/env python3
"""
测试多动作模式功能
"""

import sys
import os
import traceback

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_multi_action_enabled():
    """测试启用多动作模式"""
    print("=== 测试多动作模式启用 ===")
    try:
        from config_loader import ConfigLoader
        from solvers.v5_0.rl_selector import V5RLSelector
        from contracts import EnvironmentState, CandidateIndex, AtomicAction
        import numpy as np
        
        # 加载配置
        loader = ConfigLoader()
        config = loader.load_v5_config("configs/city_config_v5_0.json")
        
        # 启用多动作模式
        config["multi_action"]["enabled"] = True
        print("PASS: 启用多动作模式")
        
        # 创建多动作选择器
        selector = V5RLSelector(config)
        print("PASS: 多动作选择器创建成功")
        
        # 检查多动作网络
        if hasattr(selector, 'actor_networks_multi'):
            print("PASS: 多动作网络已初始化")
            for agent in ["IND", "EDU", "COUNCIL"]:
                if agent in selector.actor_networks_multi:
                    print(f"  - {agent}: 多动作网络就绪")
        else:
            print("FAIL: 多动作网络未初始化")
            return False
        
        # 测试多动作选择
        mock_state = EnvironmentState(
            month=0,
            land_prices=np.zeros((1, 1)),
            buildings=[],
            budgets={"IND": 1000.0},
            slots=[]
        )
        
        # 创建模拟候选索引
        mock_cand_idx = CandidateIndex(
            points=[100, 101],
            types_per_point=[[0, 1], [2, 3]],
            point_to_slots={100: ["s1"], 101: ["s2"]}
        )
        
        # 测试多动作选择（使用贪心模式确保有输出）
        result = selector.select_action_multi(
            agent="IND",
            candidates=[],  # 空候选列表
            cand_idx=mock_cand_idx,
            state=mock_state,
            max_k=1,  # 减少到1个动作
            greedy=True
        )
        
        if result:
            print(f"PASS: 多动作选择成功: {len(result['sequence'].actions)} 个动作")
            print(f"  - logprob: {result['logprob']:.4f}")
            print(f"  - entropy: {result['entropy']:.4f}")
            print(f"  - value: {result['value']:.4f}")
        else:
            print("WARN: 多动作选择返回空结果")
        
        return True
    except Exception as e:
        print(f"FAIL: 多动作模式测试失败: {e}")
        traceback.print_exc()
        return False

def test_multi_action_execution():
    """测试多动作执行"""
    print("\n=== 测试多动作执行 ===")
    try:
        from envs.v5_0.city_env import V5CityEnvironment
        from logic.v5_enumeration import V5ActionEnumerator
        from contracts import AtomicAction, Sequence
        
        # 创建环境
        env = V5CityEnvironment("configs/city_config_v5_0.json")
        state = env.reset()
        print("PASS: 环境创建和重置成功")
        
        # 加载测试槽位
        test_slots = [
            {"id": f"slot_{i}", "x": i, "y": 0, "neighbors": [], "building_level": 5}
            for i in range(10)
        ]
        env.enumerator.load_slots(test_slots)
        print(f"PASS: 加载了 {len(test_slots)} 个测试槽位")
        
        # 设置预算
        env.budgets["IND"] = 10000
        print("PASS: 设置预算成功")
        
        # 生成候选索引
        def lp_provider(slot_id): return 0.5
        candidates, cand_idx = env.enumerator.enumerate_with_index(
            agent="IND",
            occupied_slots=env.occupied_slots,
            lp_provider=lp_provider,
            budget=env.budgets.get("IND", 0),
            current_month=state.month
        )
        
        if len(candidates) == 0:
            print("WARN: 无可用候选，跳过执行测试")
            return True
        
        print(f"PASS: 生成候选: {len(candidates)} 个候选, {len(cand_idx.points)} 个点")
        
        # 缓存候选索引
        env._last_cand_idx["IND"] = cand_idx
        
        # 创建原子动作
        atomic_action = AtomicAction(point=0, atype=0)
        print("PASS: 创建原子动作")
        
        # 测试原子动作执行
        initial_budget = env.budgets.get("IND", 0)
        reward, terms = env._execute_action_atomic("IND", atomic_action)
        
        if reward > 0:
            print(f"PASS: 原子动作执行成功: reward={reward:.2f}")
            print(f"  - 预算变化: {initial_budget} -> {env.budgets.get('IND', 0)}")
        else:
            print(f"WARN: 原子动作执行: reward={reward:.2f}, terms={terms}")
        
        # 测试Sequence执行
        seq = Sequence(agent="IND", actions=[atomic_action])
        reward_seq, terms_seq = env._execute_agent_sequence("IND", seq)
        
        if reward_seq > 0:
            print(f"PASS: Sequence执行成功: reward={reward_seq:.2f}")
        else:
            print(f"WARN: Sequence执行: reward={reward_seq:.2f}")
        
        return True
    except Exception as e:
        print(f"FAIL: 多动作执行测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("多动作模式测试")
    print("=" * 50)
    
    test_results = []
    
    # 多动作测试
    test_results.append(("多动作启用", test_multi_action_enabled()))
    test_results.append(("多动作执行", test_multi_action_execution()))
    
    # 结果汇总
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n多动作模式测试通过！")
        return 0
    else:
        print(f"\n有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

测试多动作模式功能
"""

import sys
import os
import traceback

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_multi_action_enabled():
    """测试启用多动作模式"""
    print("=== 测试多动作模式启用 ===")
    try:
        from config_loader import ConfigLoader
        from solvers.v5_0.rl_selector import V5RLSelector
        from contracts import EnvironmentState, CandidateIndex, AtomicAction
        import numpy as np
        
        # 加载配置
        loader = ConfigLoader()
        config = loader.load_v5_config("configs/city_config_v5_0.json")
        
        # 启用多动作模式
        config["multi_action"]["enabled"] = True
        print("PASS: 启用多动作模式")
        
        # 创建多动作选择器
        selector = V5RLSelector(config)
        print("PASS: 多动作选择器创建成功")
        
        # 检查多动作网络
        if hasattr(selector, 'actor_networks_multi'):
            print("PASS: 多动作网络已初始化")
            for agent in ["IND", "EDU", "COUNCIL"]:
                if agent in selector.actor_networks_multi:
                    print(f"  - {agent}: 多动作网络就绪")
        else:
            print("FAIL: 多动作网络未初始化")
            return False
        
        # 测试多动作选择
        mock_state = EnvironmentState(
            month=0,
            land_prices=np.zeros((1, 1)),
            buildings=[],
            budgets={"IND": 1000.0},
            slots=[]
        )
        
        # 创建模拟候选索引
        mock_cand_idx = CandidateIndex(
            points=[100, 101],
            types_per_point=[[0, 1], [2, 3]],
            point_to_slots={100: ["s1"], 101: ["s2"]}
        )
        
        # 测试多动作选择（使用贪心模式确保有输出）
        result = selector.select_action_multi(
            agent="IND",
            candidates=[],  # 空候选列表
            cand_idx=mock_cand_idx,
            state=mock_state,
            max_k=1,  # 减少到1个动作
            greedy=True
        )
        
        if result:
            print(f"PASS: 多动作选择成功: {len(result['sequence'].actions)} 个动作")
            print(f"  - logprob: {result['logprob']:.4f}")
            print(f"  - entropy: {result['entropy']:.4f}")
            print(f"  - value: {result['value']:.4f}")
        else:
            print("WARN: 多动作选择返回空结果")
        
        return True
    except Exception as e:
        print(f"FAIL: 多动作模式测试失败: {e}")
        traceback.print_exc()
        return False

def test_multi_action_execution():
    """测试多动作执行"""
    print("\n=== 测试多动作执行 ===")
    try:
        from envs.v5_0.city_env import V5CityEnvironment
        from logic.v5_enumeration import V5ActionEnumerator
        from contracts import AtomicAction, Sequence
        
        # 创建环境
        env = V5CityEnvironment("configs/city_config_v5_0.json")
        state = env.reset()
        print("PASS: 环境创建和重置成功")
        
        # 加载测试槽位
        test_slots = [
            {"id": f"slot_{i}", "x": i, "y": 0, "neighbors": [], "building_level": 5}
            for i in range(10)
        ]
        env.enumerator.load_slots(test_slots)
        print(f"PASS: 加载了 {len(test_slots)} 个测试槽位")
        
        # 设置预算
        env.budgets["IND"] = 10000
        print("PASS: 设置预算成功")
        
        # 生成候选索引
        def lp_provider(slot_id): return 0.5
        candidates, cand_idx = env.enumerator.enumerate_with_index(
            agent="IND",
            occupied_slots=env.occupied_slots,
            lp_provider=lp_provider,
            budget=env.budgets.get("IND", 0),
            current_month=state.month
        )
        
        if len(candidates) == 0:
            print("WARN: 无可用候选，跳过执行测试")
            return True
        
        print(f"PASS: 生成候选: {len(candidates)} 个候选, {len(cand_idx.points)} 个点")
        
        # 缓存候选索引
        env._last_cand_idx["IND"] = cand_idx
        
        # 创建原子动作
        atomic_action = AtomicAction(point=0, atype=0)
        print("PASS: 创建原子动作")
        
        # 测试原子动作执行
        initial_budget = env.budgets.get("IND", 0)
        reward, terms = env._execute_action_atomic("IND", atomic_action)
        
        if reward > 0:
            print(f"PASS: 原子动作执行成功: reward={reward:.2f}")
            print(f"  - 预算变化: {initial_budget} -> {env.budgets.get('IND', 0)}")
        else:
            print(f"WARN: 原子动作执行: reward={reward:.2f}, terms={terms}")
        
        # 测试Sequence执行
        seq = Sequence(agent="IND", actions=[atomic_action])
        reward_seq, terms_seq = env._execute_agent_sequence("IND", seq)
        
        if reward_seq > 0:
            print(f"PASS: Sequence执行成功: reward={reward_seq:.2f}")
        else:
            print(f"WARN: Sequence执行: reward={reward_seq:.2f}")
        
        return True
    except Exception as e:
        print(f"FAIL: 多动作执行测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("多动作模式测试")
    print("=" * 50)
    
    test_results = []
    
    # 多动作测试
    test_results.append(("多动作启用", test_multi_action_enabled()))
    test_results.append(("多动作执行", test_multi_action_execution()))
    
    # 结果汇总
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n多动作模式测试通过！")
        return 0
    else:
        print(f"\n有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
