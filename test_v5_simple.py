#!/usr/bin/env python3
"""
v5.0 系统简化测试脚本
"""

import sys
import os
import traceback

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试模块导入"""
    print("=== 测试1: 模块导入 ===")
    try:
        from contracts import AtomicAction, CandidateIndex, Sequence
        from config_loader import ConfigLoader
        from logic.v5_enumeration import V5ActionEnumerator
        from envs.v5_0.city_env import V5CityEnvironment
        from solvers.v5_0.rl_selector import V5RLSelector
        print("PASS: 所有核心模块导入成功")
        return True
    except Exception as e:
        print(f"FAIL: 模块导入失败: {e}")
        return False

def test_config():
    """测试配置加载"""
    print("\n=== 测试2: 配置加载 ===")
    try:
        from config_loader import ConfigLoader
        loader = ConfigLoader()
        config = loader.load_v5_config("configs/city_config_v5_0.json")
        
        print("PASS: 配置加载成功")
        print(f"  - schema_version: {config.get('schema_version')}")
        print(f"  - multi_action.enabled: {config.get('multi_action', {}).get('enabled', False)}")
        return config
    except Exception as e:
        print(f"FAIL: 配置加载失败: {e}")
        return None

def test_data_structures():
    """测试数据结构"""
    print("\n=== 测试3: 数据结构 ===")
    try:
        from contracts import AtomicAction, CandidateIndex, Sequence
        
        # 测试 AtomicAction
        aa = AtomicAction(point=1, atype=5, meta={"test": "value"})
        assert aa.point == 1 and aa.atype == 5
        print("PASS: AtomicAction 创建成功")
        
        # 测试 CandidateIndex
        cand_idx = CandidateIndex(
            points=[100, 101],
            types_per_point=[[0, 1], [2, 3]],
            point_to_slots={100: ["s1"], 101: ["s2"]}
        )
        assert len(cand_idx.points) == 2
        print("PASS: CandidateIndex 创建成功")
        
        # 测试 Sequence 兼容性
        seq_old = Sequence(agent="IND", actions=[0, 1])
        assert len(seq_old.actions) == 2
        assert isinstance(seq_old.actions[0], AtomicAction)
        assert seq_old.actions[0].meta.get('legacy_id') == 0
        print("PASS: Sequence 兼容层工作正常")
        
        return True
    except Exception as e:
        print(f"FAIL: 数据结构测试失败: {e}")
        traceback.print_exc()
        return False

def test_environment(config):
    """测试环境"""
    print("\n=== 测试4: 环境功能 ===")
    try:
        from envs.v5_0.city_env import V5CityEnvironment
        from contracts import Sequence
        
        # 创建环境
        env = V5CityEnvironment("configs/city_config_v5_0.json")
        print("PASS: 环境创建成功")
        
        # 重置环境
        initial_state = env.reset()
        print(f"PASS: 环境重置成功: month={initial_state.month}")
        
        # 测试单动作执行
        seq = Sequence(agent="IND", actions=[3])
        print(f"PASS: 创建Sequence: {len(seq.actions)} 个动作")
        
        return env
    except Exception as e:
        print(f"FAIL: 环境测试失败: {e}")
        traceback.print_exc()
        return None

def test_selector(config):
    """测试选择器"""
    print("\n=== 测试5: 选择器功能 ===")
    try:
        from solvers.v5_0.rl_selector import V5RLSelector
        from contracts import EnvironmentState
        import numpy as np
        
        # 创建选择器
        selector = V5RLSelector(config)
        print("PASS: 选择器创建成功")
        
        # 检查多动作配置
        multi_enabled = config.get("multi_action", {}).get("enabled", False)
        print(f"  - 多动作模式: {'启用' if multi_enabled else '禁用'}")
        
        return selector
    except Exception as e:
        print(f"FAIL: 选择器测试失败: {e}")
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    print("v5.0 系统简化测试")
    print("=" * 50)
    
    test_results = []
    
    # 基础测试
    test_results.append(("模块导入", test_imports()))
    
    config = test_config()
    test_results.append(("配置加载", config is not None))
    
    test_results.append(("数据结构", test_data_structures()))
    
    if config:
        test_results.append(("环境", test_environment(config) is not None))
        test_results.append(("选择器", test_selector(config) is not None))
    
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
        print("\n所有测试通过！v5.0 系统运行正常！")
        return 0
    else:
        print(f"\n有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
"""
v5.0 系统简化测试脚本
"""

import sys
import os
import traceback

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试模块导入"""
    print("=== 测试1: 模块导入 ===")
    try:
        from contracts import AtomicAction, CandidateIndex, Sequence
        from config_loader import ConfigLoader
        from logic.v5_enumeration import V5ActionEnumerator
        from envs.v5_0.city_env import V5CityEnvironment
        from solvers.v5_0.rl_selector import V5RLSelector
        print("PASS: 所有核心模块导入成功")
        return True
    except Exception as e:
        print(f"FAIL: 模块导入失败: {e}")
        return False

def test_config():
    """测试配置加载"""
    print("\n=== 测试2: 配置加载 ===")
    try:
        from config_loader import ConfigLoader
        loader = ConfigLoader()
        config = loader.load_v5_config("configs/city_config_v5_0.json")
        
        print("PASS: 配置加载成功")
        print(f"  - schema_version: {config.get('schema_version')}")
        print(f"  - multi_action.enabled: {config.get('multi_action', {}).get('enabled', False)}")
        return config
    except Exception as e:
        print(f"FAIL: 配置加载失败: {e}")
        return None

def test_data_structures():
    """测试数据结构"""
    print("\n=== 测试3: 数据结构 ===")
    try:
        from contracts import AtomicAction, CandidateIndex, Sequence
        
        # 测试 AtomicAction
        aa = AtomicAction(point=1, atype=5, meta={"test": "value"})
        assert aa.point == 1 and aa.atype == 5
        print("PASS: AtomicAction 创建成功")
        
        # 测试 CandidateIndex
        cand_idx = CandidateIndex(
            points=[100, 101],
            types_per_point=[[0, 1], [2, 3]],
            point_to_slots={100: ["s1"], 101: ["s2"]}
        )
        assert len(cand_idx.points) == 2
        print("PASS: CandidateIndex 创建成功")
        
        # 测试 Sequence 兼容性
        seq_old = Sequence(agent="IND", actions=[0, 1])
        assert len(seq_old.actions) == 2
        assert isinstance(seq_old.actions[0], AtomicAction)
        assert seq_old.actions[0].meta.get('legacy_id') == 0
        print("PASS: Sequence 兼容层工作正常")
        
        return True
    except Exception as e:
        print(f"FAIL: 数据结构测试失败: {e}")
        traceback.print_exc()
        return False

def test_environment(config):
    """测试环境"""
    print("\n=== 测试4: 环境功能 ===")
    try:
        from envs.v5_0.city_env import V5CityEnvironment
        from contracts import Sequence
        
        # 创建环境
        env = V5CityEnvironment("configs/city_config_v5_0.json")
        print("PASS: 环境创建成功")
        
        # 重置环境
        initial_state = env.reset()
        print(f"PASS: 环境重置成功: month={initial_state.month}")
        
        # 测试单动作执行
        seq = Sequence(agent="IND", actions=[3])
        print(f"PASS: 创建Sequence: {len(seq.actions)} 个动作")
        
        return env
    except Exception as e:
        print(f"FAIL: 环境测试失败: {e}")
        traceback.print_exc()
        return None

def test_selector(config):
    """测试选择器"""
    print("\n=== 测试5: 选择器功能 ===")
    try:
        from solvers.v5_0.rl_selector import V5RLSelector
        from contracts import EnvironmentState
        import numpy as np
        
        # 创建选择器
        selector = V5RLSelector(config)
        print("PASS: 选择器创建成功")
        
        # 检查多动作配置
        multi_enabled = config.get("multi_action", {}).get("enabled", False)
        print(f"  - 多动作模式: {'启用' if multi_enabled else '禁用'}")
        
        return selector
    except Exception as e:
        print(f"FAIL: 选择器测试失败: {e}")
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    print("v5.0 系统简化测试")
    print("=" * 50)
    
    test_results = []
    
    # 基础测试
    test_results.append(("模块导入", test_imports()))
    
    config = test_config()
    test_results.append(("配置加载", config is not None))
    
    test_results.append(("数据结构", test_data_structures()))
    
    if config:
        test_results.append(("环境", test_environment(config) is not None))
        test_results.append(("选择器", test_selector(config) is not None))
    
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
        print("\n所有测试通过！v5.0 系统运行正常！")
        return 0
    else:
        print(f"\n有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
