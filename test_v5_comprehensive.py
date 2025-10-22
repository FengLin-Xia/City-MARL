#!/usr/bin/env python3
"""
v5.0 整体系统综合测试脚本
测试基础功能、兼容性、多动作机制和端到端集成
"""

import sys
import os
import time
import json
import traceback
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有核心模块导入"""
    print("=== 测试1: 模块导入 ===")
    try:
        from contracts import AtomicAction, CandidateIndex, Sequence, ActionCandidate, EnvironmentState
        from config_loader import ConfigLoader
        from logic.v5_enumeration import V5ActionEnumerator
        from envs.v5_0.city_env import V5CityEnvironment
        from solvers.v5_0.rl_selector import V5RLSelector
        print("✅ 所有核心模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n=== 测试2: 配置文件加载 ===")
    try:
        from config_loader import ConfigLoader
        loader = ConfigLoader()
        config = loader.load_v5_config("configs/city_config_v5_0.json")
        
        print(f"✅ 配置加载成功")
        print(f"  - schema_version: {config.get('schema_version')}")
        print(f"  - multi_action.enabled: {config.get('multi_action', {}).get('enabled', False)}")
        print(f"  - max_actions_per_step: {config.get('multi_action', {}).get('max_actions_per_step', 'N/A')}")
        return config
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None

def test_data_structures():
    """测试数据结构"""
    print("\n=== 测试3: 数据结构 ===")
    try:
        from contracts import AtomicAction, CandidateIndex, Sequence
        
        # 测试 AtomicAction
        aa = AtomicAction(point=1, atype=5, meta={"test": "value"})
        assert aa.point == 1 and aa.atype == 5
        print("✅ AtomicAction 创建成功")
        
        # 测试 CandidateIndex
        cand_idx = CandidateIndex(
            points=[100, 101],
            types_per_point=[[0, 1], [2, 3]],
            point_to_slots={100: ["s1"], 101: ["s2"]}
        )
        assert len(cand_idx.points) == 2
        print("✅ CandidateIndex 创建成功")
        
        # 测试 Sequence 兼容性
        seq_old = Sequence(agent="IND", actions=[0, 1])
        assert len(seq_old.actions) == 2
        assert isinstance(seq_old.actions[0], AtomicAction)
        assert seq_old.actions[0].meta.get('legacy_id') == 0
        print("✅ Sequence 兼容层工作正常")
        
        # 测试 get_legacy_ids
        legacy_ids = seq_old.get_legacy_ids()
        assert legacy_ids == [0, 1]
        print("✅ get_legacy_ids 方法正常")
        
        return True
    except Exception as e:
        print(f"❌ 数据结构测试失败: {e}")
        traceback.print_exc()
        return False

def test_enumerator(config):
    """测试枚举器"""
    print("\n=== 测试4: 枚举器功能 ===")
    try:
        from logic.v5_enumeration import V5ActionEnumerator
        
        # 创建枚举器
        enumerator = V5ActionEnumerator(config)
        print("✅ 枚举器创建成功")
        
        # 加载测试槽位
        test_slots = [
            {"id": f"slot_{i}", "x": i, "y": 0, "neighbors": [], "building_level": 5}
            for i in range(10)
        ]
        enumerator.load_slots(test_slots)
        print(f"✅ 加载了 {len(test_slots)} 个测试槽位")
        
        # 测试旧版枚举
        candidates = enumerator.enumerate_actions(
            agent="IND",
            occupied_slots=set(),
            lp_provider=lambda x: 0.5,
            budget=5000.0,
            current_month=0
        )
        print(f"✅ 旧版枚举: {len(candidates)} 个候选")
        
        # 测试新版枚举
        candidates_new, cand_idx = enumerator.enumerate_with_index(
            agent="IND",
            occupied_slots=set(),
            lp_provider=lambda x: 0.5,
            budget=5000.0,
            current_month=0
        )
        print(f"✅ 新版枚举: {len(candidates_new)} 个候选, {len(cand_idx.points)} 个点")
        
        return enumerator
    except Exception as e:
        print(f"❌ 枚举器测试失败: {e}")
        traceback.print_exc()
        return None

def test_environment(config):
    """测试环境"""
    print("\n=== 测试5: 环境功能 ===")
    try:
        from envs.v5_0.city_env import V5CityEnvironment
        from contracts import Sequence
        
        # 创建环境
        env = V5CityEnvironment("configs/city_config_v5_0.json")
        print("✅ 环境创建成功")
        
        # 重置环境
        initial_state = env.reset()
        print(f"✅ 环境重置成功: month={initial_state.month}")
        
        # 测试单动作执行（旧版模式）
        seq = Sequence(agent="IND", actions=[3])
        print(f"✅ 创建Sequence: {len(seq.actions)} 个动作")
        
        # 验证兼容层
        assert len(seq.actions) == 1
        assert hasattr(seq.actions[0], 'point')
        assert seq.actions[0].meta.get('legacy_id') == 3
        print("✅ 兼容层工作正常")
        
        return env
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        traceback.print_exc()
        return None

def test_selector(config):
    """测试选择器"""
    print("\n=== 测试6: 选择器功能 ===")
    try:
        from solvers.v5_0.rl_selector import V5RLSelector
        from contracts import EnvironmentState
        import numpy as np
        
        # 创建选择器
        selector = V5RLSelector(config)
        print("✅ 选择器创建成功")
        
        # 检查多动作配置
        multi_enabled = config.get("multi_action", {}).get("enabled", False)
        print(f"  - 多动作模式: {'启用' if multi_enabled else '禁用'}")
        
        if multi_enabled and hasattr(selector, 'actor_networks_multi'):
            print("✅ 多动作网络已初始化")
        elif not multi_enabled:
            print("✅ 单动作模式（预期）")
        else:
            print("⚠️  多动作配置异常")
        
        # 测试状态编码
        mock_state = EnvironmentState(
            month=0,
            land_prices=np.zeros((1, 1)),
            buildings=[],
            budgets={"IND": 1000.0, "EDU": 1000.0, "COUNCIL": 1000.0},
            slots=[]
        )
        
        encoded = selector._encode_state(mock_state)
        print(f"✅ 状态编码成功: shape={encoded.shape}")
        
        return selector
    except Exception as e:
        print(f"❌ 选择器测试失败: {e}")
        traceback.print_exc()
        return None

def test_multi_action_mode(config):
    """测试多动作模式"""
    print("\n=== 测试7: 多动作模式 ===")
    try:
        # 临时启用多动作
        config["multi_action"]["enabled"] = True
        
        from solvers.v5_0.rl_selector import V5RLSelector
        from contracts import EnvironmentState, CandidateIndex, AtomicAction
        import numpy as np
        import torch
        
        # 创建多动作选择器
        selector = V5RLSelector(config)
        print("✅ 多动作选择器创建成功")
        
        # 检查多动作网络
        if hasattr(selector, 'actor_networks_multi'):
            print("✅ 多动作网络已初始化")
            for agent in ["IND", "EDU", "COUNCIL"]:
                if agent in selector.actor_networks_multi:
                    print(f"  - {agent}: 多动作网络就绪")
        else:
            print("❌ 多动作网络未初始化")
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
        
        # 测试多动作选择
        result = selector.select_action_multi(
            agent="IND",
            candidates=[],  # 空候选列表
            cand_idx=mock_cand_idx,
            state=mock_state,
            max_k=2,
            greedy=True
        )
        
        if result:
            print(f"✅ 多动作选择成功: {len(result['sequence'].actions)} 个动作")
            print(f"  - logprob: {result['logprob']:.4f}")
            print(f"  - entropy: {result['entropy']:.4f}")
            print(f"  - value: {result['value']:.4f}")
        else:
            print("⚠️  多动作选择返回空结果")
        
        return True
    except Exception as e:
        print(f"❌ 多动作模式测试失败: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """端到端集成测试"""
    print("\n=== 测试8: 端到端集成 ===")
    try:
        from envs.v5_0.city_env import V5CityEnvironment
        from logic.v5_enumeration import V5ActionEnumerator
        from solvers.v5_0.rl_selector import V5RLSelector
        
        # 1. 环境初始化
        env = V5CityEnvironment("configs/city_config_v5_0.json")
        state = env.reset()
        print(f"✅ 环境初始化: month={state.month}")
        
        # 2. 枚举器
        enumerator = V5ActionEnumerator(env.config)
        test_slots = [
            {"id": f"slot_{i}", "x": i, "y": 0, "neighbors": [], "building_level": 5}
            for i in range(20)
        ]
        enumerator.load_slots(test_slots)
        print(f"✅ 枚举器初始化: {len(test_slots)} 个槽位")
        
        # 3. 选择器
        selector = V5RLSelector(env.config)
        print("✅ 选择器初始化")
        
        # 4. 单步执行测试
        success_count = 0
        for agent in ["IND", "EDU", "COUNCIL"]:
            try:
                # 枚举候选
                candidates, cand_idx = enumerator.enumerate_with_index(
                    agent=agent,
                    occupied_slots=env.occupied_slots,
                    lp_provider=lambda x: 0.5,
                    budget=env.budgets.get(agent, 1000),
                    current_month=state.month
                )
                
                if len(candidates) == 0:
                    print(f"⚠️  {agent}: 无可用候选")
                    continue
                
                print(f"✅ {agent}: {len(candidates)} 个候选, {len(cand_idx.points)} 个点")
                
                # 选择动作（单动作模式）
                sel = selector.select_action(agent, candidates, state, greedy=True)
                if sel:
                    print(f"✅ {agent} 选择: {len(sel['sequence'].actions)} 个动作")
                    success_count += 1
                else:
                    print(f"❌ {agent} 选择失败")
            except Exception as e:
                print(f"❌ {agent} 处理失败: {e}")
        
        print(f"✅ 集成测试完成: {success_count}/3 个智能体成功")
        return success_count > 0
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("v5.0 整体系统综合测试")
    print("=" * 60)
    
    test_results = []
    
    # 基础测试
    test_results.append(("模块导入", test_imports()))
    
    config = test_config_loading()
    test_results.append(("配置加载", config is not None))
    
    test_results.append(("数据结构", test_data_structures()))
    
    if config:
        test_results.append(("枚举器", test_enumerator(config) is not None))
        test_results.append(("环境", test_environment(config) is not None))
        test_results.append(("选择器", test_selector(config) is not None))
        test_results.append(("多动作模式", test_multi_action_mode(config)))
        test_results.append(("端到端集成", test_integration()))
    
    # 结果汇总
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n所有测试通过！v5.0 系统运行正常！")
        return 0
    else:
        print(f"\n有 {total - passed} 个测试失败，请检查上述错误信息")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

v5.0 整体系统综合测试脚本
测试基础功能、兼容性、多动作机制和端到端集成
"""

import sys
import os
import time
import json
import traceback
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有核心模块导入"""
    print("=== 测试1: 模块导入 ===")
    try:
        from contracts import AtomicAction, CandidateIndex, Sequence, ActionCandidate, EnvironmentState
        from config_loader import ConfigLoader
        from logic.v5_enumeration import V5ActionEnumerator
        from envs.v5_0.city_env import V5CityEnvironment
        from solvers.v5_0.rl_selector import V5RLSelector
        print("✅ 所有核心模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n=== 测试2: 配置文件加载 ===")
    try:
        from config_loader import ConfigLoader
        loader = ConfigLoader()
        config = loader.load_v5_config("configs/city_config_v5_0.json")
        
        print(f"✅ 配置加载成功")
        print(f"  - schema_version: {config.get('schema_version')}")
        print(f"  - multi_action.enabled: {config.get('multi_action', {}).get('enabled', False)}")
        print(f"  - max_actions_per_step: {config.get('multi_action', {}).get('max_actions_per_step', 'N/A')}")
        return config
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None

def test_data_structures():
    """测试数据结构"""
    print("\n=== 测试3: 数据结构 ===")
    try:
        from contracts import AtomicAction, CandidateIndex, Sequence
        
        # 测试 AtomicAction
        aa = AtomicAction(point=1, atype=5, meta={"test": "value"})
        assert aa.point == 1 and aa.atype == 5
        print("✅ AtomicAction 创建成功")
        
        # 测试 CandidateIndex
        cand_idx = CandidateIndex(
            points=[100, 101],
            types_per_point=[[0, 1], [2, 3]],
            point_to_slots={100: ["s1"], 101: ["s2"]}
        )
        assert len(cand_idx.points) == 2
        print("✅ CandidateIndex 创建成功")
        
        # 测试 Sequence 兼容性
        seq_old = Sequence(agent="IND", actions=[0, 1])
        assert len(seq_old.actions) == 2
        assert isinstance(seq_old.actions[0], AtomicAction)
        assert seq_old.actions[0].meta.get('legacy_id') == 0
        print("✅ Sequence 兼容层工作正常")
        
        # 测试 get_legacy_ids
        legacy_ids = seq_old.get_legacy_ids()
        assert legacy_ids == [0, 1]
        print("✅ get_legacy_ids 方法正常")
        
        return True
    except Exception as e:
        print(f"❌ 数据结构测试失败: {e}")
        traceback.print_exc()
        return False

def test_enumerator(config):
    """测试枚举器"""
    print("\n=== 测试4: 枚举器功能 ===")
    try:
        from logic.v5_enumeration import V5ActionEnumerator
        
        # 创建枚举器
        enumerator = V5ActionEnumerator(config)
        print("✅ 枚举器创建成功")
        
        # 加载测试槽位
        test_slots = [
            {"id": f"slot_{i}", "x": i, "y": 0, "neighbors": [], "building_level": 5}
            for i in range(10)
        ]
        enumerator.load_slots(test_slots)
        print(f"✅ 加载了 {len(test_slots)} 个测试槽位")
        
        # 测试旧版枚举
        candidates = enumerator.enumerate_actions(
            agent="IND",
            occupied_slots=set(),
            lp_provider=lambda x: 0.5,
            budget=5000.0,
            current_month=0
        )
        print(f"✅ 旧版枚举: {len(candidates)} 个候选")
        
        # 测试新版枚举
        candidates_new, cand_idx = enumerator.enumerate_with_index(
            agent="IND",
            occupied_slots=set(),
            lp_provider=lambda x: 0.5,
            budget=5000.0,
            current_month=0
        )
        print(f"✅ 新版枚举: {len(candidates_new)} 个候选, {len(cand_idx.points)} 个点")
        
        return enumerator
    except Exception as e:
        print(f"❌ 枚举器测试失败: {e}")
        traceback.print_exc()
        return None

def test_environment(config):
    """测试环境"""
    print("\n=== 测试5: 环境功能 ===")
    try:
        from envs.v5_0.city_env import V5CityEnvironment
        from contracts import Sequence
        
        # 创建环境
        env = V5CityEnvironment("configs/city_config_v5_0.json")
        print("✅ 环境创建成功")
        
        # 重置环境
        initial_state = env.reset()
        print(f"✅ 环境重置成功: month={initial_state.month}")
        
        # 测试单动作执行（旧版模式）
        seq = Sequence(agent="IND", actions=[3])
        print(f"✅ 创建Sequence: {len(seq.actions)} 个动作")
        
        # 验证兼容层
        assert len(seq.actions) == 1
        assert hasattr(seq.actions[0], 'point')
        assert seq.actions[0].meta.get('legacy_id') == 3
        print("✅ 兼容层工作正常")
        
        return env
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        traceback.print_exc()
        return None

def test_selector(config):
    """测试选择器"""
    print("\n=== 测试6: 选择器功能 ===")
    try:
        from solvers.v5_0.rl_selector import V5RLSelector
        from contracts import EnvironmentState
        import numpy as np
        
        # 创建选择器
        selector = V5RLSelector(config)
        print("✅ 选择器创建成功")
        
        # 检查多动作配置
        multi_enabled = config.get("multi_action", {}).get("enabled", False)
        print(f"  - 多动作模式: {'启用' if multi_enabled else '禁用'}")
        
        if multi_enabled and hasattr(selector, 'actor_networks_multi'):
            print("✅ 多动作网络已初始化")
        elif not multi_enabled:
            print("✅ 单动作模式（预期）")
        else:
            print("⚠️  多动作配置异常")
        
        # 测试状态编码
        mock_state = EnvironmentState(
            month=0,
            land_prices=np.zeros((1, 1)),
            buildings=[],
            budgets={"IND": 1000.0, "EDU": 1000.0, "COUNCIL": 1000.0},
            slots=[]
        )
        
        encoded = selector._encode_state(mock_state)
        print(f"✅ 状态编码成功: shape={encoded.shape}")
        
        return selector
    except Exception as e:
        print(f"❌ 选择器测试失败: {e}")
        traceback.print_exc()
        return None

def test_multi_action_mode(config):
    """测试多动作模式"""
    print("\n=== 测试7: 多动作模式 ===")
    try:
        # 临时启用多动作
        config["multi_action"]["enabled"] = True
        
        from solvers.v5_0.rl_selector import V5RLSelector
        from contracts import EnvironmentState, CandidateIndex, AtomicAction
        import numpy as np
        import torch
        
        # 创建多动作选择器
        selector = V5RLSelector(config)
        print("✅ 多动作选择器创建成功")
        
        # 检查多动作网络
        if hasattr(selector, 'actor_networks_multi'):
            print("✅ 多动作网络已初始化")
            for agent in ["IND", "EDU", "COUNCIL"]:
                if agent in selector.actor_networks_multi:
                    print(f"  - {agent}: 多动作网络就绪")
        else:
            print("❌ 多动作网络未初始化")
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
        
        # 测试多动作选择
        result = selector.select_action_multi(
            agent="IND",
            candidates=[],  # 空候选列表
            cand_idx=mock_cand_idx,
            state=mock_state,
            max_k=2,
            greedy=True
        )
        
        if result:
            print(f"✅ 多动作选择成功: {len(result['sequence'].actions)} 个动作")
            print(f"  - logprob: {result['logprob']:.4f}")
            print(f"  - entropy: {result['entropy']:.4f}")
            print(f"  - value: {result['value']:.4f}")
        else:
            print("⚠️  多动作选择返回空结果")
        
        return True
    except Exception as e:
        print(f"❌ 多动作模式测试失败: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """端到端集成测试"""
    print("\n=== 测试8: 端到端集成 ===")
    try:
        from envs.v5_0.city_env import V5CityEnvironment
        from logic.v5_enumeration import V5ActionEnumerator
        from solvers.v5_0.rl_selector import V5RLSelector
        
        # 1. 环境初始化
        env = V5CityEnvironment("configs/city_config_v5_0.json")
        state = env.reset()
        print(f"✅ 环境初始化: month={state.month}")
        
        # 2. 枚举器
        enumerator = V5ActionEnumerator(env.config)
        test_slots = [
            {"id": f"slot_{i}", "x": i, "y": 0, "neighbors": [], "building_level": 5}
            for i in range(20)
        ]
        enumerator.load_slots(test_slots)
        print(f"✅ 枚举器初始化: {len(test_slots)} 个槽位")
        
        # 3. 选择器
        selector = V5RLSelector(env.config)
        print("✅ 选择器初始化")
        
        # 4. 单步执行测试
        success_count = 0
        for agent in ["IND", "EDU", "COUNCIL"]:
            try:
                # 枚举候选
                candidates, cand_idx = enumerator.enumerate_with_index(
                    agent=agent,
                    occupied_slots=env.occupied_slots,
                    lp_provider=lambda x: 0.5,
                    budget=env.budgets.get(agent, 1000),
                    current_month=state.month
                )
                
                if len(candidates) == 0:
                    print(f"⚠️  {agent}: 无可用候选")
                    continue
                
                print(f"✅ {agent}: {len(candidates)} 个候选, {len(cand_idx.points)} 个点")
                
                # 选择动作（单动作模式）
                sel = selector.select_action(agent, candidates, state, greedy=True)
                if sel:
                    print(f"✅ {agent} 选择: {len(sel['sequence'].actions)} 个动作")
                    success_count += 1
                else:
                    print(f"❌ {agent} 选择失败")
            except Exception as e:
                print(f"❌ {agent} 处理失败: {e}")
        
        print(f"✅ 集成测试完成: {success_count}/3 个智能体成功")
        return success_count > 0
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("v5.0 整体系统综合测试")
    print("=" * 60)
    
    test_results = []
    
    # 基础测试
    test_results.append(("模块导入", test_imports()))
    
    config = test_config_loading()
    test_results.append(("配置加载", config is not None))
    
    test_results.append(("数据结构", test_data_structures()))
    
    if config:
        test_results.append(("枚举器", test_enumerator(config) is not None))
        test_results.append(("环境", test_environment(config) is not None))
        test_results.append(("选择器", test_selector(config) is not None))
        test_results.append(("多动作模式", test_multi_action_mode(config)))
        test_results.append(("端到端集成", test_integration()))
    
    # 结果汇总
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n所有测试通过！v5.0 系统运行正常！")
        return 0
    else:
        print(f"\n有 {total - passed} 个测试失败，请检查上述错误信息")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
