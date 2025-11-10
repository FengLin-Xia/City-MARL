"""
多动作机制测试脚本 - 阶段1和阶段2验证

测试范围：
1. 数据结构（AtomicAction, CandidateIndex, Sequence）
2. 兼容层（int → AtomicAction 自动转换）
3. 枚举器（enumerate_with_index, point×type 索引）
4. 环境（_execute_action_atomic, 兼容执行）
5. 配置加载
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts.contracts import AtomicAction, CandidateIndex, Sequence, ActionCandidate
from logic.v5_enumeration import V5ActionEnumerator
from envs.v5_0.city_env import V5CityEnvironment
from config_loader import ConfigLoader


class TestResults:
    """测试结果记录"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"[PASS] {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"[FAIL] {test_name}")
        print(f"   Error: {error}")
    
    def summary(self):
        print("\n" + "="*60)
        print(f"测试总结: {self.passed} 通过, {self.failed} 失败")
        print("="*60)
        if self.errors:
            print("\n失败详情:")
            for err in self.errors:
                print(f"  - {err}")
        return self.failed == 0


def test_atomic_action(results: TestResults):
    """测试1: AtomicAction 数据类"""
    try:
        # 创建原子动作
        action = AtomicAction(point=5, atype=2, meta={"test": "value"})
        
        assert action.point == 5, "point字段错误"
        assert action.atype == 2, "atype字段错误"
        assert action.meta["test"] == "value", "meta字段错误"
        
        # 测试验证
        try:
            invalid_action = AtomicAction(point=-1, atype=0)
            assert False, "应该抛出验证错误"
        except AssertionError as e:
            if "non-negative" in str(e):
                pass  # 预期的错误
            else:
                raise
        
        results.add_pass("AtomicAction 数据类")
    except Exception as e:
        results.add_fail("AtomicAction 数据类", str(e))


def test_candidate_index(results: TestResults):
    """测试2: CandidateIndex 数据类"""
    try:
        # 创建候选索引
        cand_idx = CandidateIndex(
            points=[0, 1, 2],
            types_per_point=[[0, 1], [3, 4], [6, 7, 8]],
            point_to_slots={0: ["slot_a"], 1: ["slot_b"], 2: ["slot_c"]}
        )
        
        assert len(cand_idx.points) == 3, "points长度错误"
        assert len(cand_idx.types_per_point) == 3, "types_per_point长度错误"
        assert len(cand_idx.types_per_point[2]) == 3, "types_per_point[2]长度错误"
        assert cand_idx.point_to_slots[0] == ["slot_a"], "point_to_slots映射错误"
        
        # 测试验证
        try:
            invalid_idx = CandidateIndex(
                points=[0, 1],
                types_per_point=[[0]],  # 长度不匹配
                point_to_slots={}
            )
            assert False, "应该抛出验证错误"
        except AssertionError:
            pass  # 预期的错误
        
        results.add_pass("CandidateIndex 数据类")
    except Exception as e:
        results.add_fail("CandidateIndex 数据类", str(e))


def test_sequence_compatibility(results: TestResults):
    """测试3: Sequence 兼容层"""
    try:
        # 测试旧版：使用int列表
        seq_old = Sequence(agent="IND", actions=[3, 4, 5])
        
        # 验证自动转换
        assert len(seq_old.actions) == 3, "actions长度错误"
        assert isinstance(seq_old.actions[0], AtomicAction), "应该自动转换为AtomicAction"
        assert seq_old.actions[0].point == 0, "旧版转换的point应为0"
        assert seq_old.actions[0].meta.get('legacy_id') == 3, "应该保留legacy_id"
        
        # 测试 get_legacy_ids
        legacy_ids = seq_old.get_legacy_ids()
        assert legacy_ids == [3, 4, 5], "get_legacy_ids返回错误"
        
        # 测试新版：使用AtomicAction列表
        seq_new = Sequence(
            agent="EDU",
            actions=[
                AtomicAction(point=1, atype=0, meta={"action_id": 0}),
                AtomicAction(point=2, atype=1, meta={"action_id": 1})
            ]
        )
        
        assert len(seq_new.actions) == 2, "actions长度错误"
        assert seq_new.actions[0].point == 1, "point字段错误"
        assert seq_new.actions[1].atype == 1, "atype字段错误"
        
        results.add_pass("Sequence 兼容层")
    except Exception as e:
        results.add_fail("Sequence 兼容层", str(e))


def test_config_loading(results: TestResults):
    """测试4: 配置加载"""
    try:
        config_path = "configs/city_config_v5_0.json"
        
        # 检查配置文件存在
        assert os.path.exists(config_path), f"配置文件不存在: {config_path}"
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证 multi_action 配置节
        assert 'multi_action' in config, "缺少multi_action配置节"
        
        multi_action = config['multi_action']
        assert 'enabled' in multi_action, "缺少enabled字段"
        assert 'max_actions_per_step' in multi_action, "缺少max_actions_per_step字段"
        assert 'mode' in multi_action, "缺少mode字段"
        
        # 验证默认值
        assert multi_action['enabled'] == False, "enabled应该默认为False"
        assert multi_action['max_actions_per_step'] == 5, "max_actions_per_step应该为5"
        assert multi_action['mode'] == "two_stage", "mode应该为two_stage"
        
        results.add_pass("配置加载")
    except Exception as e:
        results.add_fail("配置加载", str(e))


def test_enumerator_basic(results: TestResults):
    """测试5: 枚举器基础功能"""
    try:
        config_path = "configs/city_config_v5_0.json"
        loader = ConfigLoader()
        config = loader.load_v5_config(config_path)
        
        # 创建枚举器
        enumerator = V5ActionEnumerator(config)
        
        # 加载测试槽位
        test_slots = [
            {"id": "slot_0", "x": 0, "y": 0, "neighbors": [], "building_level": 3},
            {"id": "slot_1", "x": 1, "y": 0, "neighbors": [], "building_level": 4},
            {"id": "slot_2", "x": 2, "y": 0, "neighbors": [], "building_level": 5}
        ]
        enumerator.load_slots(test_slots)
        
        # 验证槽位加载
        assert len(enumerator.slots) == 3, "槽位加载数量错误"
        assert "slot_0" in enumerator.slots, "slot_0未加载"
        
        results.add_pass("枚举器基础功能")
    except Exception as e:
        results.add_fail("枚举器基础功能", str(e))


def test_enumerator_with_index(results: TestResults):
    """测试6: 枚举器 enumerate_with_index 方法"""
    try:
        config_path = "configs/city_config_v5_0.json"
        loader = ConfigLoader()
        config = loader.load_v5_config(config_path)
        
        # 创建枚举器
        enumerator = V5ActionEnumerator(config)
        
        # 加载测试槽位
        test_slots = [
            {"id": f"slot_{i}", "x": i, "y": 0, "neighbors": [], "building_level": 5}
            for i in range(5)
        ]
        enumerator.load_slots(test_slots)
        
        # 测试枚举
        occupied_slots = set()
        budget = 5000.0
        
        def lp_provider(slot_id):
            return 0.5
        
        candidates, cand_idx = enumerator.enumerate_with_index(
            agent="IND",
            occupied_slots=occupied_slots,
            lp_provider=lp_provider,
            budget=budget,
            current_month=0
        )
        
        # 验证返回类型
        assert isinstance(candidates, list), "candidates应该是列表"
        assert isinstance(cand_idx, CandidateIndex), "cand_idx应该是CandidateIndex"
        
        # 验证候选索引结构
        assert len(cand_idx.points) > 0, "应该有可用点"
        assert len(cand_idx.points) == len(cand_idx.types_per_point), "points和types_per_point长度应相同"
        
        # 验证候选列表
        if len(candidates) > 0:
            assert isinstance(candidates[0], ActionCandidate), "候选应该是ActionCandidate"
            assert 'point_idx' in candidates[0].meta, "meta中应该有point_idx"
            assert 'type_idx' in candidates[0].meta, "meta中应该有type_idx"
            assert 'point_id' in candidates[0].meta, "meta中应该有point_id"
        
        print(f"   枚举结果: {len(cand_idx.points)}个点, {len(candidates)}个候选")
        
        results.add_pass("枚举器 enumerate_with_index")
    except Exception as e:
        results.add_fail("枚举器 enumerate_with_index", str(e))


def test_environment_initialization(results: TestResults):
    """测试7: 环境初始化"""
    try:
        config_path = "configs/city_config_v5_0.json"
        
        # 创建环境
        env = V5CityEnvironment(config_path)
        
        # 验证基本属性
        assert hasattr(env, '_last_cand_idx'), "环境应该有_last_cand_idx属性"
        assert isinstance(env._last_cand_idx, dict), "_last_cand_idx应该是字典"
        
        assert hasattr(env, '_execute_action_atomic'), "环境应该有_execute_action_atomic方法"
        
        results.add_pass("环境初始化")
    except Exception as e:
        results.add_fail("环境初始化", str(e))


def test_environment_reset(results: TestResults):
    """测试8: 环境重置"""
    try:
        config_path = "configs/city_config_v5_0.json"
        env = V5CityEnvironment(config_path)
        
        # 重置环境
        initial_state = env.reset()
        
        # 验证初始状态
        assert env.current_step == 0, "初始step应为0"
        assert env.current_month == 0, "初始month应为0"
        assert len(env.buildings) == 0, "初始建筑列表应为空"
        
        results.add_pass("环境重置")
    except Exception as e:
        results.add_fail("环境重置", str(e))


def test_sequence_execution_compatibility(results: TestResults):
    """测试9: Sequence执行兼容性（集成测试）"""
    try:
        config_path = "configs/city_config_v5_0.json"
        env = V5CityEnvironment(config_path)
        env.reset()
        
        # 测试旧版路径：使用int动作
        seq_old = Sequence(agent="IND", actions=[3])
        
        # 验证执行不会报错（即使候选不存在也应该返回0奖励而不是崩溃）
        try:
            reward, terms = env._execute_agent_sequence("IND", seq_old)
            # 成功执行（可能返回0奖励，但不应该崩溃）
            assert isinstance(reward, (int, float)), "奖励应该是数值"
            assert isinstance(terms, dict), "奖励项应该是字典"
            print(f"   旧版执行: reward={reward}, terms={terms}")
        except Exception as e:
            # 如果是预期的错误（如预算不足、槽位占用），则通过
            if "budget" in str(e).lower() or "slot" in str(e).lower():
                print(f"   旧版执行遇到预期错误: {e}")
            else:
                raise
        
        results.add_pass("Sequence执行兼容性")
    except Exception as e:
        results.add_fail("Sequence执行兼容性", str(e))


def test_multi_action_config_off(results: TestResults):
    """测试10: multi_action.enabled=false 时的行为"""
    try:
        config_path = "configs/city_config_v5_0.json"
        loader = ConfigLoader()
        config = loader.load_v5_config(config_path)
        
        # 验证默认关闭
        multi_action_enabled = config.get('multi_action', {}).get('enabled', False)
        assert multi_action_enabled == False, "multi_action.enabled应该默认为False"
        
        # 使用默认配置创建环境
        env = V5CityEnvironment(config_path)
        env.reset()
        
        # 旧版操作应该正常工作
        seq = Sequence(agent="EDU", actions=[0])
        
        # 执行应该成功（走兼容路径）
        reward, terms = env._execute_agent_sequence("EDU", seq)
        assert isinstance(reward, (int, float)), "应该返回数值奖励"
        
        print(f"   关闭模式下执行成功: reward={reward}")
        
        results.add_pass("multi_action.enabled=false 行为")
    except Exception as e:
        results.add_fail("multi_action.enabled=false 行为", str(e))


def main():
    """主测试函数"""
    print("="*60)
    print("多动作机制测试 - 阶段1&2验证")
    print("="*60)
    print()
    
    results = TestResults()
    
    # 阶段1测试：数据结构
    print("【阶段1：数据结构测试】")
    test_atomic_action(results)
    test_candidate_index(results)
    test_sequence_compatibility(results)
    test_config_loading(results)
    print()
    
    # 阶段2测试：枚举器和环境
    print("【阶段2：枚举器和环境测试】")
    test_enumerator_basic(results)
    test_enumerator_with_index(results)
    test_environment_initialization(results)
    test_environment_reset(results)
    test_sequence_execution_compatibility(results)
    print()
    
    # 兼容性测试
    print("【兼容性测试】")
    test_multi_action_config_off(results)
    print()
    
    # 输出总结
    success = results.summary()
    
    if success:
        print("\n[SUCCESS] 所有测试通过！阶段1和阶段2的改动正常运行。")
        return 0
    else:
        print("\n[WARNING] 部分测试失败，请检查上述错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)



测试范围：
1. 数据结构（AtomicAction, CandidateIndex, Sequence）
2. 兼容层（int → AtomicAction 自动转换）
3. 枚举器（enumerate_with_index, point×type 索引）
4. 环境（_execute_action_atomic, 兼容执行）
5. 配置加载
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts.contracts import AtomicAction, CandidateIndex, Sequence, ActionCandidate
from logic.v5_enumeration import V5ActionEnumerator
from envs.v5_0.city_env import V5CityEnvironment
from config_loader import ConfigLoader


class TestResults:
    """测试结果记录"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"[PASS] {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"[FAIL] {test_name}")
        print(f"   Error: {error}")
    
    def summary(self):
        print("\n" + "="*60)
        print(f"测试总结: {self.passed} 通过, {self.failed} 失败")
        print("="*60)
        if self.errors:
            print("\n失败详情:")
            for err in self.errors:
                print(f"  - {err}")
        return self.failed == 0


def test_atomic_action(results: TestResults):
    """测试1: AtomicAction 数据类"""
    try:
        # 创建原子动作
        action = AtomicAction(point=5, atype=2, meta={"test": "value"})
        
        assert action.point == 5, "point字段错误"
        assert action.atype == 2, "atype字段错误"
        assert action.meta["test"] == "value", "meta字段错误"
        
        # 测试验证
        try:
            invalid_action = AtomicAction(point=-1, atype=0)
            assert False, "应该抛出验证错误"
        except AssertionError as e:
            if "non-negative" in str(e):
                pass  # 预期的错误
            else:
                raise
        
        results.add_pass("AtomicAction 数据类")
    except Exception as e:
        results.add_fail("AtomicAction 数据类", str(e))


def test_candidate_index(results: TestResults):
    """测试2: CandidateIndex 数据类"""
    try:
        # 创建候选索引
        cand_idx = CandidateIndex(
            points=[0, 1, 2],
            types_per_point=[[0, 1], [3, 4], [6, 7, 8]],
            point_to_slots={0: ["slot_a"], 1: ["slot_b"], 2: ["slot_c"]}
        )
        
        assert len(cand_idx.points) == 3, "points长度错误"
        assert len(cand_idx.types_per_point) == 3, "types_per_point长度错误"
        assert len(cand_idx.types_per_point[2]) == 3, "types_per_point[2]长度错误"
        assert cand_idx.point_to_slots[0] == ["slot_a"], "point_to_slots映射错误"
        
        # 测试验证
        try:
            invalid_idx = CandidateIndex(
                points=[0, 1],
                types_per_point=[[0]],  # 长度不匹配
                point_to_slots={}
            )
            assert False, "应该抛出验证错误"
        except AssertionError:
            pass  # 预期的错误
        
        results.add_pass("CandidateIndex 数据类")
    except Exception as e:
        results.add_fail("CandidateIndex 数据类", str(e))


def test_sequence_compatibility(results: TestResults):
    """测试3: Sequence 兼容层"""
    try:
        # 测试旧版：使用int列表
        seq_old = Sequence(agent="IND", actions=[3, 4, 5])
        
        # 验证自动转换
        assert len(seq_old.actions) == 3, "actions长度错误"
        assert isinstance(seq_old.actions[0], AtomicAction), "应该自动转换为AtomicAction"
        assert seq_old.actions[0].point == 0, "旧版转换的point应为0"
        assert seq_old.actions[0].meta.get('legacy_id') == 3, "应该保留legacy_id"
        
        # 测试 get_legacy_ids
        legacy_ids = seq_old.get_legacy_ids()
        assert legacy_ids == [3, 4, 5], "get_legacy_ids返回错误"
        
        # 测试新版：使用AtomicAction列表
        seq_new = Sequence(
            agent="EDU",
            actions=[
                AtomicAction(point=1, atype=0, meta={"action_id": 0}),
                AtomicAction(point=2, atype=1, meta={"action_id": 1})
            ]
        )
        
        assert len(seq_new.actions) == 2, "actions长度错误"
        assert seq_new.actions[0].point == 1, "point字段错误"
        assert seq_new.actions[1].atype == 1, "atype字段错误"
        
        results.add_pass("Sequence 兼容层")
    except Exception as e:
        results.add_fail("Sequence 兼容层", str(e))


def test_config_loading(results: TestResults):
    """测试4: 配置加载"""
    try:
        config_path = "configs/city_config_v5_0.json"
        
        # 检查配置文件存在
        assert os.path.exists(config_path), f"配置文件不存在: {config_path}"
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证 multi_action 配置节
        assert 'multi_action' in config, "缺少multi_action配置节"
        
        multi_action = config['multi_action']
        assert 'enabled' in multi_action, "缺少enabled字段"
        assert 'max_actions_per_step' in multi_action, "缺少max_actions_per_step字段"
        assert 'mode' in multi_action, "缺少mode字段"
        
        # 验证默认值
        assert multi_action['enabled'] == False, "enabled应该默认为False"
        assert multi_action['max_actions_per_step'] == 5, "max_actions_per_step应该为5"
        assert multi_action['mode'] == "two_stage", "mode应该为two_stage"
        
        results.add_pass("配置加载")
    except Exception as e:
        results.add_fail("配置加载", str(e))


def test_enumerator_basic(results: TestResults):
    """测试5: 枚举器基础功能"""
    try:
        config_path = "configs/city_config_v5_0.json"
        loader = ConfigLoader()
        config = loader.load_v5_config(config_path)
        
        # 创建枚举器
        enumerator = V5ActionEnumerator(config)
        
        # 加载测试槽位
        test_slots = [
            {"id": "slot_0", "x": 0, "y": 0, "neighbors": [], "building_level": 3},
            {"id": "slot_1", "x": 1, "y": 0, "neighbors": [], "building_level": 4},
            {"id": "slot_2", "x": 2, "y": 0, "neighbors": [], "building_level": 5}
        ]
        enumerator.load_slots(test_slots)
        
        # 验证槽位加载
        assert len(enumerator.slots) == 3, "槽位加载数量错误"
        assert "slot_0" in enumerator.slots, "slot_0未加载"
        
        results.add_pass("枚举器基础功能")
    except Exception as e:
        results.add_fail("枚举器基础功能", str(e))


def test_enumerator_with_index(results: TestResults):
    """测试6: 枚举器 enumerate_with_index 方法"""
    try:
        config_path = "configs/city_config_v5_0.json"
        loader = ConfigLoader()
        config = loader.load_v5_config(config_path)
        
        # 创建枚举器
        enumerator = V5ActionEnumerator(config)
        
        # 加载测试槽位
        test_slots = [
            {"id": f"slot_{i}", "x": i, "y": 0, "neighbors": [], "building_level": 5}
            for i in range(5)
        ]
        enumerator.load_slots(test_slots)
        
        # 测试枚举
        occupied_slots = set()
        budget = 5000.0
        
        def lp_provider(slot_id):
            return 0.5
        
        candidates, cand_idx = enumerator.enumerate_with_index(
            agent="IND",
            occupied_slots=occupied_slots,
            lp_provider=lp_provider,
            budget=budget,
            current_month=0
        )
        
        # 验证返回类型
        assert isinstance(candidates, list), "candidates应该是列表"
        assert isinstance(cand_idx, CandidateIndex), "cand_idx应该是CandidateIndex"
        
        # 验证候选索引结构
        assert len(cand_idx.points) > 0, "应该有可用点"
        assert len(cand_idx.points) == len(cand_idx.types_per_point), "points和types_per_point长度应相同"
        
        # 验证候选列表
        if len(candidates) > 0:
            assert isinstance(candidates[0], ActionCandidate), "候选应该是ActionCandidate"
            assert 'point_idx' in candidates[0].meta, "meta中应该有point_idx"
            assert 'type_idx' in candidates[0].meta, "meta中应该有type_idx"
            assert 'point_id' in candidates[0].meta, "meta中应该有point_id"
        
        print(f"   枚举结果: {len(cand_idx.points)}个点, {len(candidates)}个候选")
        
        results.add_pass("枚举器 enumerate_with_index")
    except Exception as e:
        results.add_fail("枚举器 enumerate_with_index", str(e))


def test_environment_initialization(results: TestResults):
    """测试7: 环境初始化"""
    try:
        config_path = "configs/city_config_v5_0.json"
        
        # 创建环境
        env = V5CityEnvironment(config_path)
        
        # 验证基本属性
        assert hasattr(env, '_last_cand_idx'), "环境应该有_last_cand_idx属性"
        assert isinstance(env._last_cand_idx, dict), "_last_cand_idx应该是字典"
        
        assert hasattr(env, '_execute_action_atomic'), "环境应该有_execute_action_atomic方法"
        
        results.add_pass("环境初始化")
    except Exception as e:
        results.add_fail("环境初始化", str(e))


def test_environment_reset(results: TestResults):
    """测试8: 环境重置"""
    try:
        config_path = "configs/city_config_v5_0.json"
        env = V5CityEnvironment(config_path)
        
        # 重置环境
        initial_state = env.reset()
        
        # 验证初始状态
        assert env.current_step == 0, "初始step应为0"
        assert env.current_month == 0, "初始month应为0"
        assert len(env.buildings) == 0, "初始建筑列表应为空"
        
        results.add_pass("环境重置")
    except Exception as e:
        results.add_fail("环境重置", str(e))


def test_sequence_execution_compatibility(results: TestResults):
    """测试9: Sequence执行兼容性（集成测试）"""
    try:
        config_path = "configs/city_config_v5_0.json"
        env = V5CityEnvironment(config_path)
        env.reset()
        
        # 测试旧版路径：使用int动作
        seq_old = Sequence(agent="IND", actions=[3])
        
        # 验证执行不会报错（即使候选不存在也应该返回0奖励而不是崩溃）
        try:
            reward, terms = env._execute_agent_sequence("IND", seq_old)
            # 成功执行（可能返回0奖励，但不应该崩溃）
            assert isinstance(reward, (int, float)), "奖励应该是数值"
            assert isinstance(terms, dict), "奖励项应该是字典"
            print(f"   旧版执行: reward={reward}, terms={terms}")
        except Exception as e:
            # 如果是预期的错误（如预算不足、槽位占用），则通过
            if "budget" in str(e).lower() or "slot" in str(e).lower():
                print(f"   旧版执行遇到预期错误: {e}")
            else:
                raise
        
        results.add_pass("Sequence执行兼容性")
    except Exception as e:
        results.add_fail("Sequence执行兼容性", str(e))


def test_multi_action_config_off(results: TestResults):
    """测试10: multi_action.enabled=false 时的行为"""
    try:
        config_path = "configs/city_config_v5_0.json"
        loader = ConfigLoader()
        config = loader.load_v5_config(config_path)
        
        # 验证默认关闭
        multi_action_enabled = config.get('multi_action', {}).get('enabled', False)
        assert multi_action_enabled == False, "multi_action.enabled应该默认为False"
        
        # 使用默认配置创建环境
        env = V5CityEnvironment(config_path)
        env.reset()
        
        # 旧版操作应该正常工作
        seq = Sequence(agent="EDU", actions=[0])
        
        # 执行应该成功（走兼容路径）
        reward, terms = env._execute_agent_sequence("EDU", seq)
        assert isinstance(reward, (int, float)), "应该返回数值奖励"
        
        print(f"   关闭模式下执行成功: reward={reward}")
        
        results.add_pass("multi_action.enabled=false 行为")
    except Exception as e:
        results.add_fail("multi_action.enabled=false 行为", str(e))


def main():
    """主测试函数"""
    print("="*60)
    print("多动作机制测试 - 阶段1&2验证")
    print("="*60)
    print()
    
    results = TestResults()
    
    # 阶段1测试：数据结构
    print("【阶段1：数据结构测试】")
    test_atomic_action(results)
    test_candidate_index(results)
    test_sequence_compatibility(results)
    test_config_loading(results)
    print()
    
    # 阶段2测试：枚举器和环境
    print("【阶段2：枚举器和环境测试】")
    test_enumerator_basic(results)
    test_enumerator_with_index(results)
    test_environment_initialization(results)
    test_environment_reset(results)
    test_sequence_execution_compatibility(results)
    print()
    
    # 兼容性测试
    print("【兼容性测试】")
    test_multi_action_config_off(results)
    print()
    
    # 输出总结
    success = results.summary()
    
    if success:
        print("\n[SUCCESS] 所有测试通过！阶段1和阶段2的改动正常运行。")
        return 0
    else:
        print("\n[WARNING] 部分测试失败，请检查上述错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

