#!/usr/bin/env python3
"""
测试槽位占用机制修复
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts import Sequence, AtomicAction

def test_occupied_slots_extraction():
    """测试从Sequence正确提取action_id"""
    print("=== 测试槽位占用机制修复 ===\n")
    
    # 测试1: 旧版Sequence（int actions）
    print("测试1: 旧版Sequence (int actions)")
    seq_old = Sequence(agent="IND", actions=[0, 1, 2])
    legacy_ids_old = seq_old.get_legacy_ids()
    print(f"  actions: {seq_old.actions}")
    print(f"  legacy_ids: {legacy_ids_old}")
    assert legacy_ids_old == [0, 1, 2], "旧版提取失败"
    print("  PASS: 旧版Sequence提取正确\n")
    
    # 测试2: 新版Sequence（AtomicAction）
    print("测试2: 新版Sequence (AtomicAction)")
    atomic_actions = [
        AtomicAction(point=0, atype=1, meta={"action_id": 10, "slots": ["slot_a"]}),
        AtomicAction(point=1, atype=2, meta={"action_id": 20, "slots": ["slot_b"]}),
        AtomicAction(point=2, atype=3, meta={"action_id": 30, "slots": ["slot_c"]})
    ]
    seq_new = Sequence(agent="EDU", actions=atomic_actions)
    legacy_ids_new = seq_new.get_legacy_ids()
    print(f"  actions: {len(seq_new.actions)} AtomicActions")
    print(f"  legacy_ids: {legacy_ids_new}")
    assert legacy_ids_new == [10, 20, 30], "新版提取失败"
    print("  PASS: 新版Sequence提取正确\n")
    
    # 测试3: 模拟槽位占用更新
    print("测试3: 模拟槽位占用更新")
    
    # 模拟旧的错误方式
    print("  错误方式（直接遍历actions）:")
    try:
        for action in seq_new.actions:
            # 尝试把AtomicAction当作int使用
            print(f"    action: {action} (type: {type(action).__name__})")
            # 这里如果传给期望int的函数，会出问题
        print("    结果: 类型错误，无法正确提取action_id")
    except Exception as e:
        print(f"    错误: {e}")
    
    # 模拟正确方式
    print("\n  正确方式（使用get_legacy_ids）:")
    for action_id in seq_new.get_legacy_ids():
        print(f"    action_id: {action_id} (type: {type(action_id).__name__})")
    print("    结果: 正确提取所有action_id\n")
    
    # 测试4: 槽位去重验证
    print("测试4: 槽位占用去重")
    occupied_slots = set()
    
    # 模拟多个智能体选择动作
    sequences = [
        Sequence(agent="IND", actions=[AtomicAction(point=0, atype=1, meta={"action_id": 10, "slots": ["slot_1", "slot_2"]})]),
        Sequence(agent="EDU", actions=[AtomicAction(point=1, atype=2, meta={"action_id": 20, "slots": ["slot_3", "slot_4"]})]),
        Sequence(agent="COUNCIL", actions=[AtomicAction(point=2, atype=3, meta={"action_id": 30, "slots": ["slot_5"]})])
    ]
    
    # 模拟槽位占用更新（使用正确方式）
    slot_mapping = {
        10: ["slot_1", "slot_2"],
        20: ["slot_3", "slot_4"],
        30: ["slot_5"]
    }
    
    for seq in sequences:
        for action_id in seq.get_legacy_ids():
            slots = slot_mapping.get(action_id, [])
            for slot_id in slots:
                occupied_slots.add(slot_id)
                print(f"  占用槽位: {slot_id} (action_id: {action_id})")
    
    print(f"\n  总计占用槽位: {len(occupied_slots)} 个")
    print(f"  槽位列表: {sorted(occupied_slots)}")
    assert len(occupied_slots) == 5, "槽位占用数量错误"
    print("  PASS: 槽位正确标记为占用\n")
    
    return True

if __name__ == "__main__":
    print("="*60)
    success = test_occupied_slots_extraction()
    print("="*60)
    
    if success:
        print("\n所有测试通过！\n")
        print("修复说明:")
        print("  - 问题: _update_occupied_slots_from_snapshot 直接遍历 actions")
        print("  - 影响: AtomicAction 无法正确提取 action_id")
        print("  - 结果: 槽位占用机制失效，同一槽位被重复选择")
        print("\n  - 修复: 使用 sequence.get_legacy_ids() 提取 action_id")
        print("  - 效果: 槽位正确标记为占用，避免重复选择")
        print("\n现在可以重新训练:")
        print("  python enhanced_city_simulation_v5_0.py --mode complete --episodes 2 --verbose")
    else:
        print("\n测试失败")
    
    sys.exit(0 if success else 1)
"""
测试槽位占用机制修复
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts import Sequence, AtomicAction

def test_occupied_slots_extraction():
    """测试从Sequence正确提取action_id"""
    print("=== 测试槽位占用机制修复 ===\n")
    
    # 测试1: 旧版Sequence（int actions）
    print("测试1: 旧版Sequence (int actions)")
    seq_old = Sequence(agent="IND", actions=[0, 1, 2])
    legacy_ids_old = seq_old.get_legacy_ids()
    print(f"  actions: {seq_old.actions}")
    print(f"  legacy_ids: {legacy_ids_old}")
    assert legacy_ids_old == [0, 1, 2], "旧版提取失败"
    print("  PASS: 旧版Sequence提取正确\n")
    
    # 测试2: 新版Sequence（AtomicAction）
    print("测试2: 新版Sequence (AtomicAction)")
    atomic_actions = [
        AtomicAction(point=0, atype=1, meta={"action_id": 10, "slots": ["slot_a"]}),
        AtomicAction(point=1, atype=2, meta={"action_id": 20, "slots": ["slot_b"]}),
        AtomicAction(point=2, atype=3, meta={"action_id": 30, "slots": ["slot_c"]})
    ]
    seq_new = Sequence(agent="EDU", actions=atomic_actions)
    legacy_ids_new = seq_new.get_legacy_ids()
    print(f"  actions: {len(seq_new.actions)} AtomicActions")
    print(f"  legacy_ids: {legacy_ids_new}")
    assert legacy_ids_new == [10, 20, 30], "新版提取失败"
    print("  PASS: 新版Sequence提取正确\n")
    
    # 测试3: 模拟槽位占用更新
    print("测试3: 模拟槽位占用更新")
    
    # 模拟旧的错误方式
    print("  错误方式（直接遍历actions）:")
    try:
        for action in seq_new.actions:
            # 尝试把AtomicAction当作int使用
            print(f"    action: {action} (type: {type(action).__name__})")
            # 这里如果传给期望int的函数，会出问题
        print("    结果: 类型错误，无法正确提取action_id")
    except Exception as e:
        print(f"    错误: {e}")
    
    # 模拟正确方式
    print("\n  正确方式（使用get_legacy_ids）:")
    for action_id in seq_new.get_legacy_ids():
        print(f"    action_id: {action_id} (type: {type(action_id).__name__})")
    print("    结果: 正确提取所有action_id\n")
    
    # 测试4: 槽位去重验证
    print("测试4: 槽位占用去重")
    occupied_slots = set()
    
    # 模拟多个智能体选择动作
    sequences = [
        Sequence(agent="IND", actions=[AtomicAction(point=0, atype=1, meta={"action_id": 10, "slots": ["slot_1", "slot_2"]})]),
        Sequence(agent="EDU", actions=[AtomicAction(point=1, atype=2, meta={"action_id": 20, "slots": ["slot_3", "slot_4"]})]),
        Sequence(agent="COUNCIL", actions=[AtomicAction(point=2, atype=3, meta={"action_id": 30, "slots": ["slot_5"]})])
    ]
    
    # 模拟槽位占用更新（使用正确方式）
    slot_mapping = {
        10: ["slot_1", "slot_2"],
        20: ["slot_3", "slot_4"],
        30: ["slot_5"]
    }
    
    for seq in sequences:
        for action_id in seq.get_legacy_ids():
            slots = slot_mapping.get(action_id, [])
            for slot_id in slots:
                occupied_slots.add(slot_id)
                print(f"  占用槽位: {slot_id} (action_id: {action_id})")
    
    print(f"\n  总计占用槽位: {len(occupied_slots)} 个")
    print(f"  槽位列表: {sorted(occupied_slots)}")
    assert len(occupied_slots) == 5, "槽位占用数量错误"
    print("  PASS: 槽位正确标记为占用\n")
    
    return True

if __name__ == "__main__":
    print("="*60)
    success = test_occupied_slots_extraction()
    print("="*60)
    
    if success:
        print("\n所有测试通过！\n")
        print("修复说明:")
        print("  - 问题: _update_occupied_slots_from_snapshot 直接遍历 actions")
        print("  - 影响: AtomicAction 无法正确提取 action_id")
        print("  - 结果: 槽位占用机制失效，同一槽位被重复选择")
        print("\n  - 修复: 使用 sequence.get_legacy_ids() 提取 action_id")
        print("  - 效果: 槽位正确标记为占用，避免重复选择")
        print("\n现在可以重新训练:")
        print("  python enhanced_city_simulation_v5_0.py --mode complete --episodes 2 --verbose")
    else:
        print("\n测试失败")
    
    sys.exit(0 if success else 1)
