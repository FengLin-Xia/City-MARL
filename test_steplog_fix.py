#!/usr/bin/env python3
"""
测试StepLog修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts import StepLog, AtomicAction, Sequence

def test_steplog_with_legacy_ids():
    """测试StepLog使用legacy_ids"""
    print("=== 测试StepLog修复 ===")
    
    # 测试1: 旧版Sequence -> get_legacy_ids
    seq_old = Sequence(agent="IND", actions=[0, 1, 2])
    legacy_ids = seq_old.get_legacy_ids()
    
    print(f"旧版Sequence: actions={seq_old.actions}")
    print(f"Legacy IDs: {legacy_ids}")
    
    # 创建StepLog
    step_log = StepLog(
        t=0,
        agent="IND",
        chosen=legacy_ids,
        reward_terms={"revenue": 100.0}
    )
    print(f"PASS: StepLog创建成功 (旧版)")
    print(f"  - chosen: {step_log.chosen}")
    
    # 测试2: 新版Sequence -> get_legacy_ids
    atomic_actions = [
        AtomicAction(point=0, atype=1, meta={"action_id": 1}),
        AtomicAction(point=1, atype=2, meta={"action_id": 2})
    ]
    seq_new = Sequence(agent="EDU", actions=atomic_actions)
    legacy_ids_new = seq_new.get_legacy_ids()
    
    print(f"\n新版Sequence: actions={seq_new.actions}")
    print(f"Legacy IDs: {legacy_ids_new}")
    
    # 创建StepLog
    step_log_new = StepLog(
        t=1,
        agent="EDU",
        chosen=legacy_ids_new,
        reward_terms={"revenue": 200.0}
    )
    print(f"PASS: StepLog创建成功 (新版)")
    print(f"  - chosen: {step_log_new.chosen}")
    
    # 测试3: StepLog兼容性验证
    try:
        # 测试int类型
        step_log_int = StepLog(
            t=2,
            agent="COUNCIL",
            chosen=[0, 1, 2],
            reward_terms={"revenue": 300.0}
        )
        print(f"\nPASS: StepLog支持int类型")
        
        # 测试AtomicAction类型（虽然不推荐，但应该兼容）
        step_log_atomic = StepLog(
            t=3,
            agent="IND",
            chosen=[AtomicAction(point=0, atype=1, meta={"action_id": 1})],
            reward_terms={"revenue": 400.0}
        )
        print(f"PASS: StepLog支持AtomicAction类型")
        
    except Exception as e:
        print(f"FAIL: {e}")
        return False
    
    print("\n所有测试通过！")
    return True

if __name__ == "__main__":
    success = test_steplog_with_legacy_ids()
    sys.exit(0 if success else 1)

测试StepLog修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts import StepLog, AtomicAction, Sequence

def test_steplog_with_legacy_ids():
    """测试StepLog使用legacy_ids"""
    print("=== 测试StepLog修复 ===")
    
    # 测试1: 旧版Sequence -> get_legacy_ids
    seq_old = Sequence(agent="IND", actions=[0, 1, 2])
    legacy_ids = seq_old.get_legacy_ids()
    
    print(f"旧版Sequence: actions={seq_old.actions}")
    print(f"Legacy IDs: {legacy_ids}")
    
    # 创建StepLog
    step_log = StepLog(
        t=0,
        agent="IND",
        chosen=legacy_ids,
        reward_terms={"revenue": 100.0}
    )
    print(f"PASS: StepLog创建成功 (旧版)")
    print(f"  - chosen: {step_log.chosen}")
    
    # 测试2: 新版Sequence -> get_legacy_ids
    atomic_actions = [
        AtomicAction(point=0, atype=1, meta={"action_id": 1}),
        AtomicAction(point=1, atype=2, meta={"action_id": 2})
    ]
    seq_new = Sequence(agent="EDU", actions=atomic_actions)
    legacy_ids_new = seq_new.get_legacy_ids()
    
    print(f"\n新版Sequence: actions={seq_new.actions}")
    print(f"Legacy IDs: {legacy_ids_new}")
    
    # 创建StepLog
    step_log_new = StepLog(
        t=1,
        agent="EDU",
        chosen=legacy_ids_new,
        reward_terms={"revenue": 200.0}
    )
    print(f"PASS: StepLog创建成功 (新版)")
    print(f"  - chosen: {step_log_new.chosen}")
    
    # 测试3: StepLog兼容性验证
    try:
        # 测试int类型
        step_log_int = StepLog(
            t=2,
            agent="COUNCIL",
            chosen=[0, 1, 2],
            reward_terms={"revenue": 300.0}
        )
        print(f"\nPASS: StepLog支持int类型")
        
        # 测试AtomicAction类型（虽然不推荐，但应该兼容）
        step_log_atomic = StepLog(
            t=3,
            agent="IND",
            chosen=[AtomicAction(point=0, atype=1, meta={"action_id": 1})],
            reward_terms={"revenue": 400.0}
        )
        print(f"PASS: StepLog支持AtomicAction类型")
        
    except Exception as e:
        print(f"FAIL: {e}")
        return False
    
    print("\n所有测试通过！")
    return True

if __name__ == "__main__":
    success = test_steplog_with_legacy_ids()
    sys.exit(0 if success else 1)
