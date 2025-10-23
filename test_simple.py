# -*- coding: utf-8 -*-
"""简化测试脚本 - 仅测试核心数据结构"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts.contracts import AtomicAction, CandidateIndex, Sequence

def main():
    print("="*60)
    print("核心数据结构测试")
    print("="*60)
    
    passed = 0
    failed = 0
    
    # Test 1: AtomicAction
    try:
        action = AtomicAction(point=5, atype=2, meta={"test": "value"})
        assert action.point == 5
        assert action.atype == 2
        print("[PASS] AtomicAction")
        passed += 1
    except Exception as e:
        print(f"[FAIL] AtomicAction: {e}")
        failed += 1
    
    # Test 2: CandidateIndex
    try:
        cand_idx = CandidateIndex(
            points=[0, 1, 2],
            types_per_point=[[0, 1], [3, 4], [6, 7, 8]],
            point_to_slots={0: ["slot_a"], 1: ["slot_b"], 2: ["slot_c"]}
        )
        assert len(cand_idx.points) == 3
        print("[PASS] CandidateIndex")
        passed += 1
    except Exception as e:
        print(f"[FAIL] CandidateIndex: {e}")
        failed += 1
    
    # Test 3: Sequence with int (old version)
    try:
        seq_old = Sequence(agent="IND", actions=[3, 4, 5])
        assert len(seq_old.actions) == 3
        assert isinstance(seq_old.actions[0], AtomicAction)
        assert seq_old.actions[0].meta.get('legacy_id') == 3
        legacy_ids = seq_old.get_legacy_ids()
        assert legacy_ids == [3, 4, 5]
        print("[PASS] Sequence compatibility (int -> AtomicAction)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Sequence compatibility: {e}")
        failed += 1
    
    # Test 4: Sequence with AtomicAction (new version)
    try:
        seq_new = Sequence(
            agent="EDU",
            actions=[
                AtomicAction(point=1, atype=0, meta={"action_id": 0}),
                AtomicAction(point=2, atype=1, meta={"action_id": 1})
            ]
        )
        assert len(seq_new.actions) == 2
        assert seq_new.actions[0].point == 1
        print("[PASS] Sequence with AtomicAction")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Sequence with AtomicAction: {e}")
        failed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Result: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n[SUCCESS] All core data structures working!")
        return 0
    else:
        print("\n[WARNING] Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

"""简化测试脚本 - 仅测试核心数据结构"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts.contracts import AtomicAction, CandidateIndex, Sequence

def main():
    print("="*60)
    print("核心数据结构测试")
    print("="*60)
    
    passed = 0
    failed = 0
    
    # Test 1: AtomicAction
    try:
        action = AtomicAction(point=5, atype=2, meta={"test": "value"})
        assert action.point == 5
        assert action.atype == 2
        print("[PASS] AtomicAction")
        passed += 1
    except Exception as e:
        print(f"[FAIL] AtomicAction: {e}")
        failed += 1
    
    # Test 2: CandidateIndex
    try:
        cand_idx = CandidateIndex(
            points=[0, 1, 2],
            types_per_point=[[0, 1], [3, 4], [6, 7, 8]],
            point_to_slots={0: ["slot_a"], 1: ["slot_b"], 2: ["slot_c"]}
        )
        assert len(cand_idx.points) == 3
        print("[PASS] CandidateIndex")
        passed += 1
    except Exception as e:
        print(f"[FAIL] CandidateIndex: {e}")
        failed += 1
    
    # Test 3: Sequence with int (old version)
    try:
        seq_old = Sequence(agent="IND", actions=[3, 4, 5])
        assert len(seq_old.actions) == 3
        assert isinstance(seq_old.actions[0], AtomicAction)
        assert seq_old.actions[0].meta.get('legacy_id') == 3
        legacy_ids = seq_old.get_legacy_ids()
        assert legacy_ids == [3, 4, 5]
        print("[PASS] Sequence compatibility (int -> AtomicAction)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Sequence compatibility: {e}")
        failed += 1
    
    # Test 4: Sequence with AtomicAction (new version)
    try:
        seq_new = Sequence(
            agent="EDU",
            actions=[
                AtomicAction(point=1, atype=0, meta={"action_id": 0}),
                AtomicAction(point=2, atype=1, meta={"action_id": 1})
            ]
        )
        assert len(seq_new.actions) == 2
        assert seq_new.actions[0].point == 1
        print("[PASS] Sequence with AtomicAction")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Sequence with AtomicAction: {e}")
        failed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Result: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n[SUCCESS] All core data structures working!")
        return 0
    else:
        print("\n[WARNING] Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())







