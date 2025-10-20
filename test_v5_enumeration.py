"""
测试v5.0枚举系统
"""

import json
from config_loader import ConfigLoader
from logic.v5_enumeration import V5ActionEnumerator
from logic.v5_scorer import V5ActionScorer
from logic.v5_selector import V5SequenceSelector


def test_v5_enumeration():
    """测试v5.0枚举系统"""
    print("Testing v5.0 enumeration system...")
    
    try:
        # 加载配置
        loader = ConfigLoader()
        config = loader.load_v5_config("configs/city_config_v5_0.json")
        
        # 创建枚举器
        enumerator = V5ActionEnumerator(config)
        
        # 模拟槽位数据
        slots_data = [
            {"id": "slot_1", "x": 10, "y": 10, "neighbors": ["slot_2"], "building_level": 5},
            {"id": "slot_2", "x": 11, "y": 10, "neighbors": ["slot_1"], "building_level": 5},
            {"id": "slot_3", "x": 20, "y": 20, "neighbors": [], "building_level": 3},
        ]
        enumerator.load_slots(slots_data)
        
        # 模拟地价提供函数
        def lp_provider(slot_id):
            return 0.5  # 固定地价
        
        # 测试EDU动作枚举
        print("\\nTesting EDU action enumeration...")
        edu_candidates = enumerator.enumerate_actions("EDU", set(), lp_provider, 1000.0)
        print(f"EDU candidates: {len(edu_candidates)}")
        for i, candidate in enumerate(edu_candidates[:3]):  # 显示前3个
            print(f"  Candidate {i}: action_id={candidate.id}, features_shape={candidate.features.shape}")
            print(f"    Meta: {candidate.meta}")
        
        # 测试IND动作枚举
        print("\\nTesting IND action enumeration...")
        ind_candidates = enumerator.enumerate_actions("IND", set(), lp_provider, 2000.0)
        print(f"IND candidates: {len(ind_candidates)}")
        for i, candidate in enumerate(ind_candidates[:3]):  # 显示前3个
            print(f"  Candidate {i}: action_id={candidate.id}, features_shape={candidate.features.shape}")
            print(f"    Meta: {candidate.meta}")
        
        # 测试COUNCIL动作枚举
        print("\\nTesting COUNCIL action enumeration...")
        council_candidates = enumerator.enumerate_actions("COUNCIL", set(), lp_provider, 500.0)
        print(f"COUNCIL candidates: {len(council_candidates)}")
        for i, candidate in enumerate(council_candidates[:3]):  # 显示前3个
            print(f"  Candidate {i}: action_id={candidate.id}, features_shape={candidate.features.shape}")
            print(f"    Meta: {candidate.meta}")
        
        print("\\nEnumeration test passed!")
        return True
        
    except Exception as e:
        print(f"Enumeration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_v5_scoring():
    """测试v5.0打分系统"""
    print("\\nTesting v5.0 scoring system...")
    
    try:
        # 加载配置
        loader = ConfigLoader()
        config = loader.load_v5_config("configs/city_config_v5_0.json")
        
        # 创建打分器
        scorer = V5ActionScorer(config)
        
        # 创建测试候选
        from contracts import ActionCandidate
        import numpy as np
        
        candidate = ActionCandidate(
            id=0,  # EDU_S
            features=np.array([0.0] * 32),
            meta={
                "agent": "EDU",
                "action_id": 0,
                "cost": 650,
                "reward": 160,
                "prestige": 0.2,
                "slots": ["slot_1"],
                "zone": "default",
                "lp_norm": 0.5
            }
        )
        
        # 模拟状态
        state = {
            "month": 5,
            "buildings": [],
            "budgets": {"EDU": 1000.0, "IND": 2000.0, "COUNCIL": 500.0}
        }
        
        # 计算奖励
        reward_terms = scorer.score_action(candidate, state)
        print(f"Reward terms: {reward_terms.to_dict()}")
        
        print("Scoring test passed!")
        return True
        
    except Exception as e:
        print(f"Scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_v5_selection():
    """测试v5.0选择系统"""
    print("\\nTesting v5.0 selection system...")
    
    try:
        # 加载配置
        loader = ConfigLoader()
        config = loader.load_v5_config("configs/city_config_v5_0.json")
        
        # 创建选择器
        selector = V5SequenceSelector(config)
        
        # 创建测试序列
        from contracts import Sequence
        
        sequences = [
            Sequence(agent="EDU", actions=[0]),  # EDU_S
            Sequence(agent="EDU", actions=[1]),  # EDU_M
            Sequence(agent="EDU", actions=[2]),  # EDU_L
        ]
        
        # 模拟状态
        state = {
            "month": 5,
            "buildings": [],
            "budgets": {"EDU": 1000.0, "IND": 2000.0, "COUNCIL": 500.0}
        }
        
        # 选择序列
        selected = selector.select_sequence("EDU", sequences, state, mode="greedy")
        print(f"Selected sequence: {selected}")
        
        if selected:
            # 创建步骤日志
            step_log = selector.create_step_log(
                step=5,
                agent="EDU",
                sequence=selected,
                reward_terms={"revenue": 160.0, "cost": 650.0, "prestige": 0.2},
                budget_snapshot={"EDU": 1000.0}
            )
            print(f"Step log: {step_log}")
        
        print("Selection test passed!")
        return True
        
    except Exception as e:
        print(f"Selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing v5.0 enumeration system...")
    
    # 测试枚举
    enum_ok = test_v5_enumeration()
    
    # 测试打分
    scoring_ok = test_v5_scoring()
    
    # 测试选择
    selection_ok = test_v5_selection()
    
    if enum_ok and scoring_ok and selection_ok:
        print("\\nAll v5.0 enumeration tests passed!")
    else:
        print("\\nSome v5.0 enumeration tests failed!")
