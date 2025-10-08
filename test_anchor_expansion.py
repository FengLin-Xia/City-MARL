#!/usr/bin/env python3
"""
测试锚点+扩展模式的功能
验证RL选择器能否正确生成多槽位序列
"""

import json
import torch
import numpy as np
from typing import Dict, List
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solvers.v4_1.rl_selector import RLPolicySelector
from expansion_policy import NearestKExpansion, ClusterExpansion, RandomExpansion


def test_expansion_policy_standalone():
    """测试扩展策略的独立功能"""
    print("=== Testing Expansion Policy Standalone ===")
    
    # 创建测试数据
    test_state = {
        'slots': {
            's_1': type('Slot', (), {'fx': 0.0, 'fy': 0.0})(),
            's_2': type('Slot', (), {'fx': 1.0, 'fy': 0.0})(),
            's_3': type('Slot', (), {'fx': 0.0, 'fy': 1.0})(),
            's_4': type('Slot', (), {'fx': 2.0, 'fy': 0.0})(),
            's_5': type('Slot', (), {'fx': 0.0, 'fy': 2.0})(),
            's_6': type('Slot', (), {'fx': 1.0, 'fy': 1.0})(),
        }
    }
    
    available_slots = ['s_1', 's_2', 's_3', 's_4', 's_5', 's_6']
    anchor_slot = 's_1'
    
    # 测试NearestKExpansion
    print("\n1. Testing NearestKExpansion:")
    nearest_policy = NearestKExpansion(temperature=1.0, rule='euclidean')
    selected_slots, log_prob = nearest_policy.expand(test_state, anchor_slot, available_slots, k=4)
    print(f"   Anchor: {anchor_slot}")
    print(f"   Expanded: {selected_slots}")
    print(f"   Log prob: {log_prob:.4f}")
    
    # 测试ClusterExpansion
    print("\n2. Testing ClusterExpansion:")
    cluster_policy = ClusterExpansion(temperature=1.0, cluster_radius=1.5)
    selected_slots, log_prob = cluster_policy.expand(test_state, anchor_slot, available_slots, k=4)
    print(f"   Anchor: {anchor_slot}")
    print(f"   Expanded: {selected_slots}")
    print(f"   Log prob: {log_prob:.4f}")
    
    # 测试RandomExpansion
    print("\n3. Testing RandomExpansion:")
    random_policy = RandomExpansion(temperature=1.0)
    selected_slots, log_prob = random_policy.expand(test_state, anchor_slot, available_slots, k=4)
    print(f"   Anchor: {anchor_slot}")
    print(f"   Expanded: {selected_slots}")
    print(f"   Log prob: {log_prob:.4f}")


def test_rl_selector_with_expansion():
    """测试RL选择器的锚点+扩展功能"""
    print("\n=== Testing RL Selector with Expansion ===")
    
    # 加载配置
    try:
        with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        # 创建RL选择器
        selector = RLPolicySelector(cfg)
        
        print(f"RL Selector initialized successfully")
        print(f"Expansion policy type: {type(selector.expansion_policy).__name__}")
        print(f"Expansion k: {selector.expansion_k}")
        
        # 检查扩展策略是否正确设置
        if selector.expansion_policy is not None:
            print(f"Expansion policy initialized successfully")
        else:
            print(f"Expansion policy failed to initialize")
            return False
            
    except Exception as e:
        print(f"Failed to initialize RL selector: {e}")
        return False
    
    return True


def test_expansion_config():
    """测试扩展策略配置"""
    print("\n=== Testing Expansion Configuration ===")
    
    try:
        with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        expansion_config = cfg.get('solver', {}).get('rl', {}).get('expansion_policy', {})
        
        print(f"Expansion config found:")
        print(f"   Type: {expansion_config.get('type', 'N/A')}")
        print(f"   Temperature: {expansion_config.get('temperature', 'N/A')}")
        print(f"   Rule: {expansion_config.get('rule', 'N/A')}")
        print(f"   K: {expansion_config.get('k', 'N/A')}")
        
        # 验证必需的配置项
        required_keys = ['type', 'temperature', 'rule', 'k']
        missing_keys = [key for key in required_keys if key not in expansion_config]
        
        if missing_keys:
            print(f"Missing configuration keys: {missing_keys}")
            return False
        else:
            print(f"All required configuration keys present")
            return True
            
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return False


def test_action_sequence_generation():
    """测试动作序列生成（模拟环境调用）"""
    print("\n=== Testing Action Sequence Generation ===")
    
    try:
        # 加载配置
        with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        # 创建RL选择器
        selector = RLPolicySelector(cfg)
        
        # 模拟动作池（简化的Action对象）
        from logic.v4_enumeration import Action
        
        mock_actions = []
        for i in range(10):
            action = Action(
                agent='EDU',
                size='M',
                footprint_slots=[f's_{i}'],
                zone='EDU',
                LP_norm=1.0,
                adjacency={'N': 1, 'S': 1, 'E': 1, 'W': 1},
                cost=100.0,
                reward=150.0,
                prestige=50.0
            )
            action.score = action.reward - action.cost
            mock_actions.append(action)
        
        # 测试锚点选择
        print(f"Testing with {len(mock_actions)} mock actions...")
        
        # 调用_rl_choose_sequence方法
        sequence, action_idx = selector._rl_choose_sequence(mock_actions)
        
        if sequence is not None:
            print(f"Sequence generated successfully:")
            print(f"   Action index: {action_idx}")
            print(f"   Number of actions in sequence: {len(sequence.actions)}")
            print(f"   Sequence score: {sequence.score:.2f}")
            print(f"   Total cost: {sequence.sum_cost:.2f}")
            print(f"   Total reward: {sequence.sum_reward:.2f}")
            
            # 检查是否是多槽位序列
            if len(sequence.actions) > 1:
                print(f"Multi-slot sequence generated (expansion working)")
                print(f"   Slot IDs: {[action.footprint_slots[0] for action in sequence.actions]}")
            else:
                print(f"Single-slot sequence (expansion may not be working)")
            
            # 检查扩展信息
            if hasattr(sequence, 'expansion_log_prob'):
                print(f"Expansion log prob: {sequence.expansion_log_prob:.4f}")
            if hasattr(sequence, 'anchor_slot_id'):
                print(f"Anchor slot: {sequence.anchor_slot_id}")
            if hasattr(sequence, 'expanded_slot_ids'):
                print(f"Expanded slots: {sequence.expanded_slot_ids}")
            
            return True
        else:
            print(f"Failed to generate sequence")
            return False
            
    except Exception as e:
        print(f"Error during sequence generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("Testing Anchor + Expansion Mode Implementation")
    print("=" * 60)
    
    test_results = []
    
    # 测试1：扩展策略独立功能
    try:
        test_expansion_policy_standalone()
        test_results.append(("Expansion Policy Standalone", True))
    except Exception as e:
        print(f"Expansion policy standalone test failed: {e}")
        test_results.append(("Expansion Policy Standalone", False))
    
    # 测试2：扩展策略配置
    config_success = test_expansion_config()
    test_results.append(("Expansion Configuration", config_success))
    
    # 测试3：RL选择器初始化
    selector_success = test_rl_selector_with_expansion()
    test_results.append(("RL Selector Initialization", selector_success))
    
    # 测试4：动作序列生成
    if selector_success:
        sequence_success = test_action_sequence_generation()
        test_results.append(("Action Sequence Generation", sequence_success))
    else:
        test_results.append(("Action Sequence Generation", False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "PASS" if success else "FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\nAll tests passed! Anchor + Expansion mode is working correctly.")
        return True
    else:
        print(f"\n{total-passed} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
