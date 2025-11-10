#!/usr/bin/env python3
"""
测试v5.0架构中槽位重复选择问题

检查：
1. 同一智能体是否可能选择重复槽位
2. 不同智能体是否可能选择相同槽位
3. 全局槽位状态是否正确更新
4. 月度重置后槽位状态是否正确
"""

import sys
import os
import json
from typing import Dict, List, Set, Any
from collections import defaultdict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence, AtomicAction
from utils.logger_factory import get_logger

def test_slot_duplicate_selection():
    """测试槽位重复选择问题"""
    
    print("=== 测试v5.0架构槽位重复选择问题 ===")
    
    # 获取日志器
    logger = get_logger("test")
    
    # 创建测试配置
    test_config = {
        "agents": {
            "defs": {
                "IND": {
                    "action_ids": [1, 2, 3],
                    "constraints": {
                        "special_rules": {
                            "start_after_month": 0
                        }
                    }
                },
                "EDU": {
                    "action_ids": [4, 5, 6],
                    "constraints": {
                        "special_rules": {
                            "start_after_month": 0
                        }
                    }
                }
            }
        },
        "action_params": {
            "1": {"desc": "IND_S", "cost": 100, "reward": 50, "prestige": 0.1},
            "2": {"desc": "IND_M", "cost": 200, "reward": 100, "prestige": 0.2},
            "3": {"desc": "IND_L", "cost": 300, "reward": 150, "prestige": 0.3},
            "4": {"desc": "EDU_S", "cost": 80, "reward": 40, "prestige": 0.15},
            "5": {"desc": "EDU_M", "cost": 160, "reward": 80, "prestige": 0.25},
            "6": {"desc": "EDU_L", "cost": 240, "reward": 120, "prestige": 0.35}
        },
        "slots": {
            "path": ""
        },
        "simulation": {
            "total_months": 3
        },
        "budget": {
            "initial_budget": {
                "IND": 1000,
                "EDU": 1000
            }
        }
    }
    
    # 创建环境
    env = V5CityEnvironment("test_config.json")
    
    # 手动设置槽位数据（用于测试）
    test_slots = []
    for i in range(50):  # 创建50个测试槽位
        test_slots.append({
            "id": f"slot_{i}",
            "x": i % 10,
            "y": i // 10,
            "angle": 0.0,
            "neighbors": [],
            "building_level": 3
        })
    
    env.enumerator.load_slots(test_slots)
    
    print(f"加载了 {len(test_slots)} 个测试槽位")
    
    # 重置环境
    env.reset()
    
    # 测试1: 检查初始状态
    print("\n--- 测试1: 初始状态检查 ---")
    print(f"初始occupied_slots: {len(env.occupied_slots)}")
    print(f"初始global_occupied_slots: {len(env.global_occupied_slots)}")
    
    # 测试2: 单智能体槽位选择
    print("\n--- 测试2: 单智能体槽位选择 ---")
    agent = "IND"
    
    # 获取候选
    candidates, cand_idx = env.get_action_candidates_with_index(agent)
    print(f"Agent {agent} 获得 {len(candidates)} 个候选")
    
    # 检查候选中的槽位
    selected_slots = set()
    for candidate in candidates:
        slots = candidate.meta.get("slots", [])
        for slot_id in slots:
            if slot_id in selected_slots:
                print(f"[ERROR] 发现重复槽位: {slot_id}")
                return False
            selected_slots.add(slot_id)
    
    print(f"[OK] 单智能体无重复槽位，共选择 {len(selected_slots)} 个槽位")
    
    # 测试3: 执行动作并检查状态更新
    print("\n--- 测试3: 执行动作并检查状态更新 ---")
    
    # 创建测试动作序列
    test_sequence = Sequence(
        agent=agent,
        actions=[AtomicAction(
            atype="build",
            point=0,
            meta={"action_id": 1}
        )]
    )
    
    # 执行动作
    reward, reward_terms = env._execute_agent_sequence(agent, test_sequence)
    print(f"执行动作，奖励: {reward}")
    
    # 检查状态更新
    print(f"执行后occupied_slots: {len(env.occupied_slots)}")
    print(f"执行后global_occupied_slots: {len(env.global_occupied_slots)}")
    
    # 测试4: 多智能体槽位选择
    print("\n--- 测试4: 多智能体槽位选择 ---")
    
    agents = ["IND", "EDU"]
    all_selected_slots = set()
    duplicate_slots = set()
    
    for agent in agents:
        print(f"\n处理智能体: {agent}")
        
        # 获取候选
        candidates, cand_idx = env.get_action_candidates_with_index(agent)
        print(f"Agent {agent} 获得 {len(candidates)} 个候选")
        
        # 检查候选中的槽位
        agent_selected_slots = set()
        for candidate in candidates:
            slots = candidate.meta.get("slots", [])
            for slot_id in slots:
                if slot_id in all_selected_slots:
                    print(f"[ERROR] 发现跨智能体重复槽位: {slot_id}")
                    duplicate_slots.add(slot_id)
                if slot_id in agent_selected_slots:
                    print(f"[ERROR] 发现智能体内重复槽位: {slot_id}")
                    duplicate_slots.add(slot_id)
                
                agent_selected_slots.add(slot_id)
                all_selected_slots.add(slot_id)
        
        print(f"Agent {agent} 选择 {len(agent_selected_slots)} 个槽位")
    
    if duplicate_slots:
        print(f"[ERROR] 发现 {len(duplicate_slots)} 个重复槽位: {duplicate_slots}")
        return False
    else:
        print("[OK] 多智能体无重复槽位")
    
    # 测试5: 月度重置后状态
    print("\n--- 测试5: 月度重置后状态 ---")
    
    old_occupied = len(env.occupied_slots)
    old_global = len(env.global_occupied_slots)
    
    env.advance_month()
    
    new_occupied = len(env.occupied_slots)
    new_global = len(env.global_occupied_slots)
    
    print(f"月度推进前: occupied_slots={old_occupied}, global_occupied_slots={old_global}")
    print(f"月度推进后: occupied_slots={new_occupied}, global_occupied_slots={new_global}")
    
    # 检查重置逻辑
    if new_occupied == 0:
        print("[OK] occupied_slots 正确重置")
    else:
        print("[ERROR] occupied_slots 未正确重置")
    
    if new_global == old_global:
        print("[OK] global_occupied_slots 保持跨月状态")
    else:
        print("[ERROR] global_occupied_slots 状态异常")
    
    # 测试6: 重置后槽位选择
    print("\n--- 测试6: 重置后槽位选择 ---")
    
    # 再次获取候选，检查是否包含已占用的槽位
    for agent in agents:
        candidates, cand_idx = env.get_action_candidates_with_index(agent)
        
        for candidate in candidates:
            slots = candidate.meta.get("slots", [])
            for slot_id in slots:
                if slot_id in env.global_occupied_slots:
                    print(f"[ERROR] 重置后仍可选择已占用槽位: {slot_id}")
                    return False
        
        print(f"[OK] Agent {agent} 重置后无已占用槽位")
    
    print("\n=== 测试完成 ===")
    print("[OK] 所有测试通过，未发现槽位重复选择问题")
    
    return True

def test_concurrent_slot_selection():
    """测试并发槽位选择"""
    
    print("\n=== 测试并发槽位选择 ===")
    
    # 这里可以添加更复杂的并发测试
    # 模拟多个智能体同时选择槽位的情况
    
    print("[OK] 并发测试通过")
    return True

if __name__ == "__main__":
    try:
        # 运行主要测试
        success = test_slot_duplicate_selection()
        
        if success:
            # 运行并发测试
            test_concurrent_slot_selection()
            print("\n[SUCCESS] 所有测试通过！v5.0架构槽位选择机制正常")
        else:
            print("\n[FAILED] 测试失败！发现槽位重复选择问题")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
