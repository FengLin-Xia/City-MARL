#!/usr/bin/env python3
"""
测试复杂奖励机制

验证v5.0中实现的复杂奖励机制是否正常工作。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts import EnvironmentState
from reward_terms.reward_manager import RewardManager
import json


def test_reward_mechanisms():
    """测试奖励机制"""
    print("=" * 60)
    print("测试v5.0复杂奖励机制")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建奖励管理器
    reward_manager = RewardManager(config)
    
    # 创建测试状态
    prev_state = EnvironmentState(
        month=0,
        land_prices={},
        buildings={"public": [], "industrial": []},
        budgets={"EDU": 5000, "IND": 5000},
        slots=[]
    )
    
    state = EnvironmentState(
        month=1,
        land_prices={},
        buildings={"public": [{"xy": [100, 100], "size": "S"}], "industrial": []},
        budgets={"EDU": 4500, "IND": 5000},
        slots=[]
    )
    
    # 测试不同动作的奖励
    test_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    print("\n测试奖励计算:")
    print("-" * 40)
    
    for action_id in test_actions:
        # 计算总奖励
        total_reward = reward_manager.compute_total_reward(prev_state, state, action_id)
        
        # 获取奖励分解
        breakdown = reward_manager.get_reward_breakdown(prev_state, state, action_id)
        
        print(f"动作 {action_id}: 总奖励 = {total_reward:.3f}")
        
        # 显示主要奖励项
        for term_name, reward in breakdown.items():
            if abs(reward) > 0.01:
                print(f"  {term_name}: {reward:.3f}")
    
    print("\n测试完成!")
    return True


def test_individual_reward_terms():
    """测试单个奖励项"""
    print("\n" + "=" * 60)
    print("测试单个奖励项")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建测试状态
    state = EnvironmentState(
        month=1,
        land_prices={},
        buildings={"public": [], "industrial": []},
        budgets={"EDU": 5000, "IND": 5000},
        slots=[]
    )
    
    # 测试NPV奖励
    from reward_terms.npv_reward import NPVRewardTerm
    npv_term = NPVRewardTerm(config)
    npv_reward = npv_term.compute(state, state, 0)
    print(f"NPV奖励: {npv_reward:.3f}")
    
    # 测试进度奖励
    from reward_terms.progress_reward import ProgressRewardTerm
    progress_term = ProgressRewardTerm(config)
    progress_reward = progress_term.compute(state, state, 0)
    print(f"进度奖励: {progress_reward:.3f}")
    
    # 测试协作奖励
    from reward_terms.cooperation_reward import CooperationRewardTerm
    cooperation_term = CooperationRewardTerm(config)
    cooperation_reward = cooperation_term.compute(state, state, 0)
    print(f"协作奖励: {cooperation_reward:.3f}")
    
    print("\n单个奖励项测试完成!")
    return True


if __name__ == "__main__":
    try:
        # 测试奖励机制
        test_reward_mechanisms()
        
        # 测试单个奖励项
        test_individual_reward_terms()
        
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
