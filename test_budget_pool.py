#!/usr/bin/env python3
"""
测试预算池实现

验证共享预算池功能是否正常工作
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def test_budget_pool():
    """测试预算池功能"""
    print("=" * 80)
    print("测试预算池功能")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 检查预算池状态
        pool_status = env.budget_pool_manager.get_pool_status()
        print(f"\n   预算池状态:")
        for pool_name, status in pool_status.items():
            print(f"   {pool_name}: 总预算={status['total_budget']}, 剩余={status['remaining_budget']}, 成员={status['members']}")
        
        # 重置环境
        state = env.reset()
        print(f"\n   重置后状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 检查重置后的预算池状态
        pool_status = env.budget_pool_manager.get_pool_status()
        print(f"\n   重置后预算池状态:")
        for pool_name, status in pool_status.items():
            print(f"   {pool_name}: 剩余={status['remaining_budget']}")
        
        # 测试EDU和COUNCIL的预算检查
        print(f"\n   测试预算检查:")
        
        # 检查EDU预算
        edu_can_afford_650 = env.budget_pool_manager.can_afford("EDU", 650)
        print(f"   EDU能否支付650: {edu_can_afford_650}")
        
        # 检查COUNCIL预算
        council_can_afford_570 = env.budget_pool_manager.can_afford("COUNCIL", 570)
        print(f"   COUNCIL能否支付570: {council_can_afford_570}")
        
        # 测试预算扣除
        print(f"\n   测试预算扣除:")
        
        # EDU扣除650
        edu_deduct_success = env.budget_pool_manager.deduct("EDU", 650)
        print(f"   EDU扣除650: {edu_deduct_success}")
        
        # 检查剩余预算
        remaining_after_edu = env.budget_pool_manager.get_remaining_budget("EDU")
        print(f"   EDU扣除后剩余预算: {remaining_after_edu}")
        
        # COUNCIL扣除570
        council_deduct_success = env.budget_pool_manager.deduct("COUNCIL", 570)
        print(f"   COUNCIL扣除570: {council_deduct_success}")
        
        # 检查剩余预算
        remaining_after_council = env.budget_pool_manager.get_remaining_budget("COUNCIL")
        print(f"   COUNCIL扣除后剩余预算: {remaining_after_council}")
        
        # 验证共享预算池
        print(f"\n   验证共享预算池:")
        print(f"   EDU和COUNCIL应该共享同一个预算池")
        print(f"   EDU剩余预算: {env.budget_pool_manager.get_remaining_budget('EDU')}")
        print(f"   COUNCIL剩余预算: {env.budget_pool_manager.get_remaining_budget('COUNCIL')}")
        
        # 检查是否相等
        edu_remaining = env.budget_pool_manager.get_remaining_budget("EDU")
        council_remaining = env.budget_pool_manager.get_remaining_budget("COUNCIL")
        
        if edu_remaining == council_remaining:
            print(f"   [PASS] EDU和COUNCIL共享预算池: {edu_remaining}")
        else:
            print(f"   [FAIL] EDU和COUNCIL预算不一致: EDU={edu_remaining}, COUNCIL={council_remaining}")
        
        # 测试预算不足的情况
        print(f"\n   测试预算不足:")
        large_cost = 50000  # 超过总预算
        can_afford_large = env.budget_pool_manager.can_afford("EDU", large_cost)
        print(f"   EDU能否支付50000: {can_afford_large}")
        
        if not can_afford_large:
            print(f"   [PASS] 预算不足检查正常")
        else:
            print(f"   [FAIL] 预算不足检查异常")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_budget_pool()

