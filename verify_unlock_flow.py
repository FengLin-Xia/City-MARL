#!/usr/bin/env python3
"""验证解锁流程是否正常工作"""

import json
from typing import Set

def simulate_get_unlocked_actions(current_month: int, config: dict, agent: str = "IND") -> Set[int]:
    """模拟 _get_unlocked_actions 方法"""
    print(f"\n{'='*60}")
    print(f"模拟 _get_unlocked_actions (agent={agent}, month={current_month})")
    print(f"{'='*60}")
    
    try:
        unlock_config = config.get("action_unlocks", {})
        rules = unlock_config.get("rules", [])
        print(f"\n[1] 读取配置: rules数量={len(rules)}")
        
        unlocked_actions = set()
        
        # 遍历规则
        for rule in rules:
            rule_agent = rule.get("agent")
            print(f"\n[2] 检查规则: agent={rule_agent}, after_month={rule.get('after_month')}")
            
            if rule_agent and rule_agent != agent:
                print(f"   -> 跳过（agent不匹配）")
                continue
            
            # 检查时间条件
            after_month = rule.get("after_month", 0)
            print(f"   -> 时间检查: current_month={current_month} < after_month={after_month}? {current_month < after_month}")
            
            if current_month < after_month:
                print(f"   -> 跳过（时间条件不满足）")
                continue
            
            # 检查预算条件（如果存在）
            passed = True
            if "budget_threshold" in rule:
                budget_threshold = rule["budget_threshold"]
                print(f"   -> 预算检查: budget_threshold={budget_threshold}")
                # 这里简化，假设预算足够
                passed = True
                
            if "total_budget_threshold" in rule:
                total_threshold = rule["total_budget_threshold"]
                print(f"   -> 总预算检查: total_threshold={total_threshold}")
                # 这里简化，假设预算足够
                passed = True
            
            # 如果条件满足，收集解锁动作
            if passed:
                unlock_action_ids = rule.get("unlock_action_ids", [])
                unlocked_actions.update(unlock_action_ids)
                print(f"   -> ✓ 条件满足，解锁动作: {unlock_action_ids}")
        
        # 检查是否有针对该智能体的解锁规则
        has_agent_rules = any(
            rule.get("agent") == agent for rule in rules
        )
        print(f"\n[3] 是否有针对{agent}的规则: {has_agent_rules}")
        
        # 计算基础动作
        agent_config = config["agents"]["defs"].get(agent, {})
        all_actions = set(agent_config.get("action_ids", []))
        print(f"\n[4] 所有配置的动作: {sorted(all_actions)}")
        
        # 被规则标记为"需要解锁"的动作集合
        locked_actions = set()
        for rule in rules:
            rule_agent = rule.get("agent")
            if rule_agent and rule_agent != agent:
                continue
            unlock_action_ids = rule.get("unlock_action_ids", [])
            locked_actions.update(unlock_action_ids)
        print(f"    需要解锁的动作: {sorted(locked_actions)}")
        
        basic_actions = all_actions - locked_actions
        print(f"    基础动作（all - locked）: {sorted(basic_actions)}")
        print(f"    已解锁的动作（本月）: {sorted(unlocked_actions)}")
        
        if has_agent_rules:
            final_actions = basic_actions | unlocked_actions
            print(f"\n[5] 最终返回: {sorted(final_actions)}")
            return final_actions
        else:
            print(f"\n[5] 最终返回（无规则）: {sorted(all_actions)}")
            return all_actions
            
    except Exception as e:
        print(f"\n[ERROR] 异常: {e}")
        import traceback
        traceback.print_exc()
        return set()

def simulate_enumerate_filter(unlocked_actions_set: Set[int], agent_action_ids: list) -> list:
    """模拟 _get_valid_types_for_point 中的过滤逻辑"""
    print(f"\n{'='*60}")
    print(f"模拟枚举器过滤逻辑")
    print(f"{'='*60}")
    
    action_ids = agent_action_ids.copy()
    print(f"\n[1] 原始action_ids: {action_ids}")
    print(f"    传入的unlocked_actions: {sorted(unlocked_actions_set)}")
    
    if unlocked_actions_set is not None:
        original_count = len(action_ids)
        action_ids = [aid for aid in action_ids if aid in unlocked_actions_set]
        print(f"\n[2] 过滤后action_ids: {action_ids}")
        print(f"    过滤前: {original_count}个动作")
        print(f"    过滤后: {len(action_ids)}个动作")
    else:
        print(f"\n[2] unlocked_actions为None，不进行过滤")
    
    return action_ids

if __name__ == "__main__":
    # 读取配置
    with open("configs/city_config_v5_0.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 测试不同月份
    test_months = [10, 14, 15, 16]
    
    for month in test_months:
        print(f"\n\n{'#'*60}")
        print(f"# 测试月份 {month}（当前配置：after_month=14）")
        print(f"{'#'*60}")
        
        # 模拟 _get_unlocked_actions
        final_actions = simulate_get_unlocked_actions(month, config)
        
        # 获取IND的action_ids
        ind_action_ids = config["agents"]["defs"]["IND"]["action_ids"]
        
        # 模拟枚举器过滤
        filtered_actions = simulate_enumerate_filter(final_actions, ind_action_ids)
        
        # 检查结果
        has_unlocked = {9, 10, 11}.issubset(set(filtered_actions))
        print(f"\n{'='*60}")
        print(f"最终结果检查:")
        print(f"  月份{month}: 动作9,10,11{'可用' if has_unlocked else '不可用'}")
        print(f"  过滤后的动作: {sorted(filtered_actions)}")
        
        if not has_unlocked and month >= 14:
            print(f"\n  [WARNING] 月份{month}应该解锁动作9,10,11，但实际未解锁！")

