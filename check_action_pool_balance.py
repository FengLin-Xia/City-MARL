#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.v4_enumeration import ActionEnumerator, SlotNode
from envs.v4_1.city_env import CityEnvironment
import json

def check_action_pool_balance():
    """检查动作池的平衡性"""
    
    print("=== 动作池平衡性检查 ===")
    
    # 创建环境
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    env = CityEnvironment(cfg)
    
    # 创建动作枚举器
    enumerator = ActionEnumerator(cfg['growth_v4_1']['enumeration'])
    
    # 获取槽位信息
    slots = env.slots
    
    # 统计槽位等级分布
    level_counts = {3: 0, 4: 0, 5: 0}
    for slot_id, slot in slots.items():
        # 从slots文件加载的数据结构中获取building_level
        # 根据slots_with_angle.txt的格式：x, y, angle, building_level
        if hasattr(slot, 'building_level'):
            level_counts[slot.building_level] += 1
        else:
            # 如果没有building_level属性，默认为3
            level_counts[3] += 1
    
    print(f"槽位等级分布:")
    print(f"Level 3 (只能建S): {level_counts[3]} ({level_counts[3]/len(slots)*100:.1f}%)")
    print(f"Level 4 (能建S/M): {level_counts[4]} ({level_counts[4]/len(slots)*100:.1f}%)")
    print(f"Level 5 (能建S/M/L): {level_counts[5]} ({level_counts[5]/len(slots)*100:.1f}%)")
    print(f"总槽位数: {len(slots)}")
    
    # 模拟生成动作池
    print(f"\n=== 模拟动作生成 ===")
    
    # 假设两个agent都有预算
    budgets = {"IND": 5000, "EDU": 3000}
    buildings = []  # 假设没有已建建筑
    
    try:
        actions = enumerator.enumerate_actions(
            slots=slots,
            budgets=budgets,
            buildings=buildings,
            month=1
        )
        
        # 统计动作分布
        size_counts = {'S': 0, 'M': 0, 'L': 0}
        agent_size_counts = {
            'IND': {'S': 0, 'M': 0, 'L': 0},
            'EDU': {'S': 0, 'M': 0, 'L': 0}
        }
        
        for action in actions:
            size_counts[action.size] += 1
            agent_size_counts[action.agent][action.size] += 1
        
        print(f"总动作数: {len(actions)}")
        print(f"动作尺寸分布:")
        for size, count in size_counts.items():
            percentage = count / len(actions) * 100 if actions else 0
            print(f"  {size}: {count} ({percentage:.1f}%)")
        
        print(f"\n分Agent动作分布:")
        for agent, sizes in agent_size_counts.items():
            print(f"  {agent}:")
            for size, count in sizes.items():
                percentage = count / len(actions) * 100 if actions else 0
                print(f"    {size}: {count} ({percentage:.1f}%)")
        
        # 分析不平衡程度
        s_ratio = size_counts['S'] / len(actions) if actions else 0
        m_ratio = size_counts['M'] / len(actions) if actions else 0
        l_ratio = size_counts['L'] / len(actions) if actions else 0
        
        print(f"\n=== 不平衡分析 ===")
        print(f"S型建筑占比: {s_ratio:.1%}")
        print(f"M型建筑占比: {m_ratio:.1%}")
        print(f"L型建筑占比: {l_ratio:.1%}")
        
        if s_ratio > 0.8:
            print("⚠️ 严重不平衡：S型建筑占比过高")
        elif s_ratio > 0.6:
            print("⚠️ 中度不平衡：S型建筑占比偏高")
        else:
            print("✅ 相对平衡")
            
        # 计算理论上的M/L型建筑可用性
        total_available_for_m = level_counts[4] + level_counts[5]
        total_available_for_l = level_counts[5]
        
        print(f"\n=== 理论可用性分析 ===")
        print(f"可建M型建筑的槽位: {total_available_for_m} ({total_available_for_m/len(slots)*100:.1f}%)")
        print(f"可建L型建筑的槽位: {total_available_for_l} ({total_available_for_l/len(slots)*100:.1f}%)")
        
        # 检查是否与动作分布一致
        if total_available_for_m / len(slots) > 0.2 and m_ratio < 0.1:
            print("❌ M型建筑动作生成不足：可用槽位多但生成的动作少")
        if total_available_for_l / len(slots) > 0.05 and l_ratio < 0.02:
            print("❌ L型建筑动作生成不足：可用槽位多但生成的动作少")
            
    except Exception as e:
        print(f"动作生成失败: {e}")
        print("可能的原因：预算不足或其他约束条件")

if __name__ == "__main__":
    check_action_pool_balance()
