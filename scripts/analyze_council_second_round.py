#!/usr/bin/env python3
"""
分析Council第二轮次选择的槽位情况
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json

def analyze_council_second_round():
    """分析Council第二轮次选择的槽位情况"""
    print("=== 分析Council第二轮次选择 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print(f"智能体列表: {env.rl_cfg['agents']}")
    print(f"当前月份: {env.current_month}")
    print(f"当前智能体: {env.current_agent}")
    
    # 模拟第一轮：IND选择
    print("\n--- 第一轮：IND选择 ---")
    env.current_agent = 'IND'
    ind_actions, _, _ = env.get_action_pool('IND')
    print(f"IND动作数量: {len(ind_actions)}")
    
    # 模拟IND选择一些动作（占用槽位）
    if ind_actions:
        # 选择前2个IND动作
        selected_ind_actions = ind_actions[:2]
        print(f"IND选择了{len(selected_ind_actions)}个动作")
        for i, action in enumerate(selected_ind_actions):
            print(f"  IND动作{i+1}: {action.agent}_{action.size} at {action.footprint_slots}")
            # 模拟占用槽位
            for slot_id in action.footprint_slots:
                if slot_id in env.slots:
                    env.slots[slot_id].occupied_by = 'IND'
                    env.slots[slot_id].reserved_in_turn = True
    
    # 模拟第二轮：EDU选择
    print("\n--- 第二轮：EDU选择 ---")
    env.current_agent = 'EDU'
    edu_actions, _, _ = env.get_action_pool('EDU')
    print(f"EDU动作数量: {len(edu_actions)}")
    
    # 模拟EDU选择一些动作（占用槽位）
    if edu_actions:
        # 选择前3个EDU动作
        selected_edu_actions = edu_actions[:3]
        print(f"EDU选择了{len(selected_edu_actions)}个动作")
        for i, action in enumerate(selected_edu_actions):
            print(f"  EDU动作{i+1}: {action.agent}_{action.size} at {action.footprint_slots}")
            # 模拟占用槽位
            for slot_id in action.footprint_slots:
                if slot_id in env.slots:
                    env.slots[slot_id].occupied_by = 'EDU'
                    env.slots[slot_id].reserved_in_turn = True
    
    # 模拟第三轮：Council选择
    print("\n--- 第三轮：Council选择 ---")
    env.current_agent = 'Council'
    council_actions, _, _ = env.get_action_pool('Council')
    print(f"Council动作数量: {len(council_actions)}")
    
    # 分析Council的候选槽位
    print("\n--- Council候选槽位分析 ---")
    council_candidates = env._get_candidate_slots()
    print(f"Council候选槽位数量: {len(council_candidates)}")
    print(f"Council候选槽位: {list(council_candidates)[:10]}...")
    
    # 分析已占用的槽位
    print("\n--- 已占用槽位分析 ---")
    occupied_slots = env._get_occupied_slots()
    print(f"已占用槽位数量: {len(occupied_slots)}")
    print(f"已占用槽位: {list(occupied_slots)[:10]}...")
    
    # 分析槽位占用情况
    print("\n--- 槽位占用情况分析 ---")
    ind_occupied = []
    edu_occupied = []
    council_occupied = []
    
    for slot_id, slot in env.slots.items():
        if hasattr(slot, 'occupied_by') and slot.occupied_by:
            if slot.occupied_by == 'IND':
                ind_occupied.append(slot_id)
            elif slot.occupied_by == 'EDU':
                edu_occupied.append(slot_id)
            elif slot.occupied_by == 'Council':
                council_occupied.append(slot_id)
    
    print(f"IND占用槽位: {len(ind_occupied)}个 - {ind_occupied}")
    print(f"EDU占用槽位: {len(edu_occupied)}个 - {edu_occupied}")
    print(f"Council占用槽位: {len(council_occupied)}个 - {council_occupied}")
    
    # 分析Council是否可以访问EDU和IND的槽位
    print("\n--- Council槽位访问分析 ---")
    print("Council可以访问的槽位类型:")
    print("1. 未被占用的槽位（主要选择）")
    print("2. 不能访问已被IND/EDU占用的槽位")
    
    # 检查Council候选槽位与已占用槽位的重叠
    overlap_with_ind = set(council_candidates) & set(ind_occupied)
    overlap_with_edu = set(council_candidates) & set(edu_occupied)
    
    print(f"Council候选槽位与IND占用槽位重叠: {len(overlap_with_ind)}个 - {overlap_with_ind}")
    print(f"Council候选槽位与EDU占用槽位重叠: {len(overlap_with_edu)}个 - {overlap_with_edu}")
    
    if overlap_with_ind or overlap_with_edu:
        print("WARNING: Council候选槽位包含已被占用的槽位！")
    else:
        print("OK: Council候选槽位不包含已被占用的槽位")
    
    # 分析Council的动作分布
    print("\n--- Council动作分布分析 ---")
    if council_actions:
        size_counts = {}
        for action in council_actions:
            size_counts[action.size] = size_counts.get(action.size, 0) + 1
        print(f"Council动作尺寸分布: {size_counts}")
        
        # 显示前几个动作的槽位信息
        print("前5个Council动作的槽位:")
        for i, action in enumerate(council_actions[:5]):
            print(f"  {i+1}. {action.agent}_{action.size} at {action.footprint_slots[0] if action.footprint_slots else 'N/A'}")
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    analyze_council_second_round()
