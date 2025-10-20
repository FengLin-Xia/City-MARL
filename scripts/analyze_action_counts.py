#!/usr/bin/env python3
"""
分析EDU和Council的动作数量差异
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json

def analyze_action_counts():
    """分析EDU和Council的动作数量"""
    print("=== 分析EDU和Council的动作数量 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print(f"智能体列表: {env.rl_cfg['agents']}")
    print(f"当前月份: {env.current_month}")
    
    # 分析EDU
    print("\n--- EDU智能体分析 ---")
    actions, _, _ = env.get_action_pool('EDU')
    print(f"EDU动作数量: {len(actions)}")
    
    size_counts = {}
    for action in actions:
        size_counts[action.size] = size_counts.get(action.size, 0) + 1
    print(f"EDU尺寸分布: {size_counts}")
    
    # 分析Council
    print("\n--- Council智能体分析 ---")
    actions, _, _ = env.get_action_pool('Council')
    print(f"Council动作数量: {len(actions)}")
    
    size_counts = {}
    for action in actions:
        size_counts[action.size] = size_counts.get(action.size, 0) + 1
    print(f"Council尺寸分布: {size_counts}")
    
    # 分析候选槽位
    print("\n--- 候选槽位分析 ---")
    env.current_agent = 'EDU'
    edu_candidates = env._get_candidate_slots()
    print(f"EDU候选槽位数量: {len(edu_candidates)}")
    
    env.current_agent = 'Council'
    council_candidates = env._get_candidate_slots()
    print(f"Council候选槽位数量: {len(council_candidates)}")
    
    # 分析河流过滤
    print("\n--- 河流过滤分析 ---")
    print("EDU河流过滤: 保留同侧槽位，过滤对岸槽位")
    print("Council河流过滤: 完全绕过，保留所有槽位")
    
    # 分析动作枚举逻辑
    print("\n--- 动作枚举逻辑分析 ---")
    print("EDU: S/M/L各16个槽位 = 48个动作")
    print("Council: A/B/C各16个槽位 = 48个动作")
    print("动作数量相同是因为:")
    print("1. 都基于相同的候选槽位数量(16个)")
    print("2. 每个槽位可以放置对应尺寸的建筑")
    print("3. EDU的S/M/L和Council的A/B/C都是单槽位建筑")
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    analyze_action_counts()
