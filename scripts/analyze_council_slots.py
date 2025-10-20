#!/usr/bin/env python3
"""
分析Council为什么只有16个槽位
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json

def analyze_council_slots():
    """分析Council的槽位情况"""
    print("=== 分析Council槽位情况 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print(f"智能体列表: {env.rl_cfg['agents']}")
    print(f"当前月份: {env.current_month}")
    
    # 分析ring_candidates
    print("\n--- ring_candidates分析 ---")
    from enhanced_city_simulation_v4_0 import ring_candidates
    all_candidates = ring_candidates(
        env.slots, 
        env.hubs, 
        env.current_month, 
        env.v4_cfg.get('hubs', {}), 
        tol=1.0
    )
    print(f"ring_candidates返回: {len(all_candidates)}个槽位")
    print(f"ring_candidates槽位: {list(all_candidates)[:10]}...")
    
    # 分析EDU的候选槽位
    print("\n--- EDU候选槽位分析 ---")
    env.current_agent = 'EDU'
    edu_candidates = env._get_candidate_slots()
    print(f"EDU候选槽位数量: {len(edu_candidates)}")
    print(f"EDU候选槽位: {list(edu_candidates)[:10]}...")
    
    # 分析Council的候选槽位
    print("\n--- Council候选槽位分析 ---")
    env.current_agent = 'Council'
    council_candidates = env._get_candidate_slots()
    print(f"Council候选槽位数量: {len(council_candidates)}")
    print(f"Council候选槽位: {list(council_candidates)[:10]}...")
    
    # 分析河流过滤逻辑
    print("\n--- 河流过滤逻辑分析 ---")
    print("EDU: 基础16 + 对岸0 = 16个候选槽位")
    print("Council: 完全绕过河流过滤，但只有16个候选槽位")
    print("问题：为什么Council绕过河流过滤后还是只有16个槽位？")
    
    # 分析原因
    print("\n--- 原因分析 ---")
    print("1. ring_candidates函数基于距离约束")
    print("2. 第0个月R_curr=5.0，只返回距离hub≤5.0的槽位")
    print("3. Council虽然绕过河流过滤，但仍然受ring_candidates的距离约束")
    print("4. 所以Council和EDU都只有16个距离合理的槽位")
    
    # 验证距离约束
    print("\n--- 验证距离约束 ---")
    from enhanced_city_simulation_v4_0 import compute_R
    R_prev, R_curr = compute_R(env.current_month, env.v4_cfg.get('hubs', {}), True)
    print(f"当前月份{env.current_month}的距离约束: R_curr={R_curr}")
    
    # 分析所有槽位的距离
    print("\n--- 所有槽位距离分析 ---")
    from enhanced_city_simulation_v4_0 import min_dist_to_hubs
    edu_hub = env.hubs[1] if len(env.hubs) > 1 else env.hubs[0]
    
    distances = []
    for slot_id, slot in env.slots.items():
        x = float(getattr(slot, 'fx', getattr(slot, 'x', 0.0)))
        y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
        distance = min_dist_to_hubs(x, y, [edu_hub])
        distances.append((slot_id, distance))
    
    distances.sort(key=lambda x: x[1])
    print(f"距离最近的10个槽位:")
    for slot_id, dist in distances[:10]:
        print(f"  {slot_id}: {dist:.1f}")
    
    print(f"距离最远的10个槽位:")
    for slot_id, dist in distances[-10:]:
        print(f"  {slot_id}: {dist:.1f}")
    
    # 分析距离约束的影响
    valid_slots = [s for s in distances if s[1] <= R_curr + 1.0]
    print(f"\n距离≤{R_curr + 1.0}的槽位数量: {len(valid_slots)}")
    print(f"这解释了为什么Council只有16个候选槽位")
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    analyze_council_slots()
