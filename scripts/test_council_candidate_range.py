#!/usr/bin/env python3
"""
测试Council智能体的候选槽位范围
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json

def test_council_candidate_range():
    """测试Council的候选槽位范围"""
    print("=== 测试Council候选槽位范围 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print(f"智能体列表: {env.rl_cfg['agents']}")
    print(f"hub_components: {getattr(env, 'hub_components', 'Not set')}")
    print(f"hubs: {env.hubs}")
    
    # 检查hub_components的连通域
    if hasattr(env, 'hub_components') and len(env.hub_components) >= 2:
        print(f"\n--- 连通域分析 ---")
        print(f"IND hub (索引0): 连通域 {env.hub_components[0]}")
        print(f"EDU hub (索引1): 连通域 {env.hub_components[1]}")
        
        # 分析Council会使用哪个连通域
        agent_idx = env.rl_cfg['agents'].index('Council')
        print(f"Council在agents列表中的索引: {agent_idx}")
        
        # 根据代码逻辑，Council会使用EDU的连通域
        expected_comp = env.hub_components[1] if len(env.hub_components) > 1 else env.hub_components[0]
        print(f"Council将使用的连通域: {expected_comp} (EDU的连通域)")
        
        # 但是Council完全绕过河流过滤
        print(f"\n--- Council过滤逻辑 ---")
        print("根据代码逻辑:")
        print("1. Council使用EDU的连通域 (索引1)")
        print("2. 但是Council完全绕过河流过滤")
        print("3. 这意味着Council可以选择所有槽位，不受连通域限制")
        
    # 测试Council的候选槽位
    print(f"\n--- 测试Council候选槽位 ---")
    env.current_agent = 'Council'
    
    try:
        candidates = env._get_candidate_slots()
        print(f"Council候选槽位数量: {len(candidates)}")
        
        # 分析候选槽位的分布
        if candidates:
            # 检查候选槽位是否跨越两个连通域
            ind_side_count = 0
            edu_side_count = 0
            other_side_count = 0
            
            for slot_id in list(candidates)[:10]:  # 检查前10个槽位
                slot = env.slots.get(slot_id)
                if slot:
                    slot_comp = env._get_component_of_xy(slot.x, slot.y)
                    if slot_comp == env.hub_components[0]:
                        ind_side_count += 1
                    elif slot_comp == env.hub_components[1]:
                        edu_side_count += 1
                    else:
                        other_side_count += 1
                    
                    print(f"  槽位{slot_id}: 坐标({slot.x}, {slot.y}), 连通域{slot_comp}")
            
            print(f"\n--- 候选槽位分布分析 ---")
            print(f"IND侧槽位: {ind_side_count}")
            print(f"EDU侧槽位: {edu_side_count}")
            print(f"其他连通域: {other_side_count}")
            
            if ind_side_count > 0 and edu_side_count > 0:
                print("OK Council可以选择IND侧和EDU侧的槽位")
            elif edu_side_count > 0:
                print("WARNING Council主要选择EDU侧的槽位")
            else:
                print("UNKNOWN Council的候选槽位分布异常")
        else:
            print("ERROR Council没有候选槽位")
            
    except Exception as e:
        print(f"测试Council候选槽位时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_council_candidate_range()
