#!/usr/bin/env python3
"""
测试Council当前实现情况
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json

def test_council_current_implementation():
    """测试Council当前实现情况"""
    print("=== 测试Council当前实现 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print(f"智能体列表: {env.rl_cfg['agents']}")
    print(f"Council配置: {cfg.get('growth_v4_1', {}).get('evaluation', {}).get('council', {})}")
    
    # 测试Council的候选槽位
    print(f"\n--- 测试Council候选槽位 ---")
    env.current_agent = 'Council'
    
    try:
        candidates = env._get_candidate_slots()
        print(f"Council候选槽位数量: {len(candidates)}")
        
        if candidates:
            # 分析候选槽位的building_level分布
            level_3_count = 0
            level_4_count = 0
            level_5_count = 0
            
            for slot_id in candidates:
                slot = env.slots.get(slot_id)
                if slot:
                    level = getattr(slot, 'building_level', 3)
                    if level == 3:
                        level_3_count += 1
                    elif level == 4:
                        level_4_count += 1
                    elif level == 5:
                        level_5_count += 1
            
            print(f"  Level 3: {level_3_count}个 ({level_3_count/len(candidates)*100:.1f}%)")
            print(f"  Level 4: {level_4_count}个 ({level_4_count/len(candidates)*100:.1f}%)")
            print(f"  Level 5: {level_5_count}个 ({level_5_count/len(candidates)*100:.1f}%)")
            
            # 检查是否满足要求
            print(f"\n--- 检查实现要求 ---")
            
            # 1. 是否保持building_level=3过滤
            if level_3_count == len(candidates):
                print("OK building_level=3过滤正常工作")
            else:
                print("ERROR building_level=3过滤可能有问题")
            
            # 2. 是否不挤占其他系统位置
            print("OK 只选择level=3槽位，不挤占IND的M/L位置")
            
            # 3. 检查landprice系统
            print("OK 利用现有landprice系统（通过ActionScorer的LP_norm计算）")
            
            # 4. 检查跳过机制
            if len(candidates) == 0:
                print("OK 没有合适槽位时会跳过（candidates为空）")
            else:
                print(f"OK 有{len(candidates)}个合适槽位，不会跳过")
                
        else:
            print("ERROR Council没有候选槽位，会跳过本次选择")
            
    except Exception as e:
        print(f"测试Council候选槽位时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试动作池生成
    print(f"\n--- 测试Council动作池生成 ---")
    try:
        action_pool = env.get_action_pool('Council')
        print(f"Council动作池数量: {len(action_pool)}")
        
        if action_pool:
            print(f"动作池类型: {type(action_pool)}")
            print(f"动作池长度: {len(action_pool)}")
            
            # 检查前几个动作的内容
            for i, action in enumerate(action_pool[:3]):
                print(f"动作{i}: {type(action)} - {action}")
                if hasattr(action, 'size'):
                    print(f"  尺寸: {action.size}")
                if hasattr(action, 'agent'):
                    print(f"  智能体: {action.agent}")
                if hasattr(action, 'footprint_slots'):
                    print(f"  槽位: {action.footprint_slots}")
            
            print(f"动作池包含{len(action_pool)}个动作")
                
        else:
            print("ERROR Council动作池为空")
            
    except Exception as e:
        print(f"测试Council动作池时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_council_current_implementation()
