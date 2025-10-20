#!/usr/bin/env python3
"""
测试Council智能体动作枚举
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json

def test_council_action_enumeration():
    """测试Council智能体动作枚举"""
    print("=== 测试Council智能体动作枚举 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print(f"智能体列表: {env.rl_cfg['agents']}")
    print(f"当前智能体: {env.current_agent}")
    print(f"当前月份: {env.current_month}")
    
    # 测试Council智能体动作池生成
    print("\n--- 测试Council智能体动作池 ---")
    try:
        actions, action_feats, mask = env.get_action_pool('Council')
        print(f"Council动作池大小: {len(actions)}")
        
        if actions:
            # 统计不同尺寸的动作
            size_counts = {}
            for action in actions:
                size = action.size
                size_counts[size] = size_counts.get(size, 0) + 1
            
            print(f"Council动作尺寸分布: {size_counts}")
            
            # 显示前几个动作的详细信息
            print("前5个Council动作:")
            for i, action in enumerate(actions[:5]):
                print(f"  {i+1}. {action.agent}_{action.size} at {action.footprint_slots[0] if action.footprint_slots else 'N/A'}")
        else:
            print("Council没有可用动作")
            
    except Exception as e:
        print(f"Council动作池生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试EDU智能体动作池生成（应该不包含A/B/C）
    print("\n--- 测试EDU智能体动作池 ---")
    try:
        actions, action_feats, mask = env.get_action_pool('EDU')
        print(f"EDU动作池大小: {len(actions)}")
        
        if actions:
            # 统计不同尺寸的动作
            size_counts = {}
            for action in actions:
                size = action.size
                size_counts[size] = size_counts.get(size, 0) + 1
            
            print(f"EDU动作尺寸分布: {size_counts}")
            
            # 检查是否包含A/B/C
            abc_actions = [a for a in actions if a.size in ['A', 'B', 'C']]
            if abc_actions:
                print(f"警告：EDU包含A/B/C动作: {len(abc_actions)}个")
            else:
                print("正确：EDU不包含A/B/C动作")
                
        else:
            print("EDU没有可用动作")
            
    except Exception as e:
        print(f"EDU动作池生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试IND智能体动作池生成
    print("\n--- 测试IND智能体动作池 ---")
    try:
        actions, action_feats, mask = env.get_action_pool('IND')
        print(f"IND动作池大小: {len(actions)}")
        
        if actions:
            # 统计不同尺寸的动作
            size_counts = {}
            for action in actions:
                size = action.size
                size_counts[size] = size_counts.get(size, 0) + 1
            
            print(f"IND动作尺寸分布: {size_counts}")
            
    except Exception as e:
        print(f"IND动作池生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_council_action_enumeration()
