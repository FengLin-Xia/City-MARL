#!/usr/bin/env python3
"""
检查动作池中A/B/C的实际可用性
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.v4_1.city_env import CityEnvironment
from solvers.v4_1.rl_selector import RLPolicySelector
from logic.v4_enumeration import ActionEnumerator

def debug_action_pool_abc():
    """检查动作池中A/B/C的可用性"""
    print("=== 检查动作池中A/B/C的实际可用性 ===")
    
    # 加载配置
    config_path = "configs/city_config_v4_1.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建环境
    env = CityEnvironment(config)
    obs = env.reset()
    print(f"环境初始化完成，观察空间: {type(obs)}")
    
    # 创建RL选择器
    selector = RLPolicySelector(config)
    print(f"RL选择器初始化完成")
    
    # 测试EDU agent的动作池
    print("\n=== 测试EDU agent动作池 ===")
    try:
        # 获取动作池
        action_pool = env.get_action_pool('EDU')
        print(f"EDU动作池大小: {len(action_pool)}")
        
        # 分析A/B/C动作
        print(f"动作池类型: {type(action_pool)}")
        print(f"动作池内容: {action_pool}")
        
        # 检查动作池结构
        if isinstance(action_pool, list) and len(action_pool) > 0:
            print(f"动作池第一个元素类型: {type(action_pool[0])}")
            if hasattr(action_pool[0], 'size'):
                # 这是Action对象列表
                abc_actions = []
                for action in action_pool:
                    if hasattr(action, 'size') and action.size in ['A', 'B', 'C']:
                        abc_actions.append(action)
                
                print(f"A/B/C动作数量: {len(abc_actions)}")
                
                if abc_actions:
                    print("\nA/B/C动作详情:")
                    for i, action in enumerate(abc_actions[:10]):  # 只显示前10个
                        print(f"  {i+1}. {action.size}型: 槽位={action.footprint_slots}, 成本={action.cost}, 分数={action.score:.3f}")
                        
                        # 检查是否在对岸
                        try:
                            is_other_side = selector._is_other_side_action(action)
                            print(f"     对岸检测: {is_other_side}")
                        except Exception as e:
                            print(f"     对岸检测失败: {e}")
                else:
                    print("没有找到A/B/C动作！")
                    
                # 按尺寸分组统计
                size_counts = {}
                for action in action_pool:
                    if hasattr(action, 'size'):
                        size = action.size
                        if size not in size_counts:
                            size_counts[size] = 0
                        size_counts[size] += 1
                        
                print(f"\n动作池尺寸分布: {size_counts}")
            else:
                print("动作池元素不是Action对象")
        else:
            print("动作池为空或格式不正确")
        
        # 检查A/B/C动作的过滤情况
        print(f"\n=== 检查A/B/C动作过滤 ===")
        
        # 获取所有候选槽位
        all_candidates = env._get_candidate_slots()
        print(f"EDU候选槽位总数: {len(all_candidates)}")
        
        # 检查对岸槽位
        other_side_slots = env._get_other_side_slots()
        print(f"对岸槽位数量: {len(other_side_slots)}")
        
        # 检查A/B/C在对岸槽位的可用性
        print(f"\n=== 检查A/B/C在对岸槽位的可用性 ===")
        for size in ['A', 'B', 'C']:
            print(f"\n{size}型建筑:")
            # 创建枚举器
            enumerator = ActionEnumerator(config)
            
            # 尝试在对岸槽位枚举A/B/C动作
            other_side_actions = []
            for slot_id in other_side_slots[:5]:  # 只检查前5个对岸槽位
                try:
                    # 枚举单个槽位的动作
                    slot_actions = enumerator._enumerate_single_slots([slot_id], 'EDU', [size])
                    for action in slot_actions:
                        if action.size == size:
                            other_side_actions.append(action)
                            print(f"  槽位 {slot_id}: 成本={action.cost}, 分数={action.score:.3f}")
                except Exception as e:
                    print(f"  槽位 {slot_id}: 枚举失败 - {e}")
            
            print(f"  {size}型对岸动作总数: {len(other_side_actions)}")
            
    except Exception as e:
        print(f"检查动作池失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_action_pool_abc()
