#!/usr/bin/env python3
"""
简短测试河流过滤逻辑
只运行几个步骤验证EDU的河流过滤是否正确
"""

import sys
import os
import json
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment

def test_river_filter():
    """测试河流过滤逻辑"""
    print("=== 河流过滤逻辑测试 ===")
    
    # 加载配置
    config_path = "configs/city_config_v4_1.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    # 重置环境
    obs = env.reset()
    print(f"环境重置完成，初始观察类型: {type(obs)}")
    
    # 测试EDU agent的河流过滤
    print("\n=== 测试EDU agent河流过滤 ===")
    env.current_agent = 'EDU'
    
    # 获取候选槽位
    candidates = env._get_candidate_slots()
    print(f"EDU候选槽位数: {len(candidates)}")
    
    # 枚举动作
    actions, action_feats, mask = env.get_action_pool('EDU')
    print(f"EDU动作数: {len(actions)}")
    
    # 分析动作分布
    size_counts = {}
    other_side_counts = {}
    
    for action in actions:
        size = action.size
        size_counts[size] = size_counts.get(size, 0) + 1
        
        # 检查是否在对岸（使用环境中的对岸检测逻辑）
        try:
            # 获取河流信息
            rivers = env.v4_cfg.get('terrain_features', {}).get('rivers', [])
            coords = rivers[0].get('coordinates', []) if rivers else []
            if not coords:
                from envs.v4_1.city_env import load_river_coords
                coords = load_river_coords(env.v4_cfg)
            
            if coords:
                # 计算河流中心线
                y_coords = [point[1] for point in coords]
                center_y = sum(y_coords) / len(y_coords)
                
                # 获取EDU hub位置
                hubs = env.v4_cfg.get('city', {}).get('transport_hubs', [[125, 75], [112, 121]])
                edu_hub_y = hubs[1][1] if len(hubs) > 1 else hubs[0][1]
                
                # 检查动作的第一个槽位是否在对岸
                if action.footprint_slots:
                    first_slot_id = action.footprint_slots[0]
                    slot = env.slots.get(first_slot_id)
                    if slot is not None:
                        y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
                        is_other_side = (y > center_y) != (edu_hub_y > center_y)
                        if is_other_side:
                            other_side_counts[size] = other_side_counts.get(size, 0) + 1
        except Exception as e:
            print(f"对岸检测失败: {e}")
    
    print(f"动作尺寸分布: {size_counts}")
    print(f"对岸动作分布: {other_side_counts}")
    
    # 测试IND agent的河流过滤
    print("\n=== 测试IND agent河流过滤 ===")
    env.current_agent = 'IND'
    
    # 获取候选槽位
    candidates = env._get_candidate_slots()
    print(f"IND候选槽位数: {len(candidates)}")
    
    # 枚举动作
    actions, action_feats, mask = env.get_action_pool('IND')
    print(f"IND动作数: {len(actions)}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_river_filter()
