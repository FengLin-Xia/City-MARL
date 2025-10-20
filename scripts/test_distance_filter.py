#!/usr/bin/env python3
"""
测试距离筛选逻辑
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json

def test_distance_filter():
    """测试距离筛选逻辑"""
    print("=== 测试距离筛选逻辑 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    # 设置当前智能体
    env.current_agent = 'EDU'
    
    print(f"当前月份: {env.current_month}")
    print(f"EDU hub位置: {env.hubs[1] if len(env.hubs) > 1 else env.hubs[0]}")
    
    # 获取基础候选槽位
    base_candidates = env._get_candidate_slots()
    print(f"基础候选槽位数量: {len(base_candidates)}")
    
    # 获取对岸槽位
    other_side_slots = env._get_other_side_slots()
    print(f"对岸槽位数量: {len(other_side_slots)}")
    
    # 筛选距离合理的对岸槽位
    valid_other_side_slots = env._filter_other_side_slots_by_distance(other_side_slots, base_candidates)
    print(f"距离合理的对岸槽位数量: {len(valid_other_side_slots)}")
    
    # 显示一些示例
    if valid_other_side_slots:
        print(f"有效对岸槽位示例: {valid_other_side_slots[:5]}")
        
        # 显示这些槽位的距离信息
        from enhanced_city_simulation_v4_0 import compute_R, min_dist_to_hubs
        R_prev, R_curr = compute_R(env.current_month, env.v4_cfg.get('hubs', {}), True)
        edu_hub = env.hubs[1] if len(env.hubs) > 1 else env.hubs[0]
        
        print(f"距离约束: R_curr={R_curr:.1f}")
        for slot_id in valid_other_side_slots[:3]:
            slot = env.slots.get(slot_id)
            if slot:
                x = float(getattr(slot, 'fx', getattr(slot, 'x', 0.0)))
                y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
                distance = min_dist_to_hubs(x, y, [edu_hub])
                print(f"  槽位 {slot_id}: 位置=({x:.1f}, {y:.1f}), 距离={distance:.1f}")
    
    print("=== 测试完成 ===")

if __name__ == "__main__":
    test_distance_filter()
