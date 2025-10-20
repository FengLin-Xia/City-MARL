#!/usr/bin/env python3
"""
测试building_level的分布情况
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json

def test_building_level_distribution():
    """测试building_level的分布"""
    print("=== 测试building_level分布 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    # 统计building_level分布
    level_counts = {}
    for slot_id, slot in env.slots.items():
        level = getattr(slot, 'building_level', 3)
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print(f"\n--- building_level分布 ---")
    print(f"总槽位数: {len(env.slots)}")
    for level in sorted(level_counts.keys()):
        count = level_counts[level]
        pct = count / len(env.slots) * 100
        print(f"Level {level}: {count}个槽位 ({pct:.1f}%)")
        if level == 3:
            print(f"  -> 只能建S尺寸")
        elif level == 4:
            print(f"  -> 可建S/M尺寸")
        elif level == 5:
            print(f"  -> 可建S/M/L尺寸")
    
    # 检查Council候选槽位中level=3的数量
    print(f"\n--- Council候选槽位中level=3的分布 ---")
    env.current_agent = 'Council'
    try:
        candidates = env._get_candidate_slots()
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
        
        print(f"Council候选槽位总数: {len(candidates)}")
        print(f"  Level 3: {level_3_count}个 ({level_3_count/len(candidates)*100:.1f}%)")
        print(f"  Level 4: {level_4_count}个 ({level_4_count/len(candidates)*100:.1f}%)")
        print(f"  Level 5: {level_5_count}个 ({level_5_count/len(candidates)*100:.1f}%)")
        
        print(f"\n--- 建议 ---")
        if level_3_count > 0:
            print(f"OK 可以使用level=3过滤，有{level_3_count}个候选槽位")
            print(f"   这样Council只会选择IND不能放M/L的槽位，避免抢占")
        else:
            print(f"WARNING level=3槽位数量为0，需要重新考虑过滤策略")
            
    except Exception as e:
        print(f"测试Council候选槽位时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_building_level_distribution()
