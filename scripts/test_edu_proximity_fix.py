#!/usr/bin/env python3
"""
测试EDU邻近性修复效果
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json
import numpy as np

def test_edu_proximity_fix():
    """测试EDU邻近性修复效果"""
    print("=== 测试EDU邻近性修复效果 ===")
    
    # 加载修复后的配置
    with open('configs/city_config_v4_1_fixed.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print("修复后的配置:")
    print(f"邻近性约束: {cfg['growth_v4_1']['proximity_constraint']}")
    print(f"评估配置: {cfg['growth_v4_1']['evaluation']}")
    print(f"河流过滤: {cfg.get('river_filter', {})}")
    
    # 模拟EDU在不同月份的行为
    test_months = [0, 1, 2, 3, 5, 8, 10]
    
    for test_month in test_months:
        print(f"\n--- 测试月份 {test_month} ---")
        env.current_month = test_month
        env.current_agent = 'EDU'
        
        try:
            # 获取候选槽位
            candidates = env._get_candidate_slots()
            print(f"EDU候选槽位数量: {len(candidates)}")
            
            if len(candidates) > 0:
                # 分析候选槽位的地价分布
                lp_values = []
                for slot_id in list(candidates)[:10]:  # 检查前10个
                    slot = env.slots.get(slot_id)
                    if slot:
                        # 获取地价
                        lp_provider = env._create_lp_provider()
                        lp_val = lp_provider(slot_id)
                        lp_values.append(lp_val)
                        print(f"  槽位{slot_id}: 地价={lp_val:.3f}")
                
                if lp_values:
                    print(f"  地价统计: 平均={np.mean(lp_values):.3f}, 标准差={np.std(lp_values):.3f}")
                    print(f"  地价范围: {min(lp_values):.3f} - {max(lp_values):.3f}")
                
                # 检查邻近性约束是否生效
                proximity_cfg = env.v4_cfg.get('proximity_constraint', {})
                if proximity_cfg.get('enabled', False) and test_month >= proximity_cfg.get('apply_after_month', 1):
                    print(f"  邻近性约束: 已启用 (max_distance={proximity_cfg.get('max_distance', 10.0)})")
                    
                    # 检查是否有EDU建筑作为参考
                    edu_buildings = env.buildings.get('public', [])
                    print(f"  参考EDU建筑数量: {len(edu_buildings)}")
                    
                    if len(edu_buildings) == 0:
                        print("  WARNING: 没有EDU建筑作为邻近性参考")
                    else:
                        print("  OK: 有EDU建筑作为邻近性参考")
                else:
                    print(f"  邻近性约束: 未启用 (月份{test_month} < apply_after_month)")
            else:
                print("  ERROR: EDU没有候选槽位")
                
        except Exception as e:
            print(f"  测试月份{test_month}时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 分析修复效果
    print(f"\n--- 修复效果分析 ---")
    
    # 1. 邻近性约束修复
    proximity_cfg = cfg['growth_v4_1']['proximity_constraint']
    print(f"1. 邻近性约束修复:")
    print(f"   max_distance: {proximity_cfg['max_distance']} (原来15.0)")
    print(f"   min_candidates: {proximity_cfg['min_candidates']} (原来5)")
    print(f"   edu_specific: {proximity_cfg.get('edu_specific', {})}")
    
    # 2. 邻近性奖励修复
    evaluation_cfg = cfg['growth_v4_1']['evaluation']
    print(f"\n2. 邻近性奖励修复:")
    print(f"   proximity_reward: {evaluation_cfg['proximity_reward']} (原来900.0)")
    print(f"   proximity_threshold: {evaluation_cfg['proximity_threshold']} (原来15.0)")
    print(f"   distance_penalty_coef: {evaluation_cfg['distance_penalty_coef']} (原来0.6)")
    
    # 3. 尺寸奖励修复
    size_bonus = evaluation_cfg.get('size_bonus', {})
    print(f"\n3. 尺寸奖励修复:")
    print(f"   size_bonus: {size_bonus} (原来全部为0)")
    
    # 4. 地价场跟随修复
    land_price_following = evaluation_cfg.get('edu_land_price_following', {})
    print(f"\n4. 地价场跟随修复:")
    print(f"   edu_land_price_following: {land_price_following}")
    
    # 5. 河流过滤修复
    river_filter = cfg.get('river_filter', {})
    print(f"\n5. 河流过滤修复:")
    print(f"   river_filter: {river_filter}")
    
    print(f"\n=== 预期改善 ===")
    print("1. EDU邻近性范围扩大: 15px -> 20px (普通) / 25px (EDU专用)")
    print("2. 邻近性奖励增强: 900 -> 1200")
    print("3. 尺寸奖励引导: S=100, M=200, L=300")
    print("4. 地价场跟随: 高地价奖励500, 低地价惩罚200")
    print("5. 对岸槽位过滤放宽: 距离倍数1.5x")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_edu_proximity_fix()
