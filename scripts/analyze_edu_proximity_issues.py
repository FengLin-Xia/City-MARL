#!/usr/bin/env python3
"""
分析EDU邻近性问题
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json
import numpy as np

def analyze_edu_proximity_issues():
    """分析EDU邻近性问题"""
    print("=== 分析EDU邻近性问题 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print(f"邻近性配置: {cfg.get('proximity_constraint', {})}")
    print(f"EDU评估配置: {cfg.get('evaluation', {})}")
    
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
    
    # 分析EDU的奖励配置
    print(f"\n--- EDU奖励配置分析 ---")
    evaluation_cfg = cfg.get('evaluation', {})
    
    print(f"邻近性奖励: {evaluation_cfg.get('proximity_reward', 0)}")
    print(f"邻近性阈值: {evaluation_cfg.get('proximity_threshold', 0)}")
    print(f"距离惩罚系数: {evaluation_cfg.get('distance_penalty_coef', 0)}")
    
    # 检查EDU的尺寸奖励
    size_bonus = evaluation_cfg.get('size_bonus', {})
    print(f"EDU尺寸奖励: {size_bonus}")
    
    # 检查EDU的邻近性缩放
    proximity_scale = evaluation_cfg.get('proximity_scale_EDU_by_size', {})
    if proximity_scale:
        print(f"EDU邻近性缩放: {proximity_scale}")
    else:
        print("WARNING: 没有EDU邻近性缩放配置")
    
    # 分析问题
    print(f"\n--- 问题分析 ---")
    
    # 1. 邻近性约束问题
    proximity_cfg = cfg.get('proximity_constraint', {})
    if proximity_cfg.get('enabled', False):
        max_dist = proximity_cfg.get('max_distance', 10.0)
        apply_after = proximity_cfg.get('apply_after_month', 1)
        print(f"1. 邻近性约束: 启用, max_distance={max_dist}, apply_after_month={apply_after}")
        
        if max_dist < 15.0:
            print("   WARNING: max_distance可能太小，限制EDU扩展")
        if apply_after > 0:
            print("   WARNING: apply_after_month>0，早期EDU可能没有邻近性引导")
    else:
        print("1. 邻近性约束: 未启用")
        print("   ERROR: EDU没有邻近性引导，可能分布分散")
    
    # 2. 地价场跟随问题
    proximity_reward = evaluation_cfg.get('proximity_reward', 0)
    if proximity_reward < 100:
        print(f"2. 邻近性奖励: {proximity_reward} (可能太小)")
        print("   WARNING: 邻近性奖励不足，EDU可能不重视邻近性")
    else:
        print(f"2. 邻近性奖励: {proximity_reward} (合理)")
    
    # 3. 尺寸奖励问题
    if not size_bonus or all(v == 0 for v in size_bonus.values()):
        print("3. 尺寸奖励: 全部为0")
        print("   WARNING: EDU没有尺寸奖励，可能不重视地价场")
    else:
        print(f"3. 尺寸奖励: {size_bonus}")
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    analyze_edu_proximity_issues()
