#!/usr/bin/env python3
"""
修复EDU邻近性问题
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

def fix_edu_proximity_issues():
    """修复EDU邻近性问题"""
    print("=== 修复EDU邻近性问题 ===")
    
    # 读取当前配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    print("当前配置:")
    print(f"邻近性约束: {cfg.get('growth_v4_1', {}).get('proximity_constraint', {})}")
    print(f"评估配置: {cfg.get('growth_v4_1', {}).get('evaluation', {})}")
    
    # 修复方案
    print("\n=== 修复方案 ===")
    
    # 1. 增强邻近性约束
    proximity_fixes = {
        "enabled": True,
        "max_distance": 20.0,  # 从15.0增加到20.0，给EDU更多选择
        "apply_after_month": 0,  # 立即启用
        "min_candidates": 8,  # 从5增加到8，确保有足够候选
        "edu_specific": {
            "enabled": True,
            "max_distance": 25.0,  # EDU专用更大的邻近性范围
            "relax_after_month": 3  # 3个月后放宽约束
        }
    }
    
    # 2. 增强邻近性奖励
    evaluation_fixes = {
        "proximity_threshold": 20.0,  # 从15.0增加到20.0
        "proximity_reward": 1200.0,   # 从900.0增加到1200.0
        "distance_penalty_coef": 0.8, # 从0.6增加到0.8
        "size_bonus": {
            "S": 100,  # 小建筑奖励
            "M": 200,  # 中建筑奖励  
            "L": 300   # 大建筑奖励
        },
        "edu_land_price_following": {
            "enabled": True,
            "high_land_price_bonus": 500,  # 高地价奖励
            "low_land_price_penalty": -200, # 低地价惩罚
            "land_price_threshold": 0.7  # 地价阈值
        }
    }
    
    # 3. 修复对岸槽位过滤
    river_filter_fixes = {
        "edu_other_side": {
            "enabled": True,
            "max_distance_multiplier": 1.5,  # 放宽对岸距离限制
            "min_candidates": 5,  # 确保有对岸候选
            "relax_after_month": 2  # 2个月后放宽
        }
    }
    
    print("1. 邻近性约束修复:")
    for key, value in proximity_fixes.items():
        print(f"   {key}: {value}")
    
    print("\n2. 评估配置修复:")
    for key, value in evaluation_fixes.items():
        print(f"   {key}: {value}")
    
    print("\n3. 河流过滤修复:")
    for key, value in river_filter_fixes.items():
        print(f"   {key}: {value}")
    
    # 应用修复
    print("\n=== 应用修复 ===")
    
    # 更新邻近性约束
    if 'growth_v4_1' not in cfg:
        cfg['growth_v4_1'] = {}
    if 'proximity_constraint' not in cfg['growth_v4_1']:
        cfg['growth_v4_1']['proximity_constraint'] = {}
    cfg['growth_v4_1']['proximity_constraint'].update(proximity_fixes)
    
    # 更新评估配置
    if 'evaluation' not in cfg['growth_v4_1']:
        cfg['growth_v4_1']['evaluation'] = {}
    cfg['growth_v4_1']['evaluation'].update(evaluation_fixes)
    
    # 添加河流过滤配置
    if 'river_filter' not in cfg:
        cfg['river_filter'] = {}
    cfg['river_filter'].update(river_filter_fixes)
    
    # 保存修复后的配置
    with open('configs/city_config_v4_1_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    
    print("修复完成！配置已保存到: configs/city_config_v4_1_fixed.json")
    
    # 验证修复
    print("\n=== 验证修复 ===")
    print(f"邻近性约束: {cfg['growth_v4_1']['proximity_constraint']}")
    print(f"评估配置: {cfg['growth_v4_1']['evaluation']}")
    print(f"河流过滤: {cfg.get('river_filter', {})}")
    
    print("\n=== 预期效果 ===")
    print("1. EDU将有更大的邻近性范围 (20-25px)")
    print("2. 邻近性奖励增强 (1200 vs 900)")
    print("3. 尺寸奖励引导地价场跟随")
    print("4. 对岸槽位过滤放宽")
    print("5. EDU建筑将形成更紧密的集群")
    
    return cfg

if __name__ == "__main__":
    fix_edu_proximity_issues()
