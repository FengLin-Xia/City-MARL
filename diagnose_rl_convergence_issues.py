#!/usr/bin/env python3
"""
诊断RL模型收敛问题的三个关键方面
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set
import os

def check_action_pool_distribution():
    """检查问题1：M型和L型建筑是否真的出现在动作池中？"""
    
    print("=" * 60)
    print("问题1：检查M型和L型建筑是否出现在动作池中")
    print("=" * 60)
    
    # 模拟槽位分布
    total_slots = 1000
    level_distribution = {
        3: 800,  # 80%等级3
        4: 150,  # 15%等级4  
        5: 50    # 5%等级5
    }
    
    print(f"总槽位数: {total_slots}")
    print(f"等级分布: {level_distribution}")
    
    # 计算每种尺寸的建筑数量
    s_size_count = sum(level_distribution.values())  # 所有等级都可以建S型
    m_size_count = level_distribution[4] + level_distribution[5]  # 等级4和5可以建M型
    l_size_count = level_distribution[5]  # 只有等级5可以建L型
    
    print(f"\n动作池分布:")
    print(f"S型建筑: {s_size_count} ({s_size_count/total_slots*100:.1f}%)")
    print(f"M型建筑: {m_size_count} ({m_size_count/total_slots*100:.1f}%)")
    print(f"L型建筑: {l_size_count} ({l_size_count/total_slots*100:.1f}%)")
    
    # 结论
    if m_size_count < s_size_count * 0.1:  # M型少于S型的10%
        print(f"\nX 问题发现: M型建筑数量过少 ({m_size_count/total_slots*100:.1f}%)")
        print("   建议: 增加等级4和5的槽位比例")
    else:
        print(f"\nOK M型建筑数量合理")
    
    if l_size_count < s_size_count * 0.05:  # L型少于S型的5%
        print(f"X 问题发现: L型建筑数量过少 ({l_size_count/total_slots*100:.1f}%)")
        print("   建议: 增加等级5的槽位比例")
    else:
        print(f"OK L型建筑数量合理")
    
    return {
        's_count': s_size_count,
        'm_count': m_size_count, 
        'l_count': l_size_count,
        'total': total_slots
    }

def check_reward_differences():
    """检查问题2：不同尺寸建筑的奖励差异是否足够大？"""
    
    print("\n" + "=" * 60)
    print("问题2：检查不同尺寸建筑的奖励差异")
    print("=" * 60)
    
    # 从v4_enumeration.py中提取的参数
    building_params = {
        'EDU': {
            'S': {'cost': 1.2+0.4, 'reward_base': 60*0.08, 'prestige': 0.2},
            'M': {'cost': 2.8+0.9, 'reward_base': 120*0.08, 'prestige': 0.6},
            'L': {'cost': 5.5+1.6, 'reward_base': 240*0.08, 'prestige': 1.0}
        },
        'IND': {
            'S': {'cost': 1.0+0.5, 'reward_base': 80*12*0.85/1000, 'prestige': 0.2},
            'M': {'cost': 2.2+1.0, 'reward_base': 200*12*0.85/1000, 'prestige': 0.1},
            'L': {'cost': 4.5+2.0, 'reward_base': 500*12*0.85/1000, 'prestige': -0.1}
        }
    }
    
    print("建筑参数对比:")
    for agent in ['EDU', 'IND']:
        print(f"\n{agent}建筑:")
        for size in ['S', 'M', 'L']:
            params = building_params[agent][size]
            print(f"  {size}型: 成本={params['cost']:.2f}, 收益基础={params['reward_base']:.3f}, 声望={params['prestige']}")
    
    # 计算奖励差异
    print(f"\n奖励差异分析:")
    
    for agent in ['EDU', 'IND']:
        print(f"\n{agent}建筑:")
        sizes = ['S', 'M', 'L']
        rewards = [building_params[agent][size]['reward_base'] for size in sizes]
        costs = [building_params[agent][size]['cost'] for size in sizes]
        prestiges = [building_params[agent][size]['prestige'] for size in sizes]
        
        # 计算效率 (收益/成本)
        efficiencies = [r/c if c > 0 else 0 for r, c in zip(rewards, costs)]
        
        print(f"  收益范围: {min(rewards):.3f} - {max(rewards):.3f}")
        print(f"  成本范围: {min(costs):.2f} - {max(costs):.2f}")
        print(f"  效率范围: {min(efficiencies):.3f} - {max(efficiencies):.3f}")
        
        # 检查差异是否足够大
        reward_range = max(rewards) - min(rewards)
        cost_range = max(costs) - min(costs)
        efficiency_range = max(efficiencies) - min(efficiencies)
        
        print(f"  收益差异: {reward_range:.3f}")
        print(f"  成本差异: {cost_range:.2f}")
        print(f"  效率差异: {efficiency_range:.3f}")
        
        if efficiency_range < 0.1:  # 效率差异小于0.1
            print(f"  X 问题发现: 效率差异过小 ({efficiency_range:.3f})")
            print(f"     建议: 增大不同尺寸建筑的收益差异")
        else:
            print(f"  OK 效率差异合理")
    
    return building_params

def check_state_encoding():
    """检查问题3：模型的状态编码是否包含了用地大小和智能体类型信息？"""
    
    print("\n" + "=" * 60)
    print("问题3：检查状态编码信息")
    print("=" * 60)
    
    # 分析当前的状态编码
    current_state_features = [
        "动作数量",
        "平均得分", 
        "得分标准差",
        "平均成本",
        "平均奖励",
        "平均声望",
        "最高得分",
        "最低得分", 
        "最高成本",
        "最低成本",
        "随机特征1",
        "随机特征2", 
        "随机特征3",
        "得分变异系数",
        "成本变异系数"
    ]
    
    print("当前状态编码包含的特征:")
    for i, feature in enumerate(current_state_features, 1):
        print(f"  {i:2d}. {feature}")
    
    # 检查关键信息
    print(f"\n关键信息检查:")
    
    # 检查是否包含智能体类型信息
    agent_type_info = any("智能体" in feature or "agent" in feature.lower() for feature in current_state_features)
    if not agent_type_info:
        print("X 问题发现: 状态编码中缺少智能体类型信息")
        print("   当前智能体类型信息未明确编码到状态中")
        print("   建议: 添加当前智能体类型特征")
    else:
        print("OK 包含智能体类型信息")
    
    # 检查是否包含用地大小信息
    land_size_info = any("用地" in feature or "land" in feature.lower() or "slot" in feature.lower() for feature in current_state_features)
    if not land_size_info:
        print("X 问题发现: 状态编码中缺少用地大小信息")
        print("   用地大小信息未明确编码到状态中")
        print("   建议: 添加用地大小相关特征")
    else:
        print("OK 包含用地大小信息")
    
    # 检查是否包含建筑尺寸分布信息
    size_distribution_info = any("尺寸" in feature or "size" in feature.lower() or "分布" in feature for feature in current_state_features)
    if not size_distribution_info:
        print("X 问题发现: 状态编码中缺少建筑尺寸分布信息")
        print("   无法区分当前动作池中不同尺寸建筑的比例")
        print("   建议: 添加建筑尺寸分布特征")
    else:
        print("OK 包含建筑尺寸分布信息")
    
    # 建议的状态编码改进
    print(f"\n建议的状态编码改进:")
    suggested_features = [
        "当前智能体类型 (EDU=0, IND=1)",
        "可用槽位总数",
        "S型建筑数量",
        "M型建筑数量", 
        "L型建筑数量",
        "S型建筑比例",
        "M型建筑比例",
        "L型建筑比例",
        "平均用地面积",
        "最大可用用地面积",
        "当前月份",
        "预算状态",
        "已有建筑密度",
        "距离hub的平均距离",
        "地价水平"
    ]
    
    for i, feature in enumerate(suggested_features, 1):
        print(f"  {i:2d}. {feature}")
    
    return {
        'current_features': current_state_features,
        'suggested_features': suggested_features,
        'missing_agent_info': not agent_type_info,
        'missing_land_info': not land_size_info,
        'missing_size_info': not size_distribution_info
    }

def generate_fix_recommendations():
    """生成修复建议"""
    
    print("\n" + "=" * 60)
    print("修复建议总结")
    print("=" * 60)
    
    recommendations = {
        'immediate_fixes': [
            "1. 增加探索率从10%到30-50%",
            "2. 提高温度参数从1.2到2.0", 
            "3. 添加探索衰减机制",
            "4. 给M/L型建筑探索奖励"
        ],
        'action_pool_fixes': [
            "1. 调整槽位建筑等级分布",
            "2. 增加等级4和5的槽位比例",
            "3. 限制S型建筑在动作池中的数量",
            "4. 确保每种尺寸都有合理代表"
        ],
        'reward_fixes': [
            "1. 增大不同尺寸建筑的奖励差异",
            "2. 给大建筑额外奖励",
            "3. 平衡短期和长期收益",
            "4. 添加尺寸适应性奖励"
        ],
        'state_encoding_fixes': [
            "1. 添加当前智能体类型特征",
            "2. 添加用地大小相关特征", 
            "3. 添加建筑尺寸分布特征",
            "4. 添加空间和环境特征"
        ]
    }
    
    for category, fixes in recommendations.items():
        print(f"\n{category.upper()}:")
        for fix in fixes:
            print(f"  {fix}")
    
    return recommendations

def main():
    """主函数"""
    print("RL模型收敛问题诊断")
    print("检查三个关键问题:")
    print("1. M型和L型建筑是否出现在动作池中？")
    print("2. 不同尺寸建筑的奖励差异是否足够大？") 
    print("3. 状态编码是否包含用地大小和智能体类型信息？")
    
    # 检查问题1
    action_pool_dist = check_action_pool_distribution()
    
    # 检查问题2
    reward_params = check_reward_differences()
    
    # 检查问题3
    state_encoding_info = check_state_encoding()
    
    # 生成修复建议
    recommendations = generate_fix_recommendations()
    
    # 总结
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    
    issues_found = []
    
    if action_pool_dist['m_count'] < action_pool_dist['s_count'] * 0.1:
        issues_found.append("动作池中M型建筑数量过少")
    
    if action_pool_dist['l_count'] < action_pool_dist['s_count'] * 0.05:
        issues_found.append("动作池中L型建筑数量过少")
    
    if state_encoding_info['missing_agent_info']:
        issues_found.append("状态编码缺少智能体类型信息")
    
    if state_encoding_info['missing_land_info']:
        issues_found.append("状态编码缺少用地大小信息")
    
    if state_encoding_info['missing_size_info']:
        issues_found.append("状态编码缺少建筑尺寸分布信息")
    
    if issues_found:
        print("发现的问题:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    else:
        print("未发现明显问题，需要进一步调试")
    
    print(f"\n建议优先修复顺序:")
    print("1. 立即修复: 增加探索率和温度参数")
    print("2. 动作池修复: 调整建筑等级分布") 
    print("3. 状态编码修复: 添加关键特征")
    print("4. 奖励结构修复: 增大差异")

if __name__ == "__main__":
    main()
