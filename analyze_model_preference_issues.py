#!/usr/bin/env python3
"""
深入分析模型选择偏好的三个关键问题
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set
import os

def analyze_why_s_always_optimal():
    """问题1：为什么模型认为S型建筑总是最优？"""
    
    print("=" * 60)
    print("问题1：为什么模型认为S型建筑总是最优？")
    print("=" * 60)
    
    # 从之前的分析数据
    building_efficiency = {
        'EDU': {'S': 4.18, 'M': 4.30, 'L': 4.84},
        'IND': {'S': 0.68, 'M': 0.86, 'L': 1.15}
    }
    
    print("建筑效率对比（收益/成本）:")
    for agent in ['EDU', 'IND']:
        print(f"\n{agent}建筑:")
        for size in ['S', 'M', 'L']:
            print(f"  {size}型: 效率={building_efficiency[agent][size]:.3f}")
    
    # 分析效率差异
    print(f"\n效率差异分析:")
    for agent in ['EDU', 'IND']:
        sizes = ['S', 'M', 'L']
        efficiencies = [building_efficiency[agent][s] for s in sizes]
        
        # 计算相对优势
        s_advantage_over_m = efficiencies[0] / efficiencies[1] if efficiencies[1] > 0 else 0
        s_advantage_over_l = efficiencies[0] / efficiencies[2] if efficiencies[2] > 0 else 0
        
        print(f"{agent}:")
        print(f"  S型相对M型的优势: {s_advantage_over_m:.3f}")
        print(f"  S型相对L型的优势: {s_advantage_over_l:.3f}")
        
        if s_advantage_over_m > 0.95 and s_advantage_over_l > 0.95:
            print(f"  X 问题发现: S型效率优势不明显，但模型仍偏好S型")
        else:
            print(f"  OK S型确实有优势")
    
    # 分析可用性优势
    print(f"\n可用性分析:")
    availability = {'S': 100, 'M': 20, 'L': 5}  # 百分比
    print("不同尺寸建筑的可用性:")
    for size, avail in availability.items():
        print(f"  {size}型: {avail}%的槽位可以建造")
    
    # 计算综合优势
    print(f"\n综合优势分析:")
    for agent in ['EDU', 'IND']:
        print(f"{agent}:")
        for size in ['S', 'M', 'L']:
            efficiency = building_efficiency[agent][size]
            avail = availability[size] / 100.0
            combined_score = efficiency * avail  # 效率 × 可用性
            print(f"  {size}型: 效率={efficiency:.3f} × 可用性={avail:.2f} = 综合得分={combined_score:.3f}")
    
    return building_efficiency, availability

def analyze_ml_reward_signals():
    """问题2：M/L型建筑的奖励是否足够明显？"""
    
    print("\n" + "=" * 60)
    print("问题2：M/L型建筑的奖励是否足够明显？")
    print("=" * 60)
    
    # 从v4_enumeration.py提取的详细参数
    detailed_rewards = {
        'EDU': {
            'S': {'reward': 4.60, 'cost': 1.10, 'prestige': 0.15},
            'M': {'reward': 9.25, 'cost': 2.15, 'prestige': 0.50}, 
            'L': {'reward': 18.65, 'cost': 3.85, 'prestige': 0.85}
        },
        'IND': {
            'S': {'reward': 0.72, 'cost': 1.05, 'prestige': 0.08},
            'M': {'reward': 1.64, 'cost': 1.90, 'prestige': -0.08},
            'L': {'reward': 4.10, 'cost': 3.55, 'prestige': -0.34}
        }
    }
    
    print("详细奖励对比:")
    for agent in ['EDU', 'IND']:
        print(f"\n{agent}建筑:")
        for size in ['S', 'M', 'L']:
            data = detailed_rewards[agent][size]
            print(f"  {size}型: 收益={data['reward']:.2f}, 成本={data['cost']:.2f}, 声望={data['prestige']:.2f}")
    
    # 分析奖励增长模式
    print(f"\n奖励增长模式分析:")
    for agent in ['EDU', 'IND']:
        print(f"{agent}:")
        sizes = ['S', 'M', 'L']
        rewards = [detailed_rewards[agent][s]['reward'] for s in sizes]
        costs = [detailed_rewards[agent][s]['cost'] for s in sizes]
        
        # 计算增长倍数
        m_growth = rewards[1] / rewards[0] if rewards[0] > 0 else 0
        l_growth = rewards[2] / rewards[0] if rewards[0] > 0 else 0
        cost_m_growth = costs[1] / costs[0] if costs[0] > 0 else 0
        cost_l_growth = costs[2] / costs[0] if costs[0] > 0 else 0
        
        print(f"  M型相对S型: 收益增长{m_growth:.2f}倍, 成本增长{cost_m_growth:.2f}倍")
        print(f"  L型相对S型: 收益增长{l_growth:.2f}倍, 成本增长{cost_l_growth:.2f}倍")
        
        # 分析是否明显
        if m_growth < 1.5:
            print(f"  X 问题发现: M型收益增长不够明显 ({m_growth:.2f}倍)")
        else:
            print(f"  OK M型收益增长明显")
            
        if l_growth < 2.0:
            print(f"  X 问题发现: L型收益增长不够明显 ({l_growth:.2f}倍)")
        else:
            print(f"  OK L型收益增长明显")
    
    # 分析成本效益比
    print(f"\n成本效益比分析:")
    for agent in ['EDU', 'IND']:
        print(f"{agent}:")
        for size in ['S', 'M', 'L']:
            data = detailed_rewards[agent][size]
            cost_benefit = data['reward'] / data['cost'] if data['cost'] > 0 else 0
            print(f"  {size}型: 成本效益比={cost_benefit:.3f}")
    
    return detailed_rewards

def analyze_exploration_opportunities():
    """问题3：模型是否有足够的探索机会尝试不同尺寸？"""
    
    print("\n" + "=" * 60)
    print("问题3：模型是否有足够的探索机会尝试不同尺寸？")
    print("=" * 60)
    
    # 当前探索设置
    current_exploration = {
        'epsilon': 0.1,  # 10%随机探索
        'temperature': 1.2,
        'action_pool_distribution': {'S': 80, 'M': 20, 'L': 5}  # 百分比
    }
    
    print("当前探索设置:")
    print(f"  探索率(epsilon): {current_exploration['epsilon']*100:.1f}%")
    print(f"  温度参数: {current_exploration['temperature']}")
    print(f"  动作池分布: {current_exploration['action_pool_distribution']}")
    
    # 计算实际探索机会
    print(f"\n实际探索机会分析:")
    
    # 模拟1000个决策
    total_decisions = 1000
    epsilon = current_exploration['epsilon']
    
    # 随机探索次数
    random_explorations = int(total_decisions * epsilon)
    print(f"  总决策数: {total_decisions}")
    print(f"  随机探索次数: {random_explorations}")
    print(f"  策略利用次数: {total_decisions - random_explorations}")
    
    # 在随机探索中，M/L型建筑被选中的概率
    action_dist = current_exploration['action_pool_distribution']
    total_actions = sum(action_dist.values())
    
    print(f"\n随机探索中的选择概率:")
    for size, count in action_dist.items():
        prob = count / total_actions
        expected_selections = random_explorations * prob
        print(f"  {size}型建筑: 概率={prob:.3f}, 预期选择次数={expected_selections:.1f}")
    
    # 分析问题
    m_expected = random_explorations * (action_dist['M'] / total_actions)
    l_expected = random_explorations * (action_dist['L'] / total_actions)
    
    print(f"\n探索机会评估:")
    if m_expected < 50:  # M型预期选择少于50次
        print(f"  X 问题发现: M型建筑探索机会不足 (预期{m_expected:.1f}次)")
        print(f"     建议: 增加探索率或提高M型建筑在动作池中的比例")
    else:
        print(f"  OK M型建筑探索机会充足")
    
    if l_expected < 25:  # L型预期选择少于25次
        print(f"  X 问题发现: L型建筑探索机会不足 (预期{l_expected:.1f}次)")
        print(f"     建议: 增加探索率或提高L型建筑在动作池中的比例")
    else:
        print(f"  OK L型建筑探索机会充足")
    
    # 策略利用阶段的问题
    print(f"\n策略利用阶段分析:")
    print(f"  在{total_decisions - random_explorations}次策略利用中:")
    print(f"  模型倾向于选择动作池中最常见的动作")
    print(f"  由于S型建筑占{action_dist['S']}%，模型很可能总是选择S型")
    
    if action_dist['S'] > 70:
        print(f"  X 问题发现: S型建筑在动作池中占绝对优势")
        print(f"     建议: 平衡动作池分布或限制S型建筑数量")
    else:
        print(f"  OK 动作池分布相对平衡")
    
    return current_exploration

def generate_comprehensive_analysis():
    """综合分析"""
    
    print("\n" + "=" * 60)
    print("综合分析结果")
    print("=" * 60)
    
    # 运行所有分析
    efficiency_data, availability = analyze_why_s_always_optimal()
    reward_data = analyze_ml_reward_signals()
    exploration_data = analyze_exploration_opportunities()
    
    print(f"\n关键发现总结:")
    
    issues = []
    
    # 检查问题1
    for agent in ['EDU', 'IND']:
        s_efficiency = efficiency_data[agent]['S']
        l_efficiency = efficiency_data[agent]['L']
        if s_efficiency / l_efficiency > 0.9:  # S型效率接近L型
            issues.append(f"{agent}建筑中S型效率优势不明显")
    
    # 检查问题2
    for agent in ['EDU', 'IND']:
        s_reward = reward_data[agent]['S']['reward']
        l_reward = reward_data[agent]['L']['reward']
        if l_reward / s_reward < 2.0:  # L型收益增长少于2倍
            issues.append(f"{agent}建筑中L型收益增长不够明显")
    
    # 检查问题3
    if exploration_data['epsilon'] < 0.2:  # 探索率低于20%
        issues.append("探索率过低，无法充分尝试M/L型建筑")
    
    if exploration_data['action_pool_distribution']['S'] > 70:  # S型占70%以上
        issues.append("动作池中S型建筑占比过高")
    
    if issues:
        print("发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("未发现明显问题")
    
    print(f"\n建议的修复方案:")
    print("1. 立即修复:")
    print("   - 增加探索率到30-50%")
    print("   - 提高温度参数到2.0")
    print("   - 给M/L型建筑探索奖励")
    
    print("2. 动作池优化:")
    print("   - 限制S型建筑在动作池中的数量")
    print("   - 确保M/L型建筑有足够的代表")
    
    print("3. 奖励结构调整:")
    print("   - 增大M/L型建筑的奖励差异")
    print("   - 添加尺寸适应性奖励")
    
    return issues

def main():
    """主函数"""
    print("模型选择偏好问题深度分析")
    print("分析三个关键问题:")
    print("1. 为什么模型认为S型建筑总是最优？")
    print("2. M/L型建筑的奖励是否足够明显？")
    print("3. 模型是否有足够的探索机会尝试不同尺寸？")
    
    issues = generate_comprehensive_analysis()
    
    print(f"\n" + "=" * 60)
    print("最终结论")
    print("=" * 60)
    
    if len(issues) >= 2:
        print("模型只选择S型建筑的主要原因是多方面的:")
        print("- 探索不足导致无法发现M/L型建筑的价值")
        print("- 动作池分布偏向S型建筑")
        print("- 可能缺乏足够明显的奖励信号")
    else:
        print("需要进一步调试以确定具体原因")

if __name__ == "__main__":
    main()


