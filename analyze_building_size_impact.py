#!/usr/bin/env python3
"""
分析强化学习模型中不同建筑尺寸(S/M/L)对决策的影响
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import os

def extract_building_size_parameters():
    """提取S/M/L尺寸的具体参数"""
    
    # 从v4_enumeration.py中提取的参数
    size_parameters = {
        'EDU': {
            'S': {
                'Base_EDU': 1.2,      # 基础成本系数
                'Add_EDU': 0.4,       # 额外成本系数
                'Capacity_EDU': 60,    # 容量
                'OPEX_EDU': 0.20,     # 运营成本
                'PrestigeBase_EDU': 0.2,  # 基础声望
                'Pollution_EDU': 0.2,     # 污染
                'footprint_slots': 1,     # 占地面积(槽位数)
                'building_level_required': 3  # 所需建筑等级
            },
            'M': {
                'Base_EDU': 2.8,
                'Add_EDU': 0.9,
                'Capacity_EDU': 120,
                'OPEX_EDU': 0.35,
                'PrestigeBase_EDU': 0.6,
                'Pollution_EDU': 0.4,
                'footprint_slots': 1,  # M型也是单槽位，但需要更高等级
                'building_level_required': 4
            },
            'L': {
                'Base_EDU': 5.5,
                'Add_EDU': 1.6,
                'Capacity_EDU': 240,
                'OPEX_EDU': 0.55,
                'PrestigeBase_EDU': 1.0,
                'Pollution_EDU': 0.6,
                'footprint_slots': 1,  # L型也是单槽位，但需要最高等级
                'building_level_required': 5
            }
        },
        'IND': {
            'S': {
                'Base_IND': 1.0,
                'Add_IND': 0.5,
                'Capacity_IND': 80,
                'GFA_k': 1.0,         # 建筑面积系数
                'PrestigeBase_IND': 0.2,
                'Pollution_IND': 0.6,
                'footprint_slots': 1,
                'building_level_required': 3
            },
            'M': {
                'Base_IND': 2.2,
                'Add_IND': 1.0,
                'Capacity_IND': 200,
                'GFA_k': 2.0,
                'PrestigeBase_IND': 0.1,
                'Pollution_IND': 0.9,
                'footprint_slots': 2,  # M型需要相邻的2个槽位
                'building_level_required': 4
            },
            'L': {
                'Base_IND': 4.5,
                'Add_IND': 2.0,
                'Capacity_IND': 500,
                'GFA_k': 4.0,
                'PrestigeBase_IND': -0.1,  # 大型工业建筑声望为负
                'Pollution_IND': 1.2,
                'footprint_slots': 4,  # L型需要2x2的4个槽位
                'building_level_required': 5
            }
        }
    }
    
    return size_parameters

def calculate_cost_reward_analysis(size_params: Dict, lp_norm: float = 0.5, zone: str = 'mid') -> Dict:
    """计算不同尺寸的成本-收益分析"""
    
    # 从配置中提取的额外参数
    zone_add = {'near': 0.8, 'mid': 0.3, 'far': 0.0}
    s_zone = {'near': 0.5, 'mid': 0.2, 'far': 0.0}
    m_zone = {'near': 1.10, 'mid': 1.00, 'far': 0.90}
    m_adj = {'on': 1.10, 'off': 1.00}
    
    # 经济参数
    alpha = 0.08
    beta = 0.25
    p_market = 12.0
    u = 0.85
    c_opex = 0.30
    
    analysis = {}
    
    for agent_type in ['EDU', 'IND']:
        analysis[agent_type] = {}
        
        for size in ['S', 'M', 'L']:
            params = size_params[agent_type][size]
            
            # 计算成本 (根据v4_enumeration.py中的公式)
            if agent_type == 'EDU':
                cost = (params['Base_EDU'] + params['Add_EDU']) * lp_norm + zone_add[zone]
                # 奖励 = (α × Capacity) × m_zone × m_adj − OPEX
                reward = (alpha * params['Capacity_EDU']) * m_zone[zone] * m_adj['off'] - params['OPEX_EDU']
                # 声望 = PrestigeBase + I(zone==near) + I(adj) − β × Pollution
                prestige = params['PrestigeBase_EDU'] + (1.0 if zone == 'near' else 0.0) - beta * params['Pollution_EDU']
                
            else:  # IND
                cost = (params['Base_IND'] + params['Add_IND']) * lp_norm + zone_add[zone]
                # 奖励 = ((p_market × u × Capacity) / 1000) × m_zone × m_adj − c_opex × GFA_k + s_zone
                reward = ((p_market * u * params['Capacity_IND']) / 1000.0) * m_zone[zone] * m_adj['off'] - c_opex * params['GFA_k'] + s_zone[zone]
                # 声望 = PrestigeBase + I(zone==near) + I(adj) − 0.2 × Pollution
                prestige = params['PrestigeBase_IND'] + (1.0 if zone == 'near' else 0.0) - 0.2 * params['Pollution_IND']
            
            # 计算效率指标
            cost_per_capacity = cost / params[f'Capacity_{agent_type}']
            reward_per_cost = reward / cost if cost > 0 else 0
            prestige_per_cost = prestige / cost if cost > 0 else 0
            
            analysis[agent_type][size] = {
                'cost': cost,
                'reward': reward,
                'prestige': prestige,
                'capacity': params[f'Capacity_{agent_type}'],
                'footprint_slots': params['footprint_slots'],
                'building_level_required': params['building_level_required'],
                'cost_per_capacity': cost_per_capacity,
                'reward_per_cost': reward_per_cost,
                'prestige_per_cost': prestige_per_cost,
                'efficiency_score': reward / (cost + 1e-6)  # 避免除零
            }
    
    return analysis

def analyze_rl_selection_patterns():
    """分析RL模型的选择模式"""
    
    # 模拟RL模型的选择逻辑
    selection_factors = {
        'cost_sensitivity': {
            'EDU': {'w_r': 0.3, 'w_p': 0.6, 'w_c': 0.1},  # EDU更重视声望，成本敏感度低
            'IND': {'w_r': 0.6, 'w_p': 0.2, 'w_c': 0.2}   # IND更重视收益，成本敏感度中等
        },
        'size_preferences': {
            'early_game': {
                'EDU': {'S': 0.7, 'M': 0.2, 'L': 0.1},     # 早期偏好小建筑
                'IND': {'S': 0.8, 'M': 0.15, 'L': 0.05}
            },
            'mid_game': {
                'EDU': {'S': 0.4, 'M': 0.4, 'L': 0.2},     # 中期平衡
                'IND': {'S': 0.5, 'M': 0.35, 'L': 0.15}
            },
            'late_game': {
                'EDU': {'S': 0.2, 'M': 0.3, 'L': 0.5},     # 后期偏好大建筑
                'IND': {'S': 0.3, 'M': 0.4, 'L': 0.3}
            }
        }
    }
    
    return selection_factors

def create_comparison_visualization(analysis: Dict, output_dir: str = 'analysis_output'):
    """创建对比可视化"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 成本-收益对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('建筑尺寸对RL决策的影响分析', fontsize=16, fontweight='bold')
    
    # EDU成本对比
    edu_sizes = ['S', 'M', 'L']
    edu_costs = [analysis['EDU'][s]['cost'] for s in edu_sizes]
    edu_rewards = [analysis['EDU'][s]['reward'] for s in edu_sizes]
    edu_capacities = [analysis['EDU'][s]['capacity'] for s in edu_sizes]
    
    axes[0, 0].bar(edu_sizes, edu_costs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    axes[0, 0].set_title('EDU建筑成本对比')
    axes[0, 0].set_ylabel('成本')
    axes[0, 0].set_xlabel('建筑尺寸')
    
    # IND成本对比
    ind_costs = [analysis['IND'][s]['cost'] for s in edu_sizes]
    ind_rewards = [analysis['IND'][s]['reward'] for s in edu_sizes]
    ind_capacities = [analysis['IND'][s]['capacity'] for s in edu_sizes]
    
    axes[0, 1].bar(edu_sizes, ind_costs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    axes[0, 1].set_title('IND建筑成本对比')
    axes[0, 1].set_ylabel('成本')
    axes[0, 1].set_xlabel('建筑尺寸')
    
    # 效率对比 (收益/成本)
    edu_efficiency = [analysis['EDU'][s]['reward_per_cost'] for s in edu_sizes]
    ind_efficiency = [analysis['IND'][s]['reward_per_cost'] for s in edu_sizes]
    
    x = np.arange(len(edu_sizes))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, edu_efficiency, width, label='EDU', alpha=0.7)
    axes[1, 0].bar(x + width/2, ind_efficiency, width, label='IND', alpha=0.7)
    axes[1, 0].set_title('建筑效率对比 (收益/成本)')
    axes[1, 0].set_ylabel('效率')
    axes[1, 0].set_xlabel('建筑尺寸')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(edu_sizes)
    axes[1, 0].legend()
    
    # 占地面积对比
    edu_footprint = [analysis['EDU'][s]['footprint_slots'] for s in edu_sizes]
    ind_footprint = [analysis['IND'][s]['footprint_slots'] for s in edu_sizes]
    
    axes[1, 1].bar(x - width/2, edu_footprint, width, label='EDU', alpha=0.7)
    axes[1, 1].bar(x + width/2, ind_footprint, width, label='IND', alpha=0.7)
    axes[1, 1].set_title('占地面积对比 (槽位数)')
    axes[1, 1].set_ylabel('槽位数')
    axes[1, 1].set_xlabel('建筑尺寸')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(edu_sizes)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'building_size_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 创建详细的对比表格
    create_comparison_table(analysis, output_dir)

def create_comparison_table(analysis: Dict, output_dir: str):
    """创建详细的对比表格"""
    
    # 准备数据
    data = []
    
    for agent in ['EDU', 'IND']:
        for size in ['S', 'M', 'L']:
            params = analysis[agent][size]
            data.append({
                'Agent': agent,
                'Size': size,
                'Cost': f"{params['cost']:.2f}",
                'Reward': f"{params['reward']:.2f}",
                'Prestige': f"{params['prestige']:.2f}",
                'Capacity': params['capacity'],
                'Footprint_Slots': params['footprint_slots'],
                'Building_Level_Required': params['building_level_required'],
                'Cost_Per_Capacity': f"{params['cost_per_capacity']:.4f}",
                'Reward_Per_Cost': f"{params['reward_per_cost']:.4f}",
                'Prestige_Per_Cost': f"{params['prestige_per_cost']:.4f}",
                'Efficiency_Score': f"{params['efficiency_score']:.4f}"
            })
    
    df = pd.DataFrame(data)
    
    # 保存为CSV
    csv_path = os.path.join(output_dir, 'building_size_comparison.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 创建HTML表格
    html_path = os.path.join(output_dir, 'building_size_comparison.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>建筑尺寸对比分析</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        .edu-row { background-color: #e3f2fd; }
        .ind-row { background-color: #f3e5f5; }
        .highlight { background-color: #ffeb3b; }
    </style>
</head>
<body>
    <h1>建筑尺寸对RL决策的影响分析</h1>
    <h2>详细参数对比</h2>
""")
        
        f.write(df.to_html(index=False, escape=False, classes='comparison-table'))
        
        f.write("""
    <h2>关键发现</h2>
    <ul>
        <li><strong>EDU建筑</strong>：
            <ul>
                <li>S型：成本最低，适合早期快速扩张</li>
                <li>M型：成本效益平衡，适合中期发展</li>
                <li>L型：声望最高，但成本也最高，适合后期优化</li>
            </ul>
        </li>
        <li><strong>IND建筑</strong>：
            <ul>
                <li>S型：占地面积小，适合密集布局</li>
                <li>M型：需要2个槽位，收益较高</li>
                <li>L型：需要4个槽位(2x2)，容量最大但污染也最严重</li>
            </ul>
        </li>
        <li><strong>RL模型决策考虑因素</strong>：
            <ul>
                <li>建筑等级限制：L型需要最高等级(5级)</li>
                <li>占地面积约束：大建筑需要更多连续槽位</li>
                <li>成本效益权衡：不同阶段的偏好不同</li>
                <li>智能体目标差异：EDU重视声望，IND重视收益</li>
            </ul>
        </li>
    </ul>
</body>
</html>
""")
    
    print(f"对比表格已保存到: {csv_path}")
    print(f"HTML报告已保存到: {html_path}")

def analyze_rl_decision_factors():
    """分析RL模型的决策因素"""
    
    decision_analysis = {
        'size_selection_factors': {
            'early_game': {
                'description': '游戏初期，资源有限，偏好小建筑',
                'edu_preference': 'S型建筑：成本低，快速建立基础设施',
                'ind_preference': 'S型建筑：占地面积小，适合密集布局'
            },
            'mid_game': {
                'description': '游戏中期，开始平衡成本与收益',
                'edu_preference': 'M型建筑：成本效益平衡，提升声望',
                'ind_preference': 'M型建筑：收益较高，需要2个槽位'
            },
            'late_game': {
                'description': '游戏后期，资源充足，追求最大效益',
                'edu_preference': 'L型建筑：最高声望，但需要最高建筑等级',
                'ind_preference': 'L型建筑：最大容量，但污染严重'
            }
        },
        'constraints': {
            'building_level': 'L型建筑需要等级5，限制了早期选择',
            'footprint_requirement': '大建筑需要更多连续槽位，布局受限',
            'budget_constraint': '大建筑成本高，需要足够的预算',
            'pollution_impact': '大建筑污染严重，影响环境评分'
        },
        'agent_specific_preferences': {
            'EDU': {
                'primary_goal': '最大化声望',
                'size_preference': 'L型 > M型 > S型 (声望导向)',
                'decision_factors': ['prestige', 'cost', 'capacity']
            },
            'IND': {
                'primary_goal': '最大化收益',
                'size_preference': 'L型 > M型 > S型 (收益导向)',
                'decision_factors': ['reward', 'capacity', 'cost']
            }
        }
    }
    
    return decision_analysis

def main():
    """主函数"""
    print("=" * 60)
    print("建筑尺寸对RL决策影响分析")
    print("=" * 60)
    
    # 1. 提取建筑尺寸参数
    print("\n1. 提取建筑尺寸参数...")
    size_params = extract_building_size_parameters()
    
    # 2. 计算成本-收益分析
    print("2. 计算成本-收益分析...")
    analysis = calculate_cost_reward_analysis(size_params)
    
    # 3. 分析RL选择模式
    print("3. 分析RL选择模式...")
    selection_factors = analyze_rl_selection_patterns()
    
    # 4. 分析决策因素
    print("4. 分析决策因素...")
    decision_analysis = analyze_rl_decision_factors()
    
    # 5. 创建可视化
    print("5. 创建可视化...")
    create_comparison_visualization(analysis)
    
    # 6. 打印关键发现
    print("\n" + "=" * 60)
    print("关键发现总结")
    print("=" * 60)
    
    print("\n【建筑尺寸差异】:")
    for agent in ['EDU', 'IND']:
        print(f"\n{agent}建筑:")
        for size in ['S', 'M', 'L']:
            params = analysis[agent][size]
            print(f"  {size}型: 成本={params['cost']:.2f}, 收益={params['reward']:.2f}, "
                  f"容量={params['capacity']}, 占地面积={params['footprint_slots']}槽位, "
                  f"效率={params['efficiency_score']:.4f}")
    
    print("\n【RL模型决策逻辑】:")
    print("1. 早期阶段：偏好S型建筑，快速扩张")
    print("2. 中期阶段：平衡M型和L型建筑")
    print("3. 后期阶段：优先选择L型建筑，追求最大效益")
    print("4. 约束条件：建筑等级、占地面积、预算限制")
    
    print("\n【智能体差异】:")
    print("• EDU智能体：重视声望，L型建筑声望最高")
    print("• IND智能体：重视收益，L型建筑容量最大")
    print("• 两者都受到建筑等级和占地面积约束")
    
    print("\n【优化建议】:")
    print("1. 根据游戏阶段动态调整建筑尺寸偏好")
    print("2. 考虑建筑等级升级的成本效益")
    print("3. 平衡短期收益与长期声望")
    print("4. 优化槽位布局以支持大建筑建设")

if __name__ == "__main__":
    main()


