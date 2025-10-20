#!/usr/bin/env python3
"""
分析为什么RL模型训练很快收敛并只选择S型建筑的问题
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import os

def analyze_convergence_issues():
    """分析模型收敛问题的根本原因"""
    
    issues = {
        'building_level_constraint': {
            'description': '建筑等级约束导致M/L型建筑无法建造',
            'details': {
                'default_slot_level': 3,  # 从SlotNode默认值
                's_size_requirement': 3,  # S型需要等级3
                'm_size_requirement': 4,  # M型需要等级4
                'l_size_requirement': 5,  # L型需要等级5
                'problem': '大部分槽位默认等级为3，只能建造S型建筑'
            }
        },
        'action_pool_imbalance': {
            'description': '动作池中S型建筑数量远多于M/L型',
            'details': {
                's_size_options': '所有等级3+的槽位',
                'm_size_options': '只有等级4+的槽位',
                'l_size_options': '只有等级5的槽位',
                'problem': 'S型建筑在动作池中占绝对多数，RL模型倾向于选择概率高的动作'
            }
        },
        'exploration_insufficient': {
            'description': '探索不足，无法发现M/L型建筑的长期价值',
            'details': {
                'epsilon': 0.1,  # 只有10%的探索率
                'temperature': 1.2,  # 温度参数可能不够高
                'problem': '低探索率导致模型无法充分探索M/L型建筑'
            }
        },
        'reward_structure': {
            'description': '奖励结构可能偏向短期收益',
            'details': {
                's_size_reward': '快速获得奖励，风险低',
                'm_l_size_reward': '需要更多投资，回报周期长',
                'problem': 'RL模型可能被短期奖励信号误导'
            }
        },
        'footprint_constraints': {
            'description': '占地面积约束限制M/L型建筑',
            'details': {
                's_size_footprint': 1,  # 1个槽位
                'm_size_footprint': 2,  # 需要2个相邻槽位
                'l_size_footprint': 4,  # 需要4个槽位(2x2)
                'problem': '大建筑需要更多连续空间，约束更严格'
            }
        }
    }
    
    return issues

def simulate_action_pool_distribution():
    """模拟动作池中不同尺寸建筑的数量分布"""
    
    # 模拟场景：100个槽位，不同等级分布
    total_slots = 100
    
    # 假设等级分布（这是问题的关键）
    level_distribution = {
        3: 80,  # 80%的槽位等级为3
        4: 15,  # 15%的槽位等级为4  
        5: 5    # 5%的槽位等级为5
    }
    
    # 计算每种尺寸的建筑数量
    s_size_count = sum(level_distribution.values())  # 所有等级都可以建S型
    m_size_count = level_distribution[4] + level_distribution[5]  # 等级4和5可以建M型
    l_size_count = level_distribution[5]  # 只有等级5可以建L型
    
    # 计算比例
    total_actions = s_size_count + m_size_count + l_size_count
    s_ratio = s_size_count / total_actions
    m_ratio = m_size_count / total_actions  
    l_ratio = l_size_count / total_actions
    
    return {
        's_size': {'count': s_size_count, 'ratio': s_ratio},
        'm_size': {'count': m_size_count, 'ratio': m_ratio},
        'l_size': {'count': l_size_count, 'ratio': l_ratio},
        'total_actions': total_actions
    }

def analyze_exploration_effectiveness():
    """分析探索策略的有效性"""
    
    # 当前探索设置
    current_exploration = {
        'epsilon': 0.1,  # 10%随机探索
        'temperature': 1.2,  # 温度参数
        'exploration_decay': 'none'  # 没有探索衰减
    }
    
    # 问题分析
    problems = {
        'low_epsilon': {
            'current': 0.1,
            'recommended': 0.3,  # 建议30%探索率
            'reason': '10%探索率不足以发现M/L型建筑的价值'
        },
        'static_exploration': {
            'current': 'fixed',
            'recommended': 'decay',
            'reason': '固定探索率无法在训练过程中逐步收敛'
        },
        'temperature_insufficient': {
            'current': 1.2,
            'recommended': 2.0,
            'reason': '温度参数不够高，无法增加动作多样性'
        }
    }
    
    return current_exploration, problems

def calculate_reward_bias():
    """计算奖励偏向问题"""
    
    # 基于之前的分析数据
    building_data = {
        'EDU': {
            'S': {'cost': 1.10, 'reward': 4.60, 'efficiency': 4.18},
            'M': {'cost': 2.15, 'reward': 9.25, 'efficiency': 4.30},
            'L': {'cost': 3.85, 'reward': 18.65, 'efficiency': 4.84}
        },
        'IND': {
            'S': {'cost': 1.05, 'reward': 0.72, 'efficiency': 0.68},
            'M': {'cost': 1.90, 'reward': 1.64, 'efficiency': 0.86},
            'L': {'cost': 3.55, 'reward': 4.10, 'efficiency': 1.15}
        }
    }
    
    # 分析奖励结构问题
    reward_analysis = {}
    
    for agent_type in ['EDU', 'IND']:
        agent_data = building_data[agent_type]
        
        # 计算短期vs长期收益
        short_term = agent_data['S']['reward']  # 单步奖励
        medium_term = agent_data['M']['reward'] / 2  # 假设需要2步完成
        long_term = agent_data['L']['reward'] / 4   # 假设需要4步完成
        
        reward_analysis[agent_type] = {
            'short_term_reward': short_term,
            'medium_term_reward': medium_term,
            'long_term_reward': long_term,
            'bias_toward_small': short_term > medium_term or short_term > long_term
        }
    
    return reward_analysis

def create_convergence_analysis_visualization():
    """创建收敛问题分析可视化"""
    
    os.makedirs('convergence_analysis_output', exist_ok=True)
    
    # 1. 动作池分布图
    action_dist = simulate_action_pool_distribution()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RL模型收敛问题分析', fontsize=16, fontweight='bold')
    
    # 动作池分布
    sizes = ['S型', 'M型', 'L型']
    counts = [action_dist['s_size']['count'], 
              action_dist['m_size']['count'], 
              action_dist['l_size']['count']]
    ratios = [action_dist['s_size']['ratio'], 
              action_dist['m_size']['ratio'], 
              action_dist['l_size']['ratio']]
    
    axes[0, 0].bar(sizes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    axes[0, 0].set_title('动作池中建筑数量分布')
    axes[0, 0].set_ylabel('建筑数量')
    
    # 添加比例标签
    for i, (count, ratio) in enumerate(zip(counts, ratios)):
        axes[0, 0].text(i, count + 1, f'{ratio:.1%}', ha='center', va='bottom')
    
    # 建筑等级分布
    level_dist = [80, 15, 5]  # 等级3, 4, 5的槽位数量
    level_labels = ['等级3\n(只能建S)', '等级4\n(可建S/M)', '等级5\n(可建S/M/L)']
    
    axes[0, 1].pie(level_dist, labels=level_labels, autopct='%1.1f%%', 
                   colors=['#FFB6C1', '#98FB98', '#87CEEB'])
    axes[0, 1].set_title('槽位建筑等级分布')
    
    # 探索率对比
    current_epsilon = [0.1] * 10  # 当前固定探索率
    recommended_epsilon = [0.3 * (0.9 ** i) for i in range(10)]  # 建议的衰减探索率
    
    episodes = list(range(1, 11))
    axes[1, 0].plot(episodes, current_epsilon, 'r-', label='当前固定探索率 (0.1)', linewidth=2)
    axes[1, 0].plot(episodes, recommended_epsilon, 'b-', label='建议衰减探索率', linewidth=2)
    axes[1, 0].set_title('探索策略对比')
    axes[1, 0].set_xlabel('训练Episode')
    axes[1, 0].set_ylabel('探索率 (ε)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 奖励结构分析
    reward_data = calculate_reward_bias()
    edu_rewards = [reward_data['EDU']['short_term_reward'],
                   reward_data['EDU']['medium_term_reward'],
                   reward_data['EDU']['long_term_reward']]
    ind_rewards = [reward_data['IND']['short_term_reward'],
                   reward_data['IND']['medium_term_reward'],
                   reward_data['IND']['long_term_reward']]
    
    x = np.arange(len(['短期', '中期', '长期']))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, edu_rewards, width, label='EDU', alpha=0.7)
    axes[1, 1].bar(x + width/2, ind_rewards, width, label='IND', alpha=0.7)
    axes[1, 1].set_title('不同时间尺度的奖励对比')
    axes[1, 1].set_ylabel('奖励值')
    axes[1, 1].set_xlabel('时间尺度')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['短期', '中期', '长期'])
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('convergence_analysis_output/convergence_issue_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_solutions():
    """生成解决方案"""
    
    solutions = {
        'immediate_fixes': {
            'increase_exploration': {
                'description': '增加探索率',
                'implementation': '将epsilon从0.1提升到0.3-0.5',
                'code_change': 'self.epsilon = 0.3  # 在RLPolicySelector中'
            },
            'add_exploration_decay': {
                'description': '添加探索衰减',
                'implementation': '训练过程中逐步降低探索率',
                'code_change': 'self.epsilon = max(0.05, self.epsilon * 0.995)'
            },
            'increase_temperature': {
                'description': '提高温度参数',
                'implementation': '增加策略网络的随机性',
                'code_change': 'self.temperature = 2.0  # 在Actor网络中'
            }
        },
        'architectural_fixes': {
            'fix_building_levels': {
                'description': '修复建筑等级分布',
                'implementation': '增加高等级槽位的比例',
                'code_change': 'slot.building_level = random.choice([3, 4, 5])'
            },
            'balance_action_pool': {
                'description': '平衡动作池',
                'implementation': '限制S型建筑在动作池中的数量',
                'code_change': 'cap_s_size_actions = max_actions * 0.4'
            },
            'reward_reshaping': {
                'description': '重塑奖励结构',
                'implementation': '给M/L型建筑额外的探索奖励',
                'code_change': 'exploration_bonus = 1.0 if size != "S" else 0.0'
            }
        },
        'advanced_solutions': {
            'curriculum_learning': {
                'description': '课程学习',
                'implementation': '从简单场景开始，逐步增加复杂度',
                'steps': [
                    '阶段1: 只允许S型建筑',
                    '阶段2: 引入M型建筑',
                    '阶段3: 引入L型建筑'
                ]
            },
            'hierarchical_rl': {
                'description': '分层强化学习',
                'implementation': '高层决策选择建筑类型，底层决策选择具体位置',
                'benefits': '减少动作空间，提高学习效率'
            },
            'multi_objective_optimization': {
                'description': '多目标优化',
                'implementation': '同时优化短期收益和长期发展',
                'approach': '使用Pareto前沿平衡不同目标'
            }
        }
    }
    
    return solutions

def main():
    """主函数"""
    print("=" * 60)
    print("RL模型收敛问题分析")
    print("=" * 60)
    
    # 1. 分析收敛问题
    print("\n1. 分析收敛问题...")
    issues = analyze_convergence_issues()
    
    # 2. 模拟动作池分布
    print("2. 模拟动作池分布...")
    action_dist = simulate_action_pool_distribution()
    
    # 3. 分析探索效果
    print("3. 分析探索效果...")
    exploration, problems = analyze_exploration_effectiveness()
    
    # 4. 计算奖励偏向
    print("4. 计算奖励偏向...")
    reward_bias = calculate_reward_bias()
    
    # 5. 生成解决方案
    print("5. 生成解决方案...")
    solutions = generate_solutions()
    
    # 6. 创建可视化
    print("6. 创建可视化...")
    create_convergence_analysis_visualization()
    
    # 7. 打印关键发现
    print("\n" + "=" * 60)
    print("关键问题总结")
    print("=" * 60)
    
    print("\n【根本原因】:")
    print("1. 建筑等级约束: 80%的槽位等级为3，只能建造S型建筑")
    print("2. 动作池不平衡: S型建筑在动作池中占绝对多数")
    print("3. 探索不足: 10%的探索率无法发现M/L型建筑的价值")
    print("4. 奖励偏向: 短期奖励信号偏向小建筑")
    
    print("\n【具体数据】:")
    print(f"S型建筑比例: {action_dist['s_size']['ratio']:.1%}")
    print(f"M型建筑比例: {action_dist['m_size']['ratio']:.1%}")
    print(f"L型建筑比例: {action_dist['l_size']['ratio']:.1%}")
    
    print("\n【解决方案】:")
    print("1. 立即修复:")
    print("   - 增加探索率到30-50%")
    print("   - 添加探索衰减机制")
    print("   - 提高温度参数到2.0")
    
    print("2. 架构修复:")
    print("   - 修复建筑等级分布")
    print("   - 平衡动作池比例")
    print("   - 重塑奖励结构")
    
    print("3. 高级方案:")
    print("   - 课程学习")
    print("   - 分层强化学习")
    print("   - 多目标优化")
    
    print("\n【推荐实施顺序】:")
    print("1. 首先增加探索率 (快速验证)")
    print("2. 然后修复建筑等级分布 (根本解决)")
    print("3. 最后考虑架构改进 (长期优化)")

if __name__ == "__main__":
    main()


