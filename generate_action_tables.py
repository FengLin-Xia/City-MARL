#!/usr/bin/env python3
"""
生成动作表格可视化
支持v4.0和v4.1的输出格式
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def create_action_table(month, agent, actions, output_path, budget_info=None):
    """
    创建动作表格图片
    
    参数：
    - month: 月份
    - agent: 'IND' 或 'EDU'
    - actions: 动作列表 [{'agent': 'IND', 'size': 'S', 'footprint_slots': [...], 'cost': 100, 'reward': 50}, ...]
    - output_path: 输出路径
    - budget_info: {'initial': 10000, 'final': 2701} 或 None
    """
    if not actions or len(actions) == 0:
        return
    
    # 准备数据
    table_data = []
    total_cost = 0
    total_reward = 0
    
    # 如果有budget_info，计算每个动作后的budget
    if budget_info:
        current_budget = budget_info['initial']
    
    for i, action in enumerate(actions):
        cost = int(action.get('cost', 0))
        reward = int(action.get('reward', 0))
        total_cost += cost
        total_reward += reward
        
        # Budget列
        if budget_info:
            budget_before = current_budget
            current_budget = current_budget - cost + reward
            budget_str = f"{budget_before} → {current_budget}"
        else:
            budget_str = "N/A"
        
        row = [
            str(i + 1),
            action.get('agent', agent),
            action.get('size', 'S'),
            action.get('footprint_slots', [''])[0] if action.get('footprint_slots') else '',
            str(cost),
            str(reward),
            budget_str
        ]
        table_data.append(row)
    
    # 添加总计行
    if budget_info:
        final_budget_str = f"Final: {budget_info['final']}"
    else:
        net_change = total_reward - total_cost
        final_budget_str = f"Net: {net_change:+d}"
    
    table_data.append([
        'Total',
        '',
        '',
        '',
        str(total_cost),
        str(total_reward),
        final_budget_str
    ])
    
    # 表头
    headers = ['#', 'Agent', 'Size', 'Slot', 'Cost', 'Reward', 'Budget']
    
    # 创建图表
    fig_height = max(3, len(table_data) * 0.6 + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.08, 0.12, 0.08, 0.15, 0.15, 0.15, 0.27]
    )
    
    # 样式设置
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    
    # 白色文字，透明背景
    for (i, j), cell in table.get_celld().items():
        # 表头
        if i == 0:
            cell.set_text_props(color='white', weight='bold', fontsize=12)
            cell.set_linewidth(2.0)
        # 总计行
        elif i == len(table_data):
            cell.set_text_props(color='white', weight='bold', fontsize=11)
            cell.set_linewidth(2.0)
        # 普通行
        else:
            cell.set_text_props(color='white', fontsize=10)
            cell.set_linewidth(1.0)
        
        cell.set_facecolor('none')
        cell.set_edgecolor('white')
    
    # 透明背景
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # 标题
    title = f"Month {month} - {agent} Actions"
    ax.set_title(title, color='white', fontsize=14, weight='bold', pad=20)
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='none', edgecolor='none', transparent=True)
    plt.close()
    
    print(f"  Generated: {output_path}")


def process_v4_0_output(output_dir='enhanced_simulation_v4_0_output', max_months=30):
    """处理v4.0的输出"""
    print("="*80)
    print("处理V4.0输出")
    print("="*80)
    
    debug_dir = os.path.join(output_dir, 'v4_debug')
    table_dir = os.path.join(output_dir, 'action_tables')
    os.makedirs(table_dir, exist_ok=True)
    
    # 模拟budget（未实现时）
    budgets = {'IND': 5000, 'EDU': 4000}
    
    for month in range(max_months):
        fname = f'chosen_sequence_month_{month:02d}.json'
        fpath = os.path.join(debug_dir, fname)
        
        if not os.path.exists(fpath):
            break
        
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        actions = data.get('actions', [])
        if not actions:
            continue
        
        # 按agent分组
        ind_actions = [a for a in actions if a.get('agent') == 'IND']
        edu_actions = [a for a in actions if a.get('agent') == 'EDU']
        
        # 生成IND表格
        if ind_actions:
            initial_budget = budgets['IND']
            # 计算最终budget
            for a in ind_actions:
                budgets['IND'] -= int(a.get('cost', 0))
                budgets['IND'] += int(a.get('reward', 0))
            final_budget = budgets['IND']
            
            budget_info = {'initial': initial_budget, 'final': final_budget}
            output_path = os.path.join(table_dir, f'month_{month:02d}_IND.png')
            create_action_table(month, 'IND', ind_actions, output_path, budget_info)
        
        # 生成EDU表格
        if edu_actions:
            initial_budget = budgets['EDU']
            # 计算最终budget
            for a in edu_actions:
                budgets['EDU'] -= int(a.get('cost', 0))
                budgets['EDU'] += int(a.get('reward', 0))
            final_budget = budgets['EDU']
            
            budget_info = {'initial': initial_budget, 'final': final_budget}
            output_path = os.path.join(table_dir, f'month_{month:02d}_EDU.png')
            create_action_table(month, 'EDU', edu_actions, output_path, budget_info)
    
    print(f"\nV4.0处理完成")
    print(f"输出目录: {table_dir}")
    print(f"最终Budget - IND: {budgets['IND']}, EDU: {budgets['EDU']}")


def process_v4_1_output(history_path='models/v4_1_rl/slot_selection_history.json'):
    """处理v4.1的输出"""
    print("\n" + "="*80)
    print("处理V4.1输出")
    print("="*80)
    
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found!")
        return
    
    with open(history_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    episodes = data.get('episodes', [])
    if not episodes:
        print("No episodes found!")
        return
    
    # 只处理第一个episode
    episode = episodes[0]
    steps = episode.get('steps', [])
    
    table_dir = 'enhanced_simulation_v4_1_output/action_tables'
    os.makedirs(table_dir, exist_ok=True)
    
    # 模拟budget
    budgets = {'IND': 10000, 'EDU': 10000}
    
    for step in steps:
        month = step.get('month', 0)
        agent = step.get('agent', 'IND')
        detailed_actions = step.get('detailed_actions', [])
        
        if not detailed_actions:
            continue
        
        initial_budget = budgets[agent]
        
        # 计算最终budget
        for a in detailed_actions:
            budgets[agent] -= int(a.get('cost', 0))
            budgets[agent] += int(a.get('reward', 0))
        
        final_budget = budgets[agent]
        budget_info = {'initial': initial_budget, 'final': final_budget}
        
        output_path = os.path.join(table_dir, f'month_{month:02d}_{agent}.png')
        create_action_table(month, agent, detailed_actions, output_path, budget_info)
    
    print(f"\nV4.1处理完成")
    print(f"输出目录: {table_dir}")
    print(f"最终Budget - IND: {budgets['IND']}, EDU: {budgets['EDU']}")


def main():
    parser = argparse.ArgumentParser(description='生成动作表格可视化')
    parser.add_argument('--mode', choices=['v4.0', 'v4.1', 'both'], default='both',
                       help='处理模式：v4.0, v4.1, 或 both')
    parser.add_argument('--v4_0_dir', default='enhanced_simulation_v4_0_output',
                       help='v4.0输出目录')
    parser.add_argument('--v4_1_history', default='models/v4_1_rl/slot_selection_history.json',
                       help='v4.1 slot selection history路径')
    parser.add_argument('--max_months', type=int, default=30,
                       help='最大处理月份数')
    
    args = parser.parse_args()
    
    if args.mode in ['v4.0', 'both']:
        process_v4_0_output(args.v4_0_dir, args.max_months)
    
    if args.mode in ['v4.1', 'both']:
        process_v4_1_output(args.v4_1_history)
    
    print("\n" + "="*80)
    print("所有表格生成完成！")
    print("="*80)


if __name__ == '__main__':
    main()

