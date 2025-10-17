#!/usr/bin/env python3
"""
从txt文件生成动作表格可视化
支持1015-1格式的数据
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
import re
matplotlib.use('Agg')

def parse_txt_data(txt_content):
    """
    解析txt文件内容
    格式: agent(x, y, angle)score, agent(x, y, angle)score, ...
    返回: [{'agent': 'IND', 'x': 126.197, 'y': 77.828, 'angle': 0, 'score': 6.07}, ...]
    """
    actions = []
    
    # 正则表达式匹配: agent(x, y, angle)score
    pattern = r'(\d+)\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)([\d.]+)'
    matches = re.findall(pattern, txt_content)
    
    for match in matches:
        agent_num = int(match[0])
        x = float(match[1])
        y = float(match[2])
        angle = float(match[3])
        score = float(match[4])
        
        # 根据agent数字确定类型
        if agent_num == 2:
            agent = 'EDU'
            size = 'L'
        elif agent_num == 3:
            agent = 'EDU'
            size = 'A'
        elif agent_num == 5:
            agent = 'IND'
            size = 'L'
        elif agent_num == 8:
            agent = 'EDU'
            size = 'B'
        else:
            agent = f'AGENT_{agent_num}'
            size = 'L'
        
        actions.append({
            'agent': agent,
            'x': x,
            'y': y,
            'angle': angle,
            'score': score,
            'size': size,  # 根据agent数字确定size类型
            'slot_id': f's_{len(actions)}',  # 生成slot_id
            'cost': 1000,  # 默认cost
            'reward': score * 100,  # 根据score估算reward
            'prestige': 0.3 if agent == 'EDU' else -0.03  # 默认prestige
        })
    
    return actions

def create_action_table(month, agent, actions, output_path, budget_info=None):
    """
    创建动作表格图片
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
            action.get('size', 'L'),
            f"({action.get('x', 0):.1f}, {action.get('y', 0):.1f})",
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
    headers = ['#', 'Agent', 'Size', 'Position', 'Cost', 'Reward', 'Budget']
    
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
        colWidths=[0.08, 0.12, 0.08, 0.20, 0.15, 0.15, 0.22]
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

def process_txt_data(txt_dir, max_months=30):
    """处理txt格式的数据"""
    print("="*80)
    print("处理TXT格式数据")
    print("="*80)
    
    table_dir = os.path.join(txt_dir, 'action_tables')
    os.makedirs(table_dir, exist_ok=True)
    
    # 模拟budget
    budgets = {'IND': 10000, 'EDU': 10000}
    
    for month in range(max_months + 1):  # 包括month 0
        fname = f'chosen_month_{month:02d}.txt'
        fpath = os.path.join(txt_dir, fname)
        
        if not os.path.exists(fpath):
            continue
        
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            continue
        
        # 解析数据
        actions = parse_txt_data(content)
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
    
    print(f"\nTXT处理完成")
    print(f"输出目录: {table_dir}")
    print(f"最终Budget - IND: {budgets['IND']}, EDU: {budgets['EDU']}")

def main():
    parser = argparse.ArgumentParser(description='从txt文件生成动作表格可视化')
    parser.add_argument('--txt_dir', default='enhanced_simulation_v4_1_output/v4_txt/1015-1',
                       help='txt文件目录')
    parser.add_argument('--max_months', type=int, default=30,
                       help='最大处理月份数')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.txt_dir):
        print(f"Error: {args.txt_dir} not found!")
        return
    
    process_txt_data(args.txt_dir, args.max_months)
    
    print("\n" + "="*80)
    print("所有表格生成完成！")
    print("="*80)

if __name__ == '__main__':
    main()
