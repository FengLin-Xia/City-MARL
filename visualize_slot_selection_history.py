#!/usr/bin/env python3
"""
槽位选择历史可视化脚本
用于分析训练过程中的槽位选择模式
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import argparse
import os

def load_slot_history(history_path: str):
    """加载槽位选择历史"""
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_slot_selection_patterns(history):
    """分析槽位选择模式"""
    print("=" * 60)
    print("槽位选择历史分析")
    print("=" * 60)
    
    # 基本统计
    total_episodes = len(history['episodes'])
    total_selections = history['total_selections']
    
    print(f"总Episode数: {total_episodes}")
    print(f"总选择次数: {total_selections}")
    
    if total_episodes == 0:
        print("没有Episode数据")
        return
    
    # 按智能体统计
    agent_selections = defaultdict(int)
    slot_frequency = Counter()
    monthly_selections = defaultdict(int)
    action_scores_by_agent = defaultdict(list)
    sequence_scores_by_agent = defaultdict(list)
    
    for episode in history['episodes']:
        for step in episode['steps']:
            agent = step['agent']
            month = step['month']
            
            agent_selections[agent] += 1
            monthly_selections[month] += 1
            
            # 收集选择的槽位
            for slot_list in step['selected_slots']:
                for slot in slot_list:
                    slot_frequency[slot] += 1
            
            # 收集得分
            action_scores_by_agent[agent].extend(step['action_scores'])
            sequence_scores_by_agent[agent].append(step['sequence_score'])
    
    print(f"\n智能体选择统计:")
    for agent, count in agent_selections.items():
        print(f"  {agent}: {count} 次 ({count/total_selections*100:.1f}%)")
    
    print(f"\n月度选择分布:")
    for month in sorted(monthly_selections.keys()):
        count = monthly_selections[month]
        print(f"  第{month}月: {count} 次")
    
    print(f"\n最常选择的槽位 (Top 10):")
    for slot, count in slot_frequency.most_common(10):
        print(f"  {slot}: {count} 次")
    
    print(f"\n动作得分统计:")
    for agent, scores in action_scores_by_agent.items():
        if scores:
            print(f"  {agent}: 平均={np.mean(scores):.3f}, 标准差={np.std(scores):.3f}")
    
    return {
        'agent_selections': dict(agent_selections),
        'slot_frequency': dict(slot_frequency),
        'monthly_selections': dict(monthly_selections),
        'action_scores_by_agent': {k: v for k, v in action_scores_by_agent.items()},
        'sequence_scores_by_agent': {k: v for k, v in sequence_scores_by_agent.items()}
    }

def visualize_slot_selection(history, output_dir='visualization_output'):
    """可视化槽位选择历史"""
    os.makedirs(output_dir, exist_ok=True)
    
    analysis = analyze_slot_selection_patterns(history)
    
    # 1. 智能体选择分布饼图
    plt.figure(figsize=(10, 6))
    agents = list(analysis['agent_selections'].keys())
    counts = list(analysis['agent_selections'].values())
    colors = ['#ff9999', '#66b3ff']
    
    plt.subplot(1, 2, 1)
    plt.pie(counts, labels=agents, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('智能体选择分布')
    
    # 2. 月度选择趋势
    plt.subplot(1, 2, 2)
    months = sorted(analysis['monthly_selections'].keys())
    month_counts = [analysis['monthly_selections'][m] for m in months]
    plt.bar(months, month_counts, color='skyblue', alpha=0.7)
    plt.xlabel('月份')
    plt.ylabel('选择次数')
    plt.title('月度选择趋势')
    plt.xticks(months)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slot_selection_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 动作得分分布
    if any(analysis['action_scores_by_agent'].values()):
        plt.figure(figsize=(12, 5))
        
        # EDU动作得分分布
        plt.subplot(1, 2, 1)
        edu_scores = analysis['action_scores_by_agent'].get('EDU', [])
        if edu_scores:
            plt.hist(edu_scores, bins=20, alpha=0.7, color='#ff9999', label='EDU')
            plt.axvline(np.mean(edu_scores), color='red', linestyle='--', label=f'平均: {np.mean(edu_scores):.3f}')
            plt.xlabel('动作得分')
            plt.ylabel('频次')
            plt.title('EDU动作得分分布')
            plt.legend()
        
        # IND动作得分分布
        plt.subplot(1, 2, 2)
        ind_scores = analysis['action_scores_by_agent'].get('IND', [])
        if ind_scores:
            plt.hist(ind_scores, bins=20, alpha=0.7, color='#66b3ff', label='IND')
            plt.axvline(np.mean(ind_scores), color='blue', linestyle='--', label=f'平均: {np.mean(ind_scores):.3f}')
            plt.xlabel('动作得分')
            plt.ylabel('频次')
            plt.title('IND动作得分分布')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'action_scores_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 最常选择的槽位
    if analysis['slot_frequency']:
        plt.figure(figsize=(12, 6))
        top_slots = dict(list(analysis['slot_frequency'].items())[:20])
        
        slots = list(top_slots.keys())
        counts = list(top_slots.values())
        
        plt.barh(range(len(slots)), counts, color='lightgreen', alpha=0.7)
        plt.yticks(range(len(slots)), slots)
        plt.xlabel('选择次数')
        plt.title('最常选择的槽位 (Top 20)')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_selected_slots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n可视化图表已保存到: {output_dir}/")
    print(f"  - slot_selection_overview.png: 智能体分布和月度趋势")
    print(f"  - action_scores_distribution.png: 动作得分分布")
    print(f"  - top_selected_slots.png: 最常选择的槽位")

def main():
    parser = argparse.ArgumentParser(description='槽位选择历史可视化')
    parser.add_argument('--history_path', type=str, 
                       default='models/v4_1_rl/slot_selection_history.json',
                       help='槽位选择历史文件路径')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.history_path):
        print(f"历史文件不存在: {args.history_path}")
        return
    
    print(f"加载槽位选择历史: {args.history_path}")
    history = load_slot_history(args.history_path)
    
    visualize_slot_selection(history, args.output_dir)

if __name__ == "__main__":
    main()



