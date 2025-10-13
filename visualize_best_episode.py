#!/usr/bin/env python3
"""
可视化最佳Episode的槽位选择历史
分析Episode 7 (update 7) - 最高episode_return: 148.36
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Any
import argparse

def load_slot_selection_history(history_path: str) -> Dict:
    """加载槽位选择历史数据"""
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_best_episode(history: Dict) -> Dict:
    """找到episode_return最高的episode"""
    episodes = history.get('episodes', [])
    if not episodes:
        raise ValueError("No episodes found in history")
    
    best_episode = max(episodes, key=lambda ep: ep['episode_return'])
    return best_episode

def visualize_best_episode(history_path: str, output_dir: str = None):
    """可视化最佳episode的详细信息"""
    
    # 加载数据
    history = load_slot_selection_history(history_path)
    best_episode = find_best_episode(history)
    
    if output_dir is None:
        output_dir = os.path.dirname(history_path)
    
    print(f"最佳Episode: ID {best_episode['episode_id']} (Update {best_episode['update']})")
    print(f"Episode Return: {best_episode['episode_return']:.2f}")
    print(f"总选择次数: {best_episode['summary']['total_selections']}")
    print(f"EDU选择: {best_episode['summary']['edu_selections']}, IND选择: {best_episode['summary']['ind_selections']}")
    print(f"平均动作分数: {best_episode['summary']['avg_action_score']:.4f}")
    
    steps = best_episode['steps']
    
    # 创建大图
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 月度奖励趋势 (EDU vs IND)
    ax1 = plt.subplot(3, 3, 1)
    months = [step['month'] for step in steps]
    edu_rewards = [step['reward'] for step in steps if step['agent'] == 'EDU']
    ind_rewards = [step['reward'] for step in steps if step['agent'] == 'IND']
    edu_months = [step['month'] for step in steps if step['agent'] == 'EDU']
    ind_months = [step['month'] for step in steps if step['agent'] == 'IND']
    
    ax1.plot(edu_months, edu_rewards, 'o-', label='EDU', color='blue', linewidth=2, markersize=6)
    ax1.plot(ind_months, ind_rewards, 's-', label='IND', color='red', linewidth=2, markersize=6)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Reward')
    ax1.set_title('Monthly Rewards by Agent')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 动作分数分布
    ax2 = plt.subplot(3, 3, 2)
    action_scores = [step['action_scores'][0] for step in steps]  # 每个step只有一个动作
    colors = ['blue' if step['agent'] == 'EDU' else 'red' for step in steps]
    ax2.scatter(months, action_scores, c=colors, alpha=0.7, s=60)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Action Score')
    ax2.set_title('Action Scores by Month')
    ax2.grid(True, alpha=0.3)
    
    # 3. 可用动作数量变化
    ax3 = plt.subplot(3, 3, 3)
    available_actions = [step['available_actions_count'] for step in steps]
    candidate_slots = [step['candidate_slots_count'] for step in steps]
    
    ax3.plot(months, available_actions, 'o-', label='Available Actions', color='green')
    ax3.plot(months, candidate_slots, 's-', label='Candidate Slots', color='orange')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Count')
    ax3.set_title('Available Actions vs Candidate Slots')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 智能体选择模式
    ax4 = plt.subplot(3, 3, 4)
    agent_counts = {'EDU': best_episode['summary']['edu_selections'], 
                   'IND': best_episode['summary']['ind_selections']}
    colors = ['blue', 'red']
    wedges, texts, autotexts = ax4.pie(agent_counts.values(), labels=agent_counts.keys(), 
                                      colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Agent Selection Distribution')
    
    # 5. 月度累积奖励
    ax5 = plt.subplot(3, 3, 5)
    cumulative_rewards = np.cumsum([step['reward'] for step in steps])
    ax5.plot(months, cumulative_rewards, 'o-', color='purple', linewidth=2, markersize=6)
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Cumulative Reward')
    ax5.set_title('Cumulative Reward Over Time')
    ax5.grid(True, alpha=0.3)
    
    # 6. 动作分数直方图
    ax6 = plt.subplot(3, 3, 6)
    edu_scores = [step['action_scores'][0] for step in steps if step['agent'] == 'EDU']
    ind_scores = [step['action_scores'][0] for step in steps if step['agent'] == 'IND']
    
    ax6.hist(edu_scores, bins=10, alpha=0.7, label='EDU', color='blue')
    ax6.hist(ind_scores, bins=10, alpha=0.7, label='IND', color='red')
    ax6.set_xlabel('Action Score')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Action Score Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 槽位选择热力图 (前20个最常选择的槽位)
    ax7 = plt.subplot(3, 3, 7)
    slot_counts = {}
    for step in steps:
        for slot_list in step['selected_slots']:
            for slot_id in slot_list:
                slot_counts[slot_id] = slot_counts.get(slot_id, 0) + 1
    
    if slot_counts:
        top_slots = sorted(slot_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        slot_ids = [item[0] for item in top_slots]
        counts = [item[1] for item in top_slots]
        
        bars = ax7.barh(range(len(slot_ids)), counts, color='lightgreen', alpha=0.7)
        ax7.set_yticks(range(len(slot_ids)))
        ax7.set_yticklabels(slot_ids, fontsize=8)
        ax7.set_xlabel('Selection Count')
        ax7.set_title('Top 20 Selected Slots')
        ax7.invert_yaxis()
    
    # 8. 序列分数vs奖励散点图
    ax8 = plt.subplot(3, 3, 8)
    sequence_scores = [step['sequence_score'] for step in steps]
    rewards = [step['reward'] for step in steps]
    colors = ['blue' if step['agent'] == 'EDU' else 'red' for step in steps]
    
    ax8.scatter(sequence_scores, rewards, c=colors, alpha=0.7, s=60)
    ax8.set_xlabel('Sequence Score')
    ax8.set_ylabel('Reward')
    ax8.set_title('Sequence Score vs Reward')
    ax8.grid(True, alpha=0.3)
    
    # 9. 月度效率分析 (奖励/可用动作)
    ax9 = plt.subplot(3, 3, 9)
    efficiency = [step['reward'] / step['available_actions_count'] for step in steps]
    colors = ['blue' if step['agent'] == 'EDU' else 'red' for step in steps]
    
    ax9.scatter(months, efficiency, c=colors, alpha=0.7, s=60)
    ax9.set_xlabel('Month')
    ax9.set_ylabel('Reward/Available Actions')
    ax9.set_title('Decision Efficiency by Month')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'Best Episode Analysis (Episode {best_episode["episode_id"]}, Update {best_episode["update"]})\n'
                f'Episode Return: {best_episode["episode_return"]:.2f}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'best_episode_{best_episode["episode_id"]}_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n最佳Episode分析图已保存到: {output_path}")
    
    # 输出详细统计信息
    print("\n=== 详细统计信息 ===")
    print(f"Episode ID: {best_episode['episode_id']}")
    print(f"Update: {best_episode['update']}")
    print(f"Episode Return: {best_episode['episode_return']:.4f}")
    print(f"总步数: {len(steps)}")
    print(f"EDU选择次数: {best_episode['summary']['edu_selections']}")
    print(f"IND选择次数: {best_episode['summary']['ind_selections']}")
    print(f"平均动作分数: {best_episode['summary']['avg_action_score']:.4f}")
    print(f"平均序列分数: {best_episode['summary']['avg_sequence_score']:.4f}")
    print(f"唯一槽位数量: {len(best_episode['summary']['unique_slots_selected'])}")
    
    # 月度详细分析
    print("\n=== 月度详细分析 ===")
    for step in steps:
        print(f"Month {step['month']} ({step['agent']}): "
              f"Reward={step['reward']:.2f}, "
              f"ActionScore={step['action_scores'][0]:.4f}, "
              f"SequenceScore={step['sequence_score']:.4f}, "
              f"AvailableActions={step['available_actions_count']}, "
              f"SelectedSlot={step['selected_slots'][0][0] if step['selected_slots'] else 'None'}")
    
    return best_episode

def main():
    parser = argparse.ArgumentParser(description='可视化最佳Episode的槽位选择历史')
    parser.add_argument('--history_path', type=str, 
                       default='models/v4_1_rl/slot_selection_history.json',
                       help='槽位选择历史文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认与历史文件同目录)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.history_path):
        print(f"错误: 历史文件不存在: {args.history_path}")
        return
    
    try:
        best_episode = visualize_best_episode(args.history_path, args.output_dir)
        print(f"\n可视化完成! 最佳Episode是Episode {best_episode['episode_id']}")
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



