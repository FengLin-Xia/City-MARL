#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys

def analyze_training_results():
    """分析训练结果中的建筑尺寸分布"""
    try:
        with open('models/v4_1_rl/slot_selection_history.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 统计建筑尺寸分布
    size_counts = {'S': 0, 'M': 0, 'L': 0}
    total_actions = 0

    for episode in data['episodes']:
        for step in episode['steps']:
            for action in step['detailed_actions']:
                size = action['size']
                size_counts[size] += 1
                total_actions += 1

    print('建筑尺寸分布:')
    for size, count in size_counts.items():
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        print(f'{size}: {count} ({percentage:.1f}%)')
    print(f'总计: {total_actions}')

    # 检查实际选择的建筑（通过selected_slots推断）
    selected_sizes = {'S': 0, 'M': 0, 'L': 0}
    total_selected = 0
    
    for episode in data['episodes']:
        for step in episode['steps']:
            # 通过selected_slots的数量来推断实际选择了多少个动作
            num_selected = len(step['selected_slots'])
            
            # 检查前几个动作的尺寸
            for i in range(min(num_selected, len(step['detailed_actions']))):
                action = step['detailed_actions'][i]
                size = action['size']
                selected_sizes[size] += 1
                total_selected += 1

    print('\n实际选择的建筑尺寸分布:')
    for size, count in selected_sizes.items():
        percentage = (count / total_selected * 100) if total_selected > 0 else 0
        print(f'{size}: {count} ({percentage:.1f}%)')
    print(f'总计选择: {total_selected}')
    
    # 分析M型建筑的选择情况
    print('\nM型建筑选择分析:')
    m_count = 0
    for episode in data['episodes']:
        for step in episode['steps']:
            num_selected = len(step['selected_slots'])
            for i in range(min(num_selected, len(step['detailed_actions']))):
                action = step['detailed_actions'][i]
                if action['size'] == 'M':
                    m_count += 1
                    print(f"Episode {episode['episode_id']}, Month {step['month']}: M型建筑选择")
                    print(f"  Agent: {action['agent']}, Slot: {action['slot_id']}")
                    print(f"  Cost: {action['cost']}, Reward: {action['reward']}, Score: {action['score']}")
    
    if m_count == 0:
        print("没有选择任何M型建筑")

if __name__ == "__main__":
    analyze_training_results()


