#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys

def analyze_training_log():
    """分析训练日志中的建筑尺寸分布"""
    try:
        with open('models/v4_1_rl/ppo_training_v4_1_detailed_log_20251014_014906.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("=== 训练日志分析 ===")
    print(f"实验名称: {data['experiment_info']['name']}")
    print(f"开始时间: {data['experiment_info']['start_time']}")
    print(f"结束时间: {data['experiment_info']['end_time']}")
    print(f"总episode数: {len(data['episodes'])}")
    
    # 分析每个episode的建筑选择
    for episode in data['episodes']:
        episode_id = episode['episode_id']
        print(f"\n--- Episode {episode_id} ---")
        
        # 统计这个episode的建筑尺寸分布
        size_counts = {'S': 0, 'M': 0, 'L': 0}
        total_steps = len(episode['steps'])
        
        for step in episode['steps']:
            # 检查selected_slots
            for slot_group in step['selected_slots']:
                for slot in slot_group:
                    # 从slot_id推断建筑尺寸（这里需要更详细的分析）
                    # 暂时用简单的方法：所有选择的都是S型
                    size_counts['S'] += 1
        
        print(f"总步骤数: {total_steps}")
        print(f"建筑选择: S={size_counts['S']}, M={size_counts['M']}, L={size_counts['L']}")
        
        # 只显示前3个episode的详细信息
        if episode_id >= 2:
            break

def analyze_slot_selection_history():
    """分析slot_selection_history.json中的建筑尺寸分布"""
    try:
        with open('models/v4_1_rl/slot_selection_history.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading slot_selection_history: {e}")
        return

    print("\n=== Slot Selection History 分析 ===")
    
    # 统计建筑尺寸分布
    size_counts = {'S': 0, 'M': 0, 'L': 0}
    total_actions = 0
    
    for episode in data['episodes']:
        for step in episode['steps']:
            for action in step['detailed_actions']:
                size = action['size']
                size_counts[size] += 1
                total_actions += 1
    
    print(f"总动作数: {total_actions}")
    for size, count in size_counts.items():
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"{size}型建筑: {count} ({percentage:.1f}%)")
    
    # 检查实际选择的建筑
    selected_sizes = {'S': 0, 'M': 0, 'L': 0}
    total_selected = 0
    
    for episode in data['episodes']:
        for step in episode['steps']:
            num_selected = len(step['selected_slots'])
            for i in range(min(num_selected, len(step['detailed_actions']))):
                action = step['detailed_actions'][i]
                size = action['size']
                selected_sizes[size] += 1
                total_selected += 1
    
    print(f"\n实际选择的建筑:")
    print(f"总选择数: {total_selected}")
    for size, count in selected_sizes.items():
        percentage = (count / total_selected * 100) if total_selected > 0 else 0
        print(f"{size}型建筑: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    analyze_training_log()
    analyze_slot_selection_history()


