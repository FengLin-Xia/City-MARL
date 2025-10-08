#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试训练历史中的动作得分
"""

import json
import os
import numpy as np
from typing import Dict, List

def debug_training_history():
    """调试训练历史中的动作得分"""
    print("=== 调试训练历史中的动作得分 ===")
    
    history_path = "models/v4_1_rl/slot_selection_history.json"
    if not os.path.exists(history_path):
        print(f"训练历史文件不存在: {history_path}")
        return
    
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    episodes = history.get('episodes', [])
    print(f"找到 {len(episodes)} 个episodes")
    
    # 分析每个episode的step数据
    all_action_scores = []
    all_sequence_scores = []
    
    for episode in episodes:
        episode_id = episode.get('episode_id', 0)
        steps = episode.get('steps', [])
        print(f"\nEpisode {episode_id}:")
        
        for step in steps:
            step_id = step.get('step', 0)
            month = step.get('month', 0)
            agent = step.get('agent', '')
            
            # 检查不同的得分字段
            action_scores = step.get('action_scores', [])
            sequence_score = step.get('sequence_score', 0.0)
            available_actions_count = step.get('available_actions_count', 0)
            
            if action_scores:
                all_action_scores.extend(action_scores)
                avg_action_score = np.mean(action_scores)
                print(f"  Step {step_id} (Month {month}, {agent}): {len(action_scores)} actions, avg_score={avg_action_score:.2f}")
            
            if sequence_score != 0:
                all_sequence_scores.append(sequence_score)
                print(f"    Sequence score: {sequence_score:.2f}")
            
            # 检查detailed_actions
            detailed_actions = step.get('detailed_actions', [])
            if detailed_actions:
                print(f"    Detailed actions: {len(detailed_actions)}")
                for i, action in enumerate(detailed_actions[:2]):  # 只显示前2个
                    action_score = action.get('score', 0.0)
                    print(f"      Action {i}: score={action_score:.2f}")
    
    # 统计所有得分
    print(f"\n=== 得分统计 ===")
    if all_action_scores:
        print(f"所有动作得分:")
        print(f"  数量: {len(all_action_scores)}")
        print(f"  最小值: {np.min(all_action_scores):.2f}")
        print(f"  最大值: {np.max(all_action_scores):.2f}")
        print(f"  平均值: {np.mean(all_action_scores):.2f}")
        print(f"  中位数: {np.median(all_action_scores):.2f}")
        
        # 检查负数比例
        negative_scores = [s for s in all_action_scores if s < 0]
        print(f"  负数得分: {len(negative_scores)} ({len(negative_scores)/len(all_action_scores)*100:.1f}%)")
    
    if all_sequence_scores:
        print(f"\n所有序列得分:")
        print(f"  数量: {len(all_sequence_scores)}")
        print(f"  最小值: {np.min(all_sequence_scores):.2f}")
        print(f"  最大值: {np.max(all_sequence_scores):.2f}")
        print(f"  平均值: {np.mean(all_sequence_scores):.2f}")
    
    # 检查数据来源
    print(f"\n=== 数据来源分析 ===")
    print("检查训练历史中的得分数据来源:")
    print("1. action_scores: 来自run_single_episode中的experience记录")
    print("2. sequence_score: 来自selected_sequence.score")
    print("3. detailed_actions[].score: 来自action.score")
    
    # 检查是否有score字段被错误计算
    if all_action_scores and np.mean(all_action_scores) < 0:
        print("\n*** 发现问题 ***")
        print("action_scores平均值 < 0，说明在经验收集中记录的score有问题")
        print("可能的原因:")
        print("1. selected_sequence.score计算错误")
        print("2. 或者action.score在某个环节被错误修改")

if __name__ == "__main__":
    debug_training_history()


