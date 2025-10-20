#!/usr/bin/env python3
"""
检查训练日志中的强制探索和对岸相关日志
"""

import json
import sys
from pathlib import Path

def check_training_logs():
    """检查训练日志"""
    log_file = "models/v4_1_rl/ppo_training_v4_1_detailed_log_20251018_045815.json"
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计各种日志
        force_explore_count = 0
        force_select_count = 0
        other_side_detect_count = 0
        other_side_reward_count = 0
        abc_count = 0
        
        for episode in data['episodes']:
            for step in episode['steps']:
                if 'logs' in step:
                    for log in step['logs']:
                        if '强制对岸探索' in log:
                            force_explore_count += 1
                        if '强制选择对岸' in log:
                            force_select_count += 1
                        if '对岸检测' in log:
                            other_side_detect_count += 1
                        if '对岸探索奖励' in log:
                            other_side_reward_count += 1
                        if 'A型' in log or 'B型' in log or 'C型' in log:
                            abc_count += 1
        
        print("=== 训练日志分析结果 ===")
        print(f"强制对岸探索日志: {force_explore_count}")
        print(f"强制选择对岸日志: {force_select_count}")
        print(f"对岸检测日志: {other_side_detect_count}")
        print(f"对岸探索奖励日志: {other_side_reward_count}")
        print(f"A/B/C相关日志: {abc_count}")
        
        if force_explore_count == 0 and force_select_count == 0:
            print("\nX 没有发现强制探索日志！")
            print("这说明强制探索机制可能没有触发")
        else:
            print("\nOK 发现了强制探索日志")
            
        if other_side_detect_count == 0:
            print("X 没有发现对岸检测日志！")
        else:
            print("OK 发现了对岸检测日志")
            
        if abc_count == 0:
            print("X 没有发现A/B/C相关日志！")
        else:
            print("OK 发现了A/B/C相关日志")
            
    except Exception as e:
        print(f"读取日志文件失败: {e}")

if __name__ == "__main__":
    check_training_logs()
