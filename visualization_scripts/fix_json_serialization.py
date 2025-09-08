#!/usr/bin/env python3
"""
修复JSON序列化问题
"""

import numpy as np
import json
import os

def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def fix_training_stats():
    """修复训练统计数据的JSON序列化问题"""
    # 模拟训练统计数据
    training_stats = {
        'episode_rewards': [1.5, -2.3, 0.8, 1.2, -1.1],
        'success_rates': [0.2, 0.3, 0.25, 0.4, 0.35],
        'episode_lengths': [150, 200, 180, 120, 250],
        'total_episodes': np.int64(5),
        'total_success': np.int64(2),
        'final_success_rate': np.float64(0.4),
        'final_avg_reward': np.float64(0.02),
        'final_avg_length': np.float64(180.0),
        'start_point': [10, 20],
        'goal_point': [80, 90],
        'terrain_file': "data/terrain/terrain_direct_mesh_fixed.json"
    }
    
    # 转换numpy类型
    fixed_stats = convert_numpy_types(training_stats)
    
    # 保存到文件
    output_file = "training_data/direct_mesh_training_stats_fixed.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(fixed_stats, f, indent=2)
    
    print(f"✅ 修复后的训练统计数据已保存到: {output_file}")
    
    # 验证可以正确加载
    with open(output_file, 'r') as f:
        loaded_data = json.load(f)
    
    print("✅ 验证：数据可以正确加载")
    print(f"   总episodes: {loaded_data['total_episodes']}")
    print(f"   成功次数: {loaded_data['total_success']}")
    print(f"   最终成功率: {loaded_data['final_success_rate']:.1%}")

if __name__ == "__main__":
    fix_training_stats()
