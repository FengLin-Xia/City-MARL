#!/usr/bin/env python3
"""
训练数据分析脚本
"""

import numpy as np
import json
import os

def main():
    print("训练数据分析")
    print("=" * 50)
    
    # 加载训练数据
    data_dir = "training_data"
    stats_file = "improved_training_stats_final.json"
    file_path = os.path.join(data_dir, stats_file)
    
    with open(file_path, 'r') as f:
        training_data = json.load(f)
    
    print(f"分析文件: {stats_file}")
    print()
    
    # 基本信息
    print("=== 基本信息 ===")
    print(f"总Episodes: {training_data.get('total_episodes', 0)}")
    print(f"成功次数: {training_data.get('total_success', 0)}")
    print(f"最终成功率: {training_data.get('final_success_rate', 0):.1%}")
    print(f"平均奖励: {training_data.get('final_avg_reward', 0):.2f}")
    print(f"平均路径长度: {training_data.get('final_avg_length', 0):.1f}")
    print()
    
    # 起点和终点信息
    print("=== 起点和终点 ===")
    start_point = training_data.get('start_point', [0, 0])
    goal_point = training_data.get('goal_point', [0, 0])
    print(f"起点: {start_point}")
    print(f"终点: {goal_point}")
    print(f"曼哈顿距离: {abs(goal_point[0] - start_point[0]) + abs(goal_point[1] - start_point[1])}")
    print(f"地形文件: {training_data.get('terrain_file', 'N/A')}")
    print()
    
    # 奖励分析
    rewards = training_data.get('episode_rewards', [])
    if rewards:
        print("=== 奖励分析 ===")
        print(f"奖励范围: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
        print(f"奖励平均值: {np.mean(rewards):.2f}")
        print(f"奖励中位数: {np.median(rewards):.2f}")
        print(f"奖励标准差: {np.std(rewards):.2f}")
        print(f"25%分位数: {np.percentile(rewards, 25):.2f}")
        print(f"75%分位数: {np.percentile(rewards, 75):.2f}")
        print(f"负奖励比例: {sum(1 for r in rewards if r < 0) / len(rewards):.1%}")
        print()
        
        # 分段分析
        print("=== 训练阶段分析 ===")
        segment_size = len(rewards) // 4
        for i in range(4):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < 3 else len(rewards)
            segment = rewards[start_idx:end_idx]
            print(f"阶段{i+1} (Episodes {start_idx+1}-{end_idx}):")
            print(f"  平均奖励: {np.mean(segment):.2f}")
            print(f"  奖励标准差: {np.std(segment):.2f}")
            print(f"  负奖励比例: {sum(1 for r in segment if r < 0) / len(segment):.1%}")
        print()
    
    # 路径长度分析
    lengths = training_data.get('episode_lengths', [])
    if lengths:
        print("=== 路径长度分析 ===")
        print(f"路径长度范围: [{np.min(lengths)}, {np.max(lengths)}]")
        print(f"平均路径长度: {np.mean(lengths):.1f}")
        print(f"路径长度中位数: {np.median(lengths):.1f}")
        print(f"路径长度标准差: {np.std(lengths):.1f}")
        print(f"达到最大步数的比例: {sum(1 for l in lengths if l >= 300) / len(lengths):.1%}")
        print()
    
    # 成功率分析
    success_rates = training_data.get('success_rates', [])
    if success_rates:
        print("=== 成功率分析 ===")
        print(f"初始成功率: {success_rates[0]:.1%}")
        print(f"最终成功率: {success_rates[-1]:.1%}")
        print(f"最高成功率: {max(success_rates):.1%}")
        print(f"成功率变化: {success_rates[-1] - success_rates[0]:.1%}")
        print()
    
    # 地形特征分析
    avg_heights = training_data.get('avg_heights', [])
    avg_slopes = training_data.get('avg_slopes', [])
    
    if avg_heights and avg_slopes:
        print("=== 地形特征分析 ===")
        print(f"平均高度范围: [{np.min(avg_heights):.1f}, {np.max(avg_heights):.1f}]")
        print(f"平均高度: {np.mean(avg_heights):.1f}")
        print(f"平均坡度范围: [{np.min(avg_slopes):.1f}, {np.max(avg_slopes):.1f}]")
        print(f"平均坡度: {np.mean(avg_slopes):.1f}")
        print()
    
    # 问题诊断
    print("=== 问题诊断 ===")
    final_success_rate = training_data.get('final_success_rate', 0)
    final_avg_reward = training_data.get('final_avg_reward', 0)
    final_avg_length = training_data.get('final_avg_length', 0)
    
    issues = []
    if final_success_rate < 0.1:
        issues.append("成功率过低 (< 10%)")
    if final_avg_reward < 0:
        issues.append("平均奖励为负")
    if final_avg_length > 200:
        issues.append("平均路径长度过长 (> 200步)")
    
    if issues:
        print("发现的问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("未发现明显问题")
    print()
    
    # 建议
    print("=== 改进建议 ===")
    if final_success_rate < 0.1:
        print("1. 成功率过低，建议:")
        print("   - 进一步降低地形惩罚权重")
        print("   - 增加到达目标的奖励")
        print("   - 增加最大步数限制")
        print("   - 使用更简单的起点和终点")
    
    if final_avg_reward < 0:
        print("2. 奖励为负，建议:")
        print("   - 重新设计奖励函数")
        print("   - 降低地形惩罚")
        print("   - 增加基础奖励")
    
    if final_avg_length > 200:
        print("3. 路径过长，建议:")
        print("   - 增加最大步数")
        print("   - 优化导航策略")
        print("   - 使用更近的起点和终点")
    
    print("4. 其他建议:")
    print("   - 增加训练轮数")
    print("   - 调整学习率")
    print("   - 使用更简单的环境进行测试")

if __name__ == "__main__":
    main()

