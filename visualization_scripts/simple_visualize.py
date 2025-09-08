#!/usr/bin/env python3
"""
简化版训练数据可视化脚本
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("简化版训练数据可视化")
    
    # 查找训练数据文件
    data_dir = "training_data"
    stats_files = []
    for file in os.listdir(data_dir):
        if file.startswith("improved_training_stats_") and file.endswith(".json"):
            stats_files.append(file)
    
    if not stats_files:
        print("未找到训练统计文件")
        return
    
    # 加载最新的训练数据
    latest_file = sorted(stats_files)[-1]
    file_path = os.path.join(data_dir, latest_file)
    
    with open(file_path, 'r') as f:
        training_data = json.load(f)
    
    print(f"加载了训练数据: {latest_file}")
    print(f"总episodes: {training_data.get('total_episodes', 0)}")
    print(f"成功次数: {training_data.get('total_success', 0)}")
    print(f"最终成功率: {training_data.get('final_success_rate', 0):.1%}")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练过程分析', fontsize=16)
    
    episodes = list(range(1, len(training_data.get('episode_rewards', [])) + 1))
    rewards = training_data.get('episode_rewards', [])
    success_rates = training_data.get('success_rates', [])
    lengths = training_data.get('episode_lengths', [])
    
    # 1. 成功率曲线
    if success_rates:
        axes[0, 0].plot(episodes, success_rates, 'b-', linewidth=2)
        axes[0, 0].set_title('成功率变化')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('成功率')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
    
    # 2. 奖励曲线
    if rewards:
        # 计算移动平均
        window_size = min(50, len(rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = episodes[window_size-1:]
            axes[0, 1].plot(moving_avg_episodes, moving_avg, 'r-', linewidth=2, label=f'移动平均({window_size})')
        
        axes[0, 1].plot(episodes, rewards, 'gray', alpha=0.3, linewidth=0.5)
        axes[0, 1].set_title('奖励变化')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('总奖励')
        axes[0, 1].grid(True, alpha=0.3)
        if window_size > 1:
            axes[0, 1].legend()
    
    # 3. 路径长度曲线
    if lengths:
        if window_size > 1:
            moving_avg_length = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(moving_avg_episodes, moving_avg_length, 'g-', linewidth=2, label=f'移动平均({window_size})')
        
        axes[1, 0].plot(episodes, lengths, 'gray', alpha=0.3, linewidth=0.5)
        axes[1, 0].set_title('路径长度变化')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('路径长度')
        axes[1, 0].grid(True, alpha=0.3)
        if window_size > 1:
            axes[1, 0].legend()
    
    # 4. 奖励分布直方图
    if rewards:
        axes[1, 1].hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'平均值: {np.mean(rewards):.2f}')
        axes[1, 1].set_title('奖励分布')
        axes[1, 1].set_xlabel('总奖励')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "simple_training_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练分析图已保存到: {save_path}")
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print("\n=== 训练统计信息 ===")
    print(f"总Episodes: {training_data.get('total_episodes', 0)}")
    print(f"成功次数: {training_data.get('total_success', 0)}")
    print(f"最终成功率: {training_data.get('final_success_rate', 0):.1%}")
    print(f"平均奖励: {training_data.get('final_avg_reward', 0):.2f}")
    print(f"平均路径长度: {training_data.get('final_avg_length', 0):.1f}")
    
    if rewards:
        print(f"\n奖励统计:")
        print(f"  奖励范围: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
        print(f"  奖励标准差: {np.std(rewards):.2f}")
        print(f"  奖励中位数: {np.median(rewards):.2f}")
    
    print(f"\n起点: {training_data.get('start_point', [0, 0])}")
    print(f"终点: {training_data.get('goal_point', [0, 0])}")
    print(f"地形文件: {training_data.get('terrain_file', 'N/A')}")

if __name__ == "__main__":
    main()

