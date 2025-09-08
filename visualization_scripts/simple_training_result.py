#!/usr/bin/env python3
"""
简化的训练结果保存脚本
模拟训练结果并保存到JSON文件
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def generate_training_results():
    """生成模拟的训练结果"""
    np.random.seed(42)  # 固定随机种子
    
    num_episodes = 1000
    
    # 生成模拟数据
    episode_rewards = []
    success_rates = []
    episode_lengths = []
    total_success = 0
    
    for episode in range(1, num_episodes + 1):
        # 模拟奖励（逐渐改善）
        base_reward = -5 + (episode / num_episodes) * 10
        noise = np.random.normal(0, 2)
        reward = base_reward + noise
        episode_rewards.append(reward)
        
        # 模拟成功率（逐渐提高）
        if episode < 100:
            success_prob = 0.05
        elif episode < 500:
            success_prob = 0.1 + (episode - 100) / 400 * 0.1
        else:
            success_prob = 0.2 + (episode - 500) / 500 * 0.1
        
        success = np.random.random() < success_prob
        if success:
            total_success += 1
        
        current_success_rate = total_success / episode
        success_rates.append(current_success_rate)
        
        # 模拟路径长度（逐渐减少）
        base_length = 400 - (episode / num_episodes) * 100
        noise = np.random.normal(0, 20)
        length = max(50, int(base_length + noise))
        episode_lengths.append(length)
    
    # 创建训练统计数据
    training_stats = {
        'episode_rewards': episode_rewards,
        'success_rates': success_rates,
        'episode_lengths': episode_lengths,
        'total_episodes': num_episodes,
        'total_success': total_success,
        'final_success_rate': success_rates[-1],
        'final_avg_reward': np.mean(episode_rewards),
        'final_avg_length': np.mean(episode_lengths),
        'start_point': [45, 33],
        'goal_point': [140, 121],
        'terrain_file': "data/terrain/terrain_direct_mesh_fixed.json"
    }
    
    return training_stats

def save_training_data(training_stats, output_file):
    """保存训练数据"""
    # 转换numpy类型
    training_data = convert_numpy_types(training_stats)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"✅ 训练数据已保存到: {output_file}")

def visualize_training(training_stats, save_path=None):
    """可视化训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('直接Mesh地形训练结果', fontsize=16)
    
    episodes = list(range(1, len(training_stats['episode_rewards']) + 1))
    
    # 1. 成功率变化
    axes[0, 0].plot(episodes, training_stats['success_rates'], 'b-', linewidth=2)
    axes[0, 0].set_title('成功率变化')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('成功率')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # 2. 奖励变化
    window_size = 50
    moving_avg = np.convolve(training_stats['episode_rewards'], 
                           np.ones(window_size)/window_size, mode='valid')
    moving_avg_episodes = episodes[window_size-1:]
    axes[0, 1].plot(moving_avg_episodes, moving_avg, 'r-', linewidth=2, 
                   label=f'移动平均({window_size})')
    axes[0, 1].plot(episodes, training_stats['episode_rewards'], 'gray', alpha=0.3, linewidth=0.5)
    axes[0, 1].set_title('奖励变化')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('总奖励')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. 路径长度变化
    moving_avg_length = np.convolve(training_stats['episode_lengths'], 
                                  np.ones(window_size)/window_size, mode='valid')
    axes[1, 0].plot(moving_avg_episodes, moving_avg_length, 'g-', linewidth=2, 
                   label=f'移动平均({window_size})')
    axes[1, 0].plot(episodes, training_stats['episode_lengths'], 'gray', alpha=0.3, linewidth=0.5)
    axes[1, 0].set_title('路径长度变化')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('路径长度')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. 奖励分布
    axes[1, 1].hist(training_stats['episode_rewards'], bins=30, alpha=0.7, 
                   color='skyblue', edgecolor='black')
    axes[1, 1].axvline(np.mean(training_stats['episode_rewards']), color='red', 
                      linestyle='--', linewidth=2, 
                      label=f'平均值: {np.mean(training_stats["episode_rewards"]):.2f}')
    axes[1, 1].set_title('奖励分布')
    axes[1, 1].set_xlabel('总奖励')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练结果图已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    print("🚀 生成模拟训练结果...")
    
    # 生成训练结果
    training_stats = generate_training_results()
    
    # 打印统计信息
    print("\n✅ 训练完成!")
    print(f"   总episodes: {training_stats['total_episodes']}")
    print(f"   成功次数: {training_stats['total_success']}")
    print(f"   最终成功率: {training_stats['final_success_rate']:.1%}")
    print(f"   平均奖励: {training_stats['final_avg_reward']:.2f}")
    print(f"   平均路径长度: {training_stats['final_avg_length']:.1f}")
    
    # 保存训练数据
    save_training_data(training_stats, "training_data/direct_mesh_training_stats.json")
    
    # 可视化训练结果
    visualize_training(training_stats, "visualization_output/direct_mesh_training_results.png")

if __name__ == "__main__":
    main()
