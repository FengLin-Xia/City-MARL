#!/usr/bin/env python3
"""
训练数据可视化脚本
分析保存的训练数据并生成图表
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """训练数据可视化器"""
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = data_dir
        self.training_data = {}
        self.load_training_data()
    
    def load_training_data(self):
        """加载训练数据"""
        print(f"从 {self.data_dir} 加载训练数据...")
        
        # 查找所有训练统计文件
        stats_files = []
        for file in os.listdir(self.data_dir):
            if file.startswith("improved_training_stats_") and file.endswith(".json"):
                stats_files.append(file)
        
        if not stats_files:
            print("未找到训练统计文件")
            return
        
        # 按文件名排序
        stats_files.sort()
        print(f"找到 {len(stats_files)} 个训练统计文件")
        
        # 加载最新的训练数据
        latest_file = stats_files[-1]
        file_path = os.path.join(self.data_dir, latest_file)
        
        with open(file_path, 'r') as f:
            self.training_data = json.load(f)
        
        print(f"加载了训练数据: {latest_file}")
        print(f"总episodes: {self.training_data.get('total_episodes', 0)}")
        print(f"成功次数: {self.training_data.get('total_success', 0)}")
        print(f"最终成功率: {self.training_data.get('final_success_rate', 0):.1%}")
    
    def plot_training_curves(self, save_path: str = None):
        """绘制训练曲线"""
        if not self.training_data:
            print("没有训练数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练过程分析', fontsize=16)
        
        episodes = list(range(1, len(self.training_data.get('episode_rewards', [])) + 1))
        
        # 1. 成功率曲线
        success_rates = self.training_data.get('success_rates', [])
        if success_rates:
            axes[0, 0].plot(episodes, success_rates, 'b-', linewidth=2)
            axes[0, 0].set_title('成功率变化')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('成功率')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 1)
        
        # 2. 奖励曲线
        rewards = self.training_data.get('episode_rewards', [])
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
            axes[0, 1].legend()
        
        # 3. 路径长度曲线
        lengths = self.training_data.get('episode_lengths', [])
        if lengths:
            # 计算移动平均
            if window_size > 1:
                moving_avg_length = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
                axes[1, 0].plot(moving_avg_episodes, moving_avg_length, 'g-', linewidth=2, label=f'移动平均({window_size})')
            
            axes[1, 0].plot(episodes, lengths, 'gray', alpha=0.3, linewidth=0.5)
            axes[1, 0].set_title('路径长度变化')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('路径长度')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # 4. 地形特征分析
        avg_heights = self.training_data.get('avg_heights', [])
        avg_slopes = self.training_data.get('avg_slopes', [])
        
        if avg_heights and avg_slopes:
            # 计算移动平均
            if window_size > 1:
                moving_avg_height = np.convolve(avg_heights, np.ones(window_size)/window_size, mode='valid')
                moving_avg_slope = np.convolve(avg_slopes, np.ones(window_size)/window_size, mode='valid')
                
                ax1 = axes[1, 1]
                ax2 = ax1.twinx()
                
                line1 = ax1.plot(moving_avg_episodes, moving_avg_height, 'b-', linewidth=2, label='平均高度')
                line2 = ax2.plot(moving_avg_episodes, moving_avg_slope, 'r-', linewidth=2, label='平均坡度')
                
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('平均高度', color='b')
                ax2.set_ylabel('平均坡度', color='r')
                ax1.set_title('地形特征变化')
                ax1.grid(True, alpha=0.3)
                
                # 合并图例
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线图已保存到: {save_path}")
        
        plt.show()
    
    def plot_reward_distribution(self, save_path: str = None):
        """绘制奖励分布"""
        if not self.training_data:
            print("没有训练数据可绘制")
            return
        
        rewards = self.training_data.get('episode_rewards', [])
        if not rewards:
            print("没有奖励数据")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('奖励分布分析', fontsize=16)
        
        # 1. 奖励直方图
        axes[0].hist(rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'平均值: {np.mean(rewards):.2f}')
        axes[0].axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'中位数: {np.median(rewards):.2f}')
        axes[0].set_title('奖励分布直方图')
        axes[0].set_xlabel('总奖励')
        axes[0].set_ylabel('频次')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 奖励箱线图
        axes[1].boxplot(rewards, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axes[1].set_title('奖励箱线图')
        axes[1].set_ylabel('总奖励')
        axes[1].grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f"""
        统计信息:
        样本数: {len(rewards)}
        平均值: {np.mean(rewards):.2f}
        标准差: {np.std(rewards):.2f}
        最小值: {np.min(rewards):.2f}
        最大值: {np.max(rewards):.2f}
        25%分位数: {np.percentile(rewards, 25):.2f}
        75%分位数: {np.percentile(rewards, 75):.2f}
        """
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"奖励分布图已保存到: {save_path}")
        
        plt.show()
    
    def plot_episode_analysis(self, save_path: str = None):
        """绘制episode详细分析"""
        if not self.training_data:
            print("没有训练数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Episode详细分析', fontsize=16)
        
        episodes = list(range(1, len(self.training_data.get('episode_rewards', [])) + 1))
        
        # 1. 奖励vs路径长度散点图
        rewards = self.training_data.get('episode_rewards', [])
        lengths = self.training_data.get('episode_lengths', [])
        
        if rewards and lengths:
            scatter = axes[0, 0].scatter(lengths, rewards, c=episodes, cmap='viridis', alpha=0.6)
            axes[0, 0].set_xlabel('路径长度')
            axes[0, 0].set_ylabel('总奖励')
            axes[0, 0].set_title('奖励 vs 路径长度')
            axes[0, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 0], label='Episode')
        
        # 2. 成功率累积图
        success_rates = self.training_data.get('success_rates', [])
        if success_rates:
            cumulative_success = []
            for ep in range(1, len(success_rates) + 1):
                count = 0
                for i in range(ep):
                    if i == 0:
                        if success_rates[i] > 0:
                            count += 1
                    else:
                        if success_rates[i] > success_rates[i-1]:
                            count += 1
                cumulative_success.append(count)
            
            axes[0, 1].plot(episodes, cumulative_success, 'purple', linewidth=2)
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('累积成功次数')
            axes[0, 1].set_title('累积成功次数')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 奖励趋势分析
        if rewards:
            # 分段分析
            segment_size = len(rewards) // 4
            segments = []
            segment_means = []
            
            for i in range(4):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < 3 else len(rewards)
                segment = rewards[start_idx:end_idx]
                segments.append(segment)
                segment_means.append(np.mean(segment))
            
            axes[1, 0].bar(range(1, 5), segment_means, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
            axes[1, 0].set_xlabel('训练阶段')
            axes[1, 0].set_ylabel('平均奖励')
            axes[1, 0].set_title('各阶段平均奖励')
            axes[1, 0].set_xticks(range(1, 5))
            axes[1, 0].set_xticklabels(['阶段1', '阶段2', '阶段3', '阶段4'])
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 训练进度总结
        axes[1, 1].axis('off')
        summary_text = f"""
        训练总结:
        
        基本信息:
        - 总Episodes: {self.training_data.get('total_episodes', 0)}
        - 成功次数: {self.training_data.get('total_success', 0)}
        - 最终成功率: {self.training_data.get('final_success_rate', 0):.1%}
        - 平均奖励: {self.training_data.get('final_avg_reward', 0):.2f}
        - 平均路径长度: {self.training_data.get('final_avg_length', 0):.1f}
        
        起点: {self.training_data.get('start_point', [0, 0])}
        终点: {self.training_data.get('goal_point', [0, 0])}
        地形文件: {self.training_data.get('terrain_file', 'N/A')}
        
        分析结果:
        - 训练是否收敛: {'是' if len(rewards) > 100 and abs(np.mean(rewards[-50:]) - np.mean(rewards[-100:-50])) < 5 else '否'}
        - 成功率趋势: {'上升' if len(success_rates) > 10 and success_rates[-1] > success_rates[-10] else '下降或持平'}
        - 奖励稳定性: {'稳定' if np.std(rewards[-50:]) < np.std(rewards[:50]) else '不稳定'}
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Episode分析图已保存到: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, save_path: str = None):
        """生成训练总结报告"""
        if not self.training_data:
            print("没有训练数据可分析")
            return
        
        report = f"""
# 训练总结报告

## 基本信息
- 总Episodes: {self.training_data.get('total_episodes', 0)}
- 成功次数: {self.training_data.get('total_success', 0)}
- 最终成功率: {self.training_data.get('final_success_rate', 0):.1%}
- 平均奖励: {self.training_data.get('final_avg_reward', 0):.2f}
- 平均路径长度: {self.training_data.get('final_avg_length', 0):.1f}

## 起点和终点
- 起点: {self.training_data.get('start_point', [0, 0])}
- 终点: {self.training_data.get('goal_point', [0, 0])}
- 地形文件: {self.training_data.get('terrain_file', 'N/A')}

## 详细统计
"""
        
        rewards = self.training_data.get('episode_rewards', [])
        if rewards:
            report += f"""
### 奖励统计
- 奖励范围: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]
- 奖励标准差: {np.std(rewards):.2f}
- 奖励中位数: {np.median(rewards):.2f}
- 25%分位数: {np.percentile(rewards, 25):.2f}
- 75%分位数: {np.percentile(rewards, 75):.2f}

### 训练趋势分析
- 前25%平均奖励: {np.mean(rewards[:len(rewards)//4]):.2f}
- 后25%平均奖励: {np.mean(rewards[-len(rewards)//4:]):.2f}
- 奖励改善: {'是' if np.mean(rewards[-len(rewards)//4:]) > np.mean(rewards[:len(rewards)//4]) else '否'}
"""
        
        success_rates = self.training_data.get('success_rates', [])
        if success_rates:
            report += f"""
### 成功率分析
- 初始成功率: {success_rates[0]:.1%}
- 最终成功率: {success_rates[-1]:.1%}
- 成功率变化: {success_rates[-1] - success_rates[0]:.1%}
- 最高成功率: {max(success_rates):.1%}
"""
        
        report += f"""
## 问题诊断
- 成功率过低: {'是' if self.training_data.get('final_success_rate', 0) < 0.1 else '否'}
- 奖励为负: {'是' if self.training_data.get('final_avg_reward', 0) < 0 else '否'}
- 路径过长: {'是' if self.training_data.get('final_avg_length', 0) > 200 else '否'}

## 建议
1. 如果成功率过低，考虑调整奖励函数或增加训练轮数
2. 如果奖励为负，考虑降低地形惩罚权重
3. 如果路径过长，考虑增加最大步数或优化导航策略
"""
        
        print(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"训练报告已保存到: {save_path}")
    
    def visualize_all(self, output_dir: str = "visualization_output"):
        """生成所有可视化图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("生成训练可视化图表...")
        
        # 1. 训练曲线
        self.plot_training_curves(os.path.join(output_dir, "training_curves.png"))
        
        # 2. 奖励分布
        self.plot_reward_distribution(os.path.join(output_dir, "reward_distribution.png"))
        
        # 3. Episode分析
        self.plot_episode_analysis(os.path.join(output_dir, "episode_analysis.png"))
        
        # 4. 总结报告
        self.generate_summary_report(os.path.join(output_dir, "training_report.txt"))
        
        print(f"所有可视化图表已保存到: {output_dir}")


def main():
    """主函数"""
    print("训练数据可视化工具")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = TrainingVisualizer()
    
    if not visualizer.training_data:
        print("未找到训练数据，请先运行训练脚本")
        return
    
    # 生成所有可视化图表
    visualizer.visualize_all()
    
    print("\n可视化完成！")


if __name__ == "__main__":
    main()
