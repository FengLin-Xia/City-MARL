#!/usr/bin/env python3
"""
PPO网格导航训练脚本 - 带实时可视化
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.grid_nav_env import GridNavEnv
from agents.ppo_grid_nav_agent import PPOGridNavAgent


class VisualizedPPOTrainer:
    """带可视化的PPO训练器"""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        
        # 可视化设置
        self.setup_visualization()
        
        # 训练统计
        self.episode_rewards = []
        self.success_rates = []
        self.episode_lengths = []
        
    def setup_visualization(self):
        """设置可视化"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('PPO网格导航训练 - 实时监控', fontsize=16, fontweight='bold')
        
        # 设置子图
        self.setup_subplots()
        
        plt.tight_layout()
        plt.ion()  # 开启交互模式
        
    def setup_subplots(self):
        """设置子图"""
        # 左上：当前episode路径
        self.ax1.set_title('当前Episode路径', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Y坐标', fontsize=10)
        self.ax1.set_ylabel('X坐标', fontsize=10)
        self.ax1.grid(True, alpha=0.3)
        
        # 右上：奖励曲线
        self.ax2.set_title('训练奖励曲线', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Episode', fontsize=10)
        self.ax2.set_ylabel('奖励', fontsize=10)
        self.ax2.grid(True, alpha=0.3)
        
        # 左下：成功率曲线
        self.ax3.set_title('成功率曲线', fontsize=12, fontweight='bold')
        self.ax3.set_xlabel('Episode', fontsize=10)
        self.ax3.set_ylabel('成功率 (%)', fontsize=10)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_ylim(0, 100)
        
        # 右下：平均步数曲线
        self.ax4.set_title('平均步数曲线', fontsize=12, fontweight='bold')
        self.ax4.set_xlabel('Episode', fontsize=10)
        self.ax4.set_ylabel('步数', fontsize=10)
        self.ax4.grid(True, alpha=0.3)
        
    def update_visualization(self, episode_num, total_reward, success, episode_length):
        """更新可视化"""
        # 更新数据
        self.episode_rewards.append(total_reward)
        self.success_rates.append(1.0 if success else 0.0)
        self.episode_lengths.append(episode_length)
        
        # 计算移动平均
        window = min(50, len(self.episode_rewards))
        if window > 0:
            avg_rewards = [np.mean(self.episode_rewards[max(0, i-window):i+1]) 
                          for i in range(len(self.episode_rewards))]
            avg_success = [np.mean(self.success_rates[max(0, i-window):i+1]) * 100 
                          for i in range(len(self.success_rates))]
            avg_steps = [np.mean(self.episode_lengths[max(0, i-window):i+1]) 
                        for i in range(len(self.episode_lengths))]
        else:
            avg_rewards = self.episode_rewards
            avg_success = [s * 100 for s in self.success_rates]
            avg_steps = self.episode_lengths
        
        # 清除子图
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # 重新设置子图
        self.setup_subplots()
        
        # 左上：绘制当前episode路径
        self.plot_current_episode_path()
        
        # 右上：绘制奖励曲线
        episode_numbers = list(range(1, len(self.episode_rewards) + 1))
        self.ax2.plot(episode_numbers, self.episode_rewards, 'b-', alpha=0.3, label='单次奖励')
        self.ax2.plot(episode_numbers, avg_rewards, 'r-', linewidth=2, label=f'移动平均({window})')
        self.ax2.legend()
        
        # 左下：绘制成功率曲线
        self.ax3.plot(episode_numbers, avg_success, 'g-', linewidth=2, label=f'成功率({window})')
        self.ax3.legend()
        
        # 右下：绘制步数曲线
        self.ax4.plot(episode_numbers, self.episode_lengths, 'orange', alpha=0.3, label='单次步数')
        self.ax4.plot(episode_numbers, avg_steps, 'purple', linewidth=2, label=f'平均步数({window})')
        self.ax4.legend()
        
        # 更新显示
        plt.pause(0.01)
        
    def plot_current_episode_path(self):
        """绘制当前episode的路径"""
        # 重置环境并运行一个episode来获取路径
        obs, _ = self.env.reset()
        
        path_x = [obs['position'][0]]
        path_y = [obs['position'][1]]
        
        for step in range(50):  # 最多50步
            action = self.agent.get_action(obs, training=False)
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            path_x.append(next_obs['position'][0])
            path_y.append(next_obs['position'][1])
            
            if done:
                break
                
            obs = next_obs
        
        # 绘制网格
        self.ax1.imshow(self.env.grid, cmap='gray', origin='lower', alpha=0.3)
        
        # 绘制理想路径
        ideal_x = [self.env.start[0], self.env.goal[0]]
        ideal_y = [self.env.start[1], self.env.goal[1]]
        self.ax1.plot(ideal_y, ideal_x, 'r--', linewidth=2, alpha=0.6, label='理想路径')
        
        # 绘制智能体路径
        self.ax1.plot(path_y, path_x, 'b-', linewidth=3, alpha=0.8, label='智能体路径')
        self.ax1.plot(self.env.start[1], self.env.start[0], 'go', markersize=15, 
                     label='起点', markeredgecolor='black', markeredgewidth=2)
        self.ax1.plot(self.env.goal[1], self.env.goal[0], 'ro', markersize=15, 
                     label='终点', markeredgecolor='black', markeredgewidth=2)
        self.ax1.plot(path_y[-1], path_x[-1], 'bo', markersize=10, label='当前位置')
        
        self.ax1.legend()
        
    def train_episode(self, episode_num):
        """训练一个episode并更新可视化"""
        # 收集数据
        episode_data = self.agent.collect_episode(self.env)
        
        # 更新策略
        self.agent.update_policy(episode_data)
        
        # 更新可视化
        self.update_visualization(
            episode_num, 
            episode_data['total_reward'], 
            episode_data['success'], 
            episode_data['episode_length']
        )
        
        # 打印进度
        avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
        success_rate = np.mean(self.success_rates[-50:]) if len(self.success_rates) >= 50 else np.mean(self.success_rates)
        
        print(f"Episode {episode_num:4d} | "
              f"奖励: {episode_data['total_reward']:6.1f} | "
              f"步数: {episode_data['episode_length']:3d} | "
              f"成功: {'✅' if episode_data['success'] else '❌'} | "
              f"平均奖励: {avg_reward:6.1f} | "
              f"成功率: {success_rate*100:5.1f}%")
        
        return {
            'episode': episode_num,
            'total_reward': episode_data['total_reward'],
            'episode_length': episode_data['episode_length'],
            'success': episode_data['success'],
            'avg_reward': avg_reward,
            'success_rate': success_rate
        }
        
    def train(self, num_episodes=300):
        """开始训练"""
        print("🚀 开始PPO网格导航训练（带可视化）")
        print(f"🎯 目标episodes: {num_episodes}")
        print(f"📍 固定起点: {self.env.start}")
        print(f"🎯 固定终点: {self.env.goal}")
        print(f"🖥️  设备: {self.agent.device}")
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            self.train_episode(episode)
            
            # 每100个episodes显示一次统计
            if episode % 100 == 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards[-100:])
                success_rate = np.mean(self.success_rates[-100:]) * 100
                print(f"📊 Episode {episode:4d} | 用时: {elapsed_time:.1f}s | "
                      f"平均奖励: {avg_reward:6.1f} | 成功率: {success_rate:5.1f}%")
        
        print("=" * 60)
        print("🎉 训练完成！")
        print(f"📈 最终平均奖励: {np.mean(self.episode_rewards[-50:]):.2f}")
        print(f"🎯 最终成功率: {np.mean(self.success_rates[-50:])*100:.1f}%")
        
        # 保持图形显示
        plt.ioff()
        plt.show()


def main():
    """主函数"""
    print("🚀 开始PPO网格导航可视化训练...")
    
    try:
        # 创建环境和智能体
        env = GridNavEnv()
        agent = PPOGridNavAgent(lr=3e-4)
        trainer = VisualizedPPOTrainer(env, agent)
        
        # 开始训练
        trainer.train(num_episodes=300)
        
        print("✅ 训练完成！")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

