#!/usr/bin/env python3
"""
地形PPO训练脚本 - 带可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from envs.terrain_grid_nav_env import TerrainGridNavEnv
from agents.ppo_terrain_agent import TerrainPPOAgent


class TerrainPPOVisualizedTrainer:
    """地形PPO可视化训练器"""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.setup_visualization()
        self.episode_rewards = []
        self.success_rates = []
        self.episode_lengths = []
        self.avg_heights = []
        self.avg_slopes = []
    
    def setup_visualization(self):
        """设置可视化"""
        self.fig = plt.figure(figsize=(20, 12))
        
        # 地形和路径图
        self.ax1 = plt.subplot(2, 3, 1)
        self.ax1.set_title('地形高程图与路径', fontsize=12)
        
        # 坡度图
        self.ax2 = plt.subplot(2, 3, 2)
        self.ax2.set_title('坡度图', fontsize=12)
        
        # 奖励曲线
        self.ax3 = plt.subplot(2, 3, 3)
        self.ax3.set_title('训练奖励', fontsize=12)
        
        # 成功率曲线
        self.ax4 = plt.subplot(2, 3, 4)
        self.ax4.set_title('成功率', fontsize=12)
        
        # 平均高度和坡度
        self.ax5 = plt.subplot(2, 3, 5)
        self.ax5.set_title('平均高度和坡度', fontsize=12)
        
        # 当前episode路径
        self.ax6 = plt.subplot(2, 3, 6)
        self.ax6.set_title('当前Episode路径', fontsize=12)
        
        plt.tight_layout()
        plt.ion()  # 开启交互模式
    
    def plot_terrain_and_path(self):
        """绘制地形和路径"""
        self.ax1.clear()
        
        # 绘制地形
        im1 = self.ax1.imshow(self.env.terrain, cmap='terrain', origin='lower')
        self.ax1.set_title('地形高程图与路径')
        
        # 绘制起点和终点
        self.ax1.plot(self.env.start[1], self.env.start[0], 'go', markersize=15, 
                     label='起点', markeredgecolor='black', markeredgewidth=2)
        self.ax1.plot(self.env.goal[1], self.env.goal[0], 'ro', markersize=15, 
                     label='终点', markeredgecolor='black', markeredgewidth=2)
        
        # 绘制理想路径（直线）
        ideal_x = [self.env.start[0], self.env.goal[0]]
        ideal_y = [self.env.start[1], self.env.goal[1]]
        self.ax1.plot(ideal_y, ideal_x, 'r--', linewidth=2, alpha=0.6, label='理想路径')
        
        # 绘制当前路径
        if hasattr(self, 'current_path') and len(self.current_path) > 1:
            path_x = [pos[0] for pos in self.current_path]
            path_y = [pos[1] for pos in self.current_path]
            self.ax1.plot(path_y, path_x, 'b-', linewidth=3, alpha=0.8, label='智能体路径')
        
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
    
    def plot_slope_map(self):
        """绘制坡度图"""
        self.ax2.clear()
        
        # 计算坡度图
        slope_map = np.zeros_like(self.env.terrain)
        for i in range(self.env.H):
            for j in range(self.env.W):
                from envs.terrain_grid_nav_env import calculate_slope
                slope_map[i, j] = calculate_slope(self.env.terrain, (i, j))
        
        im2 = self.ax2.imshow(slope_map, cmap='hot', origin='lower')
        self.ax2.set_title('坡度图')
        
        # 标记位置
        self.ax2.plot(self.env.start[1], self.env.start[0], 'go', markersize=10, 
                     markeredgecolor='black', markeredgewidth=2)
        self.ax2.plot(self.env.goal[1], self.env.goal[0], 'ro', markersize=10, 
                     markeredgecolor='black', markeredgewidth=2)
        self.ax2.plot(self.env.pos[1], self.env.pos[0], 'bo', markersize=8)
        
        self.ax2.grid(True, alpha=0.3)
    
    def plot_training_stats(self):
        """绘制训练统计"""
        if len(self.episode_rewards) == 0:
            return
        
        # 奖励曲线
        self.ax3.clear()
        window = min(50, len(self.episode_rewards))
        if window > 0:
            avg_rewards = [np.mean(self.episode_rewards[max(0, i-window):i+1]) 
                          for i in range(len(self.episode_rewards))]
            self.ax3.plot(avg_rewards, 'b-', alpha=0.7, label=f'平均奖励({window}ep)')
        self.ax3.plot(self.episode_rewards, 'b-', alpha=0.3, label='单次奖励')
        self.ax3.set_title('训练奖励')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('奖励')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        # 成功率曲线
        self.ax4.clear()
        window = min(50, len(self.success_rates))
        if window > 0:
            avg_success = [np.mean(self.success_rates[max(0, i-window):i+1]) 
                          for i in range(len(self.success_rates))]
            self.ax4.plot(avg_success, 'g-', alpha=0.7, label=f'平均成功率({window}ep)')
        self.ax4.plot(self.success_rates, 'g-', alpha=0.3, label='单次成功率')
        self.ax4.set_title('成功率')
        self.ax4.set_xlabel('Episode')
        self.ax4.set_ylabel('成功率')
        self.ax4.set_ylim(0, 1)
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)
        
        # 平均高度和坡度
        self.ax5.clear()
        if len(self.avg_heights) > 0:
            self.ax5.plot(self.avg_heights, 'orange', label='平均高度', alpha=0.7)
        if len(self.avg_slopes) > 0:
            ax5_twin = self.ax5.twinx()
            ax5_twin.plot(self.avg_slopes, 'red', label='平均坡度', alpha=0.7)
            ax5_twin.set_ylabel('平均坡度', color='red')
            ax5_twin.tick_params(axis='y', labelcolor='red')
        
        self.ax5.set_title('平均高度和坡度')
        self.ax5.set_xlabel('Episode')
        self.ax5.set_ylabel('平均高度', color='orange')
        self.ax5.tick_params(axis='y', labelcolor='orange')
        self.ax5.grid(True, alpha=0.3)
    
    def plot_current_episode_path(self):
        """绘制当前episode的路径"""
        self.ax6.clear()
        
        # 运行一个测试episode
        obs, _ = self.env.reset()
        path = [obs['position'].copy()]
        heights = [obs['current_height'][0]]
        slopes = [obs['current_slope'][0]]
        
        while True:
            action, _, _ = self.agent.get_action(obs)
            obs, reward, done, truncated, info = self.env.step(action)
            
            path.append(obs['position'].copy())
            heights.append(obs['current_height'][0])
            slopes.append(obs['current_slope'][0])
            
            if done or truncated:
                break
        
        # 绘制路径
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        
        # 绘制地形
        self.ax6.imshow(self.env.terrain, cmap='terrain', origin='lower', alpha=0.3)
        
        # 绘制路径
        self.ax6.plot(path_y, path_x, 'b-', linewidth=3, alpha=0.8, label='智能体路径')
        
        # 绘制起点和终点
        self.ax6.plot(self.env.start[1], self.env.start[0], 'go', markersize=12, 
                     label='起点', markeredgecolor='black', markeredgewidth=2)
        self.ax6.plot(self.env.goal[1], self.env.goal[0], 'ro', markersize=12, 
                     label='终点', markeredgecolor='black', markeredgewidth=2)
        
        # 绘制理想路径
        ideal_x = [self.env.start[0], self.env.goal[0]]
        ideal_y = [self.env.start[1], self.env.goal[1]]
        self.ax6.plot(ideal_y, ideal_x, 'r--', linewidth=2, alpha=0.6, label='理想路径')
        
        self.ax6.set_title(f'当前Episode路径 (长度: {len(path)}, 成功: {done})')
        self.ax6.legend()
        self.ax6.grid(True, alpha=0.3)
        
        # 保存当前路径用于其他图
        self.current_path = path
        
        # 计算平均高度和坡度
        avg_height = np.mean(heights)
        avg_slope = np.mean(slopes)
        self.avg_heights.append(avg_height)
        self.avg_slopes.append(avg_slope)
    
    def update_plots(self):
        """更新所有图表"""
        self.plot_terrain_and_path()
        self.plot_slope_map()
        self.plot_training_stats()
        self.plot_current_episode_path()
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def train(self, num_episodes: int = 1000, update_interval: int = 10):
        """训练"""
        print(f"开始训练 {num_episodes} 个episodes...")
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # 训练一个episode
            result = self.agent.train_episode(self.env)
            
            # 记录统计信息
            self.episode_rewards.append(result['episode_reward'])
            self.success_rates.append(result['success_rate'])
            self.episode_lengths.append(result['episode_length'])
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards[-10:])
                current_success_rate = result['success_rate']
                
                print(f"Episode {episode + 1:4d}/{num_episodes} | "
                      f"奖励: {result['episode_reward']:6.2f} | "
                      f"平均奖励: {avg_reward:6.2f} | "
                      f"成功率: {current_success_rate:.1%} | "
                      f"长度: {result['episode_length']:3d} | "
                      f"时间: {elapsed_time:.1f}s")
            
            # 更新可视化
            if (episode + 1) % update_interval == 0:
                self.update_plots()
        
        # 最终更新
        self.update_plots()
        
        # 训练完成统计
        total_time = time.time() - start_time
        final_success_rate = self.agent.success_count / self.agent.total_episodes
        avg_reward = np.mean(self.episode_rewards)
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"总时间: {total_time:.1f}秒")
        print(f"最终成功率: {final_success_rate:.1%}")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"总episodes: {self.agent.total_episodes}")
        print(f"成功episodes: {self.agent.success_count}")
        
        # 保存模型
        self.agent.save_model('terrain_ppo_model.pth')
        print("模型已保存到 terrain_ppo_model.pth")
        
        plt.ioff()
        plt.show()


def main():
    """主函数"""
    print("地形PPO训练系统")
    print("=" * 60)
    
    # 创建环境和智能体
    env = TerrainGridNavEnv(
        H=20, W=20, 
        max_steps=100,  # 减少步数限制，增加难度
        height_range=(0.0, 12.0),  # 增加高度范围，让地形更明显
        slope_penalty_weight=0.3,  # 大幅增加坡度惩罚
        height_penalty_weight=0.2   # 大幅增加高度惩罚
    )
    
    agent = TerrainPPOAgent(
        state_dim=13,  # 基础状态维度
        action_dim=4,
        hidden_dim=128,  # 增加网络容量
        lr=1e-4,  # 降低学习率，更稳定
        gamma=0.99,
        gae_lambda=0.95,
        train_pi_iters=40,  # 减少训练迭代次数，加快训练
        train_v_iters=40
    )
    
    # 创建训练器
    trainer = TerrainPPOVisualizedTrainer(env, agent)
    
    # 开始训练 - 增加episodes数量
    trainer.train(num_episodes=5000, update_interval=50)


if __name__ == "__main__":
    main()
