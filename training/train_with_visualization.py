#!/usr/bin/env python3
"""
带实时可视化的路径规划训练脚本
"""

import sys
import os
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.simple_road_env import SimpleRoadEnv
from agents.simple_ppo import SimpleActorCritic


class VisualizedTrainer:
    """带可视化的训练器"""
    
    def __init__(self, 
                 dem_size: Tuple[int, int] = (100, 100),
                 max_steps: int = 200,
                 num_episodes: int = 100,
                 save_interval: int = 25,
                 visualize_interval: int = 10):
        
        self.dem_size = dem_size
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        self.visualize_interval = visualize_interval
        
        # 创建环境
        self.env = SimpleRoadEnv(dem_size=dem_size, max_steps=max_steps)
        
        # 创建智能体
        self.actor_critic = SimpleActorCritic()
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=5e-4)  # 降低学习率
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
        # 创建保存目录
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建可视化图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.suptitle('路径规划智能体实时训练可视化', fontsize=16, fontweight='bold')
        
    def train_episode(self, episode_num: int) -> Dict:
        """训练一个episode"""
        start_time = time.time()
        
        # 重置环境
        obs, _ = self.env.reset()
        
        total_reward = 0
        episode_length = 0
        success = False
        
        # 存储episode数据
        log_probs = []
        rewards = []
        path_x = [obs['position'][0]]
        path_y = [obs['position'][1]]
        
        while episode_length < self.max_steps:
            # 转换为张量
            obs_tensor = {
                'position': torch.FloatTensor(obs['position']).unsqueeze(0),
                'goal': torch.FloatTensor(obs['goal']).unsqueeze(0),
                'heading': torch.FloatTensor(obs['heading']).unsqueeze(0),
                'distance_to_goal': torch.FloatTensor(obs['distance_to_goal']).unsqueeze(0),
                'direction_to_goal': torch.FloatTensor(obs['direction_to_goal']).unsqueeze(0),
                'local_dem': torch.FloatTensor(obs['local_dem']).unsqueeze(0)
            }
            
            # 获取动作
            action, log_prob, value = self.actor_critic.get_action(obs_tensor)
            
            # 执行动作
            next_obs, reward, done, truncated, info = self.env.step(action.numpy().squeeze())
            
            # 存储数据
            log_probs.append(log_prob)
            rewards.append(reward)
            path_x.append(next_obs['position'][0])
            path_y.append(next_obs['position'][1])
            
            # 累积奖励
            total_reward += reward
            episode_length += 1
            
            # 检查是否成功
            if done and not truncated and info.get('reason') == 'reached_goal':
                success = True
                break
            
            # 检查是否失败
            if done:
                break
            
            obs = next_obs
        
        # 改进的策略梯度更新
        if len(log_probs) > 0:
            # 计算策略梯度损失
            log_probs_tensor = torch.stack(log_probs)
            rewards_tensor = torch.FloatTensor(rewards)
            
            # 计算累积奖励（从后往前）
            cumulative_rewards = []
            running_reward = 0
            for reward in reversed(rewards):
                running_reward = reward + 0.99 * running_reward  # 折扣因子
                cumulative_rewards.insert(0, running_reward)
            
            cumulative_rewards_tensor = torch.FloatTensor(cumulative_rewards)
            
            # 标准化奖励
            if len(cumulative_rewards) > 1:
                cumulative_rewards_tensor = (cumulative_rewards_tensor - cumulative_rewards_tensor.mean()) / (cumulative_rewards_tensor.std() + 1e-8)
            
            # 策略梯度损失
            loss = -(log_probs_tensor * cumulative_rewards_tensor).mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
        
        # 更新统计
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.success_rates.append(1.0 if success else 0.0)
        
        # 计算运行时间
        episode_time = time.time() - start_time
        
        # 计算平均统计
        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
        success_rate = np.mean(self.success_rates[-100:]) if len(self.success_rates) >= 100 else np.mean(self.success_rates)
        
        # 打印进度
        print(f"Episode {episode_num:4d} | "
              f"奖励: {total_reward:6.1f} | "
              f"步数: {episode_length:3d} | "
              f"成功: {'✅' if success else '❌'} | "
              f"平均奖励: {avg_reward:6.1f} | "
              f"成功率: {success_rate*100:5.1f}% | "
              f"用时: {episode_time:.1f}s")
        
        # 可视化（如果到了可视化间隔）
        if episode_num % self.visualize_interval == 0:
            self.visualize_episode(episode_num, path_x, path_y, success)
        
        return {
            'episode': episode_num,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'success': success,
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'time': episode_time,
            'path_x': path_x,
            'path_y': path_y
        }
    
    def visualize_episode(self, episode_num: int, path_x: List[float], path_y: List[float], success: bool):
        """可视化当前episode"""
        # 计算统计信息
        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
        success_rate = np.mean(self.success_rates[-100:]) if len(self.success_rates) >= 100 else np.mean(self.success_rates)
        
        # 清空图形
        self.ax1.clear()
        self.ax2.clear()
        
        # 左图：地形和路径
        self.ax1.imshow(self.env.dem, cmap='terrain', origin='lower', alpha=0.7)
        
        # 绘制理想路径（直线）
        ideal_x = [self.env.start_pos[0], self.env.goal_pos[0]]
        ideal_y = [self.env.start_pos[1], self.env.goal_pos[1]]
        self.ax1.plot(ideal_x, ideal_y, 'r--', linewidth=2, alpha=0.6, label='理想路径')
        
        # 绘制智能体路径
        self.ax1.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8, label='智能体路径')
        self.ax1.plot(path_x[0], path_y[0], 'go', markersize=15, label='起点', markeredgecolor='black', markeredgewidth=2)
        self.ax1.plot(self.env.goal_pos[0], self.env.goal_pos[1], 'ro', markersize=15, label='终点', markeredgecolor='black', markeredgewidth=2)
        self.ax1.plot(path_x[-1], path_y[-1], 'bo', markersize=10, label='当前位置')
        
        self.ax1.set_title(f'Episode {episode_num} - 路径规划', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('X坐标', fontsize=12)
        self.ax1.set_ylabel('Y坐标', fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # 右图：训练统计
        if len(self.episode_rewards) > 1:
            # 奖励曲线
            window = min(50, len(self.episode_rewards))
            recent_rewards = self.episode_rewards[-window:]
            recent_episodes = list(range(len(self.episode_rewards) - window + 1, len(self.episode_rewards) + 1))
            
            self.ax2.plot(recent_episodes, recent_rewards, 'b-', alpha=0.7, label='Episode奖励')
            self.ax2.axhline(y=avg_reward, color='r', linestyle='--', alpha=0.8, label=f'平均奖励: {avg_reward:.1f}')
            
            self.ax2.set_title('训练进度', fontsize=14, fontweight='bold')
            self.ax2.set_xlabel('Episode', fontsize=12)
            self.ax2.set_ylabel('奖励', fontsize=12)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend()
            
            # 添加成功率信息
            success_text = f"成功率: {success_rate*100:.1f}%"
            self.ax2.text(0.02, 0.98, success_text, transform=self.ax2.transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 更新显示
        plt.tight_layout()
        plt.pause(0.1)  # 短暂暂停以显示动画效果
    
    def save_model(self, episode_num: int):
        """保存模型"""
        model_path = self.models_dir / f"visualized_road_{episode_num}.pth"
        torch.save({
            'episode': episode_num,
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'success_rates': self.success_rates
        }, model_path)
        print(f"💾 模型已保存: {model_path}")
    
    def save_results(self):
        """保存训练结果"""
        results = {
            'training_config': {
                'dem_size': self.dem_size,
                'max_steps': self.max_steps,
                'num_episodes': self.num_episodes
            },
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates
        }
        
        results_path = self.results_dir / f"visualized_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 训练结果已保存: {results_path}")
    
    def train(self):
        """开始训练"""
        print("🚀 开始带可视化的路径规划训练")
        print(f"📏 DEM尺寸: {self.dem_size}")
        print(f"⏱️ 最大步数: {self.max_steps}")
        print(f"🎯 目标episodes: {self.num_episodes}")
        print(f"📺 可视化间隔: 每{self.visualize_interval}个episodes")
        print("=" * 80)
        
        start_time = time.time()
        
        for episode in range(1, self.num_episodes + 1):
            # 训练一个episode
            episode_info = self.train_episode(episode)
            
            # 定期保存模型
            if episode % self.save_interval == 0:
                self.save_model(episode)
        
        # 训练完成
        total_time = time.time() - start_time
        
        print("=" * 80)
        print("🎉 训练完成！")
        print(f"⏱️ 总用时: {total_time/60:.1f}分钟")
        print(f"📈 最终平均奖励: {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"🎯 最终成功率: {np.mean(self.success_rates[-100:])*100:.1f}%")
        
        # 保存最终模型和结果
        self.save_model(self.num_episodes)
        self.save_results()
        
        # 显示最终可视化
        plt.show()
        
        print("💾 最终模型和结果已保存！")


if __name__ == "__main__":
    # 创建训练器
    trainer = VisualizedTrainer(
        dem_size=(100, 100),
        max_steps=200,
        num_episodes=200,  # 增加到200个episodes
        save_interval=50,
        visualize_interval=10  # 每10个episodes可视化一次
    )
    
    # 开始训练
    trainer.train()
