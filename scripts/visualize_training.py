#!/usr/bin/env python3
"""
路径规划训练可视化脚本
实时显示智能体的学习过程
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import time
from typing import Dict, List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.simple_road_env import SimpleRoadEnv
from agents.simple_ppo import SimpleActorCritic


class TrainingVisualizer:
    """训练可视化器"""
    
    def __init__(self, model_path: str = None):
        self.env = SimpleRoadEnv()
        
        # 加载训练好的模型（如果有）
        if model_path and Path(model_path).exists():
            self.actor_critic = SimpleActorCritic()
            checkpoint = torch.load(model_path, map_location='cpu')
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            print(f"✅ 已加载模型: {model_path}")
        else:
            self.actor_critic = SimpleActorCritic()
            print("🆕 使用新模型")
        
        # 设置matplotlib
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.suptitle('路径规划智能体训练可视化', fontsize=16, fontweight='bold')
        
    def visualize_episode(self, episode_num: int = 0, max_steps: int = 200, save_gif: bool = False):
        """可视化一个episode"""
        # 重置环境
        obs, _ = self.env.reset()
        
        # 存储路径用于动画
        path_x = [obs['position'][0]]
        path_y = [obs['position'][1]]
        
        # 清空图形
        self.ax.clear()
        
        # 绘制DEM
        im = self.ax.imshow(self.env.dem, cmap='terrain', origin='lower', alpha=0.7)
        plt.colorbar(im, ax=self.ax, label='高程')
        
        # 绘制理想路径（直线）
        ideal_x = [self.env.start_pos[0], self.env.goal_pos[0]]
        ideal_y = [self.env.start_pos[1], self.env.goal_pos[1]]
        self.ax.plot(ideal_x, ideal_y, 'r--', linewidth=2, alpha=0.6, label='理想路径')
        
        # 绘制起点和终点
        self.ax.plot(obs['position'][0], obs['position'][1], 'go', markersize=15, label='起点', markeredgecolor='black', markeredgewidth=2)
        self.ax.plot(obs['goal'][0], obs['goal'][1], 'ro', markersize=15, label='终点', markeredgecolor='black', markeredgewidth=2)
        
        # 设置标题和标签
        self.ax.set_title(f'Episode {episode_num} - 智能体路径规划', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X坐标', fontsize=12)
        self.ax.set_ylabel('Y坐标', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # 动画函数
        def animate(frame):
            if frame == 0:
                return
            
            # 获取动作
            obs_tensor = {
                'position': torch.FloatTensor(obs['position']).unsqueeze(0),
                'goal': torch.FloatTensor(obs['goal']).unsqueeze(0),
                'local_dem': torch.FloatTensor(obs['local_dem']).unsqueeze(0)
            }
            
            # 使用确定性动作（不随机）
            action, _, _ = self.actor_critic.get_action(obs_tensor, deterministic=True)
            
            # 执行动作
            next_obs, reward, done, truncated, info = self.env.step(action.numpy().squeeze())
            
            # 更新路径
            path_x.append(next_obs['position'][0])
            path_y.append(next_obs['position'][1])
            
            # 绘制路径
            if len(path_x) > 1:
                self.ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8, label='智能体路径')
            
            # 绘制当前位置
            self.ax.plot(next_obs['position'][0], next_obs['position'][1], 'bo', markersize=8, alpha=0.8)
            
            # 更新标题显示当前状态
            distance = np.linalg.norm(next_obs['position'] - obs['goal'])
            status = "✅ 到达目标!" if done and info.get('reason') == 'reached_goal' else f"距离目标: {distance:.1f}"
            self.ax.set_title(f'Episode {episode_num} - 步骤 {frame} - {status}', fontsize=14, fontweight='bold')
            
            # 更新观测
            nonlocal obs
            obs = next_obs
            
            # 如果完成，停止动画
            if done:
                plt.close()
                return
            
            return self.ax,
        
        # 创建动画
        anim = animation.FuncAnimation(
            self.fig, animate, frames=max_steps, 
            interval=100, blit=False, repeat=False
        )
        
        # 保存GIF（如果需要）
        if save_gif:
            gif_path = f"data/results/episode_{episode_num}_visualization.gif"
            Path("data/results").mkdir(parents=True, exist_ok=True)
            anim.save(gif_path, writer='pillow', fps=10)
            print(f"💾 动画已保存: {gif_path}")
        
        # 显示动画
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def visualize_multiple_episodes(self, num_episodes: int = 5, max_steps: int = 200):
        """可视化多个episodes"""
        print(f"🎬 开始可视化 {num_episodes} 个episodes...")
        
        for i in range(num_episodes):
            print(f"📺 播放 Episode {i+1}/{num_episodes}")
            self.visualize_episode(episode_num=i+1, max_steps=max_steps)
            time.sleep(1)  # 短暂暂停
    
    def compare_random_vs_trained(self, num_episodes: int = 3):
        """比较随机策略和训练后策略"""
        print("🔍 比较随机策略 vs 训练后策略")
        
        # 随机策略
        print("🎲 随机策略演示:")
        for i in range(num_episodes):
            print(f"  Episode {i+1}: 随机动作")
            self.visualize_random_episode(episode_num=i+1)
            time.sleep(1)
        
        # 训练后策略
        print("🧠 训练后策略演示:")
        for i in range(num_episodes):
            print(f"  Episode {i+1}: 智能体动作")
            self.visualize_episode(episode_num=i+1)
            time.sleep(1)
    
    def visualize_random_episode(self, episode_num: int = 0, max_steps: int = 200):
        """可视化随机策略的episode"""
        # 重置环境
        obs, _ = self.env.reset()
        
        # 存储路径
        path_x = [obs['position'][0]]
        path_y = [obs['position'][1]]
        
        # 清空图形
        self.ax.clear()
        
        # 绘制DEM
        im = self.ax.imshow(self.env.dem, cmap='terrain', origin='lower', alpha=0.7)
        plt.colorbar(im, ax=self.ax, label='高程')
        
        # 绘制理想路径（直线）
        ideal_x = [self.env.start_pos[0], self.env.goal_pos[0]]
        ideal_y = [self.env.start_pos[1], self.env.goal_pos[1]]
        self.ax.plot(ideal_x, ideal_y, 'r--', linewidth=2, alpha=0.6, label='理想路径')
        
        # 绘制起点和终点
        self.ax.plot(obs['position'][0], obs['position'][1], 'go', markersize=15, label='起点', markeredgecolor='black', markeredgewidth=2)
        self.ax.plot(obs['goal'][0], obs['goal'][1], 'ro', markersize=15, label='终点', markeredgecolor='black', markeredgewidth=2)
        
        # 设置标题
        self.ax.set_title(f'Episode {episode_num} - 随机策略', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X坐标', fontsize=12)
        self.ax.set_ylabel('Y坐标', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # 动画函数
        def animate(frame):
            if frame == 0:
                return
            
            # 随机动作
            action = self.env.action_space.sample()
            
            # 执行动作
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # 更新路径
            path_x.append(next_obs['position'][0])
            path_y.append(next_obs['position'][1])
            
            # 绘制路径
            if len(path_x) > 1:
                self.ax.plot(path_x, path_y, 'orange', linewidth=3, alpha=0.8, label='随机路径')
            
            # 绘制当前位置
            self.ax.plot(next_obs['position'][0], next_obs['position'][1], 'orange', marker='o', markersize=8, alpha=0.8)
            
            # 更新标题
            distance = np.linalg.norm(next_obs['position'] - obs['goal'])
            status = "✅ 到达目标!" if done and info.get('reason') == 'reached_goal' else f"距离目标: {distance:.1f}"
            self.ax.set_title(f'Episode {episode_num} - 随机策略 - 步骤 {frame} - {status}', fontsize=14, fontweight='bold')
            
            # 更新观测
            nonlocal obs
            obs = next_obs
            
            if done:
                plt.close()
                return
            
            return self.ax,
        
        # 创建动画
        anim = animation.FuncAnimation(
            self.fig, animate, frames=max_steps, 
            interval=100, blit=False, repeat=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return anim


def main():
    """主函数"""
    print("🎨 路径规划训练可视化工具")
    print("=" * 50)
    
    # 查找最新的模型
    models_dir = Path("models")
    model_files = list(models_dir.glob("simple_road_*.pth"))
    
    if model_files:
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"📁 找到最新模型: {latest_model}")
        use_model = input("是否使用这个模型进行可视化? (y/n): ").lower().startswith('y')
        model_path = str(latest_model) if use_model else None
    else:
        print("📁 未找到训练好的模型，将使用新模型")
        model_path = None
    
    # 创建可视化器
    visualizer = TrainingVisualizer(model_path)
    
    # 选择可视化模式
    print("\n🎯 选择可视化模式:")
    print("1. 可视化单个episode")
    print("2. 可视化多个episodes")
    print("3. 比较随机策略 vs 训练后策略")
    print("4. 退出")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == '1':
        episode_num = int(input("输入episode编号 (默认1): ") or "1")
        save_gif = input("是否保存GIF动画? (y/n): ").lower().startswith('y')
        visualizer.visualize_episode(episode_num=episode_num, save_gif=save_gif)
    
    elif choice == '2':
        num_episodes = int(input("输入episode数量 (默认3): ") or "3")
        visualizer.visualize_multiple_episodes(num_episodes=num_episodes)
    
    elif choice == '3':
        num_episodes = int(input("输入比较的episode数量 (默认2): ") or "2")
        visualizer.compare_random_vs_trained(num_episodes=num_episodes)
    
    elif choice == '4':
        print("👋 再见!")
        return
    
    else:
        print("❌ 无效选择")
        return


if __name__ == "__main__":
    main()
