#!/usr/bin/env python3
"""
简化版路径规划训练脚本 V2
更简单的实现，避免复杂的批处理
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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.simple_road_env import SimpleRoadEnv
from agents.simple_ppo import SimpleActorCritic


class SimpleRoadTrainerV2:
    """简化版路径规划训练器 V2"""
    
    def __init__(self, 
                 dem_size: Tuple[int, int] = (100, 100),
                 max_steps: int = 200,
                 num_episodes: int = 500,
                 save_interval: int = 50):
        
        self.dem_size = dem_size
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        
        # 创建环境
        self.env = SimpleRoadEnv(dem_size=dem_size, max_steps=max_steps)
        
        # 创建智能体
        self.actor_critic = SimpleActorCritic()
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
        # 创建保存目录
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def train_episode(self, episode_num: int) -> Dict:
        """训练一个episode（简化版）"""
        start_time = time.time()
        
        # 重置环境
        obs, _ = self.env.reset()
        
        total_reward = 0
        episode_length = 0
        success = False
        
        while episode_length < self.max_steps:
            # 转换为张量
            obs_tensor = {
                'position': torch.FloatTensor(obs['position']).unsqueeze(0),
                'goal': torch.FloatTensor(obs['goal']).unsqueeze(0),
                'local_dem': torch.FloatTensor(obs['local_dem']).unsqueeze(0)
            }
            
            # 获取动作
            action, log_prob, value = self.actor_critic.get_action(obs_tensor)
            
            # 执行动作
            next_obs, reward, done, truncated, info = self.env.step(action.numpy().squeeze())
            
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
        
        # 简单的策略梯度更新（每步都更新）
        if episode_length > 0:
            # 计算简单的损失
            loss = -log_prob.mean() * total_reward  # 简单的策略梯度
            
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
        
        return {
            'episode': episode_num,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'success': success,
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'time': episode_time
        }
    
    def save_model(self, episode_num: int):
        """保存模型"""
        model_path = self.models_dir / f"simple_road_v2_{episode_num}.pth"
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
        
        results_path = self.results_dir / f"simple_road_v2_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 训练结果已保存: {results_path}")
    
    def train(self):
        """开始训练"""
        print("🚀 开始训练简化版路径规划智能体 V2")
        print(f"📏 DEM尺寸: {self.dem_size}")
        print(f"⏱️ 最大步数: {self.max_steps}")
        print(f"🎯 目标episodes: {self.num_episodes}")
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
        
        print("💾 最终模型和结果已保存！")


if __name__ == "__main__":
    # 创建训练器
    trainer = SimpleRoadTrainerV2(
        dem_size=(100, 100),
        max_steps=200,
        num_episodes=200,  # 先训练200个episodes测试
        save_interval=50
    )
    
    # 开始训练
    trainer.train()

