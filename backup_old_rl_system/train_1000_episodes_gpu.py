#!/usr/bin/env python3
"""
GPU加速1000 Episodes训练
一千零一夜的强化学习之旅
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.terrain_road_env import TerrainRoadEnvironment
from agents.terrain_policy import TerrainPolicyNetwork

class GPUTrainer:
    """GPU加速训练器"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🎮 使用设备: {self.device}")
        
        # 查找地形数据
        terrain_dir = Path("data/terrain")
        terrain_files = list(terrain_dir.glob("terrain_continuity_boundary_*.json"))
        
        if not terrain_files:
            print("❌ 未找到地形数据文件")
            return
        
        latest_file = max(terrain_files, key=lambda x: x.stat().st_mtime)
        print(f"🗺️ 使用地形文件: {latest_file}")
        
        # 创建环境
        self.env = TerrainRoadEnvironment(mesh_file=str(latest_file))
        print(f"📏 环境网格尺寸: {self.env.grid_size}")
        
        # 创建策略网络
        self.policy_net = TerrainPolicyNetwork(
            grid_size=self.env.grid_size, 
            action_space=self.env.action_space
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
        
        print(f"🧠 策略网络参数数量: {sum(p.numel() for p in self.policy_net.parameters()):,}")
    
    def preprocess_observation(self, obs):
        """预处理观察数据到GPU"""
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).to(self.device)
            else:
                obs_tensor[key] = torch.tensor([value]).to(self.device)
        return obs_tensor
    
    def get_action(self, obs_tensor, epsilon=0.1):
        """获取动作（epsilon-greedy）"""
        if np.random.random() < epsilon:
            return np.random.randint(self.env.action_space.n)
        
        with torch.no_grad():
            action_logits, _ = self.policy_net(obs_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            return action
    
    def train_episode(self, episode_num):
        """训练一个episode"""
        # 先重置环境
        obs, _ = self.env.reset()
        total_reward = 0
        step_count = 0
        max_steps = 200
        
        # 然后开始记录episode
        self.env.start_recording()
        
        while step_count < max_steps:
            # 预处理观察
            obs_tensor = self.preprocess_observation(obs)
            
            # 获取动作
            epsilon = max(0.01, 0.1 * (0.95 ** (episode_num // 100)))  # 衰减的epsilon
            action = self.get_action(obs_tensor, epsilon)
            
            # 执行动作
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # 计算损失（简单的策略梯度）
            action_logits, value = self.policy_net(obs_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            
            # 策略梯度损失
            loss = -action_dist.log_prob(torch.tensor([action], device=self.device)) * torch.tensor([reward], device=self.device)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_reward += reward
            step_count += 1
            obs = next_obs
            
            if done or truncated:
                break
        
        # 停止记录
        episode_data = self.env.stop_recording()
        
        # 更新统计
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(step_count)
        self.total_episodes += 1
        
        if info.get('reached_target', False):
            self.success_count += 1
        
        # 保存episode（每10个episode保存一次）
        if episode_num % 10 == 0:
            if episode_data:  # 只有episode_data不为None时才保存
                self.save_episode(episode_data, episode_num)
            else:
                print(f"⚠️ Episode {episode_num} 录制数据为空，未保存JSON。")
        
        return total_reward, step_count, info.get('reached_target', False)
    
    def save_episode(self, episode_data, episode_num):
        """保存episode数据"""
        episodes_dir = Path("data/episodes")
        episodes_dir.mkdir(exist_ok=True)
        
        filename = episodes_dir / f"episode_{episode_num}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, ensure_ascii=False, indent=2)
    
    def print_progress(self, episode_num, reward, steps, reached_target, start_time):
        """打印训练进度"""
        elapsed_time = time.time() - start_time
        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
        success_rate = self.success_count / self.total_episodes * 100
        
        print(f"📊 Episode {episode_num:4d} | "
              f"奖励: {reward:6.2f} | "
              f"步数: {steps:3d} | "
              f"到达: {'✅' if reached_target else '❌'} | "
              f"平均奖励: {avg_reward:6.2f} | "
              f"成功率: {success_rate:5.1f}% | "
              f"用时: {elapsed_time/60:.1f}分钟")
    
    def train_1000_episodes(self):
        """训练1000个episodes"""
        print("🚀 开始一千零一夜的强化学习之旅！")
        print("=" * 80)
        
        start_time = time.time()
        
        for episode in range(1, 1001):
            reward, steps, reached_target = self.train_episode(episode)
            
            # 每10个episode打印一次进度
            if episode % 10 == 0:
                self.print_progress(episode, reward, steps, reached_target, start_time)
            
            # 每100个episode保存一次模型
            if episode % 100 == 0:
                self.save_model(episode)
        
        # 训练完成
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("🎉 一千零一夜训练完成！")
        print(f"⏱️ 总用时: {total_time/60:.1f}分钟")
        print(f"📈 最终平均奖励: {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"🎯 最终成功率: {self.success_count/self.total_episodes*100:.1f}%")
        print(f"💾 模型已保存到: models/terrain_policy_1000.pth")
    
    def save_model(self, episode_num):
        """保存模型"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / f"terrain_policy_{episode_num}.pth"
        torch.save({
            'episode': episode_num,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_count': self.success_count,
            'total_episodes': self.total_episodes
        }, model_path)
        
        print(f"💾 模型已保存: {model_path}")

def main():
    """主函数"""
    print("🌟 一千零一夜 GPU加速强化学习训练")
    print("=" * 80)
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ 使用CPU训练（建议使用GPU以获得更好性能）")
    
    # 创建训练器
    trainer = GPUTrainer()
    
    # 开始训练
    trainer.train_1000_episodes()
    
    print("\n🎬 训练完成！现在可以观看回放了！")
    print("运行: python tests/quick_replay.py")

if __name__ == "__main__":
    main()
