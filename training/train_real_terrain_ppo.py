#!/usr/bin/env python3
"""
使用真实地形数据的PPO训练脚本
从Flask上传的地形数据中训练智能体
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from envs.terrain_grid_nav_env import TerrainGridNavEnv
from agents.ppo_terrain_agent import TerrainPPOAgent


class RealTerrainPPOTrainer:
    """真实地形PPO训练器"""
    
    def __init__(self, terrain_file: str, start_point: Tuple[int, int], goal_point: Tuple[int, int]):
        self.terrain_file = terrain_file
        self.start_point = start_point
        self.goal_point = goal_point
        
        # 加载地形数据
        self.terrain_data = self.load_terrain_data()
        self.height_map = np.array(self.terrain_data['height_map'], dtype=np.float32)
        
        # 创建环境
        self.env = self.create_environment()
        
        # 创建智能体
        self.agent = TerrainPPOAgent(
            state_dim=13,
            action_dim=4,
            hidden_dim=256,
            lr=2e-4
        )
        
        # 训练统计
        self.episode_rewards = []
        self.success_rates = []
        self.episode_lengths = []
        self.avg_heights = []
        self.avg_slopes = []
        
        # 可视化设置
        self.fig, self.axes = None, None
        self.setup_visualization()
    
    def load_terrain_data(self) -> Dict:
        """加载地形数据"""
        print(f"加载地形数据: {self.terrain_file}")
        with open(self.terrain_file, 'r') as f:
            data = json.load(f)
        
        print(f"地形尺寸: {data['grid_size']}")
        print(f"高程范围: {data['original_bounds']['z_min']:.2f} ~ {data['original_bounds']['z_max']:.2f}")
        return data
    
    def create_environment(self) -> TerrainGridNavEnv:
        """创建基于真实地形的环境"""
        H, W = self.terrain_data['grid_size']
        
        # 创建环境，使用真实地形数据
        env = TerrainGridNavEnv(
            H=H, W=W,
            max_steps=150,  # 给足够时间找到路径
            height_range=(self.terrain_data['original_bounds']['z_min'], 
                         self.terrain_data['original_bounds']['z_max']),
            slope_penalty_weight=0.2,
            height_penalty_weight=0.15,
            custom_terrain=self.height_map,  # 使用真实地形
            fixed_start=self.start_point,
            fixed_goal=self.goal_point
        )
        
        print(f"环境创建完成: {H}x{W}, 起点{self.start_point}, 终点{self.goal_point}")
        return env
    
    def setup_visualization(self):
        """设置可视化"""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('真实地形PPO训练可视化', fontsize=16)
        plt.ion()  # 开启交互模式
        
        # 预计算坡度图，避免每次绘制重复计算
        from envs.terrain_grid_nav_env import calculate_slope
        self.slope_map = np.zeros_like(self.height_map)
        for i in range(self.height_map.shape[0]):
            for j in range(self.height_map.shape[1]):
                self.slope_map[i, j] = calculate_slope(self.height_map, (i, j))
    
    def plot_terrain_and_path(self, episode: int, path: List[Tuple[int, int]], 
                             success: bool, total_reward: float):
        """绘制地形和路径"""
        # 清除所有子图
        for ax in self.axes.flat:
            ax.clear()
        
        # 子图1: 地形高程图（不转置，origin='lower'，与路径坐标一致）
        im1 = self.axes[0, 0].imshow(self.height_map, cmap='terrain', aspect='auto', origin='lower')
        self.axes[0, 0].set_title('地形高程图')
        self.axes[0, 0].set_xlabel('X')
        self.axes[0, 0].set_ylabel('Y')
        plt.colorbar(im1, ax=self.axes[0, 0])
        
        # 绘制路径（imshow 不转置，需将 (x,y) 映射为 (y,x)）
        if path:
            xs, ys = zip(*path)
            self.axes[0, 0].plot(ys, xs, 'r-', linewidth=2, alpha=0.8)
            self.axes[0, 0].plot(ys[0], xs[0], 'go', markersize=8, label='起点')
            self.axes[0, 0].plot(ys[-1], xs[-1], 'ro', markersize=8, label='终点')
            self.axes[0, 0].legend()
        
        # 子图2: 坡度图（使用缓存的坡度图）
        im2 = self.axes[0, 1].imshow(self.slope_map, cmap='hot', aspect='auto', origin='lower')
        self.axes[0, 1].set_title('地形坡度图')
        self.axes[0, 1].set_xlabel('X')
        self.axes[0, 1].set_ylabel('Y')
        plt.colorbar(im2, ax=self.axes[0, 1])
        
        # 子图3: 训练进度
        if len(self.success_rates) > 0:
            self.axes[0, 2].plot(self.success_rates, 'b-', linewidth=2)
            self.axes[0, 2].set_title('成功率变化')
            self.axes[0, 2].set_xlabel('Episode')
            self.axes[0, 2].set_ylabel('成功率')
            self.axes[0, 2].grid(True)
            self.axes[0, 2].set_ylim(0, 1)
        
        # 子图4: 奖励变化
        if len(self.episode_rewards) > 0:
            self.axes[1, 0].plot(self.episode_rewards, 'g-', linewidth=1, alpha=0.6)
            if len(self.episode_rewards) > 10:
                window = min(50, len(self.episode_rewards) // 2)
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                self.axes[1, 0].plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2)
            self.axes[1, 0].set_title('奖励变化')
            self.axes[1, 0].set_xlabel('Episode')
            self.axes[1, 0].set_ylabel('奖励')
            self.axes[1, 0].grid(True)
        
        # 子图5: 路径长度
        if len(self.episode_lengths) > 0:
            self.axes[1, 1].plot(self.episode_lengths, 'purple', linewidth=1, alpha=0.6)
            self.axes[1, 1].set_title('路径长度')
            self.axes[1, 1].set_xlabel('Episode')
            self.axes[1, 1].set_ylabel('步数')
            self.axes[1, 1].grid(True)
        
        # 子图6: 当前episode信息
        self.axes[1, 2].text(0.1, 0.8, f'Episode: {episode}', fontsize=12)
        self.axes[1, 2].text(0.1, 0.7, f'成功: {"是" if success else "否"}', fontsize=12)
        self.axes[1, 2].text(0.1, 0.6, f'总奖励: {total_reward:.2f}', fontsize=12)
        self.axes[1, 2].text(0.1, 0.5, f'路径长度: {len(path)}', fontsize=12)
        if len(self.success_rates) > 0:
            self.axes[1, 2].text(0.1, 0.4, f'当前成功率: {self.success_rates[-1]:.1%}', fontsize=12)
        self.axes[1, 2].text(0.1, 0.3, f'起点: {self.start_point}', fontsize=12)
        self.axes[1, 2].text(0.1, 0.2, f'终点: {self.goal_point}', fontsize=12)
        self.axes[1, 2].set_xlim(0, 1)
        self.axes[1, 2].set_ylim(0, 1)
        self.axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.pause(0.01)  # 短暂暂停以更新显示
    
    def train(self, num_episodes: int = 1000, update_interval: int = 50, 
              save_interval: int = 100, render_interval: int = 10):
        """训练智能体"""
        print(f"开始训练 {num_episodes} 个episodes...")
        print(f"起点: {self.start_point}, 终点: {self.goal_point}")
        
        success_count = 0
        
        for episode in range(num_episodes):
            # 运行一个episode
            states, actions, rewards, values, log_probs, dones, path, success = \
                self.agent.collect_episode(self.env)
            
            # 更新统计
            total_reward = sum(rewards)
            episode_length = len(rewards)
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            
            if success:
                success_count += 1
            
            current_success_rate = success_count / (episode + 1)
            self.success_rates.append(current_success_rate)
            
            # 计算平均高度和坡度
            if path:
                heights = [self.height_map[x, y] for x, y in path]
                from envs.terrain_grid_nav_env import calculate_slope
                slopes = [calculate_slope(self.height_map, (x, y)) for x, y in path]
                self.avg_heights.append(np.mean(heights))
                self.avg_slopes.append(np.mean(slopes))
            else:
                self.avg_heights.append(0)
                self.avg_slopes.append(0)
            
            # 定期更新网络
            if (episode + 1) % update_interval == 0:
                self.agent.update(states, actions, rewards, values, log_probs, dones)
            
            # 定期可视化
            if (episode + 1) % render_interval == 0:
                self.plot_terrain_and_path(episode + 1, path, success, total_reward)
            
            # 定期保存模型
            if (episode + 1) % save_interval == 0:
                self.save_model(f"real_terrain_ppo_model_ep{episode+1}.pth")
                self.save_training_progress(f"real_terrain_training_progress_ep{episode+1}.png")
            
            # 打印进度
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode + 1:4d}: 成功率 = {current_success_rate:.1%} "
                      f"({success_count}/{episode + 1}), 平均奖励 = {np.mean(self.episode_rewards[-50:]):.2f}")
        
        # 训练完成
        print(f"\n训练完成!")
        print(f"最终成功率: {self.success_rates[-1]:.1%}")
        print(f"平均奖励: {np.mean(self.episode_rewards):.2f}")
        print(f"平均路径长度: {np.mean(self.episode_lengths):.1f}")
        
        # 保存最终模型和结果
        self.save_model("real_terrain_ppo_model_final.pth")
        self.save_training_progress("real_terrain_training_progress_final.png")
        
        # 测试最终性能
        self.test_final_performance()
    
    def save_model(self, filename: str):
        """保存模型"""
        self.agent.save_model(filename)
        print(f"模型已保存: {filename}")
    
    def save_training_progress(self, filename: str):
        """保存训练进度图"""
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"训练进度图已保存: {filename}")
    
    def test_final_performance(self, num_tests: int = 50):
        """测试最终性能"""
        print(f"\n测试最终性能 ({num_tests} 次)...")
        
        success_count = 0
        test_rewards = []
        test_lengths = []
        
        for i in range(num_tests):
            result = self.agent.test_episode(self.env, render=False)
            if result['success']:
                success_count += 1
            test_rewards.append(result['total_reward'])
            test_lengths.append(result['path_length'])
        
        test_success_rate = success_count / num_tests
        avg_test_reward = np.mean(test_rewards)
        avg_test_length = np.mean(test_lengths)
        
        print(f"测试结果:")
        print(f"  成功率: {test_success_rate:.1%} ({success_count}/{num_tests})")
        print(f"  平均奖励: {avg_test_reward:.2f}")
        print(f"  平均路径长度: {avg_test_length:.1f}")


def main():
    """主函数"""
    # 使用最新的地形数据
    terrain_file = "data/terrain/terrain_1755281528.json"
    
    # 选择两个固定起始点（在地形范围内）
    # 根据地形尺寸选择合适的位置
    start_point = (10, 10)  # 左下角区域
    goal_point = (120, 120)  # 右上角区域（确保在地形范围内）
    
    print("真实地形PPO训练")
    print(f"地形文件: {terrain_file}")
    print(f"起点: {start_point}")
    print(f"终点: {goal_point}")
    
    # 创建训练器
    trainer = RealTerrainPPOTrainer(terrain_file, start_point, goal_point)
    
    # 开始训练
    trainer.train(
        num_episodes=1000,
        update_interval=50,
        save_interval=100,
        render_interval=10
    )


if __name__ == "__main__":
    main()
