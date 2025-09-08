#!/usr/bin/env python3
"""
真实地形PPO训练脚本 - 只保存数据，不显示可视化
训练10000次，选择陆地上的起始点
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from envs.terrain_grid_nav_env import TerrainGridNavEnv
from agents.ppo_terrain_agent import TerrainPPOAgent


class RealTerrainDataTrainer:
    """真实地形数据训练器 - 只保存数据"""
    
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
        self.paths = []  # 保存所有路径
        self.success_flags = []  # 保存成功标志
        
        # 创建数据保存目录
        self.data_dir = "training_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
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
            max_steps=200,  # 给足够时间找到路径
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
    
    def find_land_points(self, min_height: float = 50.0) -> List[Tuple[int, int]]:
        """找到陆地上的点（高度大于阈值的点）"""
        land_points = []
        for i in range(self.height_map.shape[0]):
            for j in range(self.height_map.shape[1]):
                if self.height_map[i, j] > min_height:
                    land_points.append((i, j))
        return land_points
    
    def train(self, num_episodes: int = 10000, update_interval: int = 50, 
              save_interval: int = 500, log_interval: int = 100):
        """训练智能体"""
        print(f"开始训练 {num_episodes} 个episodes...")
        print(f"起点: {self.start_point}, 终点: {self.goal_point}")
        print(f"数据将保存到: {self.data_dir}")
        
        success_count = 0
        start_time = time.time()
        
        for episode in range(num_episodes):
            # 运行一个episode
            states, actions, rewards, values, log_probs, dones, path, success = \
                self.agent.collect_episode(self.env)
            
            # 更新统计
            total_reward = sum(rewards)
            episode_length = len(rewards)
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            self.paths.append(path)
            self.success_flags.append(success)
            
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
            
            # 定期保存数据
            if (episode + 1) % save_interval == 0:
                self.save_training_data(episode + 1)
                self.save_model(f"real_terrain_ppo_model_ep{episode+1}.pth")
            
            # 定期打印进度
            if (episode + 1) % log_interval == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_episode = elapsed_time / (episode + 1)
                remaining_episodes = num_episodes - (episode + 1)
                estimated_remaining_time = remaining_episodes * avg_time_per_episode
                
                print(f"Episode {episode + 1:5d}/{num_episodes}: "
                      f"成功率 = {current_success_rate:.1%} "
                      f"({success_count}/{episode + 1}), "
                      f"平均奖励 = {np.mean(self.episode_rewards[-log_interval:]):.2f}, "
                      f"平均路径长度 = {np.mean(self.episode_lengths[-log_interval:]):.1f}")
                print(f"  已用时: {elapsed_time/60:.1f}分钟, "
                      f"预计剩余: {estimated_remaining_time/60:.1f}分钟")
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成!")
        print(f"总用时: {total_time/60:.1f}分钟")
        print(f"最终成功率: {self.success_rates[-1]:.1%}")
        print(f"平均奖励: {np.mean(self.episode_rewards):.2f}")
        print(f"平均路径长度: {np.mean(self.episode_lengths):.1f}")
        
        # 保存最终数据
        self.save_training_data(num_episodes, is_final=True)
        self.save_model("real_terrain_ppo_model_final.pth")
        
        # 测试最终性能
        self.test_final_performance()
    
    def save_training_data(self, episode: int, is_final: bool = False):
        """保存训练数据"""
        suffix = "final" if is_final else f"ep{episode}"
        
        # 保存训练统计
        training_stats = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'success_rates': self.success_rates,
            'episode_lengths': self.episode_lengths,
            'avg_heights': self.avg_heights,
            'avg_slopes': self.avg_slopes,
            'success_flags': self.success_flags,
            'total_success': sum(self.success_flags),
            'total_episodes': len(self.success_flags),
            'final_success_rate': self.success_rates[-1] if self.success_rates else 0.0,
            'final_avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'final_avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        }
        
        stats_file = os.path.join(self.data_dir, f"training_stats_{suffix}.json")
        with open(stats_file, 'w') as f:
            json.dump(training_stats, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # 保存路径数据（只保存最近的1000个路径以避免文件过大）
        if len(self.paths) > 1000:
            recent_paths = self.paths[-1000:]
            recent_success_flags = self.success_flags[-1000:]
        else:
            recent_paths = self.paths
            recent_success_flags = self.success_flags
        
        # 将路径转换为简单的列表格式，避免循环引用
        simple_paths = []
        for path in recent_paths:
            simple_path = [[int(x), int(y)] for x, y in path]
            simple_paths.append(simple_path)
        
        paths_data = {
            'paths': simple_paths,
            'success_flags': recent_success_flags,
            'start_point': [int(self.start_point[0]), int(self.start_point[1])],
            'goal_point': [int(self.goal_point[0]), int(self.goal_point[1])],
            'terrain_file': self.terrain_file
        }
        
        paths_file = os.path.join(self.data_dir, f"paths_data_{suffix}.json")
        with open(paths_file, 'w') as f:
            json.dump(paths_data, f, indent=2)
        
        print(f"训练数据已保存: {stats_file}, {paths_file}")
    
    def save_model(self, filename: str):
        """保存模型"""
        model_path = os.path.join(self.data_dir, filename)
        self.agent.save_model(model_path)
        print(f"模型已保存: {model_path}")
    
    def test_final_performance(self, num_tests: int = 100):
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
        
        # 保存测试结果
        test_results = {
            'test_success_rate': test_success_rate,
            'test_avg_reward': avg_test_reward,
            'test_avg_length': avg_test_length,
            'test_rewards': test_rewards,
            'test_lengths': test_lengths,
            'num_tests': num_tests
        }
        
        test_file = os.path.join(self.data_dir, "final_test_results.json")
        with open(test_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"测试结果已保存: {test_file}")


def main():
    """主函数"""
    # 使用最新的地形数据
    terrain_file = "data/terrain/terrain_1755281528.json"
    
    # 加载地形数据来找到陆地上的点
    with open(terrain_file, 'r') as f:
        terrain_data = json.load(f)
    height_map = np.array(terrain_data['height_map'], dtype=np.float32)
    
    # 找到陆地上的点（高度大于50的点）
    land_points = []
    for i in range(height_map.shape[0]):
        for j in range(height_map.shape[1]):
            if height_map[i, j] > 50.0:
                land_points.append((i, j))
    
    print(f"找到 {len(land_points)} 个陆地点")
    
    # 选择陆地上的起始点
    # 选择左下角和右上角的陆地点
    start_point = (20, 20)  # 左下角区域
    goal_point = (110, 110)  # 右上角区域
    
    # 验证起始点是否在陆地上
    if height_map[start_point[0], start_point[1]] <= 50.0:
        print(f"警告：起点 {start_point} 不在陆地上，高度: {height_map[start_point[0], start_point[1]]}")
        # 寻找最近的陆地点
        for i in range(max(0, start_point[0]-10), min(height_map.shape[0], start_point[0]+10)):
            for j in range(max(0, start_point[1]-10), min(height_map.shape[1], start_point[1]+10)):
                if height_map[i, j] > 50.0:
                    start_point = (i, j)
                    print(f"调整起点为: {start_point}")
                    break
            if height_map[start_point[0], start_point[1]] > 50.0:
                break
    
    if height_map[goal_point[0], goal_point[1]] <= 50.0:
        print(f"警告：终点 {goal_point} 不在陆地上，高度: {height_map[goal_point[0], goal_point[1]]}")
        # 寻找最近的陆地点
        for i in range(max(0, goal_point[0]-10), min(height_map.shape[0], goal_point[0]+10)):
            for j in range(max(0, goal_point[1]-10), min(height_map.shape[1], goal_point[1]+10)):
                if height_map[i, j] > 50.0:
                    goal_point = (i, j)
                    print(f"调整终点为: {goal_point}")
                    break
            if height_map[goal_point[0], goal_point[1]] > 50.0:
                break
    
    print(f"起点高度: {height_map[start_point[0], start_point[1]]:.1f}")
    print(f"终点高度: {height_map[goal_point[0], goal_point[1]]:.1f}")
    
    print("真实地形PPO训练 - 数据保存模式")
    print(f"地形文件: {terrain_file}")
    print(f"起点: {start_point}")
    print(f"终点: {goal_point}")
    
    # 创建训练器
    trainer = RealTerrainDataTrainer(terrain_file, start_point, goal_point)
    
    # 开始训练
    trainer.train(
        num_episodes=10000,
        update_interval=50,
        save_interval=500,
        log_interval=100
    )


if __name__ == "__main__":
    main()
