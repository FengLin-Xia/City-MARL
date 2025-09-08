#!/usr/bin/env python3
"""
根本性解决方案 - 重新设计奖励函数和状态表示
"""

import numpy as np
import json
import time
import os
import torch
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from envs.terrain_grid_nav_env import TerrainGridNavEnv
from agents.ppo_terrain_agent import TerrainPPOAgent


class FundamentalSolutionTrainer:
    """根本性解决方案训练器"""
    
    def __init__(self, terrain_file: str, start_point: Tuple[int, int], goal_point: Tuple[int, int]):
        self.terrain_file = terrain_file
        self.start_point = start_point
        self.goal_point = goal_point
        
        # 加载地形数据
        self.terrain_data = self.load_terrain_data()
        self.height_map = np.array(self.terrain_data['height_map'], dtype=np.float32)
        
        # 计算曼哈顿距离作为基准
        self.manhattan_distance = abs(goal_point[0] - start_point[0]) + abs(goal_point[1] - start_point[1])
        print(f"曼哈顿距离: {self.manhattan_distance}")
        
        # 创建环境（使用根本性改进的奖励函数）
        self.env = self.create_environment()
        
        # 创建智能体
        self.agent = TerrainPPOAgent(
            state_dim=13,
            action_dim=4,
            hidden_dim=256,
            lr=1e-3  # 提高学习率
        )
        
        # 训练统计
        self.episode_rewards = []
        self.success_rates = []
        self.episode_lengths = []
        self.avg_heights = []
        self.avg_slopes = []
        self.success_count = 0
        self.total_episodes = 0
        
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
        
        # 创建环境，使用根本性改进的奖励函数
        env = TerrainGridNavEnv(
            H=H, W=W,
            max_steps=400,  # 增加最大步数
            height_range=(self.terrain_data['original_bounds']['z_min'], 
                         self.terrain_data['original_bounds']['z_max']),
            slope_penalty_weight=0.0,  # 完全移除坡度惩罚
            height_penalty_weight=0.0,  # 完全移除高度惩罚
            custom_terrain=self.height_map,
            fixed_start=self.start_point,
            fixed_goal=self.goal_point
        )
        
        print(f"环境创建完成: {H}x{W}, 起点{self.start_point}, 终点{self.goal_point}")
        print("根本性改进:")
        print("- 完全移除地形惩罚")
        print("- 增加最大步数")
        print("- 重新设计奖励函数")
        return env
    
    def analyze_reward_function(self):
        """分析当前奖励函数"""
        print("\n=== 奖励函数分析 ===")
        
        # 测试不同路径的奖励
        test_paths = []
        
        # 1. 直线路径
        straight_path = []
        current_pos = list(self.start_point)
        while current_pos != list(self.goal_point):
            straight_path.append(current_pos.copy())
            if current_pos[0] < self.goal_point[0]:
                current_pos[0] += 1
            elif current_pos[1] < self.goal_point[1]:
                current_pos[1] += 1
        test_paths.append(("直线路径", straight_path))
        
        # 2. 随机路径
        np.random.seed(42)
        random_path = [list(self.start_point)]
        current_pos = list(self.start_point)
        steps = 0
        while current_pos != list(self.goal_point) and steps < 200:
            action = np.random.randint(0, 4)
            next_pos = current_pos.copy()
            
            if action == 0:  # 上
                next_pos[0] = max(0, next_pos[0] - 1)
            elif action == 1:  # 右
                next_pos[1] = min(self.height_map.shape[1] - 1, next_pos[1] + 1)
            elif action == 2:  # 下
                next_pos[0] = min(self.height_map.shape[0] - 1, next_pos[0] + 1)
            elif action == 3:  # 左
                next_pos[1] = max(0, next_pos[1] - 1)
            
            if next_pos != current_pos:
                current_pos = next_pos
                random_path.append(current_pos.copy())
            steps += 1
        test_paths.append(("随机路径", random_path))
        
        # 计算各路径的奖励
        for name, path in test_paths:
            total_reward = 0
            rewards = []
            
            for i, pos in enumerate(path):
                # 计算距离奖励（改进版）
                distance = abs(self.goal_point[0] - pos[0]) + abs(self.goal_point[1] - pos[1])
                
                # 基础奖励：每步给予小奖励
                base_reward = 1.0
                
                # 距离奖励：越接近目标奖励越高
                distance_reward = max(0, self.manhattan_distance - distance) * 0.1
                
                # 进度奖励：如果距离减少，给予额外奖励
                if i > 0:
                    prev_distance = abs(self.goal_point[0] - path[i-1][0]) + abs(self.goal_point[1] - path[i-1][1])
                    progress_reward = (prev_distance - distance) * 2.0
                else:
                    progress_reward = 0
                
                step_reward = base_reward + distance_reward + progress_reward
                rewards.append(step_reward)
                total_reward += step_reward
            
            # 如果到达目标，给予额外奖励
            if path[-1] == list(self.goal_point):
                total_reward += 1000  # 大幅增加成功奖励
            
            print(f"{name}: 总奖励={total_reward:.2f}, 长度={len(path)}, 平均步奖励={np.mean(rewards):.3f}")
    
    def train(self, num_episodes: int = 1000, update_interval: int = 20, 
              save_interval: int = 100, log_interval: int = 50):
        """训练智能体"""
        print(f"开始训练 {num_episodes} 个episodes...")
        print(f"起点: {self.start_point}, 终点: {self.goal_point}")
        print(f"数据将保存到: {self.data_dir}")
        
        # 先分析奖励函数
        self.analyze_reward_function()
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # 运行一个episode
            states, actions, rewards, values, log_probs, dones, path, success = \
                self.agent.collect_episode(self.env)
            
            # 更新统计
            total_reward = sum(rewards)
            episode_length = len(rewards)
            
            self.episode_rewards.append(float(total_reward))
            self.episode_lengths.append(int(episode_length))
            
            if success:
                self.success_count += 1
            
            self.total_episodes += 1
            current_success_rate = self.success_count / self.total_episodes
            self.success_rates.append(float(current_success_rate))
            
            # 计算平均高度和坡度
            if path:
                heights = [float(self.height_map[x, y]) for x, y in path]
                from envs.terrain_grid_nav_env import calculate_slope
                slopes = [float(calculate_slope(self.height_map, (x, y))) for x, y in path]
                self.avg_heights.append(float(np.mean(heights)))
                self.avg_slopes.append(float(np.mean(slopes)))
            else:
                self.avg_heights.append(0.0)
                self.avg_slopes.append(0.0)
            
            # 定期更新网络
            if (episode + 1) % update_interval == 0:
                self.agent.update(states, actions, rewards, values, log_probs, dones)
            
            # 定期保存数据
            if (episode + 1) % save_interval == 0:
                self.save_training_data(episode + 1)
                self.save_model(f"fundamental_model_ep{episode+1}.pth")
            
            # 定期打印进度
            if (episode + 1) % log_interval == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_episode = elapsed_time / (episode + 1)
                remaining_episodes = num_episodes - (episode + 1)
                estimated_remaining_time = remaining_episodes * avg_time_per_episode
                
                recent_rewards = self.episode_rewards[-log_interval:]
                recent_lengths = self.episode_lengths[-log_interval:]
                
                print(f"Episode {episode + 1:5d}/{num_episodes}: "
                      f"成功率 = {current_success_rate:.1%} "
                      f"({self.success_count}/{self.total_episodes}), "
                      f"平均奖励 = {np.mean(recent_rewards):.2f}, "
                      f"平均路径长度 = {np.mean(recent_lengths):.1f}")
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
        self.save_model("fundamental_model_final.pth")
        
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
            'total_success': self.success_count,
            'total_episodes': self.total_episodes,
            'final_success_rate': self.success_rates[-1] if self.success_rates else 0.0,
            'final_avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'final_avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'start_point': [int(self.start_point[0]), int(self.start_point[1])],
            'goal_point': [int(self.goal_point[0]), int(self.goal_point[1])],
            'terrain_file': self.terrain_file,
            'training_type': 'fundamental_solution',
            'manhattan_distance': self.manhattan_distance
        }
        
        stats_file = os.path.join(self.data_dir, f"fundamental_stats_{suffix}.json")
        with open(stats_file, 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        print(f"训练数据已保存: {stats_file}")
    
    def save_model(self, filename: str):
        """保存模型"""
        model_path = os.path.join(self.data_dir, filename)
        self.agent.save_model(model_path)
        print(f"模型已保存: {model_path}")
    
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
            test_rewards.append(float(result['total_reward']))
            test_lengths.append(int(result['path_length']))
        
        test_success_rate = success_count / num_tests
        avg_test_reward = np.mean(test_rewards)
        avg_test_length = np.mean(test_lengths)
        
        print(f"测试结果:")
        print(f"  成功率: {test_success_rate:.1%} ({success_count}/{num_tests})")
        print(f"  平均奖励: {avg_test_reward:.2f}")
        print(f"  平均路径长度: {avg_test_length:.1f}")
        
        # 保存测试结果
        test_results = {
            'test_success_rate': float(test_success_rate),
            'test_avg_reward': float(avg_test_reward),
            'test_avg_length': float(avg_test_length),
            'test_rewards': test_rewards,
            'test_lengths': test_lengths,
            'num_tests': num_tests
        }
        
        test_file = os.path.join(self.data_dir, "fundamental_test_results.json")
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
    
    # 选择陆地上的起始点
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
    
    print("根本性解决方案 - 重新设计奖励函数和状态表示")
    print(f"地形文件: {terrain_file}")
    print(f"起点: {start_point}")
    print(f"终点: {goal_point}")
    
    # 创建训练器
    trainer = FundamentalSolutionTrainer(terrain_file, start_point, goal_point)
    
    # 开始训练
    trainer.train(
        num_episodes=1000,  # 先用1000个episodes测试
        update_interval=20,
        save_interval=100,
        log_interval=50
    )


if __name__ == "__main__":
    main()

