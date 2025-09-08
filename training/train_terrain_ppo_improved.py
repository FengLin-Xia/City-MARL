#!/usr/bin/env python3
"""
改进的地形PPO训练脚本 - 解决局部最优解问题
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from envs.terrain_grid_nav_env import TerrainGridNavEnv
from agents.ppo_terrain_agent import TerrainPPOAgent


class ImprovedTerrainPPOTrainer:
    """改进的地形PPO训练器"""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.episode_rewards = []
        self.success_rates = []
        self.episode_lengths = []
        self.avg_heights = []
        self.avg_slopes = []
        
        # 训练统计
        self.best_success_rate = 0.0
        self.stagnant_episodes = 0
        self.last_improvement = 0
        
    def adaptive_learning_rate(self, current_success_rate: float, base_lr: float = 1e-4):
        """自适应学习率 - 当性能停滞时增加探索"""
        if current_success_rate > self.best_success_rate:
            self.best_success_rate = current_success_rate
            self.stagnant_episodes = 0
            self.last_improvement = len(self.episode_rewards)
        else:
            self.stagnant_episodes += 1
        
        # 如果停滞超过100个episodes，增加学习率
        if self.stagnant_episodes > 100:
            adaptive_lr = base_lr * (1.0 + 0.1 * (self.stagnant_episodes - 100) // 50)
            return min(adaptive_lr, base_lr * 2.0)  # 最大2倍学习率
        
        return base_lr
    
    def entropy_bonus(self, current_success_rate: float):
        """熵奖励 - 鼓励探索"""
        if current_success_rate < 0.6:  # 成功率低时增加探索
            return 0.05
        elif current_success_rate < 0.8:
            return 0.02
        else:
            return 0.01
    
    def train_episode_with_improvements(self) -> Dict:
        """改进的训练episode"""
        # 收集数据
        states, actions, rewards, values, log_probs, dones = self.agent.collect_episode(self.env)
        
        # 计算当前成功率
        current_success_rate = self.agent.success_count / max(self.agent.total_episodes, 1)
        
        # 自适应学习率
        adaptive_lr = self.adaptive_learning_rate(current_success_rate)
        for param_group in self.agent.optimizer.param_groups:
            param_group['lr'] = adaptive_lr
        
        # 熵奖励
        entropy_bonus = self.entropy_bonus(current_success_rate)
        
        # 更新网络（带熵奖励）
        self.agent.update_with_entropy_bonus(states, actions, rewards, values, log_probs, dones, entropy_bonus)
        
        # 记录统计信息
        episode_reward = rewards.sum().item()
        episode_length = len(rewards)
        success = dones[-1].item() and episode_length < self.env.max_steps
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.avg_heights.append(np.mean([self.env.terrain[pos[0], pos[1]] for pos in self.get_path_from_states(states)]))
        from envs.terrain_grid_nav_env import calculate_slope
        self.avg_slopes.append(np.mean([calculate_slope(self.env.terrain, pos) for pos in self.get_path_from_states(states)]))
        
        if success:
            self.agent.success_count += 1
        self.agent.total_episodes += 1
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'success': success,
            'success_rate': self.agent.success_count / self.agent.total_episodes,
            'adaptive_lr': adaptive_lr,
            'entropy_bonus': entropy_bonus
        }
    
    def get_path_from_states(self, states):
        """从状态中提取路径"""
        path = []
        for state in states:
            # 前2维是位置
            pos = (int(state[0].item()), int(state[1].item()))
            path.append(pos)
        return path
    
    def plot_training_progress(self, save_path: str = None):
        """绘制训练进度（非实时，更稳定）"""
        if len(self.episode_rewards) == 0:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 奖励曲线
        window = min(50, len(self.episode_rewards))
        if window > 0:
            avg_rewards = [np.mean(self.episode_rewards[max(0, i-window):i+1]) 
                          for i in range(len(self.episode_rewards))]
            ax1.plot(avg_rewards, 'b-', alpha=0.7, label=f'平均奖励({window}ep)')
        ax1.plot(self.episode_rewards, 'b-', alpha=0.3, label='单次奖励')
        ax1.set_title('训练奖励')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('奖励')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 成功率曲线
        window = min(50, len(self.success_rates))
        if window > 0:
            avg_success = [np.mean(self.success_rates[max(0, i-window):i+1]) 
                          for i in range(len(self.success_rates))]
            ax2.plot(avg_success, 'g-', alpha=0.7, label=f'平均成功率({window}ep)')
        ax2.plot(self.success_rates, 'g-', alpha=0.3, label='单次成功率')
        ax2.set_title('成功率')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('成功率')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 平均高度和坡度
        if len(self.avg_heights) > 0:
            ax3.plot(self.avg_heights, 'orange', label='平均高度', alpha=0.7)
        if len(self.avg_slopes) > 0:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(self.avg_slopes, 'red', label='平均坡度', alpha=0.7)
            ax3_twin.set_ylabel('平均坡度', color='red')
            ax3_twin.tick_params(axis='y', labelcolor='red')
        
        ax3.set_title('平均高度和坡度')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('平均高度', color='orange')
        ax3.tick_params(axis='y', labelcolor='orange')
        ax3.grid(True, alpha=0.3)
        
        # 路径长度分布
        ax4.hist(self.episode_lengths, bins=30, alpha=0.7, color='purple')
        ax4.set_title('路径长度分布')
        ax4.set_xlabel('路径长度')
        ax4.set_ylabel('频次')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def train(self, num_episodes: int = 2000, plot_interval: int = 100):
        """训练"""
        print(f"开始改进训练 {num_episodes} 个episodes...")
        print("=" * 80)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # 训练一个episode
            result = self.train_episode_with_improvements()
            
            # 记录成功率
            self.success_rates.append(result['success_rate'])
            
            # 打印进度
            if (episode + 1) % 20 == 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards[-20:])
                current_success_rate = result['success_rate']
                
                print(f"Episode {episode + 1:4d}/{num_episodes} | "
                      f"奖励: {result['episode_reward']:6.2f} | "
                      f"平均奖励: {avg_reward:6.2f} | "
                      f"成功率: {current_success_rate:.1%} | "
                      f"长度: {result['episode_length']:3d} | "
                      f"学习率: {result['adaptive_lr']:.2e} | "
                      f"熵奖励: {result['entropy_bonus']:.3f} | "
                      f"时间: {elapsed_time:.1f}s")
            
            # 定期保存图表
            if (episode + 1) % plot_interval == 0:
                self.plot_training_progress(f'training_progress_{episode+1}.png')
        
        # 最终图表
        self.plot_training_progress('final_training_progress.png')
        
        # 训练完成统计
        total_time = time.time() - start_time
        final_success_rate = self.agent.success_count / self.agent.total_episodes
        avg_reward = np.mean(self.episode_rewards)
        
        print("\n" + "=" * 80)
        print("改进训练完成!")
        print(f"总时间: {total_time:.1f}秒")
        print(f"最终成功率: {final_success_rate:.1%}")
        print(f"最佳成功率: {self.best_success_rate:.1%}")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"总episodes: {self.agent.total_episodes}")
        print(f"成功episodes: {self.agent.success_count}")
        print(f"停滞episodes: {self.stagnant_episodes}")
        
        # 保存模型
        self.agent.save_model('improved_terrain_ppo_model.pth')
        print("模型已保存到 improved_terrain_ppo_model.pth")


def main():
    """主函数"""
    print("改进的地形PPO训练系统")
    print("=" * 80)
    
    # 创建环境和智能体（调整参数）
    env = TerrainGridNavEnv(
        H=20, W=20, 
        max_steps=120,  # 稍微增加步数限制
        height_range=(0.0, 10.0),  # 稍微降低高度范围
        slope_penalty_weight=0.2,  # 降低坡度惩罚
        height_penalty_weight=0.15   # 降低高度惩罚
    )
    
    agent = TerrainPPOAgent(
        state_dim=13,
        action_dim=4,
        hidden_dim=256,  # 增加网络容量
        lr=2e-4,  # 稍微增加学习率
        gamma=0.99,
        gae_lambda=0.95,
        train_pi_iters=30,  # 减少训练迭代次数
        train_v_iters=30,
        target_kl=0.02  # 增加KL散度容忍度
    )
    
    # 创建训练器
    trainer = ImprovedTerrainPPOTrainer(env, agent)
    
    # 开始训练
    trainer.train(num_episodes=2000, plot_interval=100)


if __name__ == "__main__":
    main()
