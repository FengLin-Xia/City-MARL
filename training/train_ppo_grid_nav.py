#!/usr/bin/env python3
"""
PPO网格导航训练脚本
"""

import sys
import os
import time
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.grid_nav_env import GridNavEnv
from agents.ppo_grid_nav_agent import PPOGridNavAgent, test_ppo_agent


def main():
    """主函数"""
    print("🚀 开始PPO网格导航训练...")
    
    try:
        # 创建环境和智能体
        env = GridNavEnv()
        agent = PPOGridNavAgent(lr=3e-4)
        
        print(f"📍 固定起点: {env.start}")
        print(f"🎯 固定终点: {env.goal}")
        print(f"🖥️  设备: {agent.device}")
        print("=" * 60)
        
        # 开始训练
        num_episodes = 500
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            agent.train_episode(env, episode)
            
            # 每100个episodes显示一次统计
            if episode % 100 == 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(agent.episode_rewards[-100:])
                success_rate = np.mean(agent.success_rates[-100:]) * 100
                print(f"📊 Episode {episode:4d} | 用时: {elapsed_time:.1f}s | "
                      f"平均奖励: {avg_reward:6.1f} | 成功率: {success_rate:5.1f}%")
        
        print("=" * 60)
        print("🎉 训练完成！")
        print(f"📈 最终平均奖励: {np.mean(agent.episode_rewards[-50:]):.2f}")
        print(f"🎯 最终成功率: {np.mean(agent.success_rates[-50:])*100:.1f}%")
        
        # 测试智能体
        test_ppo_agent(env, agent, num_tests=20)
        
        print("✅ 训练完成！")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
