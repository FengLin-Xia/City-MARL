#!/usr/bin/env python3
"""
测试v5.0训练指标记录功能
"""

import os
import sys
import json
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(__file__))

from trainers.v5_0.ppo_trainer import V5PPOTrainer


def test_training_metrics():
    """测试训练指标记录功能"""
    print("=" * 60)
    print("测试v5.0训练指标记录功能")
    print("=" * 60)
    
    # 配置文件路径
    config_path = "configs/city_config_v5_0.json"
    
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在: {config_path}")
        return
    
    try:
        # 初始化训练器
        print("初始化PPO训练器...")
        trainer = V5PPOTrainer(config_path)
        
        # 显示训练器配置
        print(f"训练器配置:")
        print(f"  - 学习率: {trainer.lr}")
        print(f"  - 折扣因子: {trainer.gamma}")
        print(f"  - GAE Lambda: {trainer.gae_lambda}")
        print(f"  - 裁剪参数: {trainer.clip_eps}")
        print(f"  - 熵系数: {trainer.entropy_coef}")
        print(f"  - 价值损失系数: {trainer.value_loss_coef}")
        print(f"  - 最大梯度范数: {trainer.max_grad_norm}")
        print(f"  - 温度范围: {trainer.initial_temperature} -> {trainer.final_temperature}")
        print(f"  - 智能体: {trainer.config.get('agents', {}).get('order', [])}")
        
        # 运行一个简短的训练测试
        print(f"\n运行训练测试 (1 episode)...")
        result = trainer.train(num_episodes=1, save_interval=1)
        
        # 显示训练结果
        print(f"\n训练结果:")
        print(f"  - 成功: {result.get('success', False)}")
        print(f"  - 总episode数: {result.get('total_episodes', 0)}")
        print(f"  - 训练历史长度: {len(result.get('training_history', []))}")
        
        # 显示训练历史中的指标
        training_history = result.get('training_history', [])
        if training_history:
            print(f"\n训练指标示例:")
            latest_stats = training_history[-1]
            for key, value in latest_stats.items():
                if isinstance(value, (int, float)):
                    print(f"  - {key}: {value:.4f}")
                else:
                    print(f"  - {key}: {value}")
        
        # 显示episode奖励
        episode_rewards = result.get('episode_rewards', {})
        if episode_rewards:
            print(f"\nEpisode奖励:")
            for agent, rewards in episode_rewards.items():
                if rewards:
                    avg_reward = sum(rewards) / len(rewards)
                    print(f"  - {agent}: 平均奖励 = {avg_reward:.2f}")
        
        print(f"\n✅ 训练指标记录功能测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_training_metrics()

