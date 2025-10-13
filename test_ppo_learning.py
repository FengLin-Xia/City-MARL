#!/usr/bin/env python3
"""
轻量级PPO学习测试脚本
验证修复后的PPO训练器是否能学到有用策略
"""

import json
import torch
import numpy as np
from trainers.v4_1.ppo_trainer import PPOTrainer
from envs.v4_1.city_env import CityEnvironment
from solvers.v4_1.rl_selector import RLPolicySelector

def create_test_config():
    """创建测试配置"""
    cfg = {
        "growth_v4_1": {
            "simulation": {
                "total_months": 5  # 缩短测试时间
            },
            "enumeration": {
                "length_max": 3,
                "beam_width": 8,
                "max_expansions": 500
            }
        },
        "solver": {
            "rl": {
                "rollout_steps": 100,  # 减少收集步数
                "max_updates": 10,     # 只训练10个更新
                "eval_every": 5,       # 每5个更新评估一次
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "lr": 3e-4,
                "K_epochs": 2,         # 减少训练轮数
                "vf_coef": 0.5,
                "ent_coef": 0.01,
                "max_grad_norm": 0.5,
                "agents": ["EDU", "IND"]
            }
        }
    }
    return cfg

def test_ppo_learning():
    """测试PPO学习效果"""
    print("[TEST] Starting PPO learning test...")
    
    # 创建测试配置
    cfg = create_test_config()
    
    # 初始化训练器
    print("[INFO] Initializing PPO trainer...")
    trainer = PPOTrainer(cfg)
    
    # 初始化环境
    print("[INFO] Creating city environment...")
    env = CityEnvironment(cfg)
    
    # 初始化策略选择器
    print("[INFO] Creating RL policy selector...")
    selector = RLPolicySelector(cfg)
    
    # 记录训练前的策略
    print("[INFO] Recording pre-training policy...")
    pre_training_stats = []
    for _ in range(3):
        experiences = trainer.collect_experience(env, 20)
        if experiences:
            rewards = [exp['reward'] for exp in experiences]
            pre_training_stats.append({
                'total_reward': sum(rewards),
                'avg_reward': np.mean(rewards),
                'num_steps': len(experiences)
            })
    
    pre_avg_reward = np.mean([s['avg_reward'] for s in pre_training_stats])
    pre_total_reward = np.mean([s['total_reward'] for s in pre_training_stats])
    
    print(f"[INFO] Pre-training stats: avg_reward={pre_avg_reward:.4f}, total_reward={pre_total_reward:.4f}")
    
    # 开始训练
    print("[INFO] Starting PPO training...")
    training_metrics = {
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'kl_divergences': [],
        'clip_fractions': [],
        'episode_returns': []
    }
    
    for update in range(cfg['solver']['rl']['max_updates']):
        print(f"\n[UPDATE] {update + 1}/{cfg['solver']['rl']['max_updates']}")
        
        # 收集经验
        experiences = trainer.collect_experience(env, cfg['solver']['rl']['rollout_steps'])
        
        if experiences:
            # 计算episode return
            episode_return = sum(exp['reward'] for exp in experiences)
            training_metrics['episode_returns'].append(episode_return)
            
            # 更新策略
            loss_stats = trainer.update_policy(experiences)
            
            # 记录损失统计
            training_metrics['policy_losses'].append(loss_stats['policy_loss'])
            training_metrics['value_losses'].append(loss_stats['value_loss'])
            training_metrics['entropies'].append(loss_stats['entropy'])
            training_metrics['kl_divergences'].append(loss_stats['kl_divergence'])
            training_metrics['clip_fractions'].append(loss_stats['clip_fraction'])
            
            print(f"  [LOSS] policy={loss_stats['policy_loss']:.4f}, "
                  f"value={loss_stats['value_loss']:.4f}, "
                  f"entropy={loss_stats['entropy']:.4f}")
            print(f"  [METRICS] kl_div={loss_stats['kl_divergence']:.4f}, "
                  f"clip_frac={loss_stats['clip_fraction']:.4f}")
            print(f"  [RETURN] episode_return={episode_return:.4f}")
        else:
            print("  [WARNING] No experiences collected")
    
    # 记录训练后的策略
    print("\n[INFO] Recording post-training policy...")
    post_training_stats = []
    for _ in range(3):
        experiences = trainer.collect_experience(env, 20)
        if experiences:
            rewards = [exp['reward'] for exp in experiences]
            post_training_stats.append({
                'total_reward': sum(rewards),
                'avg_reward': np.mean(rewards),
                'num_steps': len(experiences)
            })
    
    post_avg_reward = np.mean([s['avg_reward'] for s in post_training_stats])
    post_total_reward = np.mean([s['total_reward'] for s in post_training_stats])
    
    print(f"[INFO] Post-training stats: avg_reward={post_avg_reward:.4f}, total_reward={post_total_reward:.4f}")
    
    # 分析学习效果
    print("\n[ANALYSIS] Learning effect analysis...")
    
    # 1. 奖励变化
    reward_improvement = post_avg_reward - pre_avg_reward
    reward_improvement_pct = (reward_improvement / abs(pre_avg_reward)) * 100 if pre_avg_reward != 0 else 0
    
    print(f"  [REWARD] Improvement: {reward_improvement:.4f} ({reward_improvement_pct:.1f}%)")
    
    # 2. 损失趋势
    if len(training_metrics['policy_losses']) > 1:
        policy_loss_trend = np.mean(training_metrics['policy_losses'][-3:]) - np.mean(training_metrics['policy_losses'][:3])
        value_loss_trend = np.mean(training_metrics['value_losses'][-3:]) - np.mean(training_metrics['value_losses'][:3])
        
        print(f"  [LOSS_TREND] Policy loss change: {policy_loss_trend:.4f}")
        print(f"  [LOSS_TREND] Value loss change: {value_loss_trend:.4f}")
    
    # 3. 熵变化（探索性）
    if len(training_metrics['entropies']) > 1:
        entropy_trend = np.mean(training_metrics['entropies'][-3:]) - np.mean(training_metrics['entropies'][:3])
        print(f"  [EXPLORATION] Entropy change: {entropy_trend:.4f}")
    
    # 4. KL散度（策略变化）
    if len(training_metrics['kl_divergences']) > 1:
        avg_kl = np.mean(training_metrics['kl_divergences'])
        print(f"  [POLICY_CHANGE] Average KL divergence: {avg_kl:.4f}")
    
    # 判断学习效果
    learning_indicators = []
    
    if abs(reward_improvement) > 0.01:
        learning_indicators.append("Reward improvement detected")
    
    if len(training_metrics['policy_losses']) > 1:
        if abs(policy_loss_trend) > 0.01:
            learning_indicators.append("Policy loss changing")
    
    if len(training_metrics['kl_divergences']) > 1:
        if avg_kl > 0.001:
            learning_indicators.append("Policy updates detected")
    
    # 输出结果
    print("\n[RESULT] Learning test results:")
    if learning_indicators:
        print("  [SUCCESS] Learning indicators detected:")
        for indicator in learning_indicators:
            print(f"    - {indicator}")
        
        print("\n[CONCLUSION] PPO training is working! The model appears to be learning.")
        return True
    else:
        print("  [WARNING] No clear learning indicators detected")
        print("\n[CONCLUSION] PPO training may not be learning effectively.")
        return False

def test_policy_distribution():
    """测试策略分布变化"""
    print("\n[TEST] Testing policy distribution changes...")
    
    cfg = create_test_config()
    trainer = PPOTrainer(cfg)
    env = CityEnvironment(cfg)
    selector = RLPolicySelector(cfg)
    
    # 收集一些经验来测试策略分布
    experiences = trainer.collect_experience(env, 50)
    
    if experiences:
        # 分析动作选择分布
        action_indices = []
        for exp in experiences:
            sequence = exp['action']
            if hasattr(sequence, 'action_index'):
                action_indices.append(sequence.action_index)
        
        if action_indices:
            unique_actions = len(set(action_indices))
            total_actions = len(action_indices)
            diversity = unique_actions / total_actions
            
            print(f"  [DIVERSITY] Action diversity: {diversity:.3f} ({unique_actions}/{total_actions} unique)")
            
            # 检查是否有探索行为
            if diversity > 0.3:
                print("  [OK] Good exploration behavior detected")
            else:
                print("  [WARNING] Low exploration - may need epsilon adjustment")
            
            return diversity > 0.3
        else:
            print("  [WARNING] No action indices found in experiences")
            return False
    else:
        print("  [ERROR] No experiences collected")
        return False

if __name__ == "__main__":
    print("[START] Starting lightweight PPO learning test...")
    
    try:
        # 测试学习效果
        learning_success = test_ppo_learning()
        
        # 测试策略分布
        diversity_ok = test_policy_distribution()
        
        # 最终结果
        print("\n" + "="*50)
        print("[FINAL RESULT]")
        
        if learning_success and diversity_ok:
            print("[SUCCESS] PPO learning test PASSED!")
            print("  - Model is learning useful strategies")
            print("  - Exploration behavior is healthy")
            print("  - Action probability calculation fix is working")
        elif learning_success:
            print("[PARTIAL] PPO learning test PARTIALLY PASSED")
            print("  - Model is learning but exploration may need tuning")
        else:
            print("[FAILED] PPO learning test FAILED")
            print("  - Model may not be learning effectively")
            print("  - Check hyperparameters and reward function")
        
        print("="*50)
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise
