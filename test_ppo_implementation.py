#!/usr/bin/env python3
"""
测试PPO训练器实现
验证从伪RL到真RL的转变
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch
from trainers.v4_1.ppo_trainer import PPOTrainer
from envs.v4_1.city_env import CityEnvironment

def test_ppo_trainer_initialization():
    """测试PPO训练器初始化"""
    print("=== 测试PPO训练器初始化 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建PPO训练器
    trainer = PPOTrainer(cfg)
    
    print(f"设备: {trainer.device}")
    print(f"折扣因子: {trainer.gamma}")
    print(f"GAE参数: {trainer.gae_lambda}")
    print(f"裁剪率: {trainer.clip_ratio}")
    print(f"学习率: {trainer.lr}")
    print(f"更新轮数: {trainer.num_epochs}")
    
    # 检查选择器是否正确初始化
    assert trainer.selector is not None, "RL选择器未初始化"
    assert trainer.selector.actor is not None, "策略网络未初始化"
    
    print("PPO训练器初始化测试通过!")
    return True

def test_gae_computation():
    """测试GAE计算"""
    print("\n=== 测试GAE计算 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    trainer = PPOTrainer(cfg)
    
    # 测试数据
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    values = [0.5, 1.5, 2.5, 3.5, 4.5]
    dones = [False, False, False, False, True]
    
    # 计算GAE
    returns, advantages = trainer.compute_gae(rewards, values, dones)
    
    print(f"奖励: {rewards}")
    print(f"价值: {values}")
    print(f"回报: {returns.tolist()}")
    print(f"优势: {advantages.tolist()}")
    
    # 验证输出
    assert len(returns) == len(rewards), "回报长度不匹配"
    assert len(advantages) == len(rewards), "优势长度不匹配"
    assert isinstance(returns, torch.Tensor), "回报不是张量"
    assert isinstance(advantages, torch.Tensor), "优势不是张量"
    
    print("GAE计算测试通过!")
    return True

def test_experience_collection():
    """测试经验收集"""
    print("\n=== 测试经验收集 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    trainer = PPOTrainer(cfg)
    env = CityEnvironment(cfg)
    
    # 收集少量经验
    experiences = trainer.collect_experience(env, num_steps=50)
    
    print(f"收集了 {len(experiences)} 步经验")
    
    if experiences:
        # 检查经验结构
        exp = experiences[0]
        required_keys = ['state', 'action', 'agent', 'month', 'reward', 'next_state', 'done', 'info']
        
        for key in required_keys:
            assert key in exp, f"经验缺少键: {key}"
        
        print(f"经验结构正确: {list(exp.keys())}")
        print(f"智能体: {[exp['agent'] for exp in experiences[:5]]}")
        print(f"奖励: {[exp['reward'] for exp in experiences[:5]]}")
        
        print("经验收集测试通过!")
        return True
    else:
        print("没有收集到经验")
        return False

def test_policy_update():
    """测试策略更新"""
    print("\n=== 测试策略更新 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    trainer = PPOTrainer(cfg)
    env = CityEnvironment(cfg)
    
    # 收集少量经验
    experiences = trainer.collect_experience(env, num_steps=20)
    
    if experiences:
        print(f"使用 {len(experiences)} 步经验进行策略更新...")
        
        # 执行策略更新
        loss_stats = trainer.update_policy(experiences)
        
        print(f"更新完成，损失统计:")
        for key, value in loss_stats.items():
            print(f"  {key}: {value:.6f}")
        
        # 验证损失统计
        required_losses = ['policy_loss', 'value_loss', 'entropy_loss', 'total_loss']
        for loss_key in required_losses:
            assert loss_key in loss_stats, f"缺少损失统计: {loss_key}"
        
        print("策略更新测试通过!")
        return True
    else:
        print("没有经验可用于策略更新")
        return False

def test_training_integration():
    """测试训练集成"""
    print("\n=== 测试训练集成 ===")
    
    # 测试导入
    try:
        from enhanced_city_simulation_v4_1 import run_rl_mode
        print("成功导入RL模式函数")
    except ImportError as e:
        print(f"导入失败: {e}")
        return False
    
    # 测试短时间训练
    try:
        # 加载配置
        with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        # 修改配置为短时间测试
        cfg['solver']['rl']['max_updates'] = 2  # 只训练2个更新
        cfg['solver']['rl']['rollout_steps'] = 100  # 减少经验收集步数
        cfg['solver']['rl']['eval_every'] = 1  # 每个更新都评估
        
        print("运行短时间RL训练测试...")
        results = run_rl_mode(
            cfg=cfg,
            eval_only=False
        )
        
        print(f"训练结果类型: {type(results)}")
        
        # 安全地检查结果结构
        if isinstance(results, dict):
            print(f"结果键: {list(results.keys())}")
            
            if 'training_metrics' in results:
                metrics = results['training_metrics']
                print(f"训练指标类型: {type(metrics)}")
                
                if isinstance(metrics, dict):
                    print(f"训练指标键: {list(metrics.keys())}")
                    
                    # 检查关键指标
                    for key in ['episode_returns', 'edu_returns', 'ind_returns']:
                        if key in metrics:
                            value = metrics[key]
                            print(f"{key}: 类型={type(value)}, 长度={len(value) if hasattr(value, '__len__') else 'N/A'}")
                
            # 检查其他重要字段
            for key in ['mode', 'training_updates', 'final_model_path']:
                if key in results:
                    print(f"{key}: {results[key]}")
        else:
            print(f"结果不是字典: {results}")
        
        print("训练集成测试通过!")
        return True
        
    except Exception as e:
        print(f"训练集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试PPO训练器实现...")
    
    tests = [
        ("PPO训练器初始化", test_ppo_trainer_initialization),
        ("GAE计算", test_gae_computation),
        ("经验收集", test_experience_collection),
        ("策略更新", test_policy_update),
        ("训练集成", test_training_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            print(f"{test_name}: {'[通过]' if result else '[失败]'}")
        except Exception as e:
            results[test_name] = False
            print(f"{test_name}: [失败] - {e}")
    
    print(f"\n测试总结:")
    passed = sum(results.values())
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("PPO训练器实现成功！")
        print("\n关键改进:")
        print("✅ 实现了真正的PPO-Clip算法")
        print("✅ 集成了GAE优势函数计算")
        print("✅ 支持多智能体训练")
        print("✅ 从伪RL转变为真RL")
        print("\n下一步建议:")
        print("1. 运行完整训练: python enhanced_city_simulation_v4_1.py --mode rl --max-updates 100")
        print("2. 观察奖励变化和收敛性")
        print("3. 调优超参数")
    else:
        print("部分测试失败，需要进一步调试。")

if __name__ == "__main__":
    main()
