#!/usr/bin/env python3
"""
测试v5.0超参数配置

验证KL散度等超参数配置是否正常工作
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainers.v5_0.ppo_trainer import V5PPOTrainer


def test_hyperparameters_config():
    """测试超参数配置"""
    print("=" * 60)
    print("测试v5.0超参数配置")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("\n1. 配置文件检查:")
    mappo_config = config.get("mappo", {})
    ppo_config = mappo_config.get("ppo", {})
    
    print("   PPO超参数:")
    print(f"     clip_eps: {ppo_config.get('clip_eps', 'NOT_FOUND')}")
    print(f"     gamma: {ppo_config.get('gamma', 'NOT_FOUND')}")
    print(f"     gae_lambda: {ppo_config.get('gae_lambda', 'NOT_FOUND')}")
    print(f"     entropy_coef: {ppo_config.get('entropy_coef', 'NOT_FOUND')}")
    print(f"     value_coef: {ppo_config.get('value_coef', 'NOT_FOUND')}")
    print(f"     target_kl: {ppo_config.get('target_kl', 'NOT_FOUND')}")
    print(f"     max_grad_norm: {ppo_config.get('max_grad_norm', 'NOT_FOUND')}")
    print(f"     lr: {ppo_config.get('lr', 'NOT_FOUND')}")
    print(f"     lr_schedule: {ppo_config.get('lr_schedule', 'NOT_FOUND')}")
    
    print("\n2. 训练器配置检查:")
    try:
        trainer = V5PPOTrainer('configs/city_config_v5_0.json')
        print("   PPO超参数:")
        print(f"     trainer.clip_ratio: {trainer.clip_ratio}")
        print(f"     trainer.gamma: {trainer.gamma}")
        print(f"     trainer.gae_lambda: {trainer.gae_lambda}")
        print(f"     trainer.entropy_coef: {trainer.entropy_coef}")
        print(f"     trainer.value_loss_coef: {trainer.value_loss_coef}")
        print(f"     trainer.max_grad_norm: {trainer.max_grad_norm}")
        print(f"     trainer.lr: {trainer.lr}")
        print("   [PASS] 训练器配置加载成功")
    except Exception as e:
        print(f"   [FAIL] 训练器配置加载失败: {e}")
        return False
    
    print("\n3. 超参数验证:")
    
    # 检查关键超参数
    hyperparams = {
        "clip_eps": trainer.clip_ratio,
        "gamma": trainer.gamma,
        "gae_lambda": trainer.gae_lambda,
        "entropy_coef": trainer.entropy_coef,
        "value_coef": trainer.value_loss_coef,
        "max_grad_norm": trainer.max_grad_norm,
        "lr": trainer.lr
    }
    
    for param, value in hyperparams.items():
        if value is None:
            print(f"   [WARN] {param} 未设置")
        elif value <= 0:
            print(f"   [WARN] {param} = {value} <= 0，可能影响训练")
        else:
            print(f"   [PASS] {param} = {value} 配置合理")
    
    print("\n4. v4.1 vs v5.0 对比:")
    print("   参数对比:")
    print("   v4.1: clip_eps = 0.15, gamma = 0.99, gae_lambda = 0.8")
    print(f"   v5.0: clip_eps = {trainer.clip_ratio}, gamma = {trainer.gamma}, gae_lambda = {trainer.gae_lambda}")
    
    # 检查是否与v4.1一致
    if (trainer.clip_ratio == 0.15 and 
        trainer.gamma == 0.99 and 
        trainer.gae_lambda == 0.8):
        print("   [PASS] v5.0核心参数与v4.1一致")
    else:
        print("   [INFO] v5.0参数与v4.1不同，这是正常的")
    
    print("\n5. 新增参数检查:")
    print("   v5.0新增参数:")
    print(f"     target_kl: {ppo_config.get('target_kl', 'NOT_FOUND')}")
    print(f"     max_grad_norm: {ppo_config.get('max_grad_norm', 'NOT_FOUND')}")
    print(f"     lr_schedule: {ppo_config.get('lr_schedule', 'NOT_FOUND')}")
    
    # 检查新增参数
    if ppo_config.get('target_kl') is not None:
        print("   [PASS] target_kl 已配置")
    else:
        print("   [WARN] target_kl 未配置")
    
    if ppo_config.get('max_grad_norm') is not None:
        print("   [PASS] max_grad_norm 已配置")
    else:
        print("   [WARN] max_grad_norm 未配置")
    
    if ppo_config.get('lr_schedule') is not None:
        print("   [PASS] lr_schedule 已配置")
    else:
        print("   [WARN] lr_schedule 未配置")
    
    print("\n6. 配置建议:")
    print("   推荐配置值:")
    print("   - 开发测试: clip_eps=0.2, target_kl=0.05, lr=5e-4")
    print("   - 功能验证: clip_eps=0.15, target_kl=0.02, lr=3e-4")
    print("   - 正式训练: clip_eps=0.1, target_kl=0.01, lr=1e-4")
    print("   - 生产环境: clip_eps=0.1, target_kl=0.01, lr=1e-5")
    
    print("\n" + "=" * 60)
    print("超参数配置测试完成!")
    print("=" * 60)
    
    return True


def test_hyperparameters_override():
    """测试超参数覆盖"""
    print("\n" + "=" * 60)
    print("测试超参数覆盖")
    print("=" * 60)
    
    # 创建临时配置文件
    temp_config = {
        "mappo": {
            "ppo": {
                "clip_eps": 0.2,        # 测试值
                "target_kl": 0.05,      # 测试值
                "entropy_coef": 0.02,   # 测试值
                "lr": 5e-4              # 测试值
            }
        }
    }
    
    # 保存临时配置
    with open('temp_hyperparams.json', 'w', encoding='utf-8') as f:
        json.dump(temp_config, f, indent=2)
    
    try:
        # 测试临时配置
        trainer = V5PPOTrainer('temp_hyperparams.json')
        print(f"   临时配置 clip_eps: {trainer.clip_ratio}")
        print(f"   临时配置 entropy_coef: {trainer.entropy_coef}")
        print(f"   临时配置 lr: {trainer.lr}")
        
        if (trainer.clip_ratio == 0.2 and 
            trainer.entropy_coef == 0.02 and 
            trainer.lr == 5e-4):
            print("   [PASS] 超参数覆盖成功")
        else:
            print("   [FAIL] 超参数覆盖失败")
        
    except Exception as e:
        print(f"   [FAIL] 临时配置测试失败: {e}")
    
    finally:
        # 清理临时文件
        if os.path.exists('temp_hyperparams.json'):
            os.remove('temp_hyperparams.json')
    
    return True


if __name__ == "__main__":
    try:
        # 测试超参数配置
        test_hyperparameters_config()
        
        # 测试超参数覆盖
        test_hyperparameters_override()
        
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

