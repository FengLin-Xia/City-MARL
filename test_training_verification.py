#!/usr/bin/env python3
"""
测试v5.0训练是否真的在进行

验证训练器是否真的执行了梯度更新
"""

import json
import sys
import os
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainers.v5_0.ppo_trainer import V5PPOTrainer
from integration.v5_0 import V5TrainingPipeline


def test_training_verification():
    """测试训练验证"""
    print("=" * 60)
    print("测试v5.0训练验证")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("\n1. 训练器初始化检查:")
    try:
        trainer = V5PPOTrainer('configs/city_config_v5_0.json')
        print("   [PASS] 训练器初始化成功")
        
        # 检查训练器状态
        print(f"   当前更新次数: {trainer.current_update}")
        print(f"   最大更新次数: {trainer.max_updates}")
        print(f"   训练步数: {trainer.training_step}")
        
        # 检查网络参数
        print("\n2. 网络参数检查:")
        for agent in trainer.selector.actor_networks:
            actor_params = list(trainer.selector.actor_networks[agent].parameters())
            critic_params = list(trainer.selector.critic_networks[agent].parameters())
            print(f"   {agent}:")
            print(f"     Actor参数数量: {len(actor_params)}")
            print(f"     Critic参数数量: {len(critic_params)}")
            
            if actor_params:
                print(f"     第一个Actor参数形状: {actor_params[0].shape}")
                print(f"     第一个Actor参数值: {actor_params[0].data[:5]}")
            
            if critic_params:
                print(f"     第一个Critic参数形状: {critic_params[0].shape}")
                print(f"     第一个Critic参数值: {critic_params[0].data[:5]}")
        
        # 检查优化器
        print("\n3. 优化器检查:")
        for agent in trainer.optimizers:
            print(f"   {agent}:")
            print(f"     Actor优化器: {type(trainer.optimizers[agent]['actor'])}")
            print(f"     Critic优化器: {type(trainer.optimizers[agent]['critic'])}")
        
        print("\n4. 训练参数检查:")
        print(f"   学习率: {trainer.lr}")
        print(f"   裁剪比例: {trainer.clip_ratio}")
        print(f"   熵系数: {trainer.entropy_coef}")
        print(f"   价值损失系数: {trainer.value_loss_coef}")
        print(f"   最大梯度范数: {trainer.max_grad_norm}")
        print(f"   每次迭代更新次数: {trainer.updates_per_iter}")
        print(f"   最大更新次数: {trainer.max_updates}")
        
        print("\n5. 训练前参数快照:")
        # 记录训练前的参数
        pre_training_params = {}
        for agent in trainer.selector.actor_networks:
            pre_training_params[agent] = {
                'actor': [p.clone() for p in trainer.selector.actor_networks[agent].parameters()],
                'critic': [p.clone() for p in trainer.selector.critic_networks[agent].parameters()]
            }
        
        print("   参数快照已保存")
        
        print("\n6. 执行训练步骤:")
        # 收集经验
        print("   收集经验...")
        experiences = trainer.collect_experience(10)
        print(f"   收集到 {len(experiences)} 个经验")
        
        if experiences:
            # 执行训练
            print("   执行训练步骤...")
            train_stats = trainer.train_step(experiences)
            print(f"   训练统计: {train_stats}")
            
            print("\n7. 训练后参数检查:")
            # 检查参数是否改变
            params_changed = False
            for agent in trainer.selector.actor_networks:
                current_actor_params = list(trainer.selector.actor_networks[agent].parameters())
                current_critic_params = list(trainer.selector.critic_networks[agent].parameters())
                
                # 比较参数
                for i, (old_param, new_param) in enumerate(zip(pre_training_params[agent]['actor'], current_actor_params)):
                    if not torch.allclose(old_param, new_param, atol=1e-6):
                        print(f"   [PASS] {agent} Actor参数 {i} 已更新")
                        params_changed = True
                        break
                
                for i, (old_param, new_param) in enumerate(zip(pre_training_params[agent]['critic'], current_critic_params)):
                    if not torch.allclose(old_param, new_param, atol=1e-6):
                        print(f"   [PASS] {agent} Critic参数 {i} 已更新")
                        params_changed = True
                        break
            
            if not params_changed:
                print("   [WARN] 参数未发生变化，可能训练未真正执行")
            else:
                print("   [PASS] 参数已更新，训练确实执行了")
            
            print(f"\n8. 训练状态更新:")
            print(f"   当前更新次数: {trainer.current_update}")
            print(f"   训练步数: {trainer.training_step}")
            print(f"   总损失: {train_stats.get('total_loss', 0):.4f}")
            print(f"   Actor损失: {train_stats.get('actor_loss', 0):.4f}")
            print(f"   Critic损失: {train_stats.get('critic_loss', 0):.4f}")
            
        else:
            print("   [FAIL] 没有收集到经验，无法进行训练")
            return False
        
        print("\n9. 训练历史检查:")
        print(f"   训练历史记录数量: {len(trainer.training_history)}")
        if trainer.training_history:
            latest_history = trainer.training_history[-1]
            print(f"   最新训练记录: {latest_history}")
        
        print("\n10. 设备检查:")
        print(f"   使用设备: {trainer.device}")
        print(f"   是否使用GPU: {trainer.device.type == 'cuda'}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] 训练器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline():
    """测试训练管道"""
    print("\n" + "=" * 60)
    print("测试训练管道")
    print("=" * 60)
    
    try:
        # 创建训练管道
        pipeline = V5TrainingPipeline('configs/city_config_v5_0.json')
        print("   [PASS] 训练管道创建成功")
        
        # 检查管道步骤
        print(f"   管道步骤数量: {len(pipeline.pipeline.steps)}")
        for i, step in enumerate(pipeline.pipeline.steps):
            print(f"   步骤 {i+1}: {step.name}")
        
        # 执行训练管道
        print("\n   执行训练管道...")
        result = pipeline.run_training(1)  # 1个episode
        
        print(f"   训练结果: {result}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] 训练管道测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        # 测试训练验证
        success1 = test_training_verification()
        
        # 测试训练管道
        success2 = test_training_pipeline()
        
        if success1 and success2:
            print("\n" + "=" * 60)
            print("所有测试通过!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("部分测试失败，请检查训练实现!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
