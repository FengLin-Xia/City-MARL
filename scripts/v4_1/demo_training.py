#!/usr/bin/env python3
"""
v4.1 RL训练演示脚本
演示如何使用v4.1进行RL训练
"""

import sys
import os
import json
import torch
import numpy as np
from typing import Dict, Any

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from envs.v4_1.city_env import CityEnvironment
from rl.v4_1 import PPOTrainer, MAPPOTrainer


def demo_training():
    """演示训练过程"""
    print("v4.1 RL训练演示")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    print(f"求解模式: {cfg['solver']['mode']}")
    print(f"RL算法: {cfg['solver']['rl']['algo']}")
    print(f"智能体: {cfg['solver']['rl']['agents']}")
    
    # 创建环境
    env = CityEnvironment(cfg)
    print("\n环境创建成功")
    
    # 创建训练器
    if cfg['solver']['rl']['algo'] == 'ppo':
        trainer = PPOTrainer(cfg)
        print("PPO训练器创建成功")
    else:
        trainer = MAPPOTrainer(cfg)
        print("MAPPO训练器创建成功")
    
    # 演示训练循环（简化版）
    print("\n开始训练演示...")
    
    # 重置环境
    state = env.reset(seed=42)
    
    # 模拟几个训练步骤
    for update in range(3):  # 只演示3个更新
        print(f"\n--- 训练更新 {update + 1} ---")
        
        # 收集经验（简化版）
        step_count = 0
        episode_reward = 0.0
        
        while step_count < 4:  # 每个更新收集4步经验
            current_agent = state['current_agent']
            
            # 获取动作池
            actions, action_feats, mask = env.get_action_pool(current_agent)
            
            if len(actions) > 0:
                # 随机选择动作（实际应该用策略网络）
                action_idx = np.random.choice(len(actions))
                selected_action = actions[action_idx]
                
                # 执行动作
                next_state, reward, done, info = env.step(current_agent, selected_action)
                
                episode_reward += reward
                step_count += 1
                state = next_state
                
                print(f"  步骤 {step_count}: {current_agent} 奖励={reward:.4f}")
                
                if done:
                    print("  Episode完成")
                    state = env.reset(seed=42 + update)
                    break
            else:
                print(f"  {current_agent} 没有可用动作，结束收集")
                break
        
        print(f"  更新 {update + 1} 总奖励: {episode_reward:.4f}")
        
        # 这里应该调用trainer.update_policy()，但为了演示简化了
        print(f"  策略更新完成")
    
    # 保存模型
    model_path = "models/v4_1_rl/demo_model.pth"
    trainer.save_model(model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 演示评估
    print("\n--- 评估演示 ---")
    
    # 重置环境进行评估
    eval_state = env.reset(seed=123)
    eval_reward = 0.0
    eval_steps = 0
    
    while eval_steps < 6:
        current_agent = eval_state['current_agent']
        actions, action_feats, mask = env.get_action_pool(current_agent)
        
        if len(actions) > 0:
            # 贪心选择（评估时使用）
            best_action = max(actions, key=lambda a: a.score)
            next_state, reward, done, info = env.step(current_agent, best_action)
            
            eval_reward += reward
            eval_steps += 1
            eval_state = next_state
            
            print(f"  评估步骤 {eval_steps}: {current_agent} 奖励={reward:.4f}")
            
            if done:
                break
        else:
            break
    
    print(f"\n评估总奖励: {eval_reward:.4f}")
    
    # 打印最终统计
    final_stats = eval_state.get('monthly_stats', {})
    print(f"\n最终建筑统计:")
    print(f"  总建筑数: {final_stats.get('total_buildings', 0)}")
    print(f"  EDU建筑: {final_stats.get('public_buildings', 0)}")
    print(f"  IND建筑: {final_stats.get('industrial_buildings', 0)}")
    
    print("\n训练演示完成！")
    
    # 清理演示文件
    if os.path.exists(model_path):
        os.remove(model_path)
        print("演示文件已清理")


def demo_comparison():
    """演示对比模式"""
    print("\n" + "=" * 60)
    print("对比模式演示")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print("运行参数化模式...")
    
    # 参数化模式（简化演示）
    param_reward = 0.0
    param_state = env.reset(seed=42)
    param_steps = 0
    
    while param_steps < 6:
        current_agent = param_state['current_agent']
        actions, _, _ = env.get_action_pool(current_agent)
        
        if len(actions) > 0:
            # 贪心选择最佳动作
            best_action = max(actions, key=lambda a: a.score)
            next_state, reward, done, info = env.step(current_agent, best_action)
            
            param_reward += reward
            param_steps += 1
            param_state = next_state
            
            if done:
                break
        else:
            break
    
    print(f"参数化模式总奖励: {param_reward:.4f}")
    
    print("\n运行RL模式...")
    
    # RL模式（简化演示）
    rl_reward = 0.0
    rl_state = env.reset(seed=42)
    rl_steps = 0
    
    while rl_steps < 6:
        current_agent = rl_state['current_agent']
        actions, _, _ = env.get_action_pool(current_agent)
        
        if len(actions) > 0:
            # 随机选择（模拟RL策略）
            selected_action = np.random.choice(actions)
            next_state, reward, done, info = env.step(current_agent, selected_action)
            
            rl_reward += reward
            rl_steps += 1
            rl_state = next_state
            
            if done:
                break
        else:
            break
    
    print(f"RL模式总奖励: {rl_reward:.4f}")
    
    # 对比结果
    improvement = rl_reward - param_reward
    print(f"\n对比结果:")
    print(f"  参数化: {param_reward:.4f}")
    print(f"  RL:     {rl_reward:.4f}")
    print(f"  改进:   {improvement:+.4f}")
    
    if improvement > 0:
        print("  RL模式表现更好！")
    elif improvement < 0:
        print("  参数化模式表现更好")
    else:
        print("  两种模式表现相当")


def main():
    """主函数"""
    try:
        # 演示训练
        demo_training()
        
        # 演示对比
        demo_comparison()
        
    except Exception as e:
        print(f"\n[ERROR] 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

