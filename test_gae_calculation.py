#!/usr/bin/env python3
"""
测试GAE计算逻辑
"""

import torch
import numpy as np

def compute_gae_manual(rewards, values, dones, gamma=0.99, gae_lambda=0.8, next_value=0.0):
    """手动计算GAE，用于验证逻辑"""
    advantages = []
    returns = []
    
    print("=== GAE计算调试 ===")
    print(f"Rewards: {rewards}")
    print(f"Values: {values}")
    print(f"Dones: {dones}")
    print(f"Next value: {next_value}")
    print(f"Gamma: {gamma}, GAE lambda: {gae_lambda}")
    print()
    
    # 计算GAE（参考现有实现）
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[i + 1]
        
        # 计算时序差分误差
        delta = rewards[i] + gamma * next_val * (1 - dones[i]) - values[i]
        
        # 累积GAE
        gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
        
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])
        
        print(f"Step {i}: delta={delta:.6f}, gae={gae:.6f}, return={gae + values[i]:.6f}")
    
    print()
    print(f"Final advantages: {advantages}")
    print(f"Final returns: {returns}")
    
    return advantages, returns

def test_scenario_1():
    """测试场景1：所有values相同（Critic不学习的情况）"""
    print("=== 测试场景1：所有values相同 ===")
    rewards = [0.025, 0.357, 0.851, 0.056, 0.508]  # 实际的奖励序列
    values = [0.5, 0.5, 0.5, 0.5, 0.5]  # 所有value相同
    dones = [False, False, False, False, False]
    next_value = 0.5
    
    compute_gae_manual(rewards, values, dones, next_value=next_value)

def test_scenario_2():
    """测试场景2：values有差异"""
    print("\n=== 测试场景2：values有差异 ===")
    rewards = [0.025, 0.357, 0.851, 0.056, 0.508]
    values = [0.4, 0.6, 0.8, 0.3, 0.7]  # values有差异
    dones = [False, False, False, False, False]
    next_value = 0.5
    
    compute_gae_manual(rewards, values, dones, next_value=next_value)

def test_scenario_3():
    """测试场景3：简单的奖励序列"""
    print("\n=== 测试场景3：简单的奖励序列 ===")
    rewards = [1.0, 2.0, 3.0]
    values = [1.0, 2.0, 3.0]  # values等于rewards
    dones = [False, False, False]
    next_value = 0.0
    
    compute_gae_manual(rewards, values, dones, next_value=next_value)

def test_scenario_4():
    """测试场景4：极端的values相同情况"""
    print("\n=== 测试场景4：极端的values相同情况 ===")
    rewards = [0.025, 0.357, 0.851, 0.056, 0.508]
    values = [0.0, 0.0, 0.0, 0.0, 0.0]  # 所有value为0
    dones = [False, False, False, False, False]
    next_value = 0.0
    
    compute_gae_manual(rewards, values, dones, next_value=next_value)

if __name__ == "__main__":
    test_scenario_1()
    test_scenario_2()
    test_scenario_3()
    test_scenario_4()

