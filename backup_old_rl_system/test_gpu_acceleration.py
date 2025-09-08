#!/usr/bin/env python3
"""
GPU加速强化学习测试
验证从环境到训练的完整GPU加速流程
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.terrain_road_env import TerrainRoadEnvironment
from agents.terrain_policy import TerrainPolicyNetwork

def test_device_setup():
    """测试设备设置"""
    print("=== 设备设置测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device

def test_environment_gpu(device):
    """测试环境GPU化"""
    print("\n=== 环境GPU化测试 ===")
    
    # 查找地形数据
    terrain_dir = Path("data/terrain")
    terrain_files = list(terrain_dir.glob("terrain_continuity_boundary_*.json"))
    
    if not terrain_files:
        print("❌ 未找到地形数据文件")
        return None
    
    latest_file = max(terrain_files, key=lambda x: x.stat().st_mtime)
    print(f"使用地形文件: {latest_file}")
    
    # 创建环境
    env = TerrainRoadEnvironment(mesh_file=str(latest_file))
    print(f"环境网格尺寸: {env.grid_size}")
    
    # 测试观察数据GPU化
    obs, _ = env.reset()
    print(f"观察空间键: {list(obs.keys())}")
    
    # 将观察数据移到GPU
    gpu_obs = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            gpu_obs[key] = torch.from_numpy(value).to(device)
            print(f"{key}: {value.shape} -> GPU: {gpu_obs[key].device}")
        else:
            gpu_obs[key] = value
    
    # 测试GPU数据操作
    height_map_gpu = gpu_obs['height_map']
    print(f"GPU高程图统计: 最小值={height_map_gpu.min().item():.2f}, 最大值={height_map_gpu.max().item():.2f}")
    
    return env, gpu_obs

def test_network_gpu(device, env):
    """测试网络GPU化"""
    print("\n=== 网络GPU化测试 ===")
    
    # 创建策略网络
    policy_net = TerrainPolicyNetwork(grid_size=env.grid_size, action_space=env.action_space)
    policy_net = policy_net.to(device)
    print(f"策略网络已移到: {next(policy_net.parameters()).device}")
    
    # 测试前向传播
    obs, _ = env.reset()
    obs_tensor = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).to(device)  # 添加batch维度
        else:
            obs_tensor[key] = torch.tensor([value]).to(device)
    
    print("测试前向传播...")
    with torch.no_grad():
        action_probs, value = policy_net(obs_tensor)
    
    print(f"动作概率形状: {action_probs.shape}")
    print(f"价值估计: {value.item():.4f}")
    print(f"动作概率设备: {action_probs.device}")
    print(f"价值估计设备: {value.device}")
    
    return policy_net

def test_training_step(device, env, policy_net):
    """测试训练步骤"""
    print("\n=== 训练步骤测试 ===")
    
    # 创建优化器
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    
    # 执行一个完整的训练步骤
    obs, _ = env.reset()
    
    # 准备数据
    obs_tensor = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).to(device)
        else:
            obs_tensor[key] = torch.tensor([value]).to(device)
    
    # 前向传播
    action_logits, value = policy_net(obs_tensor)
    
    # 采样动作
    action_probs = torch.softmax(action_logits, dim=-1)
    action_dist = torch.distributions.Categorical(action_probs)
    action = action_dist.sample()
    
    # 执行动作
    next_obs, reward, done, truncated, info = env.step(action.item())
    
    # 计算损失（简化版本）
    loss = -action_dist.log_prob(action) * torch.tensor(reward, device=device)  # 简单的策略梯度损失
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"动作: {action.item()}")
    print(f"奖励: {reward:.4f}")
    print(f"损失: {loss.item():.4f}")
    print(f"损失设备: {loss.device}")
    print("✅ 训练步骤完成")

def test_performance_comparison(device, env, policy_net):
    """性能对比测试"""
    print("\n=== 性能对比测试 ===")
    
    # 测试GPU性能
    print("测试GPU性能...")
    start_time = time.time()
    
    for _ in range(10):
        obs, _ = env.reset()
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).to(device)
            else:
                obs_tensor[key] = torch.tensor([value]).to(device)
        
        with torch.no_grad():
            action_probs, value = policy_net(obs_tensor)
    
    gpu_time = time.time() - start_time
    print(f"GPU 10次前向传播耗时: {gpu_time:.4f}秒")
    
    # 测试CPU性能（如果可能）
    if device.type == 'cuda':
        print("测试CPU性能...")
        policy_net_cpu = policy_net.cpu()
        start_time = time.time()
        
        for _ in range(10):
            obs, _ = env.reset()
            obs_tensor = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    obs_tensor[key] = torch.from_numpy(value).unsqueeze(0)
                else:
                    obs_tensor[key] = torch.tensor([value])
            
            with torch.no_grad():
                action_probs, value = policy_net_cpu(obs_tensor)
        
        cpu_time = time.time() - start_time
        print(f"CPU 10次前向传播耗时: {cpu_time:.4f}秒")
        print(f"GPU加速比: {cpu_time/gpu_time:.2f}x")
        
        # 移回GPU
        policy_net = policy_net_cpu.to(device)

def test_memory_usage(device):
    """测试内存使用"""
    print("\n=== 内存使用测试 ===")
    
    if device.type == 'cuda':
        print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"GPU已用内存: {torch.cuda.memory_allocated(0) / 1024**3:.3f} GB")
        print(f"GPU缓存内存: {torch.cuda.memory_reserved(0) / 1024**3:.3f} GB")
        
        # 清理缓存
        torch.cuda.empty_cache()
        print("已清理GPU缓存")

def main():
    """主测试函数"""
    print("🚀 GPU加速强化学习测试开始")
    print("=" * 50)
    
    # 1. 设备设置测试
    device = test_device_setup()
    
    # 2. 环境GPU化测试
    result = test_environment_gpu(device)
    if result is None:
        print("❌ 环境测试失败，退出")
        return
    
    env, gpu_obs = result
    
    # 3. 网络GPU化测试
    policy_net = test_network_gpu(device, env)
    
    # 4. 训练步骤测试
    test_training_step(device, env, policy_net)
    
    # 5. 性能对比测试
    test_performance_comparison(device, env, policy_net)
    
    # 6. 内存使用测试
    test_memory_usage(device)
    
    print("\n" + "=" * 50)
    print("✅ GPU加速测试完成！")
    print("现在可以开始GPU加速的强化学习训练了！")

if __name__ == "__main__":
    main()
