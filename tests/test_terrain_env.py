#!/usr/bin/env python3
"""
地形环境接口测试脚本
验证环境功能和智能体交互
"""

import os
import sys
import numpy as np
import torch
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.terrain_road_env import TerrainRoadEnvironment, TerrainType
from agents.terrain_policy import TerrainPolicyNetwork, TerrainValueNetwork

def test_environment_creation():
    """测试环境创建"""
    print("=" * 50)
    print("测试环境创建")
    print("=" * 50)
    
    # 测试随机地形环境
    env = TerrainRoadEnvironment(grid_size=(20, 20), max_steps=100)
    print("✅ 随机地形环境创建成功")
    
    # 获取环境信息
    terrain_info = env.get_terrain_info()
    print(f"网格大小: {terrain_info['grid_size']}")
    print(f"高程范围: {terrain_info['height_range']}")
    print(f"地形分布: {terrain_info['terrain_distribution']}")
    print(f"道路覆盖率: {terrain_info['road_coverage']:.2%}")
    
    return env

def test_environment_reset():
    """测试环境重置"""
    print("\n" + "=" * 50)
    print("测试环境重置")
    print("=" * 50)
    
    env = TerrainRoadEnvironment(grid_size=(10, 10), max_steps=50)
    
    # 重置环境
    obs, info = env.reset()
    print("✅ 环境重置成功")
    
    # 检查观察空间
    print("观察空间:")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    # 检查智能体和目标位置
    print(f"智能体位置: {obs['agent_pos']}")
    print(f"目标位置: {obs['target_pos']}")
    print(f"资源状态: {obs['resources']}")
    
    return env, obs

def test_environment_step():
    """测试环境步进"""
    print("\n" + "=" * 50)
    print("测试环境步进")
    print("=" * 50)
    
    env, obs = test_environment_reset()
    
    # 测试所有动作
    actions = list(range(env.action_space.n))
    action_names = ["北", "南", "东", "西", "建造道路", "升级道路", "等待"]
    
    for i, action in enumerate(actions):
        print(f"\n执行动作 {i}: {action_names[i]}")
        
        # 执行动作
        next_obs, reward, done, truncated, info = env.step(action)
        
        print(f"  奖励: {reward:.2f}")
        print(f"  完成: {done}")
        print(f"  截断: {truncated}")
        print(f"  新智能体位置: {next_obs['agent_pos']}")
        print(f"  新资源状态: {next_obs['resources']}")
        
        if done:
            print("  ⚠️  Episode结束")
            break
    
    return env

def test_network_creation():
    """测试网络创建"""
    print("\n" + "=" * 50)
    print("测试网络创建")
    print("=" * 50)
    
    env = TerrainRoadEnvironment(grid_size=(20, 20))
    obs, _ = env.reset()
    
    # 测试策略网络
    policy_network = TerrainPolicyNetwork(
        grid_size=env.grid_size,
        action_space=env.action_space,
        hidden_dim=128
    )
    print("✅ 策略网络创建成功")
    
    # 测试价值网络
    value_network = TerrainValueNetwork(
        grid_size=env.grid_size,
        action_space=env.action_space,
        hidden_dim=128
    )
    print("✅ 价值网络创建成功")
    
    return env, obs, policy_network, value_network

def test_network_forward():
    """测试网络前向传播"""
    print("\n" + "=" * 50)
    print("测试网络前向传播")
    print("=" * 50)
    
    env, obs, policy_network, value_network = test_network_creation()
    
    # 转换观察为tensor
    obs_tensor = {}
    for key, value in obs.items():
        obs_tensor[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
    
    # 测试策略网络
    with torch.no_grad():
        action_logits, value = policy_network(obs_tensor)
        print(f"✅ 策略网络前向传播成功")
        print(f"  动作logits形状: {action_logits.shape}")
        print(f"  价值形状: {value.shape}")
        print(f"  动作logits: {action_logits[0]}")
        print(f"  价值: {value[0, 0]:.4f}")
    
    # 测试价值网络
    with torch.no_grad():
        q_values = value_network(obs_tensor)
        print(f"✅ 价值网络前向传播成功")
        print(f"  Q值形状: {q_values.shape}")
        print(f"  Q值: {q_values[0]}")
    
    return env, obs, policy_network, value_network

def test_agent_action():
    """测试智能体动作选择"""
    print("\n" + "=" * 50)
    print("测试智能体动作选择")
    print("=" * 50)
    
    env, obs, policy_network, value_network = test_network_forward()
    
    # 转换观察为tensor
    obs_tensor = {}
    for key, value in obs.items():
        obs_tensor[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
    
    # 测试策略网络动作选择
    with torch.no_grad():
        action, action_logits, value = policy_network.get_action(obs_tensor)
        print(f"✅ 策略网络动作选择成功")
        print(f"  选择动作: {action}")
        print(f"  动作logits: {action_logits[0]}")
        print(f"  价值: {value[0, 0]:.4f}")
    
    # 测试价值网络动作选择
    with torch.no_grad():
        action = value_network.get_action(obs_tensor, epsilon=0.0)
        print(f"✅ 价值网络动作选择成功")
        print(f"  选择动作: {action}")
    
    return env, obs, policy_network, value_network

def test_episode_simulation():
    """测试完整episode模拟"""
    print("\n" + "=" * 50)
    print("测试完整episode模拟")
    print("=" * 50)
    
    env = TerrainRoadEnvironment(grid_size=(15, 15), max_steps=50)
    policy_network = TerrainPolicyNetwork(
        grid_size=env.grid_size,
        action_space=env.action_space,
        hidden_dim=128
    )
    
    # 重置环境
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0
    
    print(f"初始智能体位置: {obs['agent_pos']}")
    print(f"目标位置: {obs['target_pos']}")
    print(f"初始资源: {obs['resources']}")
    
    # 运行episode
    while step_count < 50:
        # 转换观察为tensor
        obs_tensor = {}
        for key, value in obs.items():
            obs_tensor[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
        
        # 选择动作
        with torch.no_grad():
            action, _, _ = policy_network.get_action(obs_tensor)
        
        # 执行动作
        next_obs, reward, done, truncated, _ = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        print(f"步骤 {step_count}: 动作={action}, 奖励={reward:.2f}, "
              f"位置={next_obs['agent_pos']}, 资源={next_obs['resources']}")
        
        obs = next_obs
        
        if done or truncated:
            break
    
    print(f"\nEpisode结束:")
    print(f"  总步数: {step_count}")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  平均奖励: {total_reward/step_count:.2f}")
    print(f"  是否到达目标: {done}")
    
    return env

def test_mesh_loading():
    """测试mesh加载功能"""
    print("\n" + "=" * 50)
    print("测试mesh加载功能")
    print("=" * 50)
    
    # 创建测试mesh数据
    test_mesh = np.random.uniform(0, 100, (30, 30))
    
    # 保存为npy文件
    test_mesh_path = "test_mesh.npy"
    np.save(test_mesh_path, test_mesh)
    print(f"✅ 创建测试mesh文件: {test_mesh_path}")
    
    # 测试加载mesh
    try:
        env = TerrainRoadEnvironment(mesh_file=test_mesh_path, grid_size=(25, 25))
        print("✅ mesh加载成功")
        
        terrain_info = env.get_terrain_info()
        print(f"加载的mesh信息: {terrain_info}")
        
    except Exception as e:
        print(f"❌ mesh加载失败: {e}")
    
    # 清理测试文件
    if os.path.exists(test_mesh_path):
        os.remove(test_mesh_path)
        print(f"✅ 清理测试文件: {test_mesh_path}")

def test_performance():
    """测试性能"""
    print("\n" + "=" * 50)
    print("测试性能")
    print("=" * 50)
    
    env = TerrainRoadEnvironment(grid_size=(20, 20), max_steps=100)
    policy_network = TerrainPolicyNetwork(
        grid_size=env.grid_size,
        action_space=env.action_space,
        hidden_dim=128
    )
    
    # 测试环境步进性能
    obs, _ = env.reset()
    start_time = time.time()
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
    
    env_time = time.time() - start_time
    print(f"✅ 环境步进性能: {env_time:.4f}秒 (100步)")
    
    # 测试网络推理性能
    obs_tensor = {}
    for key, value in obs.items():
        obs_tensor[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            action, _, _ = policy_network.get_action(obs_tensor)
    
    network_time = time.time() - start_time
    print(f"✅ 网络推理性能: {network_time:.4f}秒 (100次推理)")

def main():
    """主测试函数"""
    print("开始地形环境接口测试")
    print("=" * 60)
    
    try:
        # 基础功能测试
        test_environment_creation()
        test_environment_reset()
        test_environment_step()
        
        # 网络功能测试
        test_network_creation()
        test_network_forward()
        test_agent_action()
        
        # 完整功能测试
        test_episode_simulation()
        test_mesh_loading()
        test_performance()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
