#!/usr/bin/env python3
"""
IDE端训练脚本 - 使用Flask服务器上传的地形进行强化学习训练
"""

import requests
import json
import numpy as np
import time
from envs.terrain_road_env import TerrainRoadEnvironment
from agents.terrain_policy import TerrainPolicyNetwork
from training.train_terrain_road import PPOAgent
import torch

def get_terrain_from_flask(flask_url="http://localhost:5000"):
    """从Flask服务器获取地形数据"""
    try:
        print("🌐 从Flask服务器获取地形数据...")
        
        response = requests.get(f"{flask_url}/get_terrain")
        
        if response.status_code == 200:
            result = response.json()
            terrain_data = result.get('terrain_data')
            
            if terrain_data:
                print("✅ 成功获取地形数据!")
                return terrain_data
            else:
                print("❌ 服务器上没有地形数据")
                return None
        else:
            print(f"❌ 获取地形数据失败: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Flask服务器")
        print("请确保服务器已启动: python main.py")
        return None
    except Exception as e:
        print(f"❌ 获取地形数据出错: {e}")
        return None

def create_terrain_file(terrain_data, filename="uploaded_terrain.json"):
    """将地形数据保存为文件"""
    try:
        with open(filename, 'w') as f:
            json.dump(terrain_data, f, indent=2)
        print(f"💾 地形数据已保存到: {filename}")
        return filename
    except Exception as e:
        print(f"❌ 保存地形文件失败: {e}")
        return None

def train_with_uploaded_terrain(terrain_data, training_config=None):
    """使用上传的地形数据进行训练"""
    print("🎯 开始使用上传的地形进行强化学习训练...")
    
    # 默认训练配置
    if training_config is None:
        training_config = {
            'total_timesteps': 10000,
            'learning_rate': 3e-4,
            'batch_size': 64,
            'n_steps': 2048,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5
        }
    
    try:
        # 创建环境（使用上传的地形）
        env = TerrainRoadEnvironment(
            mesh_file=None,  # 不使用文件，直接使用数据
            grid_size=tuple(terrain_data.get('grid_size', (50, 50))),
            max_steps=1000,
            render_mode=None  # 训练时不渲染
        )
        
        # 手动设置地形数据
        env.height_map = np.array(terrain_data['height_map'])
        env._generate_terrain_from_height()  # 根据高程生成地形类型
        env._set_agent_and_target()  # 重新设置智能体和目标位置
        
        print(f"📊 地形信息:")
        print(f"   网格大小: {env.grid_size}")
        print(f"   高程范围: {env.height_map.min():.2f} - {env.height_map.max():.2f}")
        print(f"   地形分布: {env.get_terrain_info()['terrain_distribution']}")
        
        # 创建策略网络
        policy_net = TerrainPolicyNetwork(
            grid_size=env.grid_size,
            action_space=env.action_space
        )
        
        print("🧠 策略网络已创建")
        print(f"   网络参数数量: {sum(p.numel() for p in policy_net.parameters())}")
        
        # 开始训练
        print("\n🚀 开始训练...")
        print("=" * 50)
        
        # 简单的训练循环示例
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=training_config['learning_rate'])
        
        episode_rewards = []
        for episode in range(100):  # 训练100个episode
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 获取动作
                obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in obs.items()}
                action_logits, _ = policy_net(obs_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
                
                # 执行动作
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                if truncated:
                    done = True
            
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}/100, 平均奖励: {avg_reward:.2f}")
        
        print("✅ 训练完成!")
        print(f"🎯 最终平均奖励: {np.mean(episode_rewards[-10:]):.2f}")
        
        return env, policy_net, episode_rewards
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        return None, None, None

def test_trained_agent(env, policy_net, num_episodes=5):
    """测试训练好的智能体"""
    print(f"\n🧪 测试训练好的智能体 ({num_episodes} 个episode)...")
    
    test_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < env.max_steps:
            # 获取动作
            obs_tensor = torch.FloatTensor(obs['height_map']).unsqueeze(0)
            action_logits, _ = policy_net({'height_map': obs_tensor})
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.argmax(action_probs, dim=1).item()
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if truncated:
                done = True
        
        test_rewards.append(episode_reward)
        if done and steps < env.max_steps:
            success_count += 1
        
        print(f"Episode {episode + 1}: 奖励={episode_reward:.2f}, 步数={steps}")
    
    success_rate = success_count / num_episodes
    avg_reward = np.mean(test_rewards)
    
    print(f"\n📊 测试结果:")
    print(f"   平均奖励: {avg_reward:.2f}")
    print(f"   成功率: {success_rate:.2%}")
    
    return test_rewards, success_rate

def main():
    """主函数"""
    print("🚀 IDE端地形强化学习训练")
    print("=" * 50)
    
    # 1. 从Flask服务器获取地形数据
    terrain_data = get_terrain_from_flask()
    if not terrain_data:
        print("❌ 无法获取地形数据，退出训练")
        return
    
    # 2. 保存地形数据到文件
    terrain_file = create_terrain_file(terrain_data)
    if not terrain_file:
        print("❌ 无法保存地形文件，退出训练")
        return
    
    # 3. 使用地形数据进行训练
    env, policy_net, training_rewards = train_with_uploaded_terrain(terrain_data)
    if env is None:
        print("❌ 训练失败")
        return
    
    # 4. 测试训练好的智能体
    test_rewards, success_rate = test_trained_agent(env, policy_net)
    
    # 5. 保存训练结果
    results = {
        'terrain_info': terrain_data,
        'training_rewards': training_rewards,
        'test_rewards': test_rewards,
        'success_rate': success_rate,
        'timestamp': time.time()
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 训练结果已保存到: training_results.json")
    print("✅ 训练流程完成!")

if __name__ == "__main__":
    main()
