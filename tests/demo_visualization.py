#!/usr/bin/env python3
"""
地形道路寻路可视化演示
展示实时寻路过程的可视化效果
"""

import numpy as np
import time
from envs.terrain_road_env import TerrainRoadEnvironment, RoadAction

def random_agent_policy(obs):
    """随机智能体策略 - 用于演示"""
    return np.random.randint(0, 7)

def simple_pathfinding_policy(obs):
    """简单寻路策略 - 朝目标方向移动"""
    agent_pos = obs['agent_pos']
    target_pos = obs['target_pos']
    
    # 计算方向
    direction = target_pos - agent_pos
    
    # 选择移动方向
    if abs(direction[0]) > abs(direction[1]):
        # 垂直移动
        if direction[0] > 0:
            return RoadAction.MOVE_SOUTH.value
        else:
            return RoadAction.MOVE_NORTH.value
    else:
        # 水平移动
        if direction[1] > 0:
            return RoadAction.MOVE_EAST.value
        else:
            return RoadAction.MOVE_WEST.value

def smart_pathfinding_policy(obs):
    """智能寻路策略 - 考虑地形和道路"""
    agent_pos = obs['agent_pos']
    target_pos = obs['target_pos']
    terrain_map = obs['terrain_map']
    road_map = obs['road_map']
    resources = obs['resources']
    
    # 如果资源充足且当前位置没有道路，考虑建设道路
    if resources[0] >= 10 and road_map[agent_pos[0], agent_pos[1]] == 0:
        # 检查周围是否有道路
        for dx, dy in [(-1,0), (1,0), (0,1), (0,-1)]:
            nx, ny = agent_pos[0] + dx, agent_pos[1] + dy
            if (0 <= nx < terrain_map.shape[0] and 
                0 <= ny < terrain_map.shape[1] and 
                road_map[nx, ny] > 0):
                return RoadAction.BUILD_ROAD.value
    
    # 优先在有道路的地方移动
    best_action = None
    best_score = float('-inf')
    
    for action in [RoadAction.MOVE_NORTH.value, RoadAction.MOVE_SOUTH.value, 
                   RoadAction.MOVE_EAST.value, RoadAction.MOVE_WEST.value]:
        
        if action == RoadAction.MOVE_NORTH.value:
            new_pos = agent_pos + np.array([-1, 0])
        elif action == RoadAction.MOVE_SOUTH.value:
            new_pos = agent_pos + np.array([1, 0])
        elif action == RoadAction.MOVE_EAST.value:
            new_pos = agent_pos + np.array([0, 1])
        else:  # MOVE_WEST
            new_pos = agent_pos + np.array([0, -1])
        
        # 检查边界
        if (0 <= new_pos[0] < terrain_map.shape[0] and 
            0 <= new_pos[1] < terrain_map.shape[1]):
            
            # 计算分数
            score = 0
            
            # 距离目标的奖励
            distance_to_target = np.linalg.norm(new_pos - target_pos)
            score -= distance_to_target * 2
            
            # 道路奖励
            if road_map[new_pos[0], new_pos[1]] > 0:
                score += 10
            
            # 地形惩罚
            terrain_type = terrain_map[new_pos[0], new_pos[1]]
            if terrain_type == 0:  # 水域
                score -= 100
            elif terrain_type == 2:  # 森林
                score -= 5
            elif terrain_type == 3:  # 山地
                score -= 10
            
            if score > best_score:
                best_score = score
                best_action = action
    
    if best_action is not None:
        return best_action
    
    # 如果找不到好的移动方向，随机移动
    return np.random.randint(0, 4)

def demo_visualization():
    """演示可视化效果"""
    print("🎮 地形道路寻路可视化演示")
    print("=" * 50)
    
    # 创建环境
    env = TerrainRoadEnvironment(
        grid_size=(30, 30),  # 较小的网格便于观察
        max_steps=200,
        render_mode='human'  # 启用可视化
    )
    
    print("✅ 环境创建成功")
    print(f"📊 网格大小: {env.grid_size}")
    print(f"🎯 目标位置: {env.target_pos}")
    print(f"🤖 智能体位置: {env.agent_pos}")
    print()
    
    # 选择策略
    strategies = {
        '1': ('随机策略', random_agent_policy),
        '2': ('简单寻路', simple_pathfinding_policy),
        '3': ('智能寻路', smart_pathfinding_policy)
    }
    
    print("请选择智能体策略:")
    for key, (name, _) in strategies.items():
        print(f"  {key}. {name}")
    
    choice = input("请输入选择 (1-3): ").strip()
    if choice not in strategies:
        choice = '1'
        print("使用默认策略: 随机策略")
    
    strategy_name, policy = strategies[choice]
    print(f"🎯 使用策略: {strategy_name}")
    print()
    
    # 运行演示
    print("🚀 开始演示...")
    print("💡 提示: 关闭图形窗口可以停止演示")
    
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0
    
    try:
        while True:
            # 选择动作
            action = policy(obs)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # 打印信息
            if step_count % 10 == 0:
                distance = np.linalg.norm(obs['agent_pos'] - obs['target_pos'])
                print(f"步骤 {step_count}: 奖励={reward:.2f}, 总奖励={total_reward:.2f}, "
                      f"距离目标={distance:.1f}, 资源={obs['resources']}")
            
            # 检查是否结束
            if done or truncated:
                break
            
            # 控制速度
            time.sleep(0.2)  # 每步暂停0.2秒
    
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
    
    # 显示结果
    print("\n" + "=" * 50)
    print("📊 演示结果:")
    print(f"  总步数: {step_count}")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  平均奖励: {total_reward/step_count:.2f}" if step_count > 0 else "  平均奖励: 0.00")
    print(f"  最终位置: {obs['agent_pos']}")
    print(f"  目标位置: {obs['target_pos']}")
    print(f"  是否到达目标: {np.array_equal(obs['agent_pos'], obs['target_pos'])}")
    
    # 关闭环境
    env.close()
    print("✅ 演示完成!")

def demo_multiple_episodes():
    """演示多个episode的效果"""
    print("🎮 多轮演示 - 观察不同策略的效果")
    print("=" * 50)
    
    strategies = [
        ('随机策略', random_agent_policy),
        ('简单寻路', simple_pathfinding_policy),
        ('智能寻路', smart_pathfinding_policy)
    ]
    
    results = {}
    
    for strategy_name, policy in strategies:
        print(f"\n🎯 测试策略: {strategy_name}")
        
        env = TerrainRoadEnvironment(
            grid_size=(20, 20),
            max_steps=100,
            render_mode='human'
        )
        
        episode_rewards = []
        success_count = 0
        
        for episode in range(3):  # 每个策略测试3个episode
            obs, _ = env.reset()
            total_reward = 0
            step_count = 0
            
            print(f"  Episode {episode + 1}: ", end="")
            
            while step_count < env.max_steps:
                action = policy(obs)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                
                if done or truncated:
                    break
                
                time.sleep(0.1)
            
            episode_rewards.append(total_reward)
            if np.array_equal(obs['agent_pos'], obs['target_pos']):
                success_count += 1
            
            print(f"奖励={total_reward:.1f}, 步数={step_count}")
        
        env.close()
        
        results[strategy_name] = {
            'avg_reward': np.mean(episode_rewards),
            'success_rate': success_count / 3,
            'rewards': episode_rewards
        }
    
    # 显示比较结果
    print("\n" + "=" * 50)
    print("📊 策略比较结果:")
    for strategy_name, result in results.items():
        print(f"\n{strategy_name}:")
        print(f"  平均奖励: {result['avg_reward']:.2f}")
        print(f"  成功率: {result['success_rate']:.1%}")
        print(f"  各轮奖励: {[f'{r:.1f}' for r in result['rewards']]}")

if __name__ == "__main__":
    print("🎮 地形道路寻路可视化演示")
    print("=" * 50)
    print("选择演示模式:")
    print("  1. 单轮详细演示")
    print("  2. 多轮策略比较")
    
    mode = input("请选择模式 (1-2): ").strip()
    
    if mode == '2':
        demo_multiple_episodes()
    else:
        demo_visualization()
