#!/usr/bin/env python3
"""
地形道路寻路回放演示
展示训练后的episode回放功能
"""

import numpy as np
import time
import os
from envs.terrain_road_env import TerrainRoadEnvironment, RoadAction

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

def record_episode():
    """记录一个episode"""
    print("🎬 开始记录episode...")
    
    # 创建环境（关闭实时渲染以提高性能）
    env = TerrainRoadEnvironment(
        grid_size=(25, 25),
        max_steps=150,
        render_mode=None  # 关闭实时渲染
    )
    
    # 开始记录
    env.start_recording()
    
    # 运行episode
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0
    
    while step_count < env.max_steps:
        action = smart_pathfinding_policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if done or truncated:
            break
    
    # 停止记录
    env.stop_recording()
    
    print(f"📊 Episode完成:")
    print(f"  总步数: {step_count}")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  是否到达目标: {np.array_equal(obs['agent_pos'], obs['target_pos'])}")
    
    return env

def demo_replay():
    """演示回放功能"""
    print("🎮 地形道路寻路回放演示")
    print("=" * 50)
    
    # 直接加载现有的episode文件
    episode_file = "episode_1755203784.json"
    
    # 检查文件是否存在
    if os.path.exists(episode_file):
        print(f"📂 加载episode文件: {episode_file}")
        env = TerrainRoadEnvironment()
        env.load_episode(episode_file)
    else:
        print(f"❌ 找不到文件: {episode_file}")
        print("🔄 尝试录制新的episode...")
        env = record_episode()
    
    # 询问是否保存
    save_choice = input("是否保存这个episode? (y/n): ").strip().lower()
    if save_choice == 'y':
        env.save_episode()
    
    # 开始回放
    print("\n🎬 开始回放...")
    print("💡 控制说明:")
    print("  - 使用滑块调整播放速度")
    print("  - 点击按钮控制播放/暂停/重置")
    print("  - 点击Save保存当前帧")
    print("  - 按Ctrl+C退出回放")
    
    env.replay_episode(speed=1.0)
    
    env.close()

def demo_multiple_episodes():
    """演示多个episode的对比"""
    print("🎮 多episode对比演示")
    print("=" * 50)
    
    episodes = []
    
    # 录制多个episode
    for i in range(3):
        print(f"\n📹 录制第 {i+1} 个episode...")
        env = record_episode()
        env.save_episode(f"episode_{i+1}.json")
        episodes.append(env)
    
    # 显示对比信息
    print("\n📊 Episode对比:")
    for i, env in enumerate(episodes):
        metadata = env.episode_history[-1] if env.episode_history else {}
        print(f"Episode {i+1}:")
        print(f"  步数: {len(env.episode_history)}")
        print(f"  最终奖励: {metadata.get('reward', 0):.2f}")
        print(f"  是否成功: {metadata.get('done', False)}")
    
    # 选择要回放的episode
    choice = input("\n请选择要回放的episode (1-3): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= 3:
        env = episodes[int(choice) - 1]
        env.replay_episode(speed=1.0)
    
    # 清理
    for env in episodes:
        env.close()

def demo_analysis():
    """演示episode分析功能"""
    print("📊 Episode分析演示")
    print("=" * 50)
    
    # 录制一个episode
    env = record_episode()
    
    if not env.episode_history:
        print("❌ 没有episode数据可分析")
        return
    
    # 分析数据
    steps = [frame['step'] for frame in env.episode_history]
    rewards = [frame['reward'] for frame in env.episode_history]
    distances = []
    road_builds = 0
    
    for frame in env.episode_history:
        # 计算到目标的距离
        agent_pos = frame['agent_pos']
        distance = np.linalg.norm(agent_pos - env.target_pos)
        distances.append(distance)
        
        # 统计道路建设
        if frame['action'] == RoadAction.BUILD_ROAD.value:
            road_builds += 1
    
    # 显示分析结果
    print("\n📈 Episode分析结果:")
    print(f"  总步数: {len(steps)}")
    print(f"  总奖励: {sum(rewards):.2f}")
    print(f"  平均奖励: {np.mean(rewards):.2f}")
    print(f"  最大奖励: {max(rewards):.2f}")
    print(f"  最小奖励: {min(rewards):.2f}")
    print(f"  道路建设次数: {road_builds}")
    print(f"  最终距离目标: {distances[-1]:.2f}")
    print(f"  是否到达目标: {env.episode_history[-1]['done']}")
    
    # 绘制分析图表
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Episode Analysis', fontsize=16)
    
    # 奖励曲线
    ax1.plot(steps, rewards, 'b-', linewidth=2)
    ax1.set_title('Reward over Time')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # 距离曲线
    ax2.plot(steps, distances, 'r-', linewidth=2)
    ax2.set_title('Distance to Target')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Distance')
    ax2.grid(True, alpha=0.3)
    
    # 累积奖励
    cumulative_rewards = np.cumsum(rewards)
    ax3.plot(steps, cumulative_rewards, 'g-', linewidth=2)
    ax3.set_title('Cumulative Reward')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Cumulative Reward')
    ax3.grid(True, alpha=0.3)
    
    # 动作分布
    actions = [frame['action'] for frame in env.episode_history]
    action_names = ['North', 'South', 'East', 'West', 'Build', 'Upgrade', 'Wait']
    action_counts = [actions.count(i) for i in range(7)]
    
    ax4.bar(action_names, action_counts, color='orange', alpha=0.7)
    ax4.set_title('Action Distribution')
    ax4.set_xlabel('Action')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    env.close()

if __name__ == "__main__":
    print("🎮 地形道路寻路回放演示")
    print("=" * 50)
    print("选择演示模式:")
    print("  1. 单episode回放")
    print("  2. 多episode对比")
    print("  3. Episode分析")
    
    mode = input("请选择模式 (1-3): ").strip()
    
    if mode == '2':
        demo_multiple_episodes()
    elif mode == '3':
        demo_analysis()
    else:
        demo_replay()
