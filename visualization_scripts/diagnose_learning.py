#!/usr/bin/env python3
"""
学习诊断脚本 - 分析奖励函数、优势学习和探索问题
"""

import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from envs.terrain_grid_nav_env import TerrainGridNavEnv
from agents.ppo_terrain_agent import TerrainPPOAgent

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_reward_function():
    """分析奖励函数设计"""
    print("=== 奖励函数分析 ===")
    
    # 加载地形数据
    terrain_file = "data/terrain/terrain_1755281528.json"
    with open(terrain_file, 'r') as f:
        terrain_data = json.load(f)
    
    height_map = np.array(terrain_data['height_map'], dtype=np.float32)
    start_point = (20, 20)
    goal_point = (110, 110)
    
    # 创建环境
    env = TerrainGridNavEnv(
        H=height_map.shape[0], W=height_map.shape[1],
        max_steps=300,
        height_range=(terrain_data['original_bounds']['z_min'], 
                     terrain_data['original_bounds']['z_max']),
        slope_penalty_weight=0.01,
        height_penalty_weight=0.005,
        custom_terrain=height_map,
        fixed_start=start_point,
        fixed_goal=goal_point
    )
    
    # 测试不同路径的奖励
    print("测试不同路径的奖励分布...")
    
    # 1. 直线路径（理想情况）
    straight_path = []
    current_pos = list(start_point)
    while current_pos != list(goal_point):
        straight_path.append(current_pos.copy())
        if current_pos[0] < goal_point[0]:
            current_pos[0] += 1
        elif current_pos[1] < goal_point[1]:
            current_pos[1] += 1
    
    # 2. 随机路径
    np.random.seed(42)
    random_path = [list(start_point)]
    current_pos = list(start_point)
    steps = 0
    while current_pos != list(goal_point) and steps < 200:
        # 随机选择动作
        action = np.random.randint(0, 4)
        next_pos = current_pos.copy()
        
        if action == 0:  # 上
            next_pos[0] = max(0, next_pos[0] - 1)
        elif action == 1:  # 右
            next_pos[1] = min(height_map.shape[1] - 1, next_pos[1] + 1)
        elif action == 2:  # 下
            next_pos[0] = min(height_map.shape[0] - 1, next_pos[0] + 1)
        elif action == 3:  # 左
            next_pos[1] = max(0, next_pos[1] - 1)
        
        if next_pos != current_pos:
            current_pos = next_pos
            random_path.append(current_pos.copy())
        steps += 1
    
    # 3. 智能体路径（模拟）
    agent_path = [list(start_point)]
    current_pos = list(start_point)
    steps = 0
    while current_pos != list(goal_point) and steps < 300:
        # 模拟智能体的动作选择（偏向目标方向）
        dx = goal_point[0] - current_pos[0]
        dy = goal_point[1] - current_pos[1]
        
        if abs(dx) > abs(dy):
            if dx > 0:
                action = 2  # 下
            else:
                action = 0  # 上
        else:
            if dy > 0:
                action = 1  # 右
            else:
                action = 3  # 左
        
        next_pos = current_pos.copy()
        if action == 0:  # 上
            next_pos[0] = max(0, next_pos[0] - 1)
        elif action == 1:  # 右
            next_pos[1] = min(height_map.shape[1] - 1, next_pos[1] + 1)
        elif action == 2:  # 下
            next_pos[0] = min(height_map.shape[0] - 1, next_pos[0] + 1)
        elif action == 3:  # 左
            next_pos[1] = max(0, next_pos[1] - 1)
        
        if next_pos != current_pos:
            current_pos = next_pos
            agent_path.append(current_pos.copy())
        steps += 1
    
    # 计算各路径的奖励
    def calculate_path_reward(path):
        total_reward = 0
        rewards = []
        
        for i, pos in enumerate(path):
            # 计算距离奖励
            distance = abs(goal_point[0] - pos[0]) + abs(goal_point[1] - pos[1])
            distance_reward = -distance * 0.1
            
            # 计算地形惩罚
            height = height_map[pos[0], pos[1]]
            height_penalty = -abs(height - height_map[start_point[0], start_point[1]]) * 0.005
            
            # 计算坡度惩罚
            from envs.terrain_grid_nav_env import calculate_slope
            slope = calculate_slope(height_map, pos)
            slope_penalty = -slope * 0.01
            
            step_reward = distance_reward + height_penalty + slope_penalty
            rewards.append(step_reward)
            total_reward += step_reward
        
        # 如果到达目标，给予额外奖励
        if path[-1] == list(goal_point):
            total_reward += 100
        
        return total_reward, rewards
    
    straight_reward, straight_rewards = calculate_path_reward(straight_path)
    random_reward, random_rewards = calculate_path_reward(random_path)
    agent_reward, agent_rewards = calculate_path_reward(agent_path)
    
    print(f"直线路径奖励: {straight_reward:.2f}, 长度: {len(straight_path)}")
    print(f"随机路径奖励: {random_reward:.2f}, 长度: {len(random_path)}")
    print(f"智能体路径奖励: {agent_reward:.2f}, 长度: {len(agent_path)}")
    
    # 分析奖励分布
    print(f"\n奖励分布分析:")
    print(f"直线路径平均步奖励: {np.mean(straight_rewards):.3f}")
    print(f"随机路径平均步奖励: {np.mean(random_rewards):.3f}")
    print(f"智能体路径平均步奖励: {np.mean(agent_rewards):.3f}")
    
    return {
        'straight_path': straight_path,
        'random_path': random_path,
        'agent_path': agent_path,
        'straight_reward': straight_reward,
        'random_reward': random_reward,
        'agent_reward': agent_reward,
        'straight_rewards': straight_rewards,
        'random_rewards': random_rewards,
        'agent_rewards': agent_rewards
    }

def analyze_advantage_learning():
    """分析优势学习"""
    print("\n=== 优势学习分析 ===")
    
    # 创建智能体
    agent = TerrainPPOAgent(
        state_dim=13,
        action_dim=4,
        hidden_dim=256,
        lr=3e-4
    )
    
    # 加载地形数据
    terrain_file = "data/terrain/terrain_1755281528.json"
    with open(terrain_file, 'r') as f:
        terrain_data = json.load(f)
    
    height_map = np.array(terrain_data['height_map'], dtype=np.float32)
    start_point = (20, 20)
    goal_point = (110, 110)
    
    # 创建环境
    env = TerrainGridNavEnv(
        H=height_map.shape[0], W=height_map.shape[1],
        max_steps=300,
        height_range=(terrain_data['original_bounds']['z_min'], 
                     terrain_data['original_bounds']['z_max']),
        slope_penalty_weight=0.01,
        height_penalty_weight=0.005,
        custom_terrain=height_map,
        fixed_start=start_point,
        fixed_goal=goal_point
    )
    
    # 收集一个episode的数据
    states, actions, rewards, values, log_probs, dones, path, success = \
        agent.collect_episode(env)
    
    # 计算优势
    advantages = []
    returns = []
    
    # 计算回报
    R = 0
    for r in reversed(rewards):
        R = r + 0.99 * R
        returns.insert(0, R)
    
    # 计算优势
    for i in range(len(rewards)):
        if i == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[i + 1]
        
        advantage = rewards[i] + 0.99 * next_value - values[i]
        advantages.append(advantage)
    
    print(f"Episode长度: {len(rewards)}")
    print(f"成功: {success}")
    print(f"总奖励: {sum(rewards):.2f}")
    print(f"平均奖励: {np.mean([float(r) for r in rewards]):.3f}")
    print(f"平均价值: {np.mean([float(v) for v in values]):.3f}")
    print(f"平均优势: {np.mean(advantages):.3f}")
    print(f"优势标准差: {np.std(advantages):.3f}")
    print(f"优势范围: [{np.min(advantages):.3f}, {np.max(advantages):.3f}]")
    
    # 分析优势分布
    positive_advantages = [a for a in advantages if a > 0]
    negative_advantages = [a for a in advantages if a < 0]
    
    print(f"正优势比例: {len(positive_advantages)/len(advantages):.1%}")
    print(f"负优势比例: {len(negative_advantages)/len(advantages):.1%}")
    
    return {
        'rewards': rewards,
        'values': values,
        'advantages': advantages,
        'returns': returns,
        'success': success
    }

def analyze_exploration():
    """分析探索问题"""
    print("\n=== 探索分析 ===")
    
    # 加载地形数据
    terrain_file = "data/terrain/terrain_1755281528.json"
    with open(terrain_file, 'r') as f:
        terrain_data = json.load(f)
    
    height_map = np.array(terrain_data['height_map'], dtype=np.float32)
    start_point = (20, 20)
    goal_point = (110, 110)
    
    # 创建环境
    env = TerrainGridNavEnv(
        H=height_map.shape[0], W=height_map.shape[1],
        max_steps=300,
        height_range=(terrain_data['original_bounds']['z_min'], 
                     terrain_data['original_bounds']['z_max']),
        slope_penalty_weight=0.01,
        height_penalty_weight=0.005,
        custom_terrain=height_map,
        fixed_start=start_point,
        fixed_goal=goal_point
    )
    
    # 创建智能体
    agent = TerrainPPOAgent(
        state_dim=13,
        action_dim=4,
        hidden_dim=256,
        lr=3e-4
    )
    
    # 运行多个episodes，分析动作分布
    action_counts = [0, 0, 0, 0]  # 上、右、下、左
    state_visits = {}
    total_episodes = 10
    
    for episode in range(total_episodes):
        states, actions, rewards, values, log_probs, dones, path, success = \
            agent.collect_episode(env)
        
        # 统计动作
        for action in actions:
            action_counts[action] += 1
        
        # 统计状态访问
        for state in states:
            state_key = tuple(state[:2])  # 只考虑位置信息
            state_visits[state_key] = state_visits.get(state_key, 0) + 1
    
    print(f"动作分布 (上、右、下、左): {action_counts}")
    print(f"动作分布比例: {[count/sum(action_counts) for count in action_counts]}")
    
    print(f"访问的不同状态数: {len(state_visits)}")
    print(f"平均每个状态访问次数: {np.mean(list(state_visits.values())):.1f}")
    
    # 分析状态访问的多样性
    visit_counts = list(state_visits.values())
    print(f"状态访问次数范围: [{min(visit_counts)}, {max(visit_counts)}]")
    print(f"状态访问标准差: {np.std(visit_counts):.1f}")
    
    return {
        'action_counts': action_counts,
        'state_visits': state_visits,
        'visit_counts': visit_counts
    }

def plot_analysis_results(reward_analysis, advantage_analysis, exploration_analysis):
    """绘制分析结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('学习问题诊断分析', fontsize=16)
    
    # 1. 奖励分布
    axes[0, 0].hist(reward_analysis['straight_rewards'], bins=20, alpha=0.7, label='直线路径', color='blue')
    axes[0, 0].hist(reward_analysis['random_rewards'], bins=20, alpha=0.7, label='随机路径', color='red')
    axes[0, 0].hist(reward_analysis['agent_rewards'], bins=20, alpha=0.7, label='智能体路径', color='green')
    axes[0, 0].set_title('步奖励分布')
    axes[0, 0].set_xlabel('步奖励')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 优势分布
    axes[0, 1].hist(advantage_analysis['advantages'], bins=30, alpha=0.7, color='orange')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('优势分布')
    axes[0, 1].set_xlabel('优势值')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 动作分布
    action_labels = ['上', '右', '下', '左']
    axes[0, 2].bar(action_labels, exploration_analysis['action_counts'], color='skyblue')
    axes[0, 2].set_title('动作分布')
    axes[0, 2].set_ylabel('选择次数')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 奖励曲线
    axes[1, 0].plot(reward_analysis['straight_rewards'], label='直线路径', color='blue')
    axes[1, 0].plot(reward_analysis['random_rewards'], label='随机路径', color='red')
    axes[1, 0].plot(reward_analysis['agent_rewards'], label='智能体路径', color='green')
    axes[1, 0].set_title('步奖励变化')
    axes[1, 0].set_xlabel('步数')
    axes[1, 0].set_ylabel('步奖励')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 价值vs回报
    values_float = [float(v) for v in advantage_analysis['values']]
    axes[1, 1].scatter(values_float, advantage_analysis['returns'], alpha=0.6)
    axes[1, 1].plot([min(values_float), max(values_float)], 
                    [min(values_float), max(values_float)], 
                    'r--', linewidth=2)
    axes[1, 1].set_title('价值 vs 回报')
    axes[1, 1].set_xlabel('预测价值')
    axes[1, 1].set_ylabel('实际回报')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 状态访问热图
    visit_counts = exploration_analysis['visit_counts']
    axes[1, 2].hist(visit_counts, bins=20, alpha=0.7, color='purple')
    axes[1, 2].set_title('状态访问次数分布')
    axes[1, 2].set_xlabel('访问次数')
    axes[1, 2].set_ylabel('状态数量')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('learning_diagnosis.png', dpi=300, bbox_inches='tight')
    print("诊断图表已保存到: learning_diagnosis.png")
    
    plt.show()

def main():
    print("学习问题诊断分析")
    print("=" * 50)
    
    # 1. 分析奖励函数
    reward_analysis = analyze_reward_function()
    
    # 2. 分析优势学习
    advantage_analysis = analyze_advantage_learning()
    
    # 3. 分析探索问题
    exploration_analysis = analyze_exploration()
    
    # 4. 绘制分析结果
    plot_analysis_results(reward_analysis, advantage_analysis, exploration_analysis)
    
    # 5. 总结和建议
    print("\n=== 诊断总结和建议 ===")
    
    # 奖励函数问题
    if np.mean(reward_analysis['straight_rewards']) < -1:
        print("1. 奖励函数问题: 步奖励过于负向")
        print("   建议: 增加基础奖励，减少地形惩罚")
    
    # 优势学习问题
    if np.mean(advantage_analysis['advantages']) < -0.5:
        print("2. 优势学习问题: 优势值过于负向")
        print("   建议: 调整基线，改进价值函数学习")
    
    # 探索问题
    action_probs = [count/sum(exploration_analysis['action_counts']) for count in exploration_analysis['action_counts']]
    if max(action_probs) > 0.4:
        print("3. 探索问题: 动作选择过于集中")
        print("   建议: 增加熵正则化，鼓励探索")
    
    print("\n4. 其他建议:")
    print("   - 使用课程学习，从简单任务开始")
    print("   - 改进状态表示，增加更多有用信息")
    print("   - 调整网络架构，增加表达能力")

if __name__ == "__main__":
    main()
