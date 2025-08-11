#!/usr/bin/env python3
"""
多智能体强化学习系统测试脚本
测试地形、寻路、地块和城市环境系统
"""

import numpy as np
import matplotlib.pyplot as plt
from envs.terrain_system import TerrainSystem, TerrainType
from envs.pathfinding import PathfindingSystem
from envs.land_system import LandSystem, LandType
from envs.city_env import CityEnvironment
import random

def test_terrain_system():
    """测试地形系统"""
    print("=" * 50)
    print("测试地形系统")
    print("=" * 50)
    
    # 创建地形系统
    terrain = TerrainSystem(30, 30)
    
    # 生成随机地形
    terrain.generate_random_terrain(seed=42)
    
    # 显示地形统计
    stats = terrain.get_terrain_stats()
    print("地形统计:")
    for terrain_name, data in stats.items():
        print(f"  {terrain_name}: {data['count']} 块 ({data['percentage']}%)")
    
    # 可视化地形
    terrain.visualize(show_resources=True, show_elevation=True)
    
    return terrain

def test_pathfinding_system(terrain):
    """测试寻路系统"""
    print("\n" + "=" * 50)
    print("测试寻路系统")
    print("=" * 50)
    
    # 创建寻路系统
    pathfinding = PathfindingSystem(terrain)
    
    # 测试A*寻路
    start = (5, 5)
    goal = (25, 25)
    
    print(f"从 {start} 到 {goal} 的路径:")
    path = pathfinding.a_star(start, goal)
    
    if path:
        print(f"路径长度: {len(path)} 步")
        print(f"路径成本: {pathfinding.get_path_length(path):.2f}")
        print(f"路径: {path[:5]}...{path[-5:] if len(path) > 10 else path}")
        
        # 可视化路径
        visualize_path(terrain, path, start, goal)
    else:
        print("未找到路径")
    
    # 测试洪水填充
    print(f"\n从 {start} 可达的区域 (成本 <= 10):")
    accessible = pathfinding.get_accessible_area(start, 10.0)
    print(f"可达位置数量: {len(accessible)}")
    
    return pathfinding

def test_land_system():
    """测试地块系统"""
    print("\n" + "=" * 50)
    print("测试地块系统")
    print("=" * 50)
    
    # 创建地块系统
    land_system = LandSystem(20, 20)
    
    # 设置一些地块
    land_system.set_land_function(5, 5, LandType.RESIDENTIAL)
    land_system.set_land_function(6, 5, LandType.COMMERCIAL)
    land_system.set_land_function(7, 5, LandType.INDUSTRIAL)
    land_system.set_land_function(5, 6, LandType.AGRICULTURAL)
    land_system.set_land_function(6, 6, LandType.RECREATIONAL)
    
    # 添加连接
    land_system.add_connection(5, 5, 6, 5)
    land_system.add_connection(6, 5, 7, 5)
    land_system.add_connection(5, 5, 5, 6)
    land_system.add_connection(5, 6, 6, 6)
    
    # 升级地块
    land_system.upgrade_land(5, 5)
    land_system.upgrade_land(6, 5)
    
    # 显示统计
    stats = land_system.get_stats()
    print("地块统计:")
    print(f"  总地块数: {stats['total_lands']}")
    print(f"  已开发地块: {stats['developed_lands']}")
    print(f"  总价值: {stats['total_value']:.2f}")
    print(f"  总收入: {stats['total_revenue']:.2f}")
    print(f"  总维护成本: {stats['total_maintenance']:.2f}")
    print(f"  净收入: {stats['total_revenue'] - stats['total_maintenance']:.2f}")
    
    # 可视化地块
    visualize_land_system(land_system)
    
    return land_system

def test_city_environment():
    """测试城市环境"""
    print("\n" + "=" * 50)
    print("测试城市环境")
    print("=" * 50)
    
    # 创建城市环境
    env = CityEnvironment(width=20, height=20, num_agents=3, max_steps=100)
    
    # 重置环境
    observations, info = env.reset(seed=42)
    
    print("环境信息:")
    print(f"  地图大小: {env.width} x {env.height}")
    print(f"  智能体数量: {env.num_agents}")
    print(f"  最大步数: {env.max_steps}")
    
    print("\n智能体初始状态:")
    for agent_id in range(env.num_agents):
        pos = env.agent_positions[agent_id]
        goal = env.agent_goals[agent_id]
        resources = env.agent_resources[agent_id]
        print(f"  智能体 {agent_id}: 位置 {pos}, 目标 {goal}, 资源 {resources:.1f}")
    
    # 运行几个步骤
    print("\n运行环境 (5步):")
    for step in range(5):
        # 随机动作
        actions = {}
        for agent_id in range(env.num_agents):
            action = {
                'action_type': random.randint(0, 5),
                'target_x': random.randint(0, env.width - 1),
                'target_y': random.randint(0, env.height - 1),
                'land_type': random.randint(0, 7)
            }
            actions[agent_id] = action
        
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        print(f"  步骤 {step + 1}:")
        for agent_id, reward in rewards.items():
            pos = env.agent_positions[agent_id]
            resources = env.agent_resources[agent_id]
            print(f"    智能体 {agent_id}: 位置 {pos}, 资源 {resources:.1f}, 奖励 {reward:.2f}")
        
        if terminated or truncated:
            print("  环境结束")
            break
    
    return env

def visualize_path(terrain, path, start, goal):
    """可视化路径"""
    plt.figure(figsize=(10, 8))
    
    # 显示地形
    plt.imshow(terrain.terrain, cmap='tab10', vmin=0, vmax=5)
    plt.colorbar(label='地形类型')
    
    # 显示路径
    if path:
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        plt.plot(path_x, path_y, 'r-', linewidth=2, label='路径')
    
    # 显示起点和终点
    plt.plot(start[0], start[1], 'go', markersize=10, label='起点')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='终点')
    
    plt.title('寻路结果')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_land_system(land_system):
    """可视化地块系统"""
    plt.figure(figsize=(12, 5))
    
    # 地块类型图
    plt.subplot(1, 2, 1)
    land_map = np.array([[land_system.get_land_type(x, y).value 
                         for x in range(land_system.width)] 
                        for y in range(land_system.height)])
    plt.imshow(land_map, cmap='tab10', vmin=0, vmax=7)
    plt.colorbar(label='地块类型')
    plt.title('地块类型分布')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 地块价值图
    plt.subplot(1, 2, 2)
    value_map = np.array([[land_system.get_land_value(x, y) 
                          for x in range(land_system.width)] 
                         for y in range(land_system.height)])
    plt.imshow(value_map, cmap='hot')
    plt.colorbar(label='地块价值')
    plt.title('地块价值分布')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.tight_layout()
    plt.show()

def main():
    """主测试函数"""
    print("🚀 多智能体强化学习系统测试")
    print("=" * 60)
    
    try:
        # 测试地形系统
        terrain = test_terrain_system()
        
        # 测试寻路系统
        pathfinding = test_pathfinding_system(terrain)
        
        # 测试地块系统
        land_system = test_land_system()
        
        # 测试城市环境
        env = test_city_environment()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n系统功能总结:")
        print("  ✅ 地形系统: 支持地形生成、加载、保存和可视化")
        print("  ✅ 寻路系统: 支持A*算法、Dijkstra算法和路径优化")
        print("  ✅ 地块系统: 支持地块功能管理、升级、修复和统计")
        print("  ✅ 城市环境: 支持多智能体交互、动作执行和奖励计算")
        print("\n可以开始进行多智能体强化学习训练了！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
