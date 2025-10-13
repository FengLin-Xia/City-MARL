#!/usr/bin/env python3
"""
测试强化学习环境可视化
验证坐标系和地形数据是否正确加载
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.terrain_road_env import TerrainRoadEnvironment, TerrainType

def visualize_environment(env, title="Terrain Road Environment"):
    """可视化环境状态"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # 1. 高程图
    im1 = axes[0, 0].imshow(env.height_map, cmap='terrain', aspect='auto')
    axes[0, 0].set_title('高程图 (Height Map)')
    axes[0, 0].set_xlabel('X坐标')
    axes[0, 0].set_ylabel('Y坐标')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 地形类型图
    terrain_colors = {
        TerrainType.WATER.value: 'blue',
        TerrainType.GRASS.value: 'lightgreen',
        TerrainType.FOREST.value: 'darkgreen',
        TerrainType.MOUNTAIN.value: 'gray',
        TerrainType.ROAD.value: 'yellow',
        TerrainType.BUILDING.value: 'red'
    }
    
    terrain_cmap = plt.cm.colors.ListedColormap(list(terrain_colors.values()))
    im2 = axes[0, 1].imshow(env.terrain_map, cmap=terrain_cmap, aspect='auto', vmin=0, vmax=5)
    axes[0, 1].set_title('地形类型图 (Terrain Map)')
    axes[0, 1].set_xlabel('X坐标')
    axes[0, 1].set_ylabel('Y坐标')
    
    # 添加地形类型标签
    terrain_labels = ['水域', '草地', '森林', '山地', '道路', '建筑']
    for i, label in enumerate(terrain_labels):
        axes[0, 1].text(0.02, 0.98 - i*0.15, f'{i}: {label}', 
                       transform=axes[0, 1].transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3. 道路网络图
    road_cmap = plt.cm.colors.ListedColormap(['white', 'yellow', 'orange', 'red'])
    im3 = axes[1, 0].imshow(env.road_map, cmap=road_cmap, aspect='auto', vmin=0, vmax=3)
    axes[1, 0].set_title('道路网络图 (Road Map)')
    axes[1, 0].set_xlabel('X坐标')
    axes[1, 0].set_ylabel('Y坐标')
    
    # 4. 综合视图（智能体位置、目标、路径）
    im4 = axes[1, 1].imshow(env.terrain_map, cmap=terrain_cmap, aspect='auto', vmin=0, vmax=5, alpha=0.7)
    axes[1, 1].set_title('综合视图 (Agent & Target)')
    axes[1, 1].set_xlabel('X坐标')
    axes[1, 1].set_ylabel('Y坐标')
    
    # 绘制智能体位置
    axes[1, 1].scatter(env.agent_pos[1], env.agent_pos[0], c='red', s=100, marker='o', label='智能体')
    
    # 绘制目标位置
    axes[1, 1].scatter(env.target_pos[1], env.target_pos[0], c='green', s=100, marker='*', label='目标')
    
    # 绘制路径
    if hasattr(env, 'agent_path') and env.agent_path:
        path_x = [pos[1] for pos in env.agent_path]
        path_y = [pos[0] for pos in env.agent_path]
        axes[1, 1].plot(path_x, path_y, 'r-', linewidth=2, alpha=0.8, label='路径')
    
    # 绘制道路
    road_positions = np.where(env.road_map > 0)
    if len(road_positions[0]) > 0:
        axes[1, 1].scatter(road_positions[1], road_positions[0], c='yellow', s=20, alpha=0.8, label='道路')
    
    axes[1, 1].legend()
    
    plt.tight_layout()
    return fig

def test_environment_with_terrain():
    """使用实际地形数据测试环境"""
    print("🧪 测试强化学习环境...")
    
    # 查找最新的地形数据文件
    terrain_dir = Path("data/terrain")
    terrain_files = list(terrain_dir.glob("terrain_continuity_boundary_*.json"))
    
    if not terrain_files:
        print("❌ 未找到地形数据文件")
        return
    
    # 使用最新的文件
    latest_file = max(terrain_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 使用地形文件: {latest_file}")
    
    # 创建环境
    env = TerrainRoadEnvironment(mesh_file=str(latest_file))
    
    print(f"📊 环境信息:")
    print(f"   - 网格尺寸: {env.grid_size}")
    print(f"   - 高程范围: {env.height_map.min():.2f} ~ {env.height_map.max():.2f}")
    print(f"   - 智能体位置: {env.agent_pos}")
    print(f"   - 目标位置: {env.target_pos}")
    print(f"   - 资源状态: {env.resources}")
    
    # 统计地形类型
    terrain_counts = np.bincount(env.terrain_map.flatten())
    terrain_names = ['水域', '草地', '森林', '山地', '道路', '建筑']
    print(f"   - 地形分布:")
    for i, count in enumerate(terrain_counts):
        if i < len(terrain_names):
            percentage = count / env.terrain_map.size * 100
            print(f"     {terrain_names[i]}: {count} ({percentage:.1f}%)")
    
    # 可视化环境
    fig = visualize_environment(env, f"Terrain Road Environment - {env.grid_size[0]}x{env.grid_size[1]}")
    
    # 测试几步动作
    print("\n🎮 测试动作执行...")
    obs, _ = env.reset()
    print(f"   - 初始观察空间: {list(obs.keys())}")
    
    # 执行几个随机动作
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"   - 步骤 {step+1}: 动作={action}, 奖励={reward:.2f}, 完成={done}")
        if done:
            break
    
    plt.show()
    return env

def test_observation_space(env=None):
    """测试观察空间"""
    print("\n🔍 测试观察空间...")
    
    if env is None:
        env = TerrainRoadEnvironment()
    
    print(f"观察空间:")
    for key, space in env.observation_space.spaces.items():
        print(f"  {key}: {space}")
    
    obs, _ = env.reset()
    print(f"\n实际观察:")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.2f}, {value.max():.2f}]")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 测试环境
    env = test_environment_with_terrain()
    
    # 测试观察空间
    test_observation_space(env)
    
    print("\n✅ 环境测试完成!")
