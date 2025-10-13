#!/usr/bin/env python3
"""
训练环境可视化脚本
可视化当前的地形环境、起始点、终点和智能体路径
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_terrain_data(terrain_file="data/terrain/terrain_direct_mesh_fixed.json"):
    """加载地形数据"""
    if not os.path.exists(terrain_file):
        print(f"❌ 地形文件不存在: {terrain_file}")
        return None
    
    with open(terrain_file, 'r') as f:
        terrain_data = json.load(f)
    
    return terrain_data

def find_land_points(terrain_data, height_threshold=0.0):
    """在陆地上找到合适的起始点和终点"""
    height_map = np.array(terrain_data['height_map'])
    mask = np.array(terrain_data['mask'])
    
    # 找到所有有效的陆地点
    valid_indices = np.where((mask) & (height_map > height_threshold))
    
    if len(valid_indices[0]) < 2:
        print("❌ 没有足够的陆地点")
        return None, None
    
    # 随机选择两个不同的点
    indices = np.random.choice(len(valid_indices[0]), 2, replace=False)
    
    start_idx = (valid_indices[0][indices[0]], valid_indices[1][indices[0]])
    goal_idx = (valid_indices[0][indices[1]], valid_indices[1][indices[1]])
    
    return start_idx, goal_idx

def generate_sample_path(start_point, goal_point, height_map, mask, max_steps=300):
    """生成一个示例路径（用于演示）"""
    path = [start_point]
    current = start_point
    
    # 简单的A*风格路径生成
    for step in range(max_steps):
        if current == goal_point:
            break
        
        # 计算到目标的方向
        dx = goal_point[0] - current[0]
        dy = goal_point[1] - current[1]
        
        # 选择下一步
        next_steps = []
        for dx_step, dy_step in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x = current[0] + dx_step
            next_y = current[1] + dy_step
            
            # 检查边界和有效性
            if (0 <= next_x < height_map.shape[0] and 
                0 <= next_y < height_map.shape[1] and 
                mask[next_x, next_y]):
                next_steps.append((next_x, next_y))
        
        if not next_steps:
            break
        
        # 选择最接近目标的下一步
        best_step = min(next_steps, key=lambda step: 
                       abs(step[0] - goal_point[0]) + abs(step[1] - goal_point[1]))
        
        current = best_step
        path.append(current)
    
    return path

def visualize_training_environment(terrain_data, start_point=None, goal_point=None, save_path=None):
    """可视化训练环境"""
    height_map = np.array(terrain_data['height_map'])
    mask = np.array(terrain_data['mask'])
    boundary_points = terrain_data['boundary_points']
    
    # 如果没有指定起始点和终点，随机生成
    if start_point is None or goal_point is None:
        start_point, goal_point = find_land_points(terrain_data)
        if start_point is None:
            print("❌ 无法找到合适的起始点和终点")
            return
    
    # 生成示例路径
    sample_path = generate_sample_path(start_point, goal_point, height_map, mask)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('训练环境可视化', fontsize=16)
    
    # 1. 地形高程图 + 路径
    valid_height_map = np.where(mask, height_map, np.nan)
    im1 = axes[0, 0].imshow(valid_height_map.T, cmap='terrain', aspect='auto', origin='lower')
    
    # 绘制路径
    if sample_path:
        path_x = [p[0] for p in sample_path]
        path_y = [p[1] for p in sample_path]
        axes[0, 0].plot(path_x, path_y, 'r-', linewidth=2, alpha=0.8, label='智能体路径')
    
    # 绘制起始点和终点
    axes[0, 0].scatter(start_point[0], start_point[1], s=100, c='green', marker='o', 
                      edgecolors='black', linewidth=2, label='起始点', zorder=5)
    axes[0, 0].scatter(goal_point[0], goal_point[1], s=100, c='red', marker='*', 
                      edgecolors='black', linewidth=2, label='目标点', zorder=5)
    
    axes[0, 0].set_title('地形高程图与路径规划')
    axes[0, 0].set_xlabel('X坐标')
    axes[0, 0].set_ylabel('Y坐标')
    axes[0, 0].legend()
    plt.colorbar(im1, ax=axes[0, 0], label='高程')
    
    # 2. 可导航区域掩码
    im2 = axes[0, 1].imshow(mask.T, cmap='gray', aspect='auto', origin='lower')
    axes[0, 1].set_title('可导航区域掩码')
    axes[0, 1].set_xlabel('X坐标')
    axes[0, 1].set_ylabel('Y坐标')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. 地形复杂度热图
    complexity_map = np.zeros_like(height_map)
    for i in range(1, height_map.shape[0] - 1):
        for j in range(1, height_map.shape[1] - 1):
            if mask[i, j]:
                neighbors = [
                    height_map[i-1, j], height_map[i+1, j],
                    height_map[i, j-1], height_map[i, j+1]
                ]
                complexity_map[i, j] = np.std(neighbors)
    
    valid_complexity = np.where(mask, complexity_map, np.nan)
    im3 = axes[1, 0].imshow(valid_complexity.T, cmap='hot', aspect='auto', origin='lower')
    axes[1, 0].set_title('地形复杂度热图')
    axes[1, 0].set_xlabel('X坐标')
    axes[1, 0].set_ylabel('Y坐标')
    plt.colorbar(im3, ax=axes[1, 0], label='复杂度')
    
    # 4. 环境信息统计
    axes[1, 1].axis('off')
    
    # 计算路径统计
    path_length = len(sample_path) if sample_path else 0
    path_height_changes = []
    if len(sample_path) > 1:
        for i in range(1, len(sample_path)):
            h1 = height_map[sample_path[i-1]]
            h2 = height_map[sample_path[i]]
            path_height_changes.append(abs(h2 - h1))
    
    avg_height_change = np.mean(path_height_changes) if path_height_changes else 0
    max_height_change = np.max(path_height_changes) if path_height_changes else 0
    
    # 计算起始点和终点的高程
    start_height = height_map[start_point]
    goal_height = height_map[goal_point]
    
    info_text = f"""
训练环境信息:

地形尺寸: {terrain_data['grid_size'][0]} x {terrain_data['grid_size'][1]}
可导航区域: {terrain_data['valid_points_count']} / {terrain_data['grid_size'][0] * terrain_data['grid_size'][1]}
覆盖率: {terrain_data['coverage_percentage']:.1f}%

起始点: ({start_point[0]}, {start_point[1]})
起始高程: {start_height:.2f}

目标点: ({goal_point[0]}, {goal_point[1]})
目标高程: {goal_height:.2f}

路径信息:
  路径长度: {path_length} 步
  平均高程变化: {avg_height_change:.3f}
  最大高程变化: {max_height_change:.3f}
  高程差: {abs(goal_height - start_height):.2f}

地形特征:
  平均高程: {np.mean(height_map[mask]):.2f}
  高程范围: [{np.min(height_map[mask]):.2f}, {np.max(height_map[mask]):.2f}]
    """
    
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练环境图已保存到: {save_path}")
    
    plt.show()

def create_animated_path(terrain_data, start_point=None, goal_point=None, save_path=None):
    """创建路径动画"""
    height_map = np.array(terrain_data['height_map'])
    mask = np.array(terrain_data['mask'])
    
    if start_point is None or goal_point is None:
        start_point, goal_point = find_land_points(terrain_data)
        if start_point is None:
            return
    
    # 生成路径
    path = generate_sample_path(start_point, goal_point, height_map, mask)
    
    # 创建动画
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制地形
    valid_height_map = np.where(mask, height_map, np.nan)
    im = ax.imshow(valid_height_map.T, cmap='terrain', aspect='auto', origin='lower')
    
    # 绘制起始点和终点
    ax.scatter(start_point[0], start_point[1], s=150, c='green', marker='o', 
              edgecolors='black', linewidth=3, label='起始点', zorder=5)
    ax.scatter(goal_point[0], goal_point[1], s=150, c='red', marker='*', 
              edgecolors='black', linewidth=3, label='目标点', zorder=5)
    
    # 初始化路径线
    line, = ax.plot([], [], 'r-', linewidth=3, alpha=0.8, label='智能体路径')
    point, = ax.plot([], [], 'ro', markersize=8, markeredgecolor='black', 
                    markeredgewidth=2, label='当前位置')
    
    ax.set_title('智能体路径动画')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.legend()
    plt.colorbar(im, ax=ax, label='高程')
    
    def animate(frame):
        if frame < len(path):
            # 更新路径线
            line.set_data([p[0] for p in path[:frame+1]], [p[1] for p in path[:frame+1]])
            # 更新当前位置
            point.set_data([path[frame][0]], [path[frame][1]])
        return line, point
    
    anim = FuncAnimation(fig, animate, frames=len(path), interval=100, blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"路径动画已保存到: {save_path}")
    
    plt.show()
    return anim

def main():
    """主函数"""
    print("🎮 可视化训练环境...")
    
    # 加载地形数据
    terrain_data = load_terrain_data()
    if terrain_data is None:
        return
    
    # 可视化训练环境
    visualize_training_environment(terrain_data, save_path="visualization_output/training_environment.png")
    
    # 创建路径动画
    print("🎬 创建路径动画...")
    create_animated_path(terrain_data, save_path="visualization_output/path_animation.gif")

if __name__ == "__main__":
    main()
