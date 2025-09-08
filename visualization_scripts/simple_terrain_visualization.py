#!/usr/bin/env python3
"""
简化的地形可视化脚本
直接显示原始地形，不使用掩码
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt

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

def visualize_raw_terrain(terrain_data, save_path=None):
    """可视化原始地形（不使用掩码）"""
    height_map = np.array(terrain_data['height_map'])
    
    print("🗺️ 地形信息:")
    print(f"   地形尺寸: {height_map.shape}")
    print(f"   高程范围: [{np.min(height_map):.3f}, {np.max(height_map):.3f}]")
    print(f"   平均高程: {np.mean(height_map):.3f}")
    print(f"   高程标准差: {np.std(height_map):.3f}")
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('原始地形可视化（无掩码）', fontsize=16)
    
    # 1. 完整地形高程图
    im1 = axes[0, 0].imshow(height_map.T, cmap='terrain', aspect='auto', origin='lower')
    axes[0, 0].set_title('完整地形高程图')
    axes[0, 0].set_xlabel('X坐标')
    axes[0, 0].set_ylabel('Y坐标')
    plt.colorbar(im1, ax=axes[0, 0], label='高程')
    
    # 2. 地形3D效果图（使用不同的颜色映射）
    im2 = axes[0, 1].imshow(height_map.T, cmap='viridis', aspect='auto', origin='lower')
    axes[0, 1].set_title('地形3D效果图')
    axes[0, 1].set_xlabel('X坐标')
    axes[0, 1].set_ylabel('Y坐标')
    plt.colorbar(im2, ax=axes[0, 1], label='高程')
    
    # 3. 地形等高线图
    X, Y = np.meshgrid(np.arange(height_map.shape[0]), np.arange(height_map.shape[1]))
    contour = axes[1, 0].contour(X, Y, height_map.T, levels=20, colors='black', alpha=0.5)
    axes[1, 0].clabel(contour, inline=True, fontsize=8)
    axes[1, 0].set_title('地形等高线图')
    axes[1, 0].set_xlabel('X坐标')
    axes[1, 0].set_ylabel('Y坐标')
    
    # 4. 高程分布直方图
    axes[1, 1].hist(height_map.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(np.mean(height_map), color='red', linestyle='--', linewidth=2, 
                      label=f'平均值: {np.mean(height_map):.2f}')
    axes[1, 1].set_title('高程分布')
    axes[1, 1].set_xlabel('高程')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"原始地形图已保存到: {save_path}")
    
    plt.show()

def visualize_terrain_sections(terrain_data, save_path=None):
    """可视化地形的不同截面"""
    height_map = np.array(terrain_data['height_map'])
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('地形截面分析', fontsize=16)
    
    # 1. 中心行截面
    center_row = height_map.shape[0] // 2
    axes[0, 0].plot(height_map[center_row, :], 'b-', linewidth=2)
    axes[0, 0].set_title(f'中心行截面 (行 {center_row})')
    axes[0, 0].set_xlabel('Y坐标')
    axes[0, 0].set_ylabel('高程')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 中心列截面
    center_col = height_map.shape[1] // 2
    axes[0, 1].plot(height_map[:, center_col], 'r-', linewidth=2)
    axes[0, 1].set_title(f'中心列截面 (列 {center_col})')
    axes[0, 1].set_xlabel('X坐标')
    axes[0, 1].set_ylabel('高程')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 对角线截面
    diagonal = np.diag(height_map)
    axes[1, 0].plot(diagonal, 'g-', linewidth=2)
    axes[1, 0].set_title('对角线截面')
    axes[1, 0].set_xlabel('位置')
    axes[1, 0].set_ylabel('高程')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 地形统计信息
    axes[1, 1].axis('off')
    
    stats_text = f"""
地形统计信息:

尺寸: {height_map.shape[0]} x {height_map.shape[1]}
总点数: {height_map.size}

高程统计:
  最小值: {np.min(height_map):.3f}
  最大值: {np.max(height_map):.3f}
  平均值: {np.mean(height_map):.3f}
  中位数: {np.median(height_map):.3f}
  标准差: {np.std(height_map):.3f}

地形特征:
  平坦区域 (变化 < 1.0): {np.sum(np.abs(height_map - np.mean(height_map)) < 1.0)} 个点
  丘陵区域 (变化 1.0-10.0): {np.sum((np.abs(height_map - np.mean(height_map)) >= 1.0) & (np.abs(height_map - np.mean(height_map)) < 10.0))} 个点
  山地区域 (变化 > 10.0): {np.sum(np.abs(height_map - np.mean(height_map)) >= 10.0)} 个点

数据质量:
  NaN值: {np.sum(np.isnan(height_map))} 个
  无穷值: {np.sum(np.isinf(height_map))} 个
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"地形截面图已保存到: {save_path}")
    
    plt.show()

def find_simple_start_goal(height_map):
    """在原始地形上找到简单的起始点和终点"""
    # 找到一些相对平坦的区域作为起始点和终点
    height_diff = np.abs(height_map - np.mean(height_map))
    flat_indices = np.where(height_diff < np.std(height_map))
    
    if len(flat_indices[0]) < 2:
        print("❌ 没有找到足够平坦的区域")
        return None, None
    
    # 随机选择两个不同的点
    indices = np.random.choice(len(flat_indices[0]), 2, replace=False)
    
    start_idx = (flat_indices[0][indices[0]], flat_indices[1][indices[0]])
    goal_idx = (flat_indices[0][indices[1]], flat_indices[1][indices[1]])
    
    return start_idx, goal_idx

def visualize_with_path(terrain_data, save_path=None):
    """可视化地形并添加路径"""
    height_map = np.array(terrain_data['height_map'])
    
    # 找到起始点和终点
    start_point, goal_point = find_simple_start_goal(height_map)
    if start_point is None:
        print("❌ 无法找到合适的起始点和终点")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制地形
    im = ax.imshow(height_map.T, cmap='terrain', aspect='auto', origin='lower')
    
    # 绘制起始点和终点
    ax.scatter(start_point[0], start_point[1], s=200, c='green', marker='o', 
              edgecolors='black', linewidth=3, label='起始点', zorder=5)
    ax.scatter(goal_point[0], goal_point[1], s=200, c='red', marker='*', 
              edgecolors='black', linewidth=3, label='目标点', zorder=5)
    
    # 添加文本标注
    start_height = height_map[start_point]
    goal_height = height_map[goal_point]
    
    ax.annotate(f'起始点\n({start_point[0]}, {start_point[1]})\n高程: {start_height:.1f}', 
               xy=(start_point[0], start_point[1]), xytext=(10, 10),
               textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.annotate(f'目标点\n({goal_point[0]}, {goal_point[1]})\n高程: {goal_height:.1f}', 
               xy=(goal_point[0], goal_point[1]), xytext=(10, -30),
               textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_title('原始地形与路径点')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.legend()
    plt.colorbar(im, ax=ax, label='高程')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"地形路径图已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    print("🗺️ 可视化原始地形（无掩码）...")
    
    # 加载地形数据
    terrain_data = load_terrain_data()
    if terrain_data is None:
        return
    
    # 可视化原始地形
    visualize_raw_terrain(terrain_data, save_path="visualization_output/raw_terrain.png")
    
    # 可视化地形截面
    visualize_terrain_sections(terrain_data, save_path="visualization_output/terrain_sections.png")
    
    # 可视化带路径的地形
    visualize_with_path(terrain_data, save_path="visualization_output/terrain_with_path.png")

if __name__ == "__main__":
    main()
