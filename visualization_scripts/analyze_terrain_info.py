#!/usr/bin/env python3
"""
地形环境信息分析脚本
分析当前地形环境的详细信息
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_terrain_info(terrain_file="data/terrain/terrain_direct_mesh_fixed.json"):
    """分析地形环境信息"""
    if not os.path.exists(terrain_file):
        print(f"❌ 地形文件不存在: {terrain_file}")
        return
    
    print("🔍 分析地形环境信息...")
    print("=" * 50)
    
    # 加载地形数据
    with open(terrain_file, 'r') as f:
        terrain_data = json.load(f)
    
    # 基本信息
    grid_size = terrain_data['grid_size']
    vertices_count = terrain_data['vertices_count']
    faces_count = terrain_data['faces_count']
    valid_points_count = terrain_data['valid_points_count']
    coverage_percentage = terrain_data['coverage_percentage']
    
    print("📊 基本信息:")
    print(f"   网格尺寸: {grid_size[0]} x {grid_size[1]} = {grid_size[0] * grid_size[1]} 个网格点")
    print(f"   原始顶点数: {vertices_count}")
    print(f"   原始面数: {faces_count}")
    print(f"   有效网格点数: {valid_points_count}")
    print(f"   覆盖率: {coverage_percentage:.1f}%")
    
    # 转换为numpy数组
    height_map = np.array(terrain_data['height_map'])
    mask = np.array(terrain_data['mask'])
    
    print(f"\n🗺️  地形数据:")
    print(f"   高程图形状: {height_map.shape}")
    print(f"   掩码形状: {mask.shape}")
    
    # 高程统计
    valid_heights = height_map[mask]
    invalid_heights = height_map[~mask]
    
    print(f"\n📈 高程统计:")
    print(f"   有效区域高程范围: [{np.min(valid_heights):.3f}, {np.max(valid_heights):.3f}]")
    print(f"   有效区域高程均值: {np.mean(valid_heights):.3f}")
    print(f"   有效区域高程标准差: {np.std(valid_heights):.3f}")
    print(f"   有效区域高程中位数: {np.median(valid_heights):.3f}")
    
    if len(invalid_heights) > 0:
        print(f"   无效区域高程范围: [{np.min(invalid_heights):.3f}, {np.max(invalid_heights):.3f}]")
        print(f"   无效区域高程均值: {np.mean(invalid_heights):.3f}")
    
    # 边界点信息
    boundary_points = terrain_data['boundary_points']
    print(f"\n🔲 边界信息:")
    print(f"   边界点数量: {len(boundary_points)}")
    
    if boundary_points:
        boundary_array = np.array(boundary_points)
        x_coords = boundary_array[:, 0]
        y_coords = boundary_array[:, 1]
        
        print(f"   边界X坐标范围: [{np.min(x_coords):.1f}, {np.max(x_coords):.1f}]")
        print(f"   边界Y坐标范围: [{np.min(y_coords):.1f}, {np.max(y_coords):.1f}]")
        print(f"   边界区域大小: {np.max(x_coords) - np.min(x_coords):.1f} x {np.max(y_coords) - np.min(y_coords):.1f}")
    
    # 地形复杂度分析
    print(f"\n🎯 地形复杂度:")
    
    # 计算坡度（简化版本）
    if len(valid_heights) > 1:
        # 计算相邻点的高程差
        height_diffs = []
        for i in range(height_map.shape[0] - 1):
            for j in range(height_map.shape[1] - 1):
                if mask[i, j] and mask[i+1, j]:
                    height_diffs.append(abs(height_map[i+1, j] - height_map[i, j]))
                if mask[i, j] and mask[i, j+1]:
                    height_diffs.append(abs(height_map[i, j+1] - height_map[i, j]))
        
        if height_diffs:
            height_diffs = np.array(height_diffs)
            print(f"   平均高程变化: {np.mean(height_diffs):.3f}")
            print(f"   最大高程变化: {np.max(height_diffs):.3f}")
            print(f"   高程变化标准差: {np.std(height_diffs):.3f}")
    
    # 地形分布分析
    print(f"\n📊 地形分布:")
    print(f"   平坦区域 (高程变化 < 0.1): {np.sum(np.abs(valid_heights - np.mean(valid_heights)) < 0.1)} 个点")
    print(f"   丘陵区域 (高程变化 0.1-1.0): {np.sum((np.abs(valid_heights - np.mean(valid_heights)) >= 0.1) & (np.abs(valid_heights - np.mean(valid_heights)) < 1.0))} 个点")
    print(f"   山地区域 (高程变化 > 1.0): {np.sum(np.abs(valid_heights - np.mean(valid_heights)) >= 1.0)} 个点")
    
    # 可导航性分析
    print(f"\n🚶 可导航性分析:")
    print(f"   总网格点数: {grid_size[0] * grid_size[1]}")
    print(f"   可导航点数: {valid_points_count}")
    print(f"   不可导航点数: {grid_size[0] * grid_size[1] - valid_points_count}")
    print(f"   可导航比例: {valid_points_count / (grid_size[0] * grid_size[1]) * 100:.1f}%")
    
    # 训练环境适用性
    print(f"\n🎮 训练环境适用性:")
    print(f"   环境大小: {grid_size[0]} x {grid_size[1]} 网格")
    print(f"   最大路径长度: {grid_size[0] + grid_size[1]} 步 (曼哈顿距离)")
    print(f"   平均路径长度估计: {np.sqrt(grid_size[0]**2 + grid_size[1]**2):.1f} 步 (欧几里得距离)")
    print(f"   建议最大步数: {int((grid_size[0] + grid_size[1]) * 1.5)} 步")
    
    return terrain_data

def visualize_terrain_analysis(terrain_data, save_path=None):
    """可视化地形分析结果"""
    height_map = np.array(terrain_data['height_map'])
    mask = np.array(terrain_data['mask'])
    boundary_points = terrain_data['boundary_points']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('地形环境详细分析', fontsize=16)
    
    # 1. 高程图
    valid_height_map = np.where(mask, height_map, np.nan)
    im1 = axes[0, 0].imshow(valid_height_map.T, cmap='terrain', aspect='auto', origin='lower')
    axes[0, 0].set_title('地形高程图')
    axes[0, 0].set_xlabel('X坐标')
    axes[0, 0].set_ylabel('Y坐标')
    plt.colorbar(im1, ax=axes[0, 0], label='高程')
    
    # 2. 掩码图
    im2 = axes[0, 1].imshow(mask.T, cmap='gray', aspect='auto', origin='lower')
    axes[0, 1].set_title('可导航区域掩码')
    axes[0, 1].set_xlabel('X坐标')
    axes[0, 1].set_ylabel('Y坐标')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. 边界点
    if boundary_points:
        boundary_array = np.array(boundary_points)
        axes[0, 2].scatter(boundary_array[:, 0], boundary_array[:, 1], s=1, alpha=0.6, c='red')
        axes[0, 2].set_title('地形边界点')
        axes[0, 2].set_xlabel('X坐标')
        axes[0, 2].set_ylabel('Y坐标')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 高程分布直方图
    valid_heights = height_map[mask]
    axes[1, 0].hist(valid_heights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(np.mean(valid_heights), color='red', linestyle='--', linewidth=2, 
                      label=f'均值: {np.mean(valid_heights):.3f}')
    axes[1, 0].set_title('高程分布')
    axes[1, 0].set_xlabel('高程')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 地形复杂度热图
    # 计算每个点的局部高程变化
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
    im3 = axes[1, 1].imshow(valid_complexity.T, cmap='hot', aspect='auto', origin='lower')
    axes[1, 1].set_title('地形复杂度热图')
    axes[1, 1].set_xlabel('X坐标')
    axes[1, 1].set_ylabel('Y坐标')
    plt.colorbar(im3, ax=axes[1, 1], label='复杂度')
    
    # 6. 统计信息
    axes[1, 2].axis('off')
    stats_text = f"""
地形环境统计信息:

网格尺寸: {terrain_data['grid_size'][0]} x {terrain_data['grid_size'][1]}
有效点数: {terrain_data['valid_points_count']}
覆盖率: {terrain_data['coverage_percentage']:.1f}%

高程统计:
  范围: [{np.min(valid_heights):.3f}, {np.max(valid_heights):.3f}]
  均值: {np.mean(valid_heights):.3f}
  标准差: {np.std(valid_heights):.3f}

边界信息:
  边界点数: {len(boundary_points)}
  可导航比例: {terrain_data['valid_points_count'] / (terrain_data['grid_size'][0] * terrain_data['grid_size'][1]) * 100:.1f}%
    """
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"地形分析图已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    # 分析地形信息
    terrain_data = analyze_terrain_info()
    
    if terrain_data:
        # 可视化分析结果
        visualize_terrain_analysis(terrain_data, "visualization_output/terrain_analysis.png")

if __name__ == "__main__":
    main()
