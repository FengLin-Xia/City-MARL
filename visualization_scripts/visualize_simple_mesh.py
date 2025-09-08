#!/usr/bin/env python3
"""
可视化简单Mesh处理结果
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_terrain_data(terrain_file="data/terrain/terrain_simple_mesh.json"):
    """加载地形数据"""
    if not os.path.exists(terrain_file):
        print(f"❌ 地形文件不存在: {terrain_file}")
        return None
    
    with open(terrain_file, 'r') as f:
        terrain_data = json.load(f)
    
    return terrain_data

def visualize_simple_mesh_result(terrain_data, save_path=None):
    """可视化简单mesh处理结果"""
    height_map = np.array(terrain_data['height_map'])
    mask = np.array(terrain_data['mask'])
    boundary_points = terrain_data['boundary_points']
    mesh_bounds = terrain_data['mesh_bounds']
    
    print("🗺️ 简单Mesh处理结果:")
    print(f"   地形尺寸: {height_map.shape}")
    print(f"   有效点数: {terrain_data['valid_points_count']} / {height_map.size}")
    print(f"   覆盖率: {terrain_data['coverage_percentage']:.1f}%")
    print(f"   高程范围: [{terrain_data['height_stats']['min']:.3f}, {terrain_data['height_stats']['max']:.3f}]")
    print(f"   平均高程: {terrain_data['height_stats']['mean']:.3f}")
    print(f"   Mesh边界: X[{mesh_bounds['x_min']:.2f}, {mesh_bounds['x_max']:.2f}], Y[{mesh_bounds['y_min']:.2f}, {mesh_bounds['y_max']:.2f}]")
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('简单Mesh边界处理结果', fontsize=16)
    
    # 1. 原始mesh边界点
    if boundary_points:
        boundary_array = np.array(boundary_points)
        axes[0, 0].scatter(boundary_array[:, 0], boundary_array[:, 1], c='red', s=1, alpha=0.6, label='边界点')
        axes[0, 0].set_title('原始Mesh边界点')
        axes[0, 0].set_xlabel('X坐标')
        axes[0, 0].set_ylabel('Y坐标')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 简单边界掩码
    axes[0, 1].imshow(mask.T, cmap='gray', aspect='auto', origin='lower')
    axes[0, 1].set_title('简单边界掩码')
    axes[0, 1].set_xlabel('X坐标')
    axes[0, 1].set_ylabel('Y坐标')
    
    # 3. 简单边界高程图
    valid_height_map = np.where(mask, height_map, np.nan)
    im3 = axes[0, 2].imshow(valid_height_map.T, cmap='terrain', aspect='auto', origin='lower')
    axes[0, 2].set_title('简单边界高程图')
    axes[0, 2].set_xlabel('X坐标')
    axes[0, 2].set_ylabel('Y坐标')
    plt.colorbar(im3, ax=axes[0, 2], label='高程')
    
    # 4. 完整高程图（无掩码）
    im4 = axes[1, 0].imshow(height_map.T, cmap='terrain', aspect='auto', origin='lower')
    axes[1, 0].set_title('完整高程图（无掩码）')
    axes[1, 0].set_xlabel('X坐标')
    axes[1, 0].set_ylabel('Y坐标')
    plt.colorbar(im4, ax=axes[1, 0], label='高程')
    
    # 5. 掩码对比
    axes[1, 1].imshow(mask.T, cmap='gray', aspect='auto', origin='lower')
    axes[1, 1].set_title('掩码对比')
    axes[1, 1].set_xlabel('X坐标')
    axes[1, 1].set_ylabel('Y坐标')
    
    # 6. 统计信息
    axes[1, 2].axis('off')
    
    info_text = f"""
简单Mesh处理结果:

网格尺寸: {height_map.shape[0]} x {height_map.shape[1]}
有效点数: {terrain_data['valid_points_count']} / {height_map.size}
覆盖率: {terrain_data['coverage_percentage']:.1f}%

Mesh边界:
  X: [{mesh_bounds['x_min']:.2f}, {mesh_bounds['x_max']:.2f}]
  Y: [{mesh_bounds['y_min']:.2f}, {mesh_bounds['y_max']:.2f}]

高程统计:
  最小值: {terrain_data['height_stats']['min']:.3f}
  最大值: {terrain_data['height_stats']['max']:.3f}
  平均值: {terrain_data['height_stats']['mean']:.3f}
  标准差: {terrain_data['height_stats']['std']:.3f}

边界点: {len(boundary_points)} 个

处理特点:
  ✓ 严格按照mesh边界
  ✓ 外部区域被排除
  ✓ 使用凸包边界
  ✓ 线性插值高程
    """
    
    axes[1, 2].text(0.05, 0.95, info_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()

def compare_with_previous(terrain_file="data/terrain/terrain_simple_mesh.json", 
                         previous_file="data/terrain/terrain_direct_mesh_fixed.json"):
    """与之前的结果对比"""
    print("🔄 与之前结果对比...")
    
    # 加载当前结果
    current_data = load_terrain_data(terrain_file)
    if current_data is None:
        return
    
    # 加载之前的结果
    if os.path.exists(previous_file):
        with open(previous_file, 'r') as f:
            previous_data = json.load(f)
        
        current_height = np.array(current_data['height_map'])
        current_mask = np.array(current_data['mask'])
        previous_height = np.array(previous_data['height_map'])
        previous_mask = np.array(previous_data['mask'])
        
        # 创建对比图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('地形处理结果对比', fontsize=16)
        
        # 当前结果
        valid_current = np.where(current_mask, current_height, np.nan)
        im1 = axes[0, 0].imshow(valid_current.T, cmap='terrain', aspect='auto', origin='lower')
        axes[0, 0].set_title('当前结果（简单Mesh）')
        axes[0, 0].set_xlabel('X坐标')
        axes[0, 0].set_ylabel('Y坐标')
        plt.colorbar(im1, ax=axes[0, 0], label='高程')
        
        axes[0, 1].imshow(current_mask.T, cmap='gray', aspect='auto', origin='lower')
        axes[0, 1].set_title('当前掩码')
        axes[0, 1].set_xlabel('X坐标')
        axes[0, 1].set_ylabel('Y坐标')
        
        # 之前的结果
        valid_previous = np.where(previous_mask, previous_height, np.nan)
        im3 = axes[1, 0].imshow(valid_previous.T, cmap='terrain', aspect='auto', origin='lower')
        axes[1, 0].set_title('之前结果（直接Mesh）')
        axes[1, 0].set_xlabel('X坐标')
        axes[1, 0].set_ylabel('Y坐标')
        plt.colorbar(im3, ax=axes[1, 0], label='高程')
        
        axes[1, 1].imshow(previous_mask.T, cmap='gray', aspect='auto', origin='lower')
        axes[1, 1].set_title('之前掩码')
        axes[1, 1].set_xlabel('X坐标')
        axes[1, 1].set_ylabel('Y坐标')
        
        # 对比信息
        axes[0, 2].axis('off')
        axes[1, 2].axis('off')
        
        current_info = f"""
当前结果（简单Mesh）:
覆盖率: {current_data['coverage_percentage']:.1f}%
有效点数: {current_data['valid_points_count']}
高程范围: [{current_data['height_stats']['min']:.1f}, {current_data['height_stats']['max']:.1f}]
        """
        
        previous_info = f"""
之前结果（直接Mesh）:
覆盖率: {previous_data['coverage_percentage']:.1f}%
有效点数: {previous_data['valid_points_count']}
高程范围: [{previous_data['height_stats']['min']:.1f}, {previous_data['height_stats']['max']:.1f}]
        """
        
        axes[0, 2].text(0.05, 0.95, current_info, transform=axes[0, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        axes[1, 2].text(0.05, 0.95, previous_info, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig("visualization_output/terrain_comparison.png", dpi=300, bbox_inches='tight')
        print("✅ 对比图已保存到: visualization_output/terrain_comparison.png")
        plt.show()
    else:
        print(f"❌ 之前的文件不存在: {previous_file}")

def main():
    """主函数"""
    print("🎨 可视化简单Mesh处理结果...")
    
    # 加载地形数据
    terrain_data = load_terrain_data()
    if terrain_data is None:
        return
    
    # 可视化结果
    visualize_simple_mesh_result(terrain_data, save_path="visualization_output/simple_mesh_result.png")
    
    # 与之前结果对比
    compare_with_previous()

if __name__ == "__main__":
    main()
