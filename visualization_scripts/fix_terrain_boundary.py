#!/usr/bin/env python3
"""
修复地形边界问题
改进插值算法，解决边界重复值问题
"""

import numpy as np
import json
import os
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def load_terrain_data(terrain_file: str):
    """加载地形数据"""
    print(f"加载地形数据: {terrain_file}")
    with open(terrain_file, 'r') as f:
        data = json.load(f)
    return data

def analyze_terrain_issues(height_map):
    """分析地形数据问题"""
    print("=== 地形数据分析 ===")
    print(f"地形尺寸: {height_map.shape}")
    print(f"高程范围: {np.min(height_map):.3f} ~ {np.max(height_map):.3f}")
    
    # 检查重复值
    unique_values, counts = np.unique(height_map, return_counts=True)
    print(f"唯一值数量: {len(unique_values)}")
    
    # 找出最常见的值
    most_common_idx = np.argmax(counts)
    most_common_value = unique_values[most_common_idx]
    most_common_count = counts[most_common_idx]
    print(f"最常见的值: {most_common_value:.6f} (出现 {most_common_count} 次)")
    print(f"最常见值占总数的比例: {most_common_count/height_map.size*100:.2f}%")
    
    # 检查边界
    border_values = np.concatenate([
        height_map[0, :],  # 上边界
        height_map[-1, :], # 下边界
        height_map[:, 0],  # 左边界
        height_map[:, -1]  # 右边界
    ])
    
    border_unique, border_counts = np.unique(border_values, return_counts=True)
    print(f"边界唯一值数量: {len(border_unique)}")
    
    # 检查边界重复值
    border_most_common_idx = np.argmax(border_counts)
    border_most_common_value = border_unique[border_most_common_idx]
    border_most_common_count = border_counts[border_most_common_idx]
    print(f"边界最常见值: {border_most_common_value:.6f} (出现 {border_most_common_count} 次)")
    print(f"边界最常见值占边界总数的比例: {border_most_common_count/len(border_values)*100:.2f}%")
    
    # 检查是否有大量重复的边界值
    if len(border_unique) < 10:
        print("⚠️ 警告：边界值过于单一，可能存在插值问题")
    
    # 检查特定重复值
    if abs(most_common_value - 1.897052) < 0.001:
        print("⚠️ 警告：检测到边界重复值 1.897052")
    
    # 检查边界区域
    print("\n=== 边界区域分析 ===")
    print("上边界前10个值:", height_map[0, :10])
    print("下边界前10个值:", height_map[-1, :10])
    print("左边界前10个值:", height_map[:10, 0])
    print("右边界前10个值:", height_map[:10, -1])
    
    return {
        'total_unique': len(unique_values),
        'most_common_value': most_common_value,
        'most_common_count': most_common_count,
        'border_unique': len(border_unique),
        'border_most_common_value': border_most_common_value,
        'border_most_common_count': border_most_common_count
    }

def improved_terrain_interpolation(vertices, faces, grid_size, boundary=None):
    """改进的地形插值算法"""
    print("=== 改进地形插值 ===")
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # 提取坐标和高程
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    heights = vertices[:, 2]
    
    # 使用边界信息或自动计算
    if boundary:
        x_min, x_max = boundary['x_min'], boundary['x_max']
        y_min, y_max = boundary['y_min'], boundary['y_max']
        print("使用提供的边界信息")
    else:
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        print("自动计算边界信息")
    
    print(f"边界: X({x_min:.3f}~{x_max:.3f}), Y({y_min:.3f}~{y_max:.3f})")
    print(f"目标网格: {grid_size}")
    
    # 创建网格点
    grid_x = np.linspace(x_min, x_max, grid_size[0])
    grid_y = np.linspace(y_min, y_max, grid_size[1])
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y, indexing='ij')
    
    # 准备插值点
    points = np.column_stack((x_coords, y_coords))
    values = heights
    
    # 使用scipy的griddata进行插值
    print("执行插值...")
    height_map = griddata(points, values, (grid_X, grid_Y), method='linear', fill_value=np.nan)
    
    # 处理NaN值（边界外的点）
    if np.any(np.isnan(height_map)):
        print(f"发现 {np.sum(np.isnan(height_map))} 个NaN值，使用最近邻插值填充")
        height_map_nn = griddata(points, values, (grid_X, grid_Y), method='nearest')
        height_map = np.where(np.isnan(height_map), height_map_nn, height_map)
    
    # 应用高斯平滑减少噪声
    print("应用高斯平滑...")
    height_map = gaussian_filter(height_map, sigma=0.5)
    
    # 确保数据类型
    height_map = height_map.astype(np.float32)
    
    print(f"插值完成，高程范围: {np.min(height_map):.3f} ~ {np.max(height_map):.3f}")
    
    return height_map

def create_improved_terrain_data(original_data, improved_height_map):
    """创建改进的地形数据"""
    improved_data = original_data.copy()
    improved_data['height_map'] = improved_height_map.tolist()
    
    # 更新统计信息
    improved_data['vertices_count'] = len(original_data.get('height_map', []))
    improved_data['faces_count'] = len(original_data.get('height_map', []))
    
    return improved_data

def visualize_comparison(original_height_map, improved_height_map, save_path=None):
    """可视化对比"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('地形数据对比分析', fontsize=16)
    
    # 原始地形
    im1 = axes[0, 0].imshow(original_height_map, cmap='terrain', aspect='auto')
    axes[0, 0].set_title('原始地形高程图')
    axes[0, 0].set_xlabel('X坐标')
    axes[0, 0].set_ylabel('Y坐标')
    plt.colorbar(im1, ax=axes[0, 0], label='高程')
    
    # 改进地形
    im2 = axes[0, 1].imshow(improved_height_map, cmap='terrain', aspect='auto')
    axes[0, 1].set_title('改进地形高程图')
    axes[0, 1].set_xlabel('X坐标')
    axes[0, 1].set_ylabel('Y坐标')
    plt.colorbar(im2, ax=axes[0, 1], label='高程')
    
    # 差异图
    diff = improved_height_map - original_height_map
    im3 = axes[0, 2].imshow(diff, cmap='RdBu_r', aspect='auto')
    axes[0, 2].set_title('差异图 (改进 - 原始)')
    axes[0, 2].set_xlabel('X坐标')
    axes[0, 2].set_ylabel('Y坐标')
    plt.colorbar(im3, ax=axes[0, 2], label='高程差异')
    
    # 原始高程分布
    axes[1, 0].hist(original_height_map.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_title('原始高程分布')
    axes[1, 0].set_xlabel('高程')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 改进高程分布
    axes[1, 1].hist(improved_height_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].set_title('改进高程分布')
    axes[1, 1].set_xlabel('高程')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 边界对比
    # 提取边界值
    orig_border = np.concatenate([
        original_height_map[0, :], original_height_map[-1, :],
        original_height_map[:, 0], original_height_map[:, -1]
    ])
    improved_border = np.concatenate([
        improved_height_map[0, :], improved_height_map[-1, :],
        improved_height_map[:, 0], improved_height_map[:, -1]
    ])
    
    axes[1, 2].hist(orig_border, bins=30, alpha=0.7, color='red', label='原始边界', edgecolor='black')
    axes[1, 2].hist(improved_border, bins=30, alpha=0.7, color='blue', label='改进边界', edgecolor='black')
    axes[1, 2].set_title('边界值分布对比')
    axes[1, 2].set_xlabel('高程')
    axes[1, 2].set_ylabel('频次')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    # 使用现有的地形数据
    terrain_file = "data/terrain/terrain_1755281528.json"
    
    if not os.path.exists(terrain_file):
        print(f"地形文件不存在: {terrain_file}")
        return
    
    # 加载原始数据
    original_data = load_terrain_data(terrain_file)
    original_height_map = np.array(original_data['height_map'])
    
    # 分析问题
    issues = analyze_terrain_issues(original_height_map)
    
    # 计算边界值总数
    border_values = np.concatenate([
        original_height_map[0, :],  # 上边界
        original_height_map[-1, :], # 下边界
        original_height_map[:, 0],  # 左边界
        original_height_map[:, -1]  # 右边界
    ])
    
    # 检查是否需要修复
    if (issues['border_unique'] < 10 or 
        issues['most_common_count'] > original_height_map.size * 0.1 or
        issues['border_most_common_count'] > len(border_values) * 0.3 or
        abs(issues['most_common_value'] - 1.897052) < 0.001):
        print("⚠️ 检测到地形数据问题，需要修复")
        
        # 由于我们没有原始的OBJ文件，我们需要从现有数据重建
        print("尝试从现有数据重建地形...")
        
        # 创建输出目录
        output_dir = "data/terrain"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成改进的地形数据
        # 这里我们使用一个简化的方法：重新插值现有数据
        H, W = original_height_map.shape
        
        # 创建坐标网格
        x_coords = np.arange(W)
        y_coords = np.arange(H)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # 找到非重复值的点
        unique_mask = original_height_map != issues['most_common_value']
        valid_points = np.column_stack((X[unique_mask], Y[unique_mask]))
        valid_values = original_height_map[unique_mask]
        
        if len(valid_points) > 100:  # 确保有足够的有效点
            print(f"使用 {len(valid_points)} 个有效点进行重新插值")
            
            # 重新插值
            improved_height_map = griddata(
                valid_points, valid_values, 
                (X, Y), method='cubic', fill_value=np.nan
            )
            
            # 处理NaN值
            if np.any(np.isnan(improved_height_map)):
                improved_height_map_nn = griddata(
                    valid_points, valid_values, 
                    (X, Y), method='nearest'
                )
                improved_height_map = np.where(
                    np.isnan(improved_height_map), 
                    improved_height_map_nn, 
                    improved_height_map
                )
            
            # 应用平滑
            improved_height_map = gaussian_filter(improved_height_map, sigma=0.5)
            improved_height_map = improved_height_map.astype(np.float32)
            
            # 创建改进的数据
            improved_data = create_improved_terrain_data(original_data, improved_height_map)
            
            # 保存改进的数据
            improved_file = os.path.join(output_dir, "terrain_improved.json")
            with open(improved_file, 'w') as f:
                json.dump(improved_data, f, indent=2)
            
            print(f"改进的地形数据已保存到: {improved_file}")
            
            # 分析改进效果
            improved_issues = analyze_terrain_issues(improved_height_map)
            
            # 可视化对比
            visualize_comparison(
                original_height_map, improved_height_map,
                save_path="visualization_output/terrain_comparison.png"
            )
            
            print("\n=== 修复效果 ===")
            print(f"原始边界唯一值: {issues['border_unique']} -> 改进后: {improved_issues['border_unique']}")
            print(f"原始最常见值出现次数: {issues['most_common_count']} -> 改进后: {improved_issues['most_common_count']}")
            
        else:
            print("❌ 有效点太少，无法进行有效修复")
            print("建议：重新从Blender导出OBJ文件")
    
    else:
        print("✅ 地形数据看起来正常，无需修复")

if __name__ == "__main__":
    main()
