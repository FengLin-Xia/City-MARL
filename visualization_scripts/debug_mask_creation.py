#!/usr/bin/env python3
"""
调试掩码创建问题
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

def debug_mask_creation():
    """调试掩码创建过程"""
    print("🔍 调试掩码创建过程...")
    
    # 模拟有序边界数据
    ordered_boundary = {
        'boundary_loops': [
            # 主边界
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]],
            # 内部空洞
            [[3.0, 3.0, 0.0], [7.0, 3.0, 0.0], [7.0, 7.0, 0.0], [3.0, 7.0, 0.0]]
        ],
        'loop_count': 2,
        'total_points': 8
    }
    
    # 模拟网格参数
    grid_size = (150, 150)
    x_min, x_max = 0.0, 10.0
    y_min, y_max = 0.0, 10.0
    
    print(f"📊 网格参数:")
    print(f"   网格大小: {grid_size}")
    print(f"   X范围: [{x_min}, {x_max}]")
    print(f"   Y范围: [{y_min}, {y_max}]")
    
    # 创建网格坐标
    grid_x, grid_y = grid_size
    x_coords = np.linspace(x_min, x_max, grid_x)
    y_coords = np.linspace(y_min, y_max, grid_y)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.column_stack((X.flatten(), Y.flatten()))
    
    print(f"📊 网格点:")
    print(f"   总点数: {len(points)}")
    print(f"   前5个点: {points[:5]}")
    
    # 处理每个边界循环
    boundary_loops = ordered_boundary['boundary_loops']
    mask = np.zeros(grid_size, dtype=bool)
    
    for i, loop in enumerate(boundary_loops):
        print(f"\n🔄 处理边界循环 {i+1}:")
        print(f"   循环点数: {len(loop)}")
        print(f"   循环点: {loop}")
        
        # 只取XY坐标（忽略Z坐标）
        loop_2d = np.array([[point[0], point[1]] for point in loop])
        print(f"   2D循环点: {loop_2d}")
        
        # 创建路径
        path = Path(loop_2d)
        print(f"   路径创建成功")
        
        # 检查哪些点在路径内
        inside = path.contains_points(points)
        inside = inside.reshape(grid_size)
        
        print(f"   内部点数: {np.sum(inside)}")
        print(f"   内部点比例: {np.sum(inside)/inside.size*100:.1f}%")
        
        # 更新掩码（主边界为True，内部空洞为False）
        if i == 0:  # 主边界
            mask = mask | inside
            print(f"   主边界掩码更新后: {np.sum(mask)} 个有效点")
        else:  # 内部空洞
            mask = mask & (~inside)
            print(f"   内部空洞掩码更新后: {np.sum(mask)} 个有效点")
    
    print(f"\n📊 最终结果:")
    print(f"   有效点数: {np.sum(mask)} / {mask.size}")
    print(f"   覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
    
    # 可视化结果
    visualize_debug_result(mask, boundary_loops, x_min, x_max, y_min, y_max)
    
    return mask

def visualize_debug_result(mask, boundary_loops, x_min, x_max, y_min, y_max):
    """可视化调试结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示掩码
    im1 = ax1.imshow(mask.T, cmap='gray', origin='lower', 
                     extent=[x_min, x_max, y_min, y_max], aspect='equal')
    ax1.set_title('生成的掩码')
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    
    # 显示边界
    colors = ['red', 'blue']
    for i, loop in enumerate(boundary_loops):
        loop_array = np.array(loop)
        ax2.plot(loop_array[:, 0], loop_array[:, 1], color=colors[i], 
                linewidth=2, marker='o', label=f'边界 {i+1}')
        ax2.fill(loop_array[:, 0], loop_array[:, 1], alpha=0.3, color=colors[i])
    
    ax2.set_xlim(x_min-1, x_max+1)
    ax2.set_ylim(y_min-1, y_max+1)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_title('原始边界')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("debug_mask_result.png", dpi=300, bbox_inches='tight')
    print("✅ 调试结果已保存到: debug_mask_result.png")
    plt.show()

if __name__ == "__main__":
    debug_mask_creation()
