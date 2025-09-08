#!/usr/bin/env python3
"""
测试边界提取功能
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def test_boundary_visualization():
    """测试边界可视化"""
    # 模拟有序边界数据
    test_boundary = {
        'boundary_loops': [
            # 主边界
            [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
            # 内部空洞
            [[3, 3, 0], [7, 3, 0], [7, 7, 0], [3, 7, 0]]
        ],
        'loop_count': 2,
        'total_points': 8
    }
    
    # 可视化测试
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = ['red', 'blue']
    for i, loop in enumerate(test_boundary['boundary_loops']):
        loop_array = np.array(loop)
        ax.plot(loop_array[:, 0], loop_array[:, 1], color=colors[i], linewidth=2, marker='o')
        ax.fill(loop_array[:, 0], loop_array[:, 1], alpha=0.3, color=colors[i])
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('测试边界可视化')
    
    plt.show()
    
    print("✅ 边界可视化测试完成")

def test_mask_creation():
    """测试掩码创建"""
    # 模拟有序边界
    boundary_loops = [
        # 主边界
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        # 内部空洞
        [[3, 3], [7, 3], [7, 7], [3, 7]]
    ]
    
    # 创建网格
    grid_size = (20, 20)
    mask = np.zeros(grid_size, dtype=bool)
    
    # 使用matplotlib的Path来填充
    from matplotlib.path import Path
    
    # 创建网格点
    x = np.arange(grid_size[0])
    y = np.arange(grid_size[1])
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.flatten(), Y.flatten()))
    
    # 对每个边界循环
    for loop in boundary_loops:
        loop_array = np.array(loop)
        path = Path(loop_array)
        
        # 检查哪些点在路径内
        inside = path.contains_points(points)
        inside = inside.reshape(grid_size)
        
        # 更新掩码（主边界为True，内部空洞为False）
        if len(boundary_loops) == 1 or loop == boundary_loops[0]:  # 主边界
            mask = mask | inside
        else:  # 内部空洞
            mask = mask & (~inside)
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始边界
    for i, loop in enumerate(boundary_loops):
        loop_array = np.array(loop)
        ax1.plot(loop_array[:, 0], loop_array[:, 1], linewidth=2, marker='o')
        ax1.fill(loop_array[:, 0], loop_array[:, 1], alpha=0.3)
    
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1, 11)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title('原始边界')
    
    # 生成的掩码
    ax2.imshow(mask.T, cmap='gray', origin='lower', extent=[0, 20, 0, 20])
    ax2.set_aspect('equal')
    ax2.set_title('生成的掩码')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ 掩码创建测试完成")
    print(f"   掩码覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
    print(f"   有效点数: {np.sum(mask)} / {mask.size}")

if __name__ == "__main__":
    print("🧪 测试边界提取功能...")
    test_boundary_visualization()
    test_mask_creation()
    print("✅ 所有测试完成")
