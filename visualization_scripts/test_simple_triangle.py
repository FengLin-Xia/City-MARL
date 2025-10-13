#!/usr/bin/env python3
"""
测试最简单的三角形填充
"""

import numpy as np
import matplotlib.pyplot as plt

def test_simple_triangle():
    """测试最简单的三角形"""
    print("🧪 测试简单三角形填充...")
    
    # 创建一个简单的三角形
    vertices = np.array([
        [0, 0, 10],    # 左下角，高度10
        [10, 0, 20],   # 右下角，高度20  
        [5, 10, 15]    # 顶部，高度15
    ])
    
    faces = np.array([[0, 1, 2]])  # 一个三角形
    
    # 网格参数
    grid_size = (20, 20)  # 20x20网格
    x_min, x_max = -1, 11
    y_min, y_max = -1, 11
    
    print(f"三角形顶点: {vertices}")
    print(f"网格尺寸: {grid_size}")
    print(f"坐标范围: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    # 直接实现简单的三角面填充算法
    def simple_triangle_fill(vertices, faces, grid_size, x_min, x_max, y_min, y_max):
        """最简单的三角面填充算法"""
        W, H = grid_size
        dx = (x_max - x_min) / W
        dy = (y_max - y_min) / H
        
        # 初始化
        Z = np.full((H, W), np.nan, dtype=np.float32)
        M = np.zeros((H, W), dtype=bool)
        
        # 像素中心坐标
        xx = x_min + (np.arange(W) + 0.5) * dx
        yy = y_max - (np.arange(H) + 0.5) * dy  # y方向向下
        
        print(f"像素坐标范围: x=[{xx[0]:.3f}, {xx[-1]:.3f}], y=[{yy[-1]:.3f}, {yy[0]:.3f}]")
        
        # 对每个三角形
        for face_idx, (a, b, c) in enumerate(faces):
            xa, ya, za = vertices[a]
            xb, yb, zb = vertices[b]
            xc, yc, zc = vertices[c]
            
            print(f"处理三角形{face_idx}: 顶点({xa:.1f},{ya:.1f},{za:.1f}), ({xb:.1f},{yb:.1f},{zb:.1f}), ({xc:.1f},{yc:.1f},{zc:.1f})")
            
            # 对每个像素
            covered_pixels = 0
            for j in range(H):
                for i in range(W):
                    px, py = xx[i], yy[j]
                    
                    # 重心坐标计算
                    def crossz(x1, y1, x2, y2):
                        return x1 * y2 - x2 * y1
                    
                    area = crossz(xb - xa, yb - ya, xc - xa, yc - ya)
                    if abs(area) < 1e-12:
                        continue
                    
                    w0 = crossz(xb - px, yb - py, xc - px, yc - py) / area
                    w1 = crossz(xc - px, yc - py, xa - px, ya - py) / area
                    w2 = 1.0 - w0 - w1
                    
                    # 检查是否在三角形内
                    if w0 >= 0 and w1 >= 0 and w2 >= 0:
                        z_val = w0 * za + w1 * zb + w2 * zc
                        
                        # 如果这个三角形更高，就更新
                        if np.isnan(Z[j, i]) or z_val > Z[j, i]:
                            Z[j, i] = z_val
                            M[j, i] = True
                            covered_pixels += 1
            
            print(f"  三角形{face_idx}覆盖了{covered_pixels}个像素")
        
        return Z.T, M.T  # 转置回(W,H)格式
    
    height_map, mask = simple_triangle_fill(vertices, faces, grid_size, x_min, x_max, y_min, y_max)
    
    print(f"结果形状: height_map={height_map.shape}, mask={mask.shape}")
    print(f"掩码覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
    
    if not np.all(np.isnan(height_map)):
        valid_heights = height_map[~np.isnan(height_map)]
        print(f"高度范围: [{np.min(valid_heights):.3f}, {np.max(valid_heights):.3f}]")
    
    # 可视化结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 掩码
    axes[0].imshow(mask.T, cmap='gray', origin='lower', aspect='equal')
    axes[0].set_title('掩码')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # 2. 高度图
    masked_height = np.where(mask, height_map, np.nan)
    im = axes[1].imshow(masked_height.T, cmap='terrain', origin='lower', aspect='equal')
    axes[1].set_title('高度图')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im, ax=axes[1])
    
    # 3. 原始三角形
    axes[2].scatter(vertices[:, 0], vertices[:, 1], c=vertices[:, 2], 
                   cmap='terrain', s=100, edgecolors='black')
    axes[2].set_title('原始三角形')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(True)
    
    # 添加三角形边界
    triangle_x = [vertices[0, 0], vertices[1, 0], vertices[2, 0], vertices[0, 0]]
    triangle_y = [vertices[0, 1], vertices[1, 1], vertices[2, 1], vertices[0, 1]]
    axes[2].plot(triangle_x, triangle_y, 'r--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("simple_triangle_test.png", dpi=300, bbox_inches='tight')
    print("✅ 结果已保存到: simple_triangle_test.png")
    plt.show()

if __name__ == "__main__":
    test_simple_triangle()
