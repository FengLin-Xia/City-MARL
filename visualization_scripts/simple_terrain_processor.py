#!/usr/bin/env python3
"""
简单直接的地形处理器
避免复杂的边界处理，直接使用mesh顶点进行插值
"""

import numpy as np
import json
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SimpleTerrainProcessor:
    """简单地形处理器"""
    
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.height_map = None
        
    def load_obj_file(self, obj_filepath: str) -> bool:
        """加载OBJ文件"""
        try:
            vertices = []
            faces = []
            
            with open(obj_filepath, 'r') as f:
                for line in f:
                    if line.startswith('v '):  # 顶点
                        parts = line.strip().split()[1:]
                        if len(parts) >= 3:
                            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    elif line.startswith('f '):  # 面
                        parts = line.strip().split()[1:]
                        if len(parts) >= 3:
                            # 只取前3个顶点，确保所有面都是三角形
                            face = [int(part.split('/')[0]) - 1 for part in parts[:3]]
                            faces.append(face)
            
            if not vertices:
                print("❌ 没有找到顶点数据")
                return False
            
            self.vertices = np.array(vertices)
            self.faces = np.array(faces)
            
            print(f"✅ 成功加载OBJ文件")
            print(f"   顶点数: {len(vertices)}")
            print(f"   面数: {len(faces)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载OBJ文件失败: {e}")
            return False
    
    def analyze_mesh(self) -> Dict:
        """分析mesh特征"""
        if self.vertices is None:
            return None
        
        # 提取坐标
        x_coords = self.vertices[:, 0]
        y_coords = self.vertices[:, 1]
        heights = self.vertices[:, 2]
        
        # 计算边界
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = heights.min(), heights.max()
        
        # 计算统计信息
        x_span = x_max - x_min
        y_span = y_max - y_min
        aspect_ratio = x_span / y_span
        
        analysis = {
            'x_min': float(x_min),
            'x_max': float(x_max),
            'y_min': float(y_min),
            'y_max': float(y_max),
            'z_min': float(z_min),
            'z_max': float(z_max),
            'x_span': float(x_span),
            'y_span': float(y_span),
            'aspect_ratio': float(aspect_ratio),
            'vertex_density': len(self.vertices) / (x_span * y_span)
        }
        
        print(f"📊 Mesh分析:")
        print(f"   X范围: {x_min:.3f} ~ {x_max:.3f} (跨度: {x_span:.3f})")
        print(f"   Y范围: {y_min:.3f} ~ {y_max:.3f} (跨度: {y_span:.3f})")
        print(f"   Z范围: {z_min:.3f} ~ {z_max:.3f}")
        print(f"   宽高比: {aspect_ratio:.3f}")
        print(f"   顶点密度: {analysis['vertex_density']:.1f} 顶点/单位面积")
        
        return analysis
    
    def create_height_map(self, grid_size: Tuple[int, int] = (150, 150)) -> np.ndarray:
        """创建高程图"""
        if self.vertices is None:
            return None
        
        # 分析mesh
        analysis = self.analyze_mesh()
        if analysis is None:
            return None
        
        # 根据宽高比确定网格尺寸
        aspect_ratio = analysis['aspect_ratio']
        if aspect_ratio > 1:  # 宽大于高
            grid_x = grid_size[0]
            grid_y = int(grid_size[0] / aspect_ratio)
        else:  # 高大于宽
            grid_y = grid_size[1]
            grid_x = int(grid_size[1] * aspect_ratio)
        
        actual_grid_size = (grid_x, grid_y)
        print(f"📐 实际网格尺寸: {actual_grid_size}")
        
        # 提取顶点数据
        x_coords = self.vertices[:, 0]
        y_coords = self.vertices[:, 1]
        heights = self.vertices[:, 2]
        
        # 创建网格坐标
        grid_x_coords = np.linspace(analysis['x_min'], analysis['x_max'], grid_x)
        grid_y_coords = np.linspace(analysis['y_min'], analysis['y_max'], grid_y)
        X, Y = np.meshgrid(grid_x_coords, grid_y_coords, indexing='ij')
        
        # 准备插值点
        points = np.column_stack((x_coords, y_coords))
        values = heights
        
        # 执行插值
        print("🔄 执行高程插值...")
        height_map = griddata(points, values, (X, Y), method='linear', fill_value=np.nan)
        
        # 处理NaN值
        nan_count = np.sum(np.isnan(height_map))
        if nan_count > 0:
            print(f"   发现 {nan_count} 个NaN值，使用最近邻插值填充")
            height_map_nn = griddata(points, values, (X, Y), method='nearest')
            height_map = np.where(np.isnan(height_map), height_map_nn, height_map)
        
        # 确保数据类型
        height_map = height_map.astype(np.float32)
        
        print(f"✅ 高程插值完成")
        print(f"   网格尺寸: {height_map.shape}")
        print(f"   高程范围: {np.min(height_map):.3f} ~ {np.max(height_map):.3f}")
        
        return height_map
    
    def process_terrain(self, obj_filepath: str, grid_size: Tuple[int, int] = (150, 150)) -> Dict:
        """处理地形数据"""
        print("🚀 开始简单地形处理")
        print("=" * 50)
        
        # 1. 加载OBJ文件
        if not self.load_obj_file(obj_filepath):
            return None
        
        # 2. 创建高程图
        height_map = self.create_height_map(grid_size)
        if height_map is None:
            return None
        
        # 3. 保存结果
        self.height_map = height_map
        
        # 4. 创建输出数据
        analysis = self.analyze_mesh()
        result = {
            'height_map': height_map.tolist(),
            'grid_size': height_map.shape,
            'vertices_count': len(self.vertices),
            'faces_count': len(self.faces),
            'mesh_analysis': analysis,
            'valid_points_count': int(height_map.size),
            'coverage_percentage': 100.0
        }
        
        print("\n✅ 地形处理完成!")
        print(f"   网格尺寸: {height_map.shape}")
        print(f"   总点数: {height_map.size}")
        print(f"   覆盖率: 100%")
        
        return result
    
    def visualize_result(self, save_path: Optional[str] = None):
        """可视化结果"""
        if self.height_map is None:
            print("❌ 没有处理结果可可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('简单地形处理结果', fontsize=16)
        
        # 1. 原始mesh顶点分布
        if self.vertices is not None:
            x_coords = self.vertices[:, 0]
            y_coords = self.vertices[:, 1]
            
            axes[0, 0].scatter(x_coords, y_coords, s=1, alpha=0.5, c='blue')
            axes[0, 0].set_title('原始Mesh顶点分布')
            axes[0, 0].set_xlabel('X坐标')
            axes[0, 0].set_ylabel('Y坐标')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_aspect('equal')
        
        # 2. 高程图
        im1 = axes[0, 1].imshow(self.height_map.T, cmap='terrain', aspect='auto', origin='lower')
        axes[0, 1].set_title('高程图')
        axes[0, 1].set_xlabel('X坐标')
        axes[0, 1].set_ylabel('Y坐标')
        plt.colorbar(im1, ax=axes[0, 1], label='高程')
        
        # 3. 高程分布直方图
        axes[1, 0].hist(self.height_map.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('高程分布')
        axes[1, 0].set_xlabel('高程')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 3D视图
        from mpl_toolkits.mplot3d import Axes3D
        ax3d = fig.add_subplot(2, 2, 4, projection='3d')
        
        # 创建网格
        grid_x, grid_y = self.height_map.shape
        x = np.arange(grid_x)
        y = np.arange(grid_y)
        X, Y = np.meshgrid(x, y)
        
        # 绘制3D表面 - 确保维度匹配
        surf = ax3d.plot_surface(X, Y, self.height_map.T, cmap='terrain', 
                                linewidth=0, antialiased=True)
        ax3d.set_title('3D地形视图')
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('高程')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"处理结果图已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    # 检查是否有OBJ文件
    obj_filepath = "uploads/terrain.obj"
    
    if not os.path.exists(obj_filepath):
        print(f"❌ OBJ文件不存在: {obj_filepath}")
        print("请先从Blender导出地形OBJ文件")
        return
    
    # 创建处理器
    processor = SimpleTerrainProcessor()
    
    # 处理地形
    result = processor.process_terrain(obj_filepath, grid_size=(150, 150))
    
    if result is None:
        print("❌ 地形处理失败")
        return
    
    # 保存结果
    output_dir = "data/terrain"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "terrain_simple.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ 处理结果已保存到: {output_file}")
    
    # 可视化结果
    processor.visualize_result(
        save_path="visualization_output/simple_terrain.png"
    )


if __name__ == "__main__":
    main()
