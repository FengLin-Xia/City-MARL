#!/usr/bin/env python3
"""
严格Mesh边界处理器
严格按照mesh的实际边界创建掩码，确保外部区域被完全排除
"""

import numpy as np
import json
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from scipy.spatial import ConvexHull
import alphashape

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class StrictMeshProcessor:
    """严格Mesh边界处理器"""
    
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.height_map = None
        self.mask = None
        self.boundary_points = None
        self.mesh_bounds = None
        
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
                            # 处理多边形面，只取前三个顶点
                            face = []
                            for part in parts[:3]:  # 只取前3个顶点
                                vertex_part = part.split('/')[0]  # 只取顶点索引
                                try:
                                    vertex_idx = int(vertex_part) - 1  # OBJ索引从1开始
                                    face.append(vertex_idx)
                                except ValueError:
                                    continue
                            
                            if len(face) == 3:  # 确保有3个有效顶点
                                faces.append(face)
            
            if not vertices:
                print("❌ 没有找到顶点数据")
                return False
            
            self.vertices = np.array(vertices)
            self.faces = np.array(faces)
            
            print(f"✅ 成功加载OBJ文件")
            print(f"   顶点数: {len(vertices)}")
            print(f"   面数: {len(faces)}")
            print(f"   顶点范围: X[{np.min(vertices[:, 0]):.2f}, {np.max(vertices[:, 0]):.2f}], Y[{np.min(vertices[:, 1]):.2f}, {np.max(vertices[:, 1]):.2f}]")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载OBJ文件失败: {e}")
            return False
    
    def extract_strict_boundary(self) -> List[Tuple[float, float]]:
        """严格提取mesh边界点"""
        if self.vertices is None:
            return None
        
        print("🔄 严格提取mesh边界点...")
        
        # 找到边界边（只属于一个面的边）
        edge_count = {}
        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i+1) % 3]
                edge = tuple(sorted([v1, v2]))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        print(f"   找到 {len(boundary_edges)} 条边界边")
        
        # 收集所有边界顶点
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)
        
        # 转换为坐标
        boundary_points = []
        for vertex_idx in boundary_vertices:
            vertex = self.vertices[vertex_idx]
            boundary_points.append((vertex[0], vertex[1]))
        
        print(f"   边界顶点数: {len(boundary_points)}")
        
        # 记录mesh的实际边界
        x_coords = [p[0] for p in boundary_points]
        y_coords = [p[1] for p in boundary_points]
        self.mesh_bounds = {
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords)
        }
        
        print(f"   Mesh实际边界: X[{self.mesh_bounds['x_min']:.2f}, {self.mesh_bounds['x_max']:.2f}], Y[{self.mesh_bounds['y_min']:.2f}, {self.mesh_bounds['y_max']:.2f}]")
        
        self.boundary_points = boundary_points
        return boundary_points
    
    def create_strict_mask(self, grid_size: Tuple[int, int]) -> np.ndarray:
        """创建严格的mesh边界掩码"""
        if self.boundary_points is None or self.mesh_bounds is None:
            print("❌ 没有边界数据")
            return None
        
        grid_x, grid_y = grid_size
        
        print("🔄 创建严格mesh边界掩码...")
        print(f"   网格尺寸: {grid_x} x {grid_y}")
        print(f"   Mesh边界: X[{self.mesh_bounds['x_min']:.2f}, {self.mesh_bounds['x_max']:.2f}], Y[{self.mesh_bounds['y_min']:.2f}, {self.mesh_bounds['y_max']:.2f}]")
        
        # 创建网格坐标（严格限制在mesh边界内）
        x_coords_grid = np.linspace(self.mesh_bounds['x_min'], self.mesh_bounds['x_max'], grid_x)
        y_coords_grid = np.linspace(self.mesh_bounds['y_min'], self.mesh_bounds['y_max'], grid_y)
        X, Y = np.meshgrid(x_coords_grid, y_coords_grid, indexing='ij')
        
        # 创建掩码（初始为False）
        mask = np.zeros((grid_x, grid_y), dtype=bool)
        
        # 使用Alpha Shape创建更精确的边界
        try:
            print("   使用Alpha Shape创建精确边界...")
            boundary_array = np.array(self.boundary_points)
            
            # 计算合适的alpha值
            alpha = 0.1  # 可以调整这个值
            alpha_shape = alphashape.alphashape(boundary_array, alpha=alpha)
            
            if alpha_shape.is_empty:
                print("   Alpha Shape为空，使用凸包...")
                hull = ConvexHull(boundary_array)
                hull_points = boundary_array[hull.vertices]
                
                # 对每个网格点判断是否在凸包内
                for i in range(grid_x):
                    for j in range(grid_y):
                        point = np.array([X[i, j], Y[i, j]])
                        if self._point_in_polygon(point, hull_points):
                            mask[i, j] = True
            else:
                print("   使用Alpha Shape边界...")
                # 获取Alpha Shape的边界坐标
                if hasattr(alpha_shape, 'exterior'):
                    boundary_coords = np.array(alpha_shape.exterior.coords)
                    
                    # 对每个网格点判断是否在Alpha Shape内
                    for i in range(grid_x):
                        for j in range(grid_y):
                            point = np.array([X[i, j], Y[i, j]])
                            if self._point_in_polygon(point, boundary_coords):
                                mask[i, j] = True
                else:
                    print("   Alpha Shape没有外部边界，使用凸包...")
                    hull = ConvexHull(boundary_array)
                    hull_points = boundary_array[hull.vertices]
                    
                    for i in range(grid_x):
                        for j in range(grid_y):
                            point = np.array([X[i, j], Y[i, j]])
                            if self._point_in_polygon(point, hull_points):
                                mask[i, j] = True
                                
        except Exception as e:
            print(f"   Alpha Shape计算失败: {e}")
            print("   使用凸包作为备选方案...")
            boundary_array = np.array(self.boundary_points)
            hull = ConvexHull(boundary_array)
            hull_points = boundary_array[hull.vertices]
            
            for i in range(grid_x):
                for j in range(grid_y):
                    point = np.array([X[i, j], Y[i, j]])
                    if self._point_in_polygon(point, hull_points):
                        mask[i, j] = True
        
        print(f"✅ 严格边界掩码完成")
        print(f"   有效网格点数: {np.sum(mask)} / {mask.size}")
        print(f"   覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
        
        self.mask = mask
        return mask
    
    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """判断点是否在多边形内（射线法）"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def interpolate_strict_height_map(self, grid_size: Tuple[int, int]) -> np.ndarray:
        """在严格边界内插值高程图"""
        if self.vertices is None or self.mesh_bounds is None:
            return None
        
        grid_x, grid_y = grid_size
        
        print("🔄 在严格边界内插值高程图...")
        
        # 提取顶点数据
        vertices_x = self.vertices[:, 0]
        vertices_y = self.vertices[:, 1]
        heights = self.vertices[:, 2]
        
        # 创建网格坐标（严格限制在mesh边界内）
        x_coords_grid = np.linspace(self.mesh_bounds['x_min'], self.mesh_bounds['x_max'], grid_x)
        y_coords_grid = np.linspace(self.mesh_bounds['y_min'], self.mesh_bounds['y_max'], grid_y)
        X, Y = np.meshgrid(x_coords_grid, y_coords_grid, indexing='ij')
        
        # 准备插值点
        points = np.column_stack((vertices_x, vertices_y))
        xi = np.column_stack((X.flatten(), Y.flatten()))
        
        # 执行插值
        print("   执行插值...")
        interpolated_heights = griddata(points, heights, xi, method='linear', fill_value=0.0)
        
        # 重塑为网格
        height_map = interpolated_heights.reshape((grid_x, grid_y))
        
        # 应用严格掩码
        if self.mask is not None:
            print("   应用严格掩码...")
            height_map = np.where(self.mask, height_map, 0.0)
        
        print(f"✅ 严格高程图完成")
        print(f"   高程范围: [{np.min(height_map):.3f}, {np.max(height_map):.3f}]")
        print(f"   平均高程: {np.mean(height_map):.3f}")
        
        self.height_map = height_map
        return height_map
    
    def process_terrain(self, obj_filepath: str, grid_size: Tuple[int, int] = (150, 150)) -> Dict:
        """处理地形数据"""
        print("🚀 开始严格地形处理...")
        
        # 1. 加载OBJ文件
        if not self.load_obj_file(obj_filepath):
            return None
        
        # 2. 提取严格边界
        boundary_points = self.extract_strict_boundary()
        if boundary_points is None:
            return None
        
        # 3. 创建严格掩码
        mask = self.create_strict_mask(grid_size)
        if mask is None:
            return None
        
        # 4. 插值高程图
        height_map = self.interpolate_strict_height_map(grid_size)
        if height_map is None:
            return None
        
        # 5. 准备结果数据
        result = {
            'height_map': height_map.tolist(),
            'mask': mask.tolist(),
            'boundary_points': boundary_points,
            'mesh_bounds': self.mesh_bounds,
            'grid_size': grid_size,
            'valid_points_count': int(np.sum(mask)),
            'coverage_percentage': float(np.sum(mask)/mask.size*100),
            'height_stats': {
                'min': float(np.min(height_map)),
                'max': float(np.max(height_map)),
                'mean': float(np.mean(height_map)),
                'std': float(np.std(height_map))
            }
        }
        
        print("✅ 严格地形处理完成")
        return result
    
    def visualize_result(self, save_path: str = None):
        """可视化处理结果"""
        if self.height_map is None or self.mask is None:
            print("❌ 没有数据可可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('严格Mesh边界处理结果', fontsize=16)
        
        # 1. 原始mesh边界
        if self.boundary_points:
            boundary_array = np.array(self.boundary_points)
            axes[0, 0].scatter(boundary_array[:, 0], boundary_array[:, 1], c='red', s=1, alpha=0.6, label='边界点')
            axes[0, 0].set_title('原始Mesh边界点')
            axes[0, 0].set_xlabel('X坐标')
            axes[0, 0].set_ylabel('Y坐标')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 严格掩码
        axes[0, 1].imshow(self.mask.T, cmap='gray', aspect='auto', origin='lower')
        axes[0, 1].set_title('严格边界掩码')
        axes[0, 1].set_xlabel('X坐标')
        axes[0, 1].set_ylabel('Y坐标')
        
        # 3. 严格高程图
        valid_height_map = np.where(self.mask, self.height_map, np.nan)
        im3 = axes[1, 0].imshow(valid_height_map.T, cmap='terrain', aspect='auto', origin='lower')
        axes[1, 0].set_title('严格边界高程图')
        axes[1, 0].set_xlabel('X坐标')
        axes[1, 0].set_ylabel('Y坐标')
        plt.colorbar(im3, ax=axes[1, 0], label='高程')
        
        # 4. 统计信息
        axes[1, 1].axis('off')
        
        info_text = f"""
严格处理结果:

网格尺寸: {self.height_map.shape[0]} x {self.height_map.shape[1]}
有效点数: {np.sum(self.mask)} / {self.mask.size}
覆盖率: {np.sum(self.mask)/self.mask.size*100:.1f}%

Mesh边界:
  X: [{self.mesh_bounds['x_min']:.2f}, {self.mesh_bounds['x_max']:.2f}]
  Y: [{self.mesh_bounds['y_min']:.2f}, {self.mesh_bounds['y_max']:.2f}]

高程统计:
  最小值: {np.min(self.height_map):.3f}
  最大值: {np.max(self.height_map):.3f}
  平均值: {np.mean(self.height_map):.3f}
  标准差: {np.std(self.height_map):.3f}

边界点: {len(self.boundary_points)} 个
        """
        
        axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    processor = StrictMeshProcessor()
    
    # 处理地形
    obj_file = "data/terrain/terrain.obj"
    result = processor.process_terrain(obj_file, grid_size=(150, 150))
    
    if result:
        # 保存结果
        output_file = "data/terrain/terrain_strict_mesh.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ 结果已保存到: {output_file}")
        
        # 可视化结果
        processor.visualize_result("visualization_output/strict_mesh_result.png")


if __name__ == "__main__":
    main()
