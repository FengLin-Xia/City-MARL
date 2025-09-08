#!/usr/bin/env python3
"""
精确mesh边界地形处理器
使用实际的mesh边界而不是凸包来创建掩码
"""

import numpy as np
import json
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Set

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ExactMeshProcessor:
    """精确mesh边界处理器"""
    
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.height_map = None
        self.mask = None
        self.boundary_vertices = None
        
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
    
    def extract_mesh_boundary(self) -> List[Tuple[float, float]]:
        """提取mesh的实际边界"""
        if self.vertices is None or self.faces is None:
            return None
        
        print("🔄 提取mesh边界...")
        
        # 1. 找到所有边
        edges = {}  # (v1, v2) -> count
        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i+1)%3]
                # 确保边的方向一致
                edge = tuple(sorted([v1, v2]))
                edges[edge] = edges.get(edge, 0) + 1
        
        # 2. 找到边界边（只出现一次的边）
        boundary_edges = []
        for edge, count in edges.items():
            if count == 1:  # 边界边
                boundary_edges.append(edge)
        
        print(f"   找到 {len(boundary_edges)} 条边界边")
        
        # 3. 构建边界多边形
        if not boundary_edges:
            print("❌ 没有找到边界边")
            return None
        
        # 连接边界边形成多边形
        boundary_vertices = self._connect_boundary_edges(boundary_edges)
        
        if boundary_vertices:
            print(f"   边界多边形顶点数: {len(boundary_vertices)}")
            self.boundary_vertices = boundary_vertices
            return boundary_vertices
        else:
            print("❌ 无法构建边界多边形")
            return None
    
    def _connect_boundary_edges(self, boundary_edges: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """连接边界边形成多边形"""
        if not boundary_edges:
            return None
        
        # 创建边的邻接表
        adjacency = {}
        for v1, v2 in boundary_edges:
            if v1 not in adjacency:
                adjacency[v1] = []
            if v2 not in adjacency:
                adjacency[v2] = []
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
        
        # 找到起始点（度数最小的点）
        start_vertex = min(adjacency.keys(), key=lambda v: len(adjacency[v]))
        
        # 构建多边形路径
        path = [start_vertex]
        current = start_vertex
        visited = set([start_vertex])
        
        while len(path) < len(boundary_edges):
            # 找到下一个未访问的邻居
            next_vertex = None
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    next_vertex = neighbor
                    break
            
            if next_vertex is None:
                # 如果找不到未访问的邻居，尝试闭合多边形
                for neighbor in adjacency[current]:
                    if neighbor == start_vertex and len(path) > 2:
                        next_vertex = neighbor
                        break
                break
            
            path.append(next_vertex)
            visited.add(next_vertex)
            current = next_vertex
        
        # 转换为2D坐标
        boundary_coords = []
        for vertex_idx in path:
            vertex = self.vertices[vertex_idx]
            boundary_coords.append((vertex[0], vertex[1]))
        
        return boundary_coords
    
    def create_exact_mask(self, grid_size: Tuple[int, int]) -> np.ndarray:
        """使用精确mesh边界创建掩码"""
        if self.boundary_vertices is None:
            print("❌ 没有边界顶点数据")
            return None
        
        # 计算边界范围
        x_coords = [v[0] for v in self.boundary_vertices]
        y_coords = [v[1] for v in self.boundary_vertices]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        grid_x, grid_y = grid_size
        
        # 创建网格坐标
        x_coords_grid = np.linspace(x_min, x_max, grid_x)
        y_coords_grid = np.linspace(y_min, y_max, grid_y)
        X, Y = np.meshgrid(x_coords_grid, y_coords_grid, indexing='ij')
        
        # 创建掩码（初始为False）
        mask = np.zeros((grid_x, grid_y), dtype=bool)
        
        print("🔄 创建精确mesh掩码...")
        
        # 对每个网格点判断是否在边界多边形内
        boundary_array = np.array(self.boundary_vertices)
        
        for i in range(grid_x):
            for j in range(grid_y):
                point = np.array([X[i, j], Y[i, j]])
                
                # 使用射线法判断点是否在多边形内
                if self._point_in_polygon(point, boundary_array):
                    mask[i, j] = True
        
        print(f"✅ 精确mesh掩码完成")
        print(f"   有效网格点数: {np.sum(mask)} / {mask.size}")
        print(f"   覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
        
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
    
    def interpolate_height_map(self, grid_size: Tuple[int, int], mask: np.ndarray) -> np.ndarray:
        """在有效区域内插值高程图"""
        if self.boundary_vertices is None:
            return None
        
        # 计算边界范围
        x_coords = [v[0] for v in self.boundary_vertices]
        y_coords = [v[1] for v in self.boundary_vertices]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        grid_x, grid_y = grid_size
        
        # 提取顶点数据
        vertices_x = self.vertices[:, 0]
        vertices_y = self.vertices[:, 1]
        heights = self.vertices[:, 2]
        
        # 创建网格坐标
        grid_x_coords = np.linspace(x_min, x_max, grid_x)
        grid_y_coords = np.linspace(y_min, y_max, grid_y)
        X, Y = np.meshgrid(grid_x_coords, grid_y_coords, indexing='ij')
        
        # 准备插值点
        points = np.column_stack((vertices_x, vertices_y))
        values = heights
        
        # 执行插值
        print("🔄 执行高程插值...")
        height_map = griddata(points, values, (X, Y), method='linear', fill_value=np.nan)
        
        # 处理NaN值
        if np.any(np.isnan(height_map)):
            print(f"   发现 {np.sum(np.isnan(height_map))} 个NaN值，使用最近邻插值填充")
            height_map_nn = griddata(points, values, (X, Y), method='nearest')
            height_map = np.where(np.isnan(height_map), height_map_nn, height_map)
        
        # 应用掩码：将无效区域设为NaN
        height_map = np.where(mask, height_map, np.nan)
        
        print(f"✅ 高程插值完成")
        print(f"   有效高程点数: {np.sum(~np.isnan(height_map))}")
        print(f"   高程范围: {np.nanmin(height_map):.3f} ~ {np.nanmax(height_map):.3f}")
        
        return height_map
    
    def process_terrain(self, obj_filepath: str, grid_size: Tuple[int, int] = (150, 150)) -> Dict:
        """处理地形数据"""
        print("🚀 开始精确mesh边界地形处理")
        print("=" * 50)
        
        # 1. 加载OBJ文件
        if not self.load_obj_file(obj_filepath):
            return None
        
        # 2. 提取mesh边界
        boundary_vertices = self.extract_mesh_boundary()
        if boundary_vertices is None:
            return None
        
        # 3. 创建精确掩码
        mask = self.create_exact_mask(grid_size)
        if mask is None:
            return None
        
        # 4. 插值高程图
        height_map = self.interpolate_height_map(grid_size, mask)
        if height_map is None:
            return None
        
        # 5. 保存结果
        self.height_map = height_map
        self.mask = mask
        
        # 6. 创建输出数据
        result = {
            'height_map': height_map.tolist(),
            'mask': mask.tolist(),
            'grid_size': grid_size,
            'vertices_count': len(self.vertices),
            'faces_count': len(self.faces),
            'boundary_vertices': boundary_vertices,
            'valid_points_count': int(np.sum(mask)),
            'coverage_percentage': float(np.sum(mask)/mask.size*100)
        }
        
        print("\n✅ 地形处理完成!")
        print(f"   网格尺寸: {grid_size}")
        print(f"   有效点数: {result['valid_points_count']}")
        print(f"   覆盖率: {result['coverage_percentage']:.1f}%")
        
        return result
    
    def visualize_result(self, save_path: Optional[str] = None):
        """可视化结果"""
        if self.height_map is None or self.mask is None:
            print("❌ 没有处理结果可可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('精确Mesh边界地形处理结果', fontsize=16)
        
        # 1. 原始mesh顶点分布 + 边界
        if self.vertices is not None:
            x_coords = self.vertices[:, 0]
            y_coords = self.vertices[:, 1]
            
            axes[0, 0].scatter(x_coords, y_coords, s=1, alpha=0.5, c='blue')
            
            # 绘制边界
            if self.boundary_vertices:
                boundary_x = [v[0] for v in self.boundary_vertices]
                boundary_y = [v[1] for v in self.boundary_vertices]
                # 闭合边界
                boundary_x.append(boundary_x[0])
                boundary_y.append(boundary_y[0])
                axes[0, 0].plot(boundary_x, boundary_y, 'r-', linewidth=2, label='Mesh边界')
            
            axes[0, 0].set_title('原始Mesh顶点分布与边界')
            axes[0, 0].set_xlabel('X坐标')
            axes[0, 0].set_ylabel('Y坐标')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_aspect('equal')
            axes[0, 0].legend()
        
        # 2. 精确掩码
        im1 = axes[0, 1].imshow(self.mask.T, cmap='gray', aspect='auto', origin='lower')
        axes[0, 1].set_title('精确Mesh掩码')
        axes[0, 1].set_xlabel('X坐标')
        axes[0, 1].set_ylabel('Y坐标')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 3. 高程图（只显示有效区域）
        valid_height_map = np.where(self.mask, self.height_map, np.nan)
        im2 = axes[1, 0].imshow(valid_height_map.T, cmap='terrain', aspect='auto', origin='lower')
        axes[1, 0].set_title('高程图（精确边界）')
        axes[1, 0].set_xlabel('X坐标')
        axes[1, 0].set_ylabel('Y坐标')
        plt.colorbar(im2, ax=axes[1, 0], label='高程')
        
        # 4. 边界对比
        axes[1, 1].imshow(self.height_map.T, cmap='terrain', aspect='auto', origin='lower')
        axes[1, 1].set_title('完整高程图（包含无效区域）')
        axes[1, 1].set_xlabel('X坐标')
        axes[1, 1].set_ylabel('Y坐标')
        
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
    processor = ExactMeshProcessor()
    
    # 处理地形
    result = processor.process_terrain(obj_filepath, grid_size=(150, 150))
    
    if result is None:
        print("❌ 地形处理失败")
        return
    
    # 保存结果
    output_dir = "data/terrain"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "terrain_exact_mesh.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ 处理结果已保存到: {output_file}")
    
    # 可视化结果
    processor.visualize_result(
        save_path="visualization_output/exact_mesh_terrain.png"
    )


if __name__ == "__main__":
    main()
