#!/usr/bin/env python3
"""
简化的Mesh处理器
"""

import numpy as np
import json
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from scipy.spatial import ConvexHull

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SimpleMeshProcessor:
    """简化的Mesh处理器"""
    
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
            
            print("🔍 开始加载OBJ文件...")
            
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
            
            print(f"✅ 成功加载OBJ文件")
            print(f"   顶点数: {len(vertices)}")
            print(f"   面数: {len(faces)}")
            
            # 转换为numpy数组
            self.vertices = np.array(vertices)
            self.faces = np.array(faces)
            
            # 检查数组形状
            print(f"   顶点数组形状: {self.vertices.shape}")
            print(f"   面数组形状: {self.faces.shape}")
            
            # 安全地计算范围
            try:
                x_min, x_max = np.min(self.vertices[:, 0]), np.max(self.vertices[:, 0])
                y_min, y_max = np.min(self.vertices[:, 1]), np.max(self.vertices[:, 1])
                print(f"   顶点范围: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")
            except Exception as e:
                print(f"   计算顶点范围时出错: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载OBJ文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_simple_boundary(self) -> List[Tuple[float, float]]:
        """提取简单的mesh边界点"""
        if self.vertices is None or self.faces is None:
            print("❌ 没有mesh数据")
            return None
        
        print("🔄 提取简单边界点...")
        
        try:
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
                if vertex_idx < len(self.vertices):
                    vertex = self.vertices[vertex_idx]
                    boundary_points.append((vertex[0], vertex[1]))
            
            print(f"   边界顶点数: {len(boundary_points)}")
            
            # 记录mesh的实际边界
            if boundary_points:
                x_coords = [p[0] for p in boundary_points]
                y_coords = [p[1] for p in boundary_points]
                self.mesh_bounds = {
                    'x_min': min(x_coords),
                    'x_max': max(x_coords),
                    'y_min': min(y_coords),
                    'y_max': max(y_coords)
                }
                
                print(f"   Mesh边界: X[{self.mesh_bounds['x_min']:.2f}, {self.mesh_bounds['x_max']:.2f}], Y[{self.mesh_bounds['y_min']:.2f}, {self.mesh_bounds['y_max']:.2f}]")
            
            self.boundary_points = boundary_points
            return boundary_points
            
        except Exception as e:
            print(f"❌ 提取边界时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_simple_mask(self, grid_size: Tuple[int, int]) -> np.ndarray:
        """创建严格的mesh边界掩码（像素填充法）"""
        if self.boundary_points is None or self.mesh_bounds is None:
            print("❌ 没有边界数据")
            return None
        
        grid_x, grid_y = grid_size
        
        print("🔄 创建严格边界掩码（像素填充法）...")
        print(f"   目标网格尺寸: {grid_x} x {grid_y}")
        print(f"   原始边界点数: {len(self.boundary_points)}")
        
        try:
            # 1. 创建高分辨率掩码（比如3000x3000）
            high_res_size = 3000
            print(f"   创建高分辨率掩码: {high_res_size} x {high_res_size}")
            
            # 计算边界点的范围
            boundary_array = np.array(self.boundary_points)
            x_min, x_max = np.min(boundary_array[:, 0]), np.max(boundary_array[:, 0])
            y_min, y_max = np.min(boundary_array[:, 1]), np.max(boundary_array[:, 1])
            
            # 创建高分辨率掩码
            high_res_mask = np.zeros((high_res_size, high_res_size), dtype=bool)
            
            # 2. 将边界点映射到高分辨率像素
            print("   映射边界点到像素...")
            for point in self.boundary_points:
                x, y = point
                # 映射到高分辨率坐标
                pixel_x = int((x - x_min) / (x_max - x_min) * (high_res_size - 1))
                pixel_y = int((y - y_min) / (y_max - y_min) * (high_res_size - 1))
                
                # 确保在范围内
                pixel_x = max(0, min(high_res_size - 1, pixel_x))
                pixel_y = max(0, min(high_res_size - 1, pixel_y))
                
                # 标记边界像素
                high_res_mask[pixel_x, pixel_y] = True
            
            # 3. 使用形态学操作填充内部
            print("   填充内部区域...")
            from scipy import ndimage
            
            # 使用形态学膨胀连接边界
            kernel = np.ones((3, 3), dtype=bool)
            dilated = ndimage.binary_dilation(high_res_mask, structure=kernel, iterations=2)
            
            # 使用flood fill填充内部
            filled = ndimage.binary_fill_holes(dilated)
            
            # 4. 降采样到目标分辨率
            print("   降采样到目标分辨率...")
            # 计算缩放因子
            scale_x = high_res_size / grid_x
            scale_y = high_res_size / grid_y
            
            # 创建目标掩码
            mask = np.zeros((grid_x, grid_y), dtype=bool)
            
            # 降采样：如果高分辨率中大部分像素是True，则目标像素为True
            for i in range(grid_x):
                for j in range(grid_y):
                    # 计算高分辨率中的对应区域
                    start_x = int(i * scale_x)
                    end_x = int((i + 1) * scale_x)
                    start_y = int(j * scale_y)
                    end_y = int((j + 1) * scale_y)
                    
                    # 检查该区域中True像素的比例
                    region = filled[start_x:end_x, start_y:end_y]
                    if np.sum(region) > region.size * 0.5:  # 超过50%为True
                        mask[i, j] = True
            
            print(f"✅ 严格边界掩码完成")
            print(f"   有效网格点数: {np.sum(mask)} / {mask.size}")
            print(f"   覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
            
            self.mask = mask
            return mask
            
        except Exception as e:
            print(f"❌ 创建掩码时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _simplify_boundary_points(self, boundary_points: List[Tuple[float, float]], max_points: int = 1000) -> List[Tuple[float, float]]:
        """简化边界点，保留轮廓形状"""
        if len(boundary_points) <= max_points:
            return boundary_points
        
        # 使用Douglas-Peucker算法简化
        def perpendicular_distance(point, line_start, line_end):
            """计算点到线段的垂直距离"""
            x, y = point
            x1, y1 = line_start
            x2, y2 = line_end
            
            if x1 == x2 and y1 == y2:
                return np.sqrt((x - x1)**2 + (y - y1)**2)
            
            # 计算点到直线的距离
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            distance = abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)
            return distance
        
        def douglas_peucker(points, epsilon):
            """Douglas-Peucker算法"""
            if len(points) <= 2:
                return points
            
            # 找到距离最远的点
            max_distance = 0
            max_index = 0
            
            for i in range(1, len(points) - 1):
                distance = perpendicular_distance(points[i], points[0], points[-1])
                if distance > max_distance:
                    max_distance = distance
                    max_index = i
            
            # 如果最大距离小于阈值，则简化
            if max_distance <= epsilon:
                return [points[0], points[-1]]
            
            # 递归处理两段
            left = douglas_peucker(points[:max_index + 1], epsilon)
            right = douglas_peucker(points[max_index:], epsilon)
            
            return left[:-1] + right
        
        # 转换为numpy数组
        points_array = np.array(boundary_points)
        
        # 计算合适的epsilon值
        x_range = np.max(points_array[:, 0]) - np.min(points_array[:, 0])
        y_range = np.max(points_array[:, 1]) - np.min(points_array[:, 1])
        epsilon = min(x_range, y_range) * 0.01  # 1%的边界范围
        
        # 应用Douglas-Peucker算法
        simplified = douglas_peucker(boundary_points, epsilon)
        
        # 如果简化后仍然太多，进一步减少
        if len(simplified) > max_points:
            # 均匀采样
            indices = np.linspace(0, len(simplified) - 1, max_points, dtype=int)
            simplified = [simplified[i] for i in indices]
        
        return simplified
    
    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """判断点是否在多边形内（射线法）"""
        try:
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
        except Exception as e:
            print(f"   点内多边形判断出错: {e}")
            return False
    
    def interpolate_simple_height_map(self, grid_size: Tuple[int, int]) -> np.ndarray:
        """在简单边界内插值高程图"""
        if self.vertices is None or self.mesh_bounds is None:
            return None
        
        grid_x, grid_y = grid_size
        
        print("🔄 在简单边界内插值高程图...")
        
        try:
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
            
            # 应用简单掩码
            if self.mask is not None:
                print("   应用简单掩码...")
                height_map = np.where(self.mask, height_map, 0.0)
            
            print(f"✅ 简单高程图完成")
            print(f"   高程范围: [{np.min(height_map):.3f}, {np.max(height_map):.3f}]")
            print(f"   平均高程: {np.mean(height_map):.3f}")
            
            self.height_map = height_map
            return height_map
            
        except Exception as e:
            print(f"❌ 插值高程图时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_terrain(self, obj_filepath: str, grid_size: Tuple[int, int] = (150, 150)) -> Dict:
        """处理地形数据"""
        print("🚀 开始简单地形处理...")
        
        # 1. 加载OBJ文件
        if not self.load_obj_file(obj_filepath):
            return None
        
        # 2. 提取简单边界
        boundary_points = self.extract_simple_boundary()
        if boundary_points is None:
            return None
        
        # 3. 创建简单掩码
        mask = self.create_simple_mask(grid_size)
        if mask is None:
            return None
        
        # 4. 插值高程图
        height_map = self.interpolate_simple_height_map(grid_size)
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
        
        print("✅ 简单地形处理完成")
        return result


def main():
    """主函数"""
    processor = SimpleMeshProcessor()
    
    # 处理地形
    obj_file = "data/terrain/terrain.obj"
    result = processor.process_terrain(obj_file, grid_size=(150, 150))
    
    if result:
        # 保存结果
        output_file = "data/terrain/terrain_simple_mesh.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
