#!/usr/bin/env python3
"""
改进的地形处理器
正确处理mesh边界，避免边界填充问题
"""

import numpy as np
import json
import os
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ImprovedTerrainProcessor:
    """改进的地形处理器"""
    
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.height_map = None
        self.mask = None  # 有效区域掩码
        
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
            print(f"   所有面已转换为三角形")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载OBJ文件失败: {e}")
            return False
    
    def calculate_mesh_boundary(self) -> Dict:
        """计算mesh的实际边界"""
        if self.vertices is None:
            return None
        
        # 提取2D投影（X-Y平面）
        x_coords = self.vertices[:, 0]
        y_coords = self.vertices[:, 1]
        heights = self.vertices[:, 2]
        
        # 计算边界
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = heights.min(), heights.max()
        
        # 计算凸包边界
        points_2d = np.column_stack((x_coords, y_coords))
        hull = ConvexHull(points_2d)
        boundary_points = points_2d[hull.vertices]
        
        boundary = {
            'x_min': float(x_min),
            'x_max': float(x_max),
            'y_min': float(y_min),
            'y_max': float(y_max),
            'z_min': float(z_min),
            'z_max': float(z_max),
            'boundary_points': boundary_points.tolist(),
            'hull_vertices': hull.vertices.tolist()
        }
        
        print(f"📏 Mesh边界信息:")
        print(f"   X范围: {x_min:.3f} ~ {x_max:.3f}")
        print(f"   Y范围: {y_min:.3f} ~ {y_max:.3f}")
        print(f"   Z范围: {z_min:.3f} ~ {z_max:.3f}")
        print(f"   凸包顶点数: {len(hull.vertices)}")
        
        return boundary
    
    def create_mesh_mask(self, grid_size: Tuple[int, int], boundary: Dict) -> np.ndarray:
        """创建mesh有效区域的掩码"""
        x_min, x_max = boundary['x_min'], boundary['x_max']
        y_min, y_max = boundary['y_min'], boundary['y_max']
        grid_x, grid_y = grid_size
        
        # 创建网格坐标
        x_coords = np.linspace(x_min, x_max, grid_x)
        y_coords = np.linspace(y_min, y_max, grid_y)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # 创建掩码（初始为False）
        mask = np.zeros((grid_x, grid_y), dtype=bool)
        
        # 使用凸包边界点判断每个网格点是否在mesh内
        boundary_points = np.array(boundary['boundary_points'])
        
        # 简化的点内判断：检查点是否在凸包内
        for i in range(grid_x):
            for j in range(grid_y):
                point = np.array([X[i, j], Y[i, j]])
                
                # 使用射线法判断点是否在多边形内
                if self._point_in_polygon(point, boundary_points):
                    mask[i, j] = True
        
        print(f"✅ 创建mesh掩码完成")
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
    
    def interpolate_height_map(self, grid_size: Tuple[int, int], boundary: Dict, mask: np.ndarray) -> np.ndarray:
        """在有效区域内插值高程图"""
        x_min, x_max = boundary['x_min'], boundary['x_max']
        y_min, y_max = boundary['y_min'], boundary['y_max']
        grid_x, grid_y = grid_size
        
        # 提取顶点数据
        x_coords = self.vertices[:, 0]
        y_coords = self.vertices[:, 1]
        heights = self.vertices[:, 2]
        
        # 创建网格坐标
        grid_x_coords = np.linspace(x_min, x_max, grid_x)
        grid_y_coords = np.linspace(y_min, y_max, grid_y)
        X, Y = np.meshgrid(grid_x_coords, grid_y_coords, indexing='ij')
        
        # 准备插值点
        points = np.column_stack((x_coords, y_coords))
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
        print("🚀 开始处理地形数据")
        print("=" * 50)
        
        # 1. 加载OBJ文件
        if not self.load_obj_file(obj_filepath):
            return None
        
        # 2. 计算mesh边界
        boundary = self.calculate_mesh_boundary()
        if boundary is None:
            return None
        
        # 3. 创建mesh掩码
        mask = self.create_mesh_mask(grid_size, boundary)
        
        # 4. 插值高程图
        height_map = self.interpolate_height_map(grid_size, boundary, mask)
        
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
            'boundary': boundary,
            'valid_points_count': int(np.sum(mask)),
            'coverage_percentage': float(np.sum(mask)/mask.size*100)
        }
        
        print("\n✅ 地形处理完成!")
        print(f"   网格尺寸: {grid_size}")
        print(f"   有效点数: {result['valid_points_count']}")
        print(f"   覆盖率: {result['coverage_percentage']:.1f}%")
        
        return result
    
    def visualize_processing_result(self, save_path: Optional[str] = None):
        """可视化处理结果"""
        if self.height_map is None or self.mask is None:
            print("❌ 没有处理结果可可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('改进地形处理结果', fontsize=16)
        
        # 1. 原始mesh边界
        if self.vertices is not None:
            x_coords = self.vertices[:, 0]
            y_coords = self.vertices[:, 1]
            
            axes[0, 0].scatter(x_coords, y_coords, s=1, alpha=0.5, c='blue')
            axes[0, 0].set_title('原始Mesh顶点分布')
            axes[0, 0].set_xlabel('X坐标')
            axes[0, 0].set_ylabel('Y坐标')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 有效区域掩码
        im1 = axes[0, 1].imshow(self.mask.T, cmap='gray', aspect='auto', origin='lower')
        axes[0, 1].set_title('有效区域掩码')
        axes[0, 1].set_xlabel('X坐标')
        axes[0, 1].set_ylabel('Y坐标')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 3. 高程图（只显示有效区域）
        valid_height_map = np.where(self.mask, self.height_map, np.nan)
        im2 = axes[1, 0].imshow(valid_height_map.T, cmap='terrain', aspect='auto', origin='lower')
        axes[1, 0].set_title('高程图（有效区域）')
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
    processor = ImprovedTerrainProcessor()
    
    # 处理地形
    result = processor.process_terrain(obj_filepath, grid_size=(150, 150))
    
    if result is None:
        print("❌ 地形处理失败")
        return
    
    # 保存结果
    output_dir = "data/terrain"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "terrain_mesh_aware.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ 处理结果已保存到: {output_file}")
    
    # 可视化结果
    processor.visualize_processing_result(
        save_path="visualization_output/mesh_aware_terrain.png"
    )


if __name__ == "__main__":
    main()
