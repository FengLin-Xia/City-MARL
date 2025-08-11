"""
地形系统模块
支持地形导入、管理和查询功能
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import json
import os

class TerrainType:
    """地形类型枚举"""
    WATER = 0
    GRASS = 1
    FOREST = 2
    MOUNTAIN = 3
    ROAD = 4
    BUILDING = 5
    
    @staticmethod
    def get_name(terrain_id: int) -> str:
        """获取地形类型名称"""
        names = {
            0: "水域",
            1: "草地",
            2: "森林", 
            3: "山地",
            4: "道路",
            5: "建筑"
        }
        return names.get(terrain_id, "未知")
    
    @staticmethod
    def get_cost(terrain_id: int) -> float:
        """获取地形移动成本"""
        costs = {
            0: float('inf'),  # 水域不可通行
            1: 1.0,           # 草地
            2: 2.0,           # 森林
            3: 3.0,           # 山地
            4: 0.5,           # 道路
            5: float('inf')   # 建筑不可通行
        }
        return costs.get(terrain_id, float('inf'))

class TerrainSystem:
    """地形系统类"""
    
    def __init__(self, width: int = 50, height: int = 50):
        """
        初始化地形系统
        
        Args:
            width: 地形宽度
            height: 地形高度
        """
        self.width = width
        self.height = height
        self.terrain = np.full((height, width), TerrainType.GRASS, dtype=np.int32)
        self.elevation = np.zeros((height, width), dtype=np.float32)
        self.resources = np.zeros((height, width), dtype=np.float32)
        
        # 地形属性
        self.terrain_properties = {
            TerrainType.WATER: {
                'name': '水域',
                'movement_cost': float('inf'),
                'resource_value': 0.0,
                'color': 'blue'
            },
            TerrainType.GRASS: {
                'name': '草地',
                'movement_cost': 1.0,
                'resource_value': 1.0,
                'color': 'green'
            },
            TerrainType.FOREST: {
                'name': '森林',
                'movement_cost': 2.0,
                'resource_value': 3.0,
                'color': 'darkgreen'
            },
            TerrainType.MOUNTAIN: {
                'name': '山地',
                'movement_cost': 3.0,
                'resource_value': 5.0,
                'color': 'gray'
            },
            TerrainType.ROAD: {
                'name': '道路',
                'movement_cost': 0.5,
                'resource_value': 0.0,
                'color': 'yellow'
            },
            TerrainType.BUILDING: {
                'name': '建筑',
                'movement_cost': float('inf'),
                'resource_value': 0.0,
                'color': 'red'
            }
        }
    
    def load_from_file(self, filepath: str) -> bool:
        """
        从文件加载地形数据
        
        Args:
            filepath: 地形文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            if filepath.endswith('.npy'):
                # 加载numpy数组
                terrain_data = np.load(filepath)
                if terrain_data.shape == (self.height, self.width):
                    self.terrain = terrain_data
                    return True
            elif filepath.endswith('.json'):
                # 加载JSON格式
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.terrain = np.array(data['terrain'])
                    if 'elevation' in data:
                        self.elevation = np.array(data['elevation'])
                    if 'resources' in data:
                        self.resources = np.array(data['resources'])
                    return True
            elif filepath.endswith('.txt'):
                # 加载文本格式
                terrain_data = np.loadtxt(filepath, dtype=np.int32)
                if terrain_data.shape == (self.height, self.width):
                    self.terrain = terrain_data
                    return True
        except Exception as e:
            print(f"加载地形文件失败: {e}")
        return False
    
    def save_to_file(self, filepath: str) -> bool:
        """
        保存地形数据到文件
        
        Args:
            filepath: 保存文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            if filepath.endswith('.npy'):
                np.save(filepath, self.terrain)
            elif filepath.endswith('.json'):
                data = {
                    'terrain': self.terrain.tolist(),
                    'elevation': self.elevation.tolist(),
                    'resources': self.resources.tolist(),
                    'width': self.width,
                    'height': self.height
                }
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            elif filepath.endswith('.txt'):
                np.savetxt(filepath, self.terrain, fmt='%d')
            return True
        except Exception as e:
            print(f"保存地形文件失败: {e}")
        return False
    
    def generate_random_terrain(self, seed: Optional[int] = None):
        """
        生成随机地形
        
        Args:
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 生成基础地形
        self.terrain = np.random.choice([
            TerrainType.GRASS, TerrainType.FOREST, TerrainType.MOUNTAIN
        ], size=(self.height, self.width), p=[0.6, 0.3, 0.1])
        
        # 生成水域
        water_mask = np.random.random((self.height, self.width)) < 0.1
        self.terrain[water_mask] = TerrainType.WATER
        
        # 生成道路网络
        self._generate_road_network()
        
        # 生成高程
        self._generate_elevation()
        
        # 生成资源分布
        self._generate_resources()
    
    def _generate_road_network(self):
        """生成道路网络"""
        # 简单的道路生成：连接几个关键点
        key_points = [
            (self.width // 4, self.height // 4),
            (3 * self.width // 4, self.height // 4),
            (self.width // 4, 3 * self.height // 4),
            (3 * self.width // 4, 3 * self.height // 4),
            (self.width // 2, self.height // 2)
        ]
        
        # 连接关键点
        for i in range(len(key_points) - 1):
            start = key_points[i]
            end = key_points[i + 1]
            self._draw_line(start, end, TerrainType.ROAD)
    
    def _draw_line(self, start: Tuple[int, int], end: Tuple[int, int], terrain_type: int):
        """绘制直线"""
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        if dx > dy:
            steps = dx
        else:
            steps = dy
        
        if steps == 0:
            return
        
        x_increment = float(dx) / steps
        y_increment = float(dy) / steps
        
        x = x0
        y = y0
        
        for i in range(int(steps) + 1):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.width and 0 <= iy < self.height:
                self.terrain[iy, ix] = terrain_type
            x += x_increment
            y += y_increment
    
    def _generate_elevation(self):
        """生成高程数据"""
        # 使用简单的噪声生成高程
        x = np.linspace(0, 4, self.width)
        y = np.linspace(0, 4, self.height)
        X, Y = np.meshgrid(x, y)
        
        # 多层噪声
        self.elevation = (
            np.sin(X + Y) * 0.3 +
            np.sin(2 * X) * 0.2 +
            np.sin(2 * Y) * 0.2 +
            np.random.random((self.height, self.width)) * 0.3
        )
        
        # 根据地形类型调整高程
        self.elevation[self.terrain == TerrainType.MOUNTAIN] += 2.0
        self.elevation[self.terrain == TerrainType.WATER] -= 1.0
    
    def _generate_resources(self):
        """生成资源分布"""
        # 根据地形类型生成资源
        for terrain_type in [TerrainType.GRASS, TerrainType.FOREST, TerrainType.MOUNTAIN]:
            mask = self.terrain == terrain_type
            if terrain_type == TerrainType.FOREST:
                self.resources[mask] = np.random.exponential(2.0, size=mask.sum())
            elif terrain_type == TerrainType.MOUNTAIN:
                self.resources[mask] = np.random.exponential(3.0, size=mask.sum())
            else:
                self.resources[mask] = np.random.exponential(1.0, size=mask.sum())
    
    def get_terrain_at(self, x: int, y: int) -> int:
        """获取指定位置的地形类型"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.terrain[y, x]
        return TerrainType.WATER  # 边界外默认为水域
    
    def get_movement_cost(self, x: int, y: int) -> float:
        """获取指定位置的移动成本"""
        terrain_type = self.get_terrain_at(x, y)
        return TerrainType.get_cost(terrain_type)
    
    def get_resource_value(self, x: int, y: int) -> float:
        """获取指定位置的资源价值"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.resources[y, x]
        return 0.0
    
    def get_elevation(self, x: int, y: int) -> float:
        """获取指定位置的高程"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.elevation[y, x]
        return 0.0
    
    def is_passable(self, x: int, y: int) -> bool:
        """检查位置是否可通行"""
        return self.get_movement_cost(x, y) < float('inf')
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """获取相邻的可通行位置"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_passable(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
    
    def visualize(self, show_resources: bool = False, show_elevation: bool = False):
        """可视化地形"""
        fig, axes = plt.subplots(1, 3 if show_resources and show_elevation else 2, 
                                figsize=(15, 5))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # 地形类型可视化
        terrain_colors = [self.terrain_properties[t]['color'] for t in range(6)]
        terrain_plot = axes[0].imshow(self.terrain, cmap='tab10', vmin=0, vmax=5)
        axes[0].set_title('地形类型')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # 高程可视化
        if show_elevation:
            elevation_plot = axes[1].imshow(self.elevation, cmap='terrain')
            axes[1].set_title('高程')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            plt.colorbar(elevation_plot, ax=axes[1])
        
        # 资源可视化
        if show_resources:
            resource_plot = axes[-1].imshow(self.resources, cmap='hot')
            axes[-1].set_title('资源分布')
            axes[-1].set_xlabel('X')
            axes[-1].set_ylabel('Y')
            plt.colorbar(resource_plot, ax=axes[-1])
        
        plt.tight_layout()
        plt.show()
    
    def get_terrain_stats(self) -> Dict:
        """获取地形统计信息"""
        stats = {}
        for terrain_type in range(6):
            count = np.sum(self.terrain == terrain_type)
            percentage = count / (self.width * self.height) * 100
            stats[TerrainType.get_name(terrain_type)] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        return stats
