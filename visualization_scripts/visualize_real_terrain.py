#!/usr/bin/env python3
"""
真实地形可视化脚本
显示Flask上传的地形数据和训练路径
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class RealTerrainVisualizer:
    """真实地形可视化器"""
    
    def __init__(self, terrain_file: str):
        self.terrain_file = terrain_file
        self.terrain_data = self.load_terrain_data()
        self.height_map = np.array(self.terrain_data['height_map'], dtype=np.float32)
        self.H, self.W = self.height_map.shape
        
        print(f"地形尺寸: {self.H}x{self.W}")
        print(f"高程范围: {np.min(self.height_map):.1f} ~ {np.max(self.height_map):.1f}")
    
    def load_terrain_data(self) -> Dict:
        """加载地形数据"""
        print(f"加载地形数据: {self.terrain_file}")
        with open(self.terrain_file, 'r') as f:
            data = json.load(f)
        return data
    
    def calculate_slope_map(self) -> np.ndarray:
        """计算坡度图"""
        slope_map = np.zeros_like(self.height_map)
        
        for i in range(self.H):
            for j in range(self.W):
                # 计算8邻域的坡度
                max_slope = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.H and 0 <= nj < self.W:
                            height_diff = abs(self.height_map[ni, nj] - self.height_map[i, j])
                            distance = np.sqrt(di**2 + dj**2)
                            slope = height_diff / distance if distance > 0 else 0
                            max_slope = max(max_slope, slope)
                
                slope_map[i, j] = max_slope
        
        return slope_map
    
    def visualize_terrain_overview(self, save_path: Optional[str] = None):
        """可视化地形概览"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('真实地形概览', fontsize=16)
        
        # 1. 地形高程图
        im1 = axes[0, 0].imshow(self.height_map, cmap='terrain', aspect='auto')
        axes[0, 0].set_title('地形高程图')
        axes[0, 0].set_xlabel('X坐标')
        axes[0, 0].set_ylabel('Y坐标')
        plt.colorbar(im1, ax=axes[0, 0], label='高程')
        
        # 2. 坡度图
        slope_map = self.calculate_slope_map()
        im2 = axes[0, 1].imshow(slope_map, cmap='hot', aspect='auto')
        axes[0, 1].set_title('坡度图')
        axes[0, 1].set_xlabel('X坐标')
        axes[0, 1].set_ylabel('Y坐标')
        plt.colorbar(im2, ax=axes[0, 1], label='坡度')
        
        # 3. 高程分布直方图
        axes[1, 0].hist(self.height_map.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('高程分布')
        axes[1, 0].set_xlabel('高程')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 坡度分布直方图
        axes[1, 1].hist(slope_map.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('坡度分布')
        axes[1, 1].set_xlabel('坡度')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"地形概览图已保存到: {save_path}")
        
        plt.show()
    
    def visualize_training_paths(self, training_data_file: str, save_path: Optional[str] = None):
        """可视化训练路径"""
        # 加载训练数据
        with open(training_data_file, 'r') as f:
            training_data = json.load(f)
        
        start_point = training_data.get('start_point', [0, 0])
        goal_point = training_data.get('goal_point', [0, 0])
        
        print(f"起点: {start_point}")
        print(f"终点: {goal_point}")
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 地形高程图 + 起点终点
        im1 = axes[0].imshow(self.height_map, cmap='terrain', aspect='auto')
        axes[0].set_title('地形高程图与起终点')
        axes[0].set_xlabel('X坐标')
        axes[0].set_ylabel('Y坐标')
        
        # 标记起点和终点
        axes[0].plot(start_point[1], start_point[0], 'go', markersize=15, 
                    label='起点', markeredgecolor='black', markeredgewidth=2)
        axes[0].plot(goal_point[1], goal_point[0], 'ro', markersize=15, 
                    label='终点', markeredgecolor='black', markeredgewidth=2)
        
        # 绘制理想路径（直线）
        ideal_x = [start_point[0], goal_point[0]]
        ideal_y = [start_point[1], goal_point[1]]
        axes[0].plot(ideal_y, ideal_x, 'r--', linewidth=3, alpha=0.8, label='理想路径')
        
        axes[0].legend()
        plt.colorbar(im1, ax=axes[0], label='高程')
        
        # 2. 坡度图 + 起点终点
        slope_map = self.calculate_slope_map()
        im2 = axes[1].imshow(slope_map, cmap='hot', aspect='auto')
        axes[1].set_title('坡度图与起终点')
        axes[1].set_xlabel('X坐标')
        axes[1].set_ylabel('Y坐标')
        
        # 标记起点和终点
        axes[1].plot(start_point[1], start_point[0], 'go', markersize=15, 
                    label='起点', markeredgecolor='black', markeredgewidth=2)
        axes[1].plot(goal_point[1], goal_point[0], 'ro', markersize=15, 
                    label='终点', markeredgecolor='black', markeredgewidth=2)
        
        axes[1].legend()
        plt.colorbar(im2, ax=axes[1], label='坡度')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练路径图已保存到: {save_path}")
        
        plt.show()
        
        # 打印地形信息
        print(f"\n=== 地形信息 ===")
        print(f"起点高程: {self.height_map[start_point[0], start_point[1]]:.1f}")
        print(f"终点高程: {self.height_map[goal_point[0], goal_point[1]]:.1f}")
        print(f"起点坡度: {slope_map[start_point[0], start_point[1]]:.3f}")
        print(f"终点坡度: {slope_map[goal_point[0], goal_point[1]]:.3f}")
        print(f"曼哈顿距离: {abs(goal_point[0] - start_point[0]) + abs(goal_point[1] - start_point[1])}")
        
        # 计算路径难度
        ideal_path_length = abs(goal_point[0] - start_point[0]) + abs(goal_point[1] - start_point[1])
        print(f"理想路径长度: {ideal_path_length}")
        
        # 分析路径上的地形特征
        path_heights = []
        path_slopes = []
        
        # 计算理想路径上的地形特征
        current_pos = list(start_point)
        while current_pos != list(goal_point):
            path_heights.append(self.height_map[current_pos[0], current_pos[1]])
            path_slopes.append(slope_map[current_pos[0], current_pos[1]])
            
            if current_pos[0] < goal_point[0]:
                current_pos[0] += 1
            elif current_pos[1] < goal_point[1]:
                current_pos[1] += 1
        
        if path_heights:
            print(f"路径平均高程: {np.mean(path_heights):.1f}")
            print(f"路径最大高程: {np.max(path_heights):.1f}")
            print(f"路径最小高程: {np.min(path_heights):.1f}")
            print(f"路径平均坡度: {np.mean(path_slopes):.3f}")
            print(f"路径最大坡度: {np.max(path_slopes):.3f}")
    
    def visualize_3d_terrain(self, save_path: Optional[str] = None):
        """3D地形可视化"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建网格
        x = np.arange(self.W)
        y = np.arange(self.H)
        X, Y = np.meshgrid(x, y)
        
        # 绘制3D地形
        surf = ax.plot_surface(X, Y, self.height_map, cmap='terrain', 
                              linewidth=0, antialiased=True)
        
        ax.set_title('3D地形图')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_zlabel('高程')
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D地形图已保存到: {save_path}")
        
        plt.show()
    
    def find_land_points(self, min_height: float = 50.0) -> List[Tuple[int, int]]:
        """找到陆地上的点"""
        land_points = []
        for i in range(self.H):
            for j in range(self.W):
                if self.height_map[i, j] > min_height:
                    land_points.append((i, j))
        return land_points
    
    def suggest_start_goal_points(self, min_height: float = 50.0) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """建议起点和终点"""
        land_points = self.find_land_points(min_height)
        
        if len(land_points) < 2:
            print("警告：陆地点太少，无法找到合适的起终点")
            return (0, 0), (self.H-1, self.W-1)
        
        # 选择距离较远的两个点
        max_distance = 0
        best_start = land_points[0]
        best_goal = land_points[1]
        
        for i, start in enumerate(land_points):
            for j, goal in enumerate(land_points[i+1:], i+1):
                distance = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
                if distance > max_distance:
                    max_distance = distance
                    best_start = start
                    best_goal = goal
        
        return best_start, best_goal


def main():
    """主函数"""
    # 使用最新的地形数据
    terrain_file = "data/terrain/terrain_1755281528.json"
    
    if not os.path.exists(terrain_file):
        print(f"地形文件不存在: {terrain_file}")
        return
    
    # 创建可视化器
    visualizer = RealTerrainVisualizer(terrain_file)
    
    # 创建输出目录
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("真实地形可视化")
    print("=" * 50)
    
    # 1. 地形概览
    print("1. 生成地形概览...")
    visualizer.visualize_terrain_overview(
        save_path=os.path.join(output_dir, "terrain_overview.png")
    )
    
    # 2. 3D地形图
    print("2. 生成3D地形图...")
    visualizer.visualize_3d_terrain(
        save_path=os.path.join(output_dir, "terrain_3d.png")
    )
    
    # 3. 建议起终点
    print("3. 分析地形并建议起终点...")
    start_point, goal_point = visualizer.suggest_start_goal_points()
    print(f"建议起点: {start_point}")
    print(f"建议终点: {goal_point}")
    
    # 4. 如果有训练数据，可视化训练路径
    training_data_dir = "training_data"
    if os.path.exists(training_data_dir):
        training_files = [f for f in os.listdir(training_data_dir) 
                         if f.endswith('_final.json')]
        
        if training_files:
            latest_training_file = sorted(training_files)[-1]
            training_file_path = os.path.join(training_data_dir, latest_training_file)
            
            print(f"4. 可视化训练路径 (使用 {latest_training_file})...")
            visualizer.visualize_training_paths(
                training_file_path,
                save_path=os.path.join(output_dir, "training_paths.png")
            )
    
    print(f"\n所有可视化图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()

