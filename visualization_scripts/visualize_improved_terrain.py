#!/usr/bin/env python3
"""
使用修复后地形数据的可视化脚本
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ImprovedTerrainVisualizer:
    """改进地形可视化器"""
    
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
        fig.suptitle('修复后地形概览', fontsize=16)
        
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
    
    def analyze_terrain_quality(self):
        """分析地形质量"""
        print("=== 地形质量分析 ===")
        
        # 检查重复值
        unique_values, counts = np.unique(self.height_map, return_counts=True)
        most_common_idx = np.argmax(counts)
        most_common_value = unique_values[most_common_idx]
        most_common_count = counts[most_common_idx]
        
        print(f"唯一值数量: {len(unique_values)}")
        print(f"最常见值: {most_common_value:.6f} (出现 {most_common_count} 次)")
        print(f"最常见值比例: {most_common_count/self.height_map.size*100:.2f}%")
        
        # 检查边界
        border_values = np.concatenate([
            self.height_map[0, :], self.height_map[-1, :],
            self.height_map[:, 0], self.height_map[:, -1]
        ])
        border_unique = len(np.unique(border_values))
        print(f"边界唯一值数量: {border_unique}")
        
        # 评估质量
        quality_score = 0
        if len(unique_values) > 10000:
            quality_score += 30
        if most_common_count/self.height_map.size < 0.05:
            quality_score += 30
        if border_unique > 400:
            quality_score += 40
        
        print(f"地形质量评分: {quality_score}/100")
        
        if quality_score >= 80:
            print("✅ 地形质量优秀")
        elif quality_score >= 60:
            print("⚠️ 地形质量良好")
        else:
            print("❌ 地形质量需要改进")
        
        return quality_score


def main():
    """主函数"""
    # 使用修复后的地形数据
    terrain_file = "data/terrain/terrain_improved.json"
    
    if not os.path.exists(terrain_file):
        print(f"修复后的地形文件不存在: {terrain_file}")
        print("请先运行 fix_terrain_boundary.py 修复地形数据")
        return
    
    # 创建可视化器
    visualizer = ImprovedTerrainVisualizer(terrain_file)
    
    # 创建输出目录
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("修复后地形可视化")
    print("=" * 50)
    
    # 1. 地形质量分析
    print("1. 分析地形质量...")
    quality_score = visualizer.analyze_terrain_quality()
    
    # 2. 地形概览
    print("2. 生成地形概览...")
    visualizer.visualize_terrain_overview(
        save_path=os.path.join(output_dir, "improved_terrain_overview.png")
    )
    
    # 3. 建议起终点
    print("3. 分析地形并建议起终点...")
    start_point, goal_point = visualizer.suggest_start_goal_points()
    print(f"建议起点: {start_point}")
    print(f"建议终点: {goal_point}")
    
    # 4. 分析建议路径的难度
    slope_map = visualizer.calculate_slope_map()
    print(f"\n=== 建议路径分析 ===")
    print(f"起点高程: {visualizer.height_map[start_point[0], start_point[1]]:.1f}")
    print(f"终点高程: {visualizer.height_map[goal_point[0], goal_point[1]]:.1f}")
    print(f"起点坡度: {slope_map[start_point[0], start_point[1]]:.3f}")
    print(f"终点坡度: {slope_map[goal_point[0], goal_point[1]]:.3f}")
    print(f"曼哈顿距离: {abs(goal_point[0] - start_point[0]) + abs(goal_point[1] - start_point[1])}")
    
    print(f"\n所有可视化图表已保存到: {output_dir}")
    print(f"地形质量评分: {quality_score}/100")


if __name__ == "__main__":
    main()

