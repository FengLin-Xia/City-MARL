#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算向量方向脚本
以hub2 (112, 121) 作为主要吸引点，考虑地块边界干扰
输出格式：x, y, angle (向右为0度，顺时针到360度)
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict
import math

class VectorDirectionCalculator:
    def __init__(self, hub2_x: float = 112.0, hub2_y: float = 121.0):
        """
        初始化向量方向计算器
        
        Args:
            hub2_x: hub2的x坐标
            hub2_y: hub2的y坐标
        """
        self.hub2_x = hub2_x
        self.hub2_y = hub2_y
        self.parcel_boundaries = []
        
    def load_parcel_data(self, parcel_file: str):
        """
        加载地块数据
        
        Args:
            parcel_file: 地块数据文件路径
        """
        print(f"正在加载地块数据: {parcel_file}")
        
        with open(parcel_file, 'r') as f:
            content = f.read().strip()
            
        # 解析地块数据 - 每个地块用[]包围
        parcels = []
        current_parcel = []
        in_parcel = False
        
        for line in content.split('\n'):
            line = line.strip()
            if line == '[':
                in_parcel = True
                current_parcel = []
            elif line == ']':
                if in_parcel and current_parcel:
                    parcels.append(current_parcel)
                in_parcel = False
                current_parcel = []
            elif in_parcel and line and not line.startswith('[') and not line.startswith(']'):
                # 解析坐标行: x, y, z
                try:
                    coords = [float(x.strip()) for x in line.split(',')]
                    if len(coords) >= 2:
                        current_parcel.append((coords[0], coords[1]))
                except ValueError:
                    continue
                    
        self.parcel_boundaries = parcels
        print(f"成功加载 {len(parcels)} 个地块")
        
    def calculate_angle_to_hub2(self, x: float, y: float) -> float:
        """
        计算从点(x,y)到hub2的角度
        
        Args:
            x: 点的x坐标
            y: 点的y坐标
            
        Returns:
            角度（度），向右为0度，顺时针到360度
        """
        dx = self.hub2_x - x
        dy = self.hub2_y - y
        
        # 计算角度（弧度）
        angle_rad = math.atan2(dy, dx)
        
        # 转换为度，并调整到0-360度范围（向右为0度）
        angle_deg = math.degrees(angle_rad)
        
        # 调整角度：向右为0度，顺时针为正
        if angle_deg < 0:
            angle_deg += 360
            
        return angle_deg
    
    def calculate_boundary_interference(self, x: float, y: float, interference_strength: float = 0.1) -> float:
        """
        计算地块边界的干扰角度
        
        Args:
            x: 点的x坐标
            y: 点的y坐标
            interference_strength: 干扰强度（0-1之间）
            
        Returns:
            干扰角度（度）
        """
        total_interference_x = 0.0
        total_interference_y = 0.0
        total_weight = 0.0
        
        for parcel in self.parcel_boundaries:
            if len(parcel) < 3:  # 至少需要3个点才能形成边界
                continue
                
            # 计算到地块边界的最近距离和方向
            min_dist = float('inf')
            closest_point = None
            
            for i in range(len(parcel)):
                p1 = parcel[i]
                p2 = parcel[(i + 1) % len(parcel)]
                
                # 计算点到线段的距离
                dist, closest = self._point_to_line_distance(x, y, p1, p2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point = closest
            
            if closest_point and min_dist < 50:  # 只考虑50单位内的边界
                # 计算干扰向量（指向边界）
                dx = closest_point[0] - x
                dy = closest_point[1] - y
                
                # 距离越近，干扰越强
                weight = 1.0 / (min_dist + 1.0)
                
                total_interference_x += dx * weight
                total_interference_y += dy * weight
                total_weight += weight
        
        if total_weight > 0:
            # 归一化干扰向量
            avg_interference_x = total_interference_x / total_weight
            avg_interference_y = total_interference_y / total_weight
            
            # 计算干扰角度
            interference_angle = math.degrees(math.atan2(avg_interference_y, avg_interference_x))
            if interference_angle < 0:
                interference_angle += 360
                
            return interference_angle * interference_strength
        
        return 0.0
    
    def _point_to_line_distance(self, px: float, py: float, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, Tuple[float, float]]:
        """
        计算点到线段的距离和最近点
        
        Args:
            px, py: 点的坐标
            p1, p2: 线段的两个端点
            
        Returns:
            (距离, 最近点坐标)
        """
        x1, y1 = p1
        x2, y2 = p2
        
        # 线段长度的平方
        line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
        
        if line_length_sq == 0:
            # 线段退化为点
            dist = math.sqrt((px - x1)**2 + (py - y1)**2)
            return dist, (x1, y1)
        
        # 计算参数t，表示最近点在线段上的位置
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
        
        # 最近点坐标
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        
        # 距离
        dist = math.sqrt((px - closest_x)**2 + (py - closest_y)**2)
        
        return dist, (closest_x, closest_y)
    
    def calculate_vector_directions(self, points: List[Tuple[float, float]], 
                                 hub_weight: float = 0.8, 
                                 boundary_weight: float = 0.2) -> List[Tuple[float, float, float]]:
        """
        计算所有点的向量方向
        
        Args:
            points: 点列表 [(x, y), ...]
            hub_weight: hub2吸引的权重
            boundary_weight: 边界干扰的权重
            
        Returns:
            结果列表 [(x, y, angle), ...]
        """
        results = []
        
        print(f"正在计算 {len(points)} 个点的向量方向...")
        
        for i, (x, y) in enumerate(points):
            if i % 100 == 0:
                print(f"处理进度: {i}/{len(points)}")
            
            # 计算到hub2的角度
            hub_angle = self.calculate_angle_to_hub2(x, y)
            
            # 计算边界干扰角度
            boundary_angle = self.calculate_boundary_interference(x, y)
            
            # 加权合成最终角度
            if boundary_weight > 0 and boundary_angle != 0:
                # 使用向量合成
                hub_rad = math.radians(hub_angle)
                boundary_rad = math.radians(boundary_angle)
                
                # 计算加权向量
                final_x = hub_weight * math.cos(hub_rad) + boundary_weight * math.cos(boundary_rad)
                final_y = hub_weight * math.sin(hub_rad) + boundary_weight * math.sin(boundary_rad)
                
                # 计算最终角度
                final_angle = math.degrees(math.atan2(final_y, final_x))
                if final_angle < 0:
                    final_angle += 360
            else:
                final_angle = hub_angle
            
            results.append((x, y, final_angle))
        
        return results
    
    def save_results(self, results: List[Tuple[float, float, float]], output_file: str):
        """
        保存结果到文件
        
        Args:
            results: 结果列表
            output_file: 输出文件路径
        """
        print(f"正在保存结果到: {output_file}")
        
        with open(output_file, 'w') as f:
            for x, y, angle in results:
                f.write(f"{x:.6f},{y:.6f},{angle:.2f}\n")
        
        print(f"成功保存 {len(results)} 个点的向量方向数据")

def main():
    """主函数"""
    print("=== 向量方向计算器 ===")
    
    # 初始化计算器
    calculator = VectorDirectionCalculator(hub2_x=112.0, hub2_y=121.0)
    
    # 加载地块数据
    parcel_file = "parcel.txt"
    if not os.path.exists(parcel_file):
        print(f"错误：找不到地块数据文件 {parcel_file}")
        return
    
    calculator.load_parcel_data(parcel_file)
    
    # 从slots_with_angle.txt加载点数据
    points_file = "slots_with_angle.txt"
    if not os.path.exists(points_file):
        print(f"错误：找不到点数据文件 {points_file}")
        return
    
    print(f"正在加载点数据: {points_file}")
    points = []
    
    with open(points_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
                        points.append((x, y))
                except ValueError:
                    continue
    
    print(f"成功加载 {len(points)} 个点")
    
    # 计算向量方向
    results = calculator.calculate_vector_directions(
        points, 
        hub_weight=0.8,  # hub2吸引权重
        boundary_weight=0.2  # 边界干扰权重
    )
    
    # 保存结果
    output_file = "vector_directions.txt"
    calculator.save_results(results, output_file)
    
    print("=== 计算完成 ===")
    print(f"结果已保存到: {output_file}")
    print(f"格式: x, y, angle (向右为0度，顺时针到360度)")

if __name__ == "__main__":
    main()











