#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算向量方向脚本 - 使用repulsion_outputs中的最新地块点数据
以hub2 (112, 121) 作为主要吸引点，考虑地块边界干扰
输出格式：x, y, angle (向右为0度，顺时针到360度)
"""

import numpy as np
import json
import os
import glob
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
        加载地块边界数据
        
        Args:
            parcel_file: 地块边界数据文件路径
        """
        print(f"正在加载地块边界数据: {parcel_file}")
        
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
        print(f"成功加载 {len(parcels)} 个地块边界")
        
    def load_repulsion_points(self, repulsion_dir: str):
        """
        加载repulsion_outputs中的最新地块点数据
        
        Args:
            repulsion_dir: repulsion_outputs目录路径
            
        Returns:
            所有地块点的列表 [(x, y, parcel_id), ...]
        """
        print(f"正在加载repulsion数据: {repulsion_dir}")
        
        # 找到最新的时间戳目录
        timestamp_dirs = []
        for item in os.listdir(repulsion_dir):
            item_path = os.path.join(repulsion_dir, item)
            if os.path.isdir(item_path) and item.startswith('parcel_'):
                timestamp_dirs.append(item)
        
        if not timestamp_dirs:
            # 如果没有子目录，直接在当前目录查找
            csv_files = glob.glob(os.path.join(repulsion_dir, "parcel_*_final_points_*.csv"))
        else:
            # 使用最新的时间戳目录
            latest_timestamp = max(timestamp_dirs)
            csv_files = glob.glob(os.path.join(repulsion_dir, latest_timestamp, "parcel_*_final_points_*.csv"))
        
        if not csv_files:
            # 如果还是找不到，尝试直接查找CSV文件
            csv_files = glob.glob(os.path.join(repulsion_dir, "parcel_*_final_points_*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"在 {repulsion_dir} 中找不到地块点数据文件")
        
        print(f"找到 {len(csv_files)} 个地块点数据文件")
        
        all_points = []
        
        for csv_file in sorted(csv_files):
            # 从文件名提取地块ID
            filename = os.path.basename(csv_file)
            if 'parcel_' in filename:
                try:
                    parcel_id = int(filename.split('_')[1])
                except:
                    parcel_id = 0
            else:
                parcel_id = 0
            
            print(f"正在处理地块 {parcel_id}: {filename}")
            
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                
            # 跳过标题行
            for line in lines[1:]:
                line = line.strip()
                if line:
                    try:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            x = float(parts[0])
                            y = float(parts[1])
                            all_points.append((x, y, parcel_id))
                    except ValueError:
                        continue
        
        print(f"成功加载 {len(all_points)} 个地块点")
        return all_points
    
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
    
    def calculate_vector_directions(self, points: List[Tuple[float, float, int]], 
                                 hub_weight: float = 0.8, 
                                 boundary_weight: float = 0.2) -> List[Tuple[float, float, float, int]]:
        """
        计算所有点的向量方向
        
        Args:
            points: 点列表 [(x, y, parcel_id), ...]
            hub_weight: hub2吸引的权重
            boundary_weight: 边界干扰的权重
            
        Returns:
            结果列表 [(x, y, angle, parcel_id), ...]
        """
        results = []
        
        print(f"正在计算 {len(points)} 个点的向量方向...")
        
        for i, (x, y, parcel_id) in enumerate(points):
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
            
            results.append((x, y, final_angle, parcel_id))
        
        return results
    
    def save_results(self, results: List[Tuple[float, float, float, int]], output_file: str):
        """
        保存结果到文件
        
        Args:
            results: 结果列表
            output_file: 输出文件路径
        """
        print(f"正在保存结果到: {output_file}")
        
        with open(output_file, 'w') as f:
            for x, y, angle, parcel_id in results:
                f.write(f"{x:.6f},{y:.6f},{angle:.2f},{parcel_id}\n")
        
        print(f"成功保存 {len(results)} 个点的向量方向数据")

def main():
    """主函数"""
    print("=== 向量方向计算器 (Repulsion数据) ===")
    
    # 初始化计算器
    calculator = VectorDirectionCalculator(hub2_x=112.0, hub2_y=121.0)
    
    # 加载地块边界数据
    parcel_file = "parcel.txt"
    if not os.path.exists(parcel_file):
        print(f"错误：找不到地块边界数据文件 {parcel_file}")
        return
    
    calculator.load_parcel_data(parcel_file)
    
    # 加载repulsion数据
    repulsion_dir = "repulsion_outputs"
    if not os.path.exists(repulsion_dir):
        print(f"错误：找不到repulsion数据目录 {repulsion_dir}")
        return
    
    try:
        points = calculator.load_repulsion_points(repulsion_dir)
    except FileNotFoundError as e:
        print(f"错误：{e}")
        return
    
    # 计算向量方向
    results = calculator.calculate_vector_directions(
        points, 
        hub_weight=0.8,  # hub2吸引权重
        boundary_weight=0.2  # 边界干扰权重
    )
    
    # 保存结果
    output_file = "vector_directions_repulsion.txt"
    calculator.save_results(results, output_file)
    
    print("=== 计算完成 ===")
    print(f"结果已保存到: {output_file}")
    print(f"格式: x, y, angle, parcel_id (向右为0度，顺时针到360度)")

if __name__ == "__main__":
    main()

