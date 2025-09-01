import numpy as np
import heapq
from typing import List, Tuple, Optional

class MoveLogic:
    def __init__(self, grid_size: Tuple[int, int] = (256, 256)):
        self.grid_size = grid_size
        self.grid_w, self.grid_h = grid_size
        
    def move_towards(self, current_pos: List[float], target_pos: List[float], 
                    speed_px: float = 4, method: str = "linear") -> List[float]:
        """移动朝向目标位置"""
        if method == "linear":
            return self._linear_move(current_pos, target_pos, speed_px)
        elif method == "astar":
            return self._astar_move(current_pos, target_pos, speed_px)
        else:
            return self._linear_move(current_pos, target_pos, speed_px)
    
    def _linear_move(self, current_pos: List[float], target_pos: List[float], 
                    speed_px: float) -> List[float]:
        """直线移动"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance <= speed_px:
            # 已经到达目标
            return target_pos.copy()
        
        # 标准化方向向量
        dx = dx / distance * speed_px
        dy = dy / distance * speed_px
        
        new_x = current_pos[0] + dx
        new_y = current_pos[1] + dy
        
        # 确保在网格范围内
        new_x = max(0, min(self.grid_w - 1, new_x))
        new_y = max(0, min(self.grid_h - 1, new_y))
        
        return [new_x, new_y]
    
    def _astar_move(self, current_pos: List[float], target_pos: List[float], 
                   speed_px: float) -> List[float]:
        """A*路径规划移动"""
        # 简化版A*，只考虑障碍物
        path = self._astar_path(current_pos, target_pos)
        
        if not path or len(path) < 2:
            return self._linear_move(current_pos, target_pos, speed_px)
        
        # 沿着路径移动一步
        next_pos = path[1] if len(path) > 1 else path[0]
        return self._linear_move(current_pos, next_pos, speed_px)
    
    def _astar_path(self, start: List[float], goal: List[float]) -> List[List[float]]:
        """A*路径规划算法"""
        # 简化的A*实现，主要用于演示
        # 在实际应用中，这里会考虑地形障碍、道路网络等
        
        # 将浮点坐标转换为网格坐标
        start_grid = [int(start[0]), int(start[1])]
        goal_grid = [int(goal[0]), int(goal[1])]
        
        # 如果起点和终点很近，直接返回直线路径
        if self._manhattan_distance(start_grid, goal_grid) < 10:
            return [start, goal]
        
        # 简化的路径：先水平移动，再垂直移动
        path = [start]
        
        # 水平移动
        if abs(start[0] - goal[0]) > 1:
            mid_x = goal[0]
            mid_y = start[1]
            path.append([mid_x, mid_y])
        
        # 垂直移动
        if abs(start[1] - goal[1]) > 1:
            path.append(goal)
        
        return path
    
    def _manhattan_distance(self, pos1: List[int], pos2: List[int]) -> int:
        """曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def reached(self, current_pos: List[float], target_pos: List[float], 
               threshold: float = 5.0) -> bool:
        """检查是否到达目标"""
        distance = np.sqrt((current_pos[0] - target_pos[0])**2 + 
                          (current_pos[1] - target_pos[1])**2)
        return distance <= threshold
    
    def calculate_local_cost(self, pos: List[float], heat_map: np.ndarray) -> float:
        """计算局部移动成本"""
        x, y = int(pos[0]), int(pos[1])
        
        # 确保坐标在有效范围内
        x = max(0, min(self.grid_w - 1, x))
        y = max(0, min(self.grid_h - 1, y))
        
        # 基础成本
        base_cost = 1.0
        
        # 热力图影响（拥挤度）
        if heat_map is not None and x < heat_map.shape[0] and y < heat_map.shape[1]:
            heat_value = heat_map[x, y]
            # 热力值越高，移动成本越高
            heat_cost = heat_value * 0.1
            base_cost += heat_cost
        
        return base_cost

