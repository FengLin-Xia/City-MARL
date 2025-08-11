"""
寻路系统模块
支持A*算法和路径规划
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict, Set
from collections import deque
import math

class PathfindingSystem:
    """寻路系统类"""
    
    def __init__(self, terrain_system):
        """
        初始化寻路系统
        
        Args:
            terrain_system: 地形系统实例
        """
        self.terrain = terrain_system
        self.width = terrain_system.width
        self.height = terrain_system.height
        
        # 缓存路径结果
        self.path_cache = {}
        self.cache_size = 1000
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int], 
               heuristic_type: str = 'manhattan') -> Optional[List[Tuple[int, int]]]:
        """
        A*寻路算法
        
        Args:
            start: 起始位置 (x, y)
            goal: 目标位置 (x, y)
            heuristic_type: 启发式函数类型 ('manhattan', 'euclidean', 'diagonal')
            
        Returns:
            路径列表，如果找不到路径则返回None
        """
        # 检查缓存
        cache_key = (start, goal, heuristic_type)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # 检查起点和终点是否可通行
        if not self.terrain.is_passable(*start) or not self.terrain.is_passable(*goal):
            return None
        
        # 初始化
        open_set = []
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal, heuristic_type)}
        
        heapq.heappush(open_set, (f_score[start], start))
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                # 缓存结果
                self._cache_path(cache_key, path)
                return path
            
            closed_set.add(current)
            
            # 检查所有邻居
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # 计算从起点到邻居的成本
                tentative_g_score = g_score[current] + self._get_movement_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal, heuristic_type)
                    
                    # 如果邻居不在开放集中，添加它
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 没有找到路径
        return None
    
    def dijkstra(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Dijkstra算法
        
        Args:
            start: 起始位置
            goal: 目标位置
            
        Returns:
            最短路径
        """
        if not self.terrain.is_passable(*start) or not self.terrain.is_passable(*goal):
            return None
        
        # 初始化
        distances = {start: 0}
        previous = {}
        unvisited = set()
        
        # 初始化所有可通行位置
        for y in range(self.height):
            for x in range(self.width):
                if self.terrain.is_passable(x, y):
                    unvisited.add((x, y))
                    if (x, y) not in distances:
                        distances[(x, y)] = float('inf')
        
        current = start
        
        while unvisited:
            if current == goal:
                break
            
            # 找到未访问节点中距离最小的
            if current not in unvisited:
                break
                
            unvisited.remove(current)
            
            # 更新邻居距离
            for neighbor in self._get_neighbors(current):
                if neighbor in unvisited:
                    distance = distances[current] + self._get_movement_cost(current, neighbor)
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current
            
            # 选择下一个当前节点
            if unvisited:
                current = min(unvisited, key=lambda x: distances[x])
            else:
                break
        
        # 重建路径
        if goal in previous or goal == start:
            path = []
            current = goal
            while current in previous:
                path.append(current)
                current = previous[current]
            path.append(start)
            path.reverse()
            return path
        
        return None
    
    def flood_fill(self, start: Tuple[int, int], max_distance: float = float('inf')) -> Dict[Tuple[int, int], float]:
        """
        洪水填充算法，计算从起点到所有可达位置的距离
        
        Args:
            start: 起始位置
            max_distance: 最大距离限制
            
        Returns:
            位置到距离的映射
        """
        if not self.terrain.is_passable(*start):
            return {}
        
        distances = {start: 0}
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            current, current_distance = queue.popleft()
            
            if current_distance >= max_distance:
                continue
            
            for neighbor in self._get_neighbors(current):
                if neighbor not in visited:
                    neighbor_distance = current_distance + self._get_movement_cost(current, neighbor)
                    if neighbor_distance < max_distance:
                        distances[neighbor] = neighbor_distance
                        visited.add(neighbor)
                        queue.append((neighbor, neighbor_distance))
        
        return distances
    
    def find_nearest_passable(self, position: Tuple[int, int], max_radius: int = 10) -> Optional[Tuple[int, int]]:
        """
        找到最近的可通行位置
        
        Args:
            position: 目标位置
            max_radius: 最大搜索半径
            
        Returns:
            最近的可通行位置
        """
        if self.terrain.is_passable(*position):
            return position
        
        # 从近到远搜索
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) == radius:  # 曼哈顿距离
                        nx, ny = position[0] + dx, position[1] + dy
                        if (0 <= nx < self.width and 0 <= ny < self.height and 
                            self.terrain.is_passable(nx, ny)):
                            return (nx, ny)
        
        return None
    
    def get_accessible_area(self, start: Tuple[int, int], max_cost: float) -> Set[Tuple[int, int]]:
        """
        获取在给定成本限制内可到达的区域
        
        Args:
            start: 起始位置
            max_cost: 最大成本限制
            
        Returns:
            可到达的位置集合
        """
        accessible = set()
        if not self.terrain.is_passable(*start):
            return accessible
        
        queue = deque([(start, 0)])
        visited = {start}
        accessible.add(start)
        
        while queue:
            current, current_cost = queue.popleft()
            
            for neighbor in self._get_neighbors(current):
                if neighbor not in visited:
                    neighbor_cost = current_cost + self._get_movement_cost(current, neighbor)
                    if neighbor_cost <= max_cost:
                        visited.add(neighbor)
                        accessible.add(neighbor)
                        queue.append((neighbor, neighbor_cost))
        
        return accessible
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int], 
                   heuristic_type: str) -> float:
        """计算启发式值"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        if heuristic_type == 'manhattan':
            return dx + dy
        elif heuristic_type == 'euclidean':
            return math.sqrt(dx * dx + dy * dy)
        elif heuristic_type == 'diagonal':
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        else:
            return dx + dy  # 默认曼哈顿距离
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取邻居位置（包括对角线）"""
        neighbors = []
        x, y = pos
        
        # 四个方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 可选：包括对角线方向
        # directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
        #               (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                self.terrain.is_passable(nx, ny)):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def _get_movement_cost(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """计算两点间的移动成本"""
        # 基础成本是目标位置的移动成本
        base_cost = self.terrain.get_movement_cost(*pos2)
        
        # 如果是对角线移动，增加额外成本
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        if dx == 1 and dy == 1:
            base_cost *= 1.414  # sqrt(2)
        
        return base_cost
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _cache_path(self, key: Tuple, path: List[Tuple[int, int]]):
        """缓存路径结果"""
        if len(self.path_cache) >= self.cache_size:
            # 简单的LRU：删除第一个条目
            first_key = next(iter(self.path_cache))
            del self.path_cache[first_key]
        
        self.path_cache[key] = path
    
    def clear_cache(self):
        """清空路径缓存"""
        self.path_cache.clear()
    
    def get_path_length(self, path: List[Tuple[int, int]]) -> float:
        """计算路径总长度"""
        if not path:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self._get_movement_cost(path[i], path[i + 1])
        
        return total_cost
    
    def optimize_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        优化路径，移除不必要的中间点
        
        Args:
            path: 原始路径
            
        Returns:
            优化后的路径
        """
        if len(path) <= 2:
            return path
        
        optimized = [path[0]]
        current = 0
        
        while current < len(path) - 1:
            # 尝试跳过中间点
            for i in range(len(path) - 1, current, -1):
                if self._has_line_of_sight(path[current], path[i]):
                    optimized.append(path[i])
                    current = i
                    break
            else:
                # 如果没有找到可跳过的点，移动到下一个
                current += 1
                if current < len(path):
                    optimized.append(path[current])
        
        return optimized
    
    def _has_line_of_sight(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """检查两点间是否有直线视野"""
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        if dx > dy:
            steps = dx
        else:
            steps = dy
        
        if steps == 0:
            return True
        
        x_increment = float(dx) / steps
        y_increment = float(dy) / steps
        
        x = x0
        y = y0
        
        for i in range(int(steps) + 1):
            ix, iy = int(round(x)), int(round(y))
            if not self.terrain.is_passable(ix, iy):
                return False
            x += x_increment
            y += y_increment
        
        return True
