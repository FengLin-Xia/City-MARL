#!/usr/bin/env python3
"""
轨迹系统模块
实现真实通勤轨迹和热力图
"""

import numpy as np
from typing import List, Dict, Tuple
import math

class TrajectorySystem:
    """轨迹系统：管理居民通勤轨迹和热力图"""
    
    def __init__(self, map_size: Tuple[int, int], config: Dict):
        """初始化轨迹系统"""
        self.map_size = map_size
        self.config = config
        
        # 初始化热力图
        self.commute_heatmap = np.zeros((map_size[1], map_size[0]))
        self.commercial_heatmap = np.zeros((map_size[1], map_size[0]))
        self.combined_heatmap = np.zeros((map_size[1], map_size[0]))
        
        # 轨迹配置
        self.trajectory_config = config.get('trajectory_config', {})
        self.decay_rate = config.get('trajectory_decay_rate', 0.8)
        
        # 轨迹类型
        self.trajectory_types = {
            'commute': {
                'color': self.trajectory_config.get('commute_color', '#0066CC'),
                'intensity': 1.0,
                'description': '住宅到工作地点的通勤轨迹'
            },
            'commercial': {
                'color': self.trajectory_config.get('commercial_color', '#CC3300'),
                'intensity': 0.7,
                'description': '购物和娱乐活动轨迹'
            }
        }
    
    def update_trajectories(self, residents: List[Dict], city_state: Dict):
        """更新居民轨迹"""
        # 不清空热力图，而是累积轨迹
        # 注释掉清空操作，让热力图保持历史累积
        # self.commute_heatmap.fill(0)
        # self.commercial_heatmap.fill(0)
        
        # 为每个居民生成轨迹
        for resident in residents:
            if resident.get('home') and resident.get('workplace'):
                self._generate_resident_trajectory(resident, city_state)
        
        # 更新综合热力图
        self._update_combined_heatmap()
    
    def _generate_resident_trajectory(self, resident: Dict, city_state: Dict):
        """为单个居民生成轨迹"""
        home_id = resident['home']
        workplace_id = resident['workplace']
        
        # 获取住宅和工作地点位置
        home_pos = self._get_building_position(home_id, city_state)
        workplace_pos = self._get_building_position(workplace_id, city_state)
        
        if home_pos and workplace_pos:
            # 生成通勤轨迹
            commute_path = self._generate_path(home_pos, workplace_pos)
            self._add_trajectory_to_heatmap(commute_path, 'commute')
            
            # 生成商业活动轨迹（从工作地点到购物地点）
            commercial_buildings = city_state.get('commercial', [])
            if commercial_buildings:
                # 选择最近或最繁忙的商业建筑
                shopping_pos = self._select_shopping_destination(workplace_pos, commercial_buildings)
                if shopping_pos:
                    commercial_path = self._generate_path(workplace_pos, shopping_pos)
                    self._add_trajectory_to_heatmap(commercial_path, 'commercial')
    
    def _get_building_position(self, building_id: str, city_state: Dict) -> List[int]:
        """获取建筑位置"""
        # 查找住宅建筑
        for building in city_state.get('residential', []):
            if building['id'] == building_id:
                return building['xy']
        
        # 查找商业建筑
        for building in city_state.get('commercial', []):
            if building['id'] == building_id:
                return building['xy']
        
        return None
    
    def _generate_path(self, start_pos: List[int], end_pos: List[int]) -> List[List[int]]:
        """生成两点间的路径（简化版A*算法）"""
        path = []
        current_pos = start_pos.copy()
        
        # 简单的直线路径生成
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # 计算步数
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return [start_pos]
        
        # 生成路径点
        for i in range(steps + 1):
            x = int(start_pos[0] + (dx * i) / steps)
            y = int(start_pos[1] + (dy * i) / steps)
            
            # 确保在边界内
            x = max(0, min(x, self.map_size[0] - 1))
            y = max(0, min(y, self.map_size[1] - 1))
            
            path.append([x, y])
        
        return path
    
    def _select_shopping_destination(self, workplace_pos: List[int], commercial_buildings: List[Dict]) -> List[int]:
        """选择购物目的地"""
        if not commercial_buildings:
            return None
        
        # 选择最近的商业建筑
        min_distance = float('inf')
        best_destination = None
        
        for building in commercial_buildings:
            distance = self._calculate_distance(workplace_pos, building['xy'])
            if distance < min_distance:
                min_distance = distance
                best_destination = building['xy']
        
        return best_destination
    
    def _calculate_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _add_trajectory_to_heatmap(self, path: List[List[int]], trajectory_type: str):
        """将轨迹添加到热力图"""
        intensity = self.trajectory_types[trajectory_type]['intensity']
        
        for pos in path:
            x, y = pos[0], pos[1]
            # 确保坐标在地图范围内
            if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                # 热力图矩阵使用 [y, x] 索引（行，列）
                if trajectory_type == 'commute':
                    self.commute_heatmap[y, x] += intensity
                elif trajectory_type == 'commercial':
                    self.commercial_heatmap[y, x] += intensity
    
    def _update_combined_heatmap(self):
        """更新综合热力图"""
        # 简单叠加
        self.combined_heatmap = self.commute_heatmap + self.commercial_heatmap
    
    def apply_decay(self):
        """应用热力图衰减"""
        decay_rate = self.decay_rate
        self.commute_heatmap *= decay_rate
        self.commercial_heatmap *= decay_rate
        self.combined_heatmap *= decay_rate
    
    def get_heatmap_data(self) -> Dict:
        """获取热力图数据"""
        return {
            'commute_heatmap': self.commute_heatmap.copy(),
            'commercial_heatmap': self.commercial_heatmap.copy(),
            'combined_heatmap': self.combined_heatmap.copy(),
            'trajectory_types': self.trajectory_types
        }
    
    def get_trajectory_stats(self) -> Dict:
        """获取轨迹统计信息"""
        return {
            'commute_intensity': {
                'max': float(np.max(self.commute_heatmap)),
                'avg': float(np.mean(self.commute_heatmap)),
                'total': float(np.sum(self.commute_heatmap))
            },
            'commercial_intensity': {
                'max': float(np.max(self.commercial_heatmap)),
                'avg': float(np.mean(self.commercial_heatmap)),
                'total': float(np.sum(self.commercial_heatmap))
            },
            'combined_intensity': {
                'max': float(np.max(self.combined_heatmap)),
                'avg': float(np.mean(self.combined_heatmap)),
                'total': float(np.sum(self.combined_heatmap))
            }
        }
