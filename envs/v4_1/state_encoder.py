"""
v4.1 状态编码器
将城市环境状态编码为神经网络输入
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any


class CityStateEncoder:
    """城市状态编码器"""
    
    def __init__(self, map_size: Tuple[int, int] = (200, 200)):
        self.map_size = map_size
        self.W, self.H = map_size
        
        # 状态维度定义
        self.grid_channels = 5  # 占用、功能、锁期、地价、河距
        self.global_stats_dim = 15  # 月份、预算、计数等
        self.action_feat_dim = 20  # 动作特征维度
        
    def encode_state(self, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码环境状态为神经网络输入
        
        Args:
            state: 环境状态字典
            
        Returns:
            grid_data: [1, grid_channels, H, W] - 栅格数据
            global_stats: [1, global_stats_dim] - 全局统计
        """
        # 1. 编码栅格数据
        grid_data = self._encode_grid_data(state)
        
        # 2. 编码全局统计
        global_stats = self._encode_global_stats(state)
        
        return grid_data, global_stats
    
    def _encode_grid_data(self, state: Dict[str, Any]) -> torch.Tensor:
        """编码栅格数据"""
        # 初始化栅格通道
        grid_channels = torch.zeros(self.grid_channels, self.H, self.W, dtype=torch.float32)
        
        # 通道0: 占用状态 (0=空闲, 1=占用)
        occupancy_map = torch.zeros(self.H, self.W, dtype=torch.float32)
        
        # 通道1: 功能类型 (0=无, 1=public, 2=industrial)
        function_map = torch.zeros(self.H, self.W, dtype=torch.float32)
        
        # 通道2: 锁定期 (0=无锁定, >0=剩余锁定月数)
        lock_period_map = torch.zeros(self.H, self.W, dtype=torch.float32)
        
        # 通道3: 地价场
        land_price_map = torch.from_numpy(state.get('land_price_field', np.zeros((self.H, self.W))))
        
        # 通道4: 河岸距离
        river_distance_map = self._compute_river_distance_map(state)
        
        # 填充占用和功能信息
        self._fill_building_info(
            state, 
            occupancy_map, 
            function_map, 
            lock_period_map
        )
        
        # 组装栅格数据
        grid_channels[0] = occupancy_map
        grid_channels[1] = function_map
        grid_channels[2] = lock_period_map
        grid_channels[3] = land_price_map
        grid_channels[4] = river_distance_map
        
        # 添加batch维度
        return grid_channels.unsqueeze(0)  # [1, C, H, W]
    
    def _encode_global_stats(self, state: Dict[str, Any]) -> torch.Tensor:
        """编码全局统计信息"""
        stats = []
        
        # 基础信息
        month = state.get('month', 0)
        current_agent = state.get('current_agent', 'EDU')
        agent_turn = state.get('agent_turn', 0)
        
        # 归一化月份
        total_months = 290  # 从配置中获取
        normalized_month = month / total_months
        
        # 智能体编码 (one-hot)
        agent_encoding = [0.0, 0.0]  # [EDU, IND]
        if current_agent == 'EDU':
            agent_encoding[0] = 1.0
        elif current_agent == 'IND':
            agent_encoding[1] = 1.0
        
        # 回合信息
        normalized_turn = agent_turn / 2.0  # 假设2个智能体
        
        stats.extend([
            normalized_month,
            normalized_turn,
            *agent_encoding
        ])
        
        # 建筑统计
        monthly_stats = state.get('monthly_stats', {})
        total_buildings = monthly_stats.get('total_buildings', 0)
        public_buildings = monthly_stats.get('public_buildings', 0)
        industrial_buildings = monthly_stats.get('industrial_buildings', 0)
        
        # 归一化建筑数量
        max_buildings = 1000  # 假设最大建筑数
        stats.extend([
            total_buildings / max_buildings,
            public_buildings / max_buildings,
            industrial_buildings / max_buildings
        ])
        
        # 候选区域统计
        candidate_slots = state.get('candidate_slots', set())
        occupied_slots = state.get('occupied_slots', set())
        
        candidate_ratio = len(candidate_slots) / max(1, len(candidate_slots) + len(occupied_slots))
        occupied_ratio = len(occupied_slots) / max(1, len(candidate_slots) + len(occupied_slots))
        
        stats.extend([
            candidate_ratio,
            occupied_ratio
        ])
        
        # Hub距离统计
        hubs = state.get('hubs', [])
        if hubs:
            # 计算到最近hub的平均距离
            hub_distances = self._compute_hub_distances(state, hubs)
            avg_hub_distance = np.mean(hub_distances) if hub_distances else 0.0
            stats.append(min(avg_hub_distance / 100.0, 1.0))  # 归一化到[0,1]
        else:
            stats.append(0.0)
        
        # 奖励历史统计
        monthly_rewards = monthly_stats.get('monthly_rewards', {})
        for agent in ['EDU', 'IND']:
            rewards = monthly_rewards.get(agent, [])
            if rewards:
                avg_reward = np.mean(rewards)
                stats.append(np.tanh(avg_reward))  # 使用tanh归一化
            else:
                stats.append(0.0)
        
        # 填充到固定维度
        while len(stats) < self.global_stats_dim:
            stats.append(0.0)
        
        # 截断到固定维度
        stats = stats[:self.global_stats_dim]
        
        return torch.tensor(stats, dtype=torch.float32).unsqueeze(0)  # [1, global_stats_dim]
    
    def _compute_river_distance_map(self, state: Dict[str, Any]) -> torch.Tensor:
        """计算河岸距离图"""
        river_coords = state.get('river_coords', [])
        
        if not river_coords:
            return torch.zeros(self.H, self.W, dtype=torch.float32)
        
        river_distance = torch.zeros(self.H, self.W, dtype=torch.float32)
        
        for y in range(self.H):
            for x in range(self.W):
                # 计算到最近河流点的距离
                min_dist = float('inf')
                for rx, ry in river_coords:
                    dist = np.sqrt((x - rx)**2 + (y - ry)**2)
                    min_dist = min(min_dist, dist)
                
                # 归一化距离
                river_distance[y, x] = min(min_dist / 50.0, 1.0)  # 假设最大影响距离50像素
        
        return river_distance
    
    def _fill_building_info(self, state: Dict[str, Any], 
                          occupancy_map: torch.Tensor,
                          function_map: torch.Tensor, 
                          lock_period_map: torch.Tensor):
        """填充建筑信息到栅格"""
        buildings = state.get('buildings', {})
        
        # 填充public建筑
        for building in buildings.get('public', []):
            xy = building.get('xy', [0, 0])
            x, y = int(round(xy[0])), int(round(xy[1]))
            
            if 0 <= x < self.W and 0 <= y < self.H:
                occupancy_map[y, x] = 1.0
                function_map[y, x] = 1.0
                
                # 计算锁定期（简化处理）
                month_built = building.get('month', 0)
                current_month = state.get('month', 0)
                lock_period = max(0, 3 - (current_month - month_built))  # 假设锁定3个月
                lock_period_map[y, x] = lock_period / 3.0  # 归一化
        
        # 填充industrial建筑
        for building in buildings.get('industrial', []):
            xy = building.get('xy', [0, 0])
            x, y = int(round(xy[0])), int(round(xy[1]))
            
            if 0 <= x < self.W and 0 <= y < self.H:
                occupancy_map[y, x] = 1.0
                function_map[y, x] = 2.0
                
                # 计算锁定期
                month_built = building.get('month', 0)
                current_month = state.get('month', 0)
                lock_period = max(0, 3 - (current_month - month_built))
                lock_period_map[y, x] = lock_period / 3.0
    
    def _compute_hub_distances(self, state: Dict[str, Any], hubs: List[List[float]]) -> List[float]:
        """计算到hub的距离"""
        distances = []
        occupied_slots = state.get('occupied_slots', set())
        
        for slot_id in occupied_slots:
            # 这里需要从slots中获取坐标，简化处理
            # 实际应该从state中获取slot信息
            pass
        
        return distances
    
    def encode_action_features(self, actions: List, state: Dict[str, Any]) -> torch.Tensor:
        """编码动作特征"""
        if not actions:
            return torch.zeros(1, 0, self.action_feat_dim, dtype=torch.float32)
        
        features = []
        for action in actions:
            feat = self._extract_single_action_features(action, state)
            features.append(feat)
        
        return torch.stack(features).unsqueeze(0)  # [1, num_actions, action_feat_dim]
    
    def _extract_single_action_features(self, action, state: Dict[str, Any]) -> torch.Tensor:
        """提取单个动作的特征"""
        features = []
        
        # 基础动作特征
        if hasattr(action, 'score'):
            features.extend([
                float(action.score),
                float(getattr(action, 'cost', 0.0)),
                float(getattr(action, 'reward', 0.0)),
                float(getattr(action, 'prestige', 0.0)),
                len(getattr(action, 'footprint_slots', []))
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 位置特征
        footprint_slots = getattr(action, 'footprint_slots', [])
        if footprint_slots:
            # 计算中心位置
            center_x, center_y = self._compute_action_center(footprint_slots, state)
            features.extend([
                center_x / self.W,
                center_y / self.H
            ])
        else:
            features.extend([0.0, 0.0])
        
        # 地价特征
        if footprint_slots:
            prices = self._get_action_land_prices(footprint_slots, state)
            features.extend([
                np.mean(prices),
                np.std(prices),
                np.min(prices),
                np.max(prices)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 邻接特征
        adjacency_score = self._compute_adjacency_score(footprint_slots, state)
        features.append(adjacency_score)
        
        # 河流距离特征
        river_distance = self._compute_action_river_distance(footprint_slots, state)
        features.append(river_distance)
        
        # Hub距离特征
        hub_distance = self._compute_action_hub_distance(footprint_slots, state)
        features.append(hub_distance)
        
        # 填充到固定维度
        while len(features) < self.action_feat_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.action_feat_dim], dtype=torch.float32)
    
    def _compute_action_center(self, footprint_slots: List[str], state: Dict[str, Any]) -> Tuple[float, float]:
        """计算动作中心位置"""
        # 简化实现，实际需要从state中获取slot坐标信息
        return 0.0, 0.0
    
    def _get_action_land_prices(self, footprint_slots: List[str], state: Dict[str, Any]) -> List[float]:
        """获取动作涉及的地价"""
        # 简化实现
        return [0.0] * len(footprint_slots)
    
    def _compute_adjacency_score(self, footprint_slots: List[str], state: Dict[str, Any]) -> float:
        """计算邻接得分"""
        # 简化实现
        return 0.0
    
    def _compute_action_river_distance(self, footprint_slots: List[str], state: Dict[str, Any]) -> float:
        """计算动作到河流的距离"""
        # 简化实现
        return 0.0
    
    def _compute_action_hub_distance(self, footprint_slots: List[str], state: Dict[str, Any]) -> float:
        """计算动作到hub的距离"""
        # 简化实现
        return 0.0

