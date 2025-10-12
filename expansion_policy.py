#!/usr/bin/env python3
"""
锚点扩展策略模块
实现基于锚点槽位的多槽位序列扩展
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
import math


class ExpansionPolicy(ABC):
    """锚点扩展策略基类"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    @abstractmethod
    def expand(self, state: Dict[str, Any], anchor_slot_id: str, 
               available_slots: List[str], k: int = 5) -> Tuple[List[str], float]:
        """
        基于锚点槽位扩展生成多槽位序列
        
        Args:
            state: 环境状态信息
            anchor_slot_id: 锚点槽位ID
            available_slots: 可用槽位列表
            k: 扩展的槽位数量
            
        Returns:
            Tuple[扩展后的槽位列表, 扩展的log概率]
        """
        pass


class NearestKExpansion(ExpansionPolicy):
    """基于距离的最近K个槽位扩展策略"""
    
    def __init__(self, temperature: float = 1.0, rule: str = "euclidean"):
        super().__init__(temperature)
        self.rule = rule  # "euclidean", "manhattan", "weighted"
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """计算两个位置之间的距离"""
        if self.rule == "euclidean":
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        elif self.rule == "manhattan":
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        elif self.rule == "weighted":
            # 加权距离：x方向权重0.6，y方向权重0.4
            return 0.6 * abs(pos1[0] - pos2[0]) + 0.4 * abs(pos1[1] - pos2[1])
        else:
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _get_slot_position(self, slot_id: str, slots: Dict) -> Optional[Tuple[float, float]]:
        """获取槽位的位置信息"""
        slot = slots.get(slot_id)
        if slot is None:
            return None
        
        # 尝试从不同属性获取位置
        if hasattr(slot, 'fx') and hasattr(slot, 'fy'):
            return (float(slot.fx), float(slot.fy))
        elif hasattr(slot, 'x') and hasattr(slot, 'y'):
            return (float(slot.x), float(slot.y))
        else:
            return None
    
    def expand(self, state: Dict[str, Any], anchor_slot_id: str, 
               available_slots: List[str], k: int = 5) -> Tuple[List[str], float]:
        """
        基于锚点槽位扩展生成最近的k个槽位序列
        """
        if not available_slots or anchor_slot_id not in available_slots:
            return [anchor_slot_id], 0.0
        
        # 获取槽位位置信息
        slots = state.get('slots', {})
        anchor_pos = self._get_slot_position(anchor_slot_id, slots)
        
        if anchor_pos is None:
            # 如果无法获取锚点位置，返回锚点本身
            return [anchor_slot_id], 0.0
        
        # 计算所有可用槽位到锚点的距离
        slot_distances = []
        valid_slots = []
        
        for slot_id in available_slots:
            slot_pos = self._get_slot_position(slot_id, slots)
            if slot_pos is not None:
                distance = self._calculate_distance(anchor_pos, slot_pos)
                slot_distances.append((slot_id, distance))
                valid_slots.append(slot_id)
        
        if not slot_distances:
            return [anchor_slot_id], 0.0
        
        # 按距离排序，选择最近的k个槽位
        slot_distances.sort(key=lambda x: x[1])
        
        # 确保锚点在结果中（即使它不是最近的）
        selected_slots = [anchor_slot_id]
        selected_distances = [0.0]  # 锚点到自己的距离为0
        
        for slot_id, distance in slot_distances:
            if slot_id != anchor_slot_id and len(selected_slots) < k:
                selected_slots.append(slot_id)
                selected_distances.append(distance)
        
        # 计算扩展的log概率（基于距离的softmax）
        if len(selected_distances) > 1:
            # 使用距离的倒数作为权重，距离越近权重越高
            weights = [1.0 / (d + 1e-6) for d in selected_distances[1:]]  # 排除锚点
            weights = [w / self.temperature for w in weights]
            
            # 归一化权重
            weight_sum = sum(weights)
            if weight_sum > 0:
                normalized_weights = [w / weight_sum for w in weights]
                expansion_log_prob = sum(w * math.log(w + 1e-8) for w in normalized_weights if w > 0)
            else:
                expansion_log_prob = 0.0
        else:
            expansion_log_prob = 0.0
        
        return selected_slots, expansion_log_prob


class ClusterExpansion(ExpansionPolicy):
    """基于聚类的槽位扩展策略"""
    
    def __init__(self, temperature: float = 1.0, cluster_radius: float = 10.0):
        super().__init__(temperature)
        self.cluster_radius = cluster_radius
    
    def expand(self, state: Dict[str, Any], anchor_slot_id: str, 
               available_slots: List[str], k: int = 5) -> Tuple[List[str], float]:
        """
        基于锚点槽位聚类扩展生成槽位序列
        """
        if not available_slots or anchor_slot_id not in available_slots:
            return [anchor_slot_id], 0.0
        
        slots = state.get('slots', {})
        anchor_pos = self._get_slot_position(anchor_slot_id, slots)
        
        if anchor_pos is None:
            return [anchor_slot_id], 0.0
        
        # 找到锚点半径内的槽位
        cluster_slots = []
        for slot_id in available_slots:
            if slot_id == anchor_slot_id:
                continue
                
            slot_pos = self._get_slot_position(slot_id, slots)
            if slot_pos is None:
                continue
            
            distance = math.sqrt((anchor_pos[0] - slot_pos[0])**2 + (anchor_pos[1] - slot_pos[1])**2)
            if distance <= self.cluster_radius:
                cluster_slots.append((slot_id, distance))
        
        # 按距离排序，选择最近的k个
        cluster_slots.sort(key=lambda x: x[1])
        selected_slots = [anchor_slot_id] + [slot_id for slot_id, _ in cluster_slots[:k-1]]
        
        # 计算扩展概率
        expansion_log_prob = -len(selected_slots) * 0.1  # 简单的概率模型
        
        return selected_slots, expansion_log_prob
    
    def _get_slot_position(self, slot_id: str, slots: Dict) -> Optional[Tuple[float, float]]:
        """获取槽位的位置信息"""
        slot = slots.get(slot_id)
        if slot is None:
            return None
        
        if hasattr(slot, 'fx') and hasattr(slot, 'fy'):
            return (float(slot.fx), float(slot.fy))
        elif hasattr(slot, 'x') and hasattr(slot, 'y'):
            return (float(slot.x), float(slot.y))
        else:
            return None


class RandomExpansion(ExpansionPolicy):
    """随机扩展策略（用于对比测试）"""
    
    def expand(self, state: Dict[str, Any], anchor_slot_id: str, 
               available_slots: List[str], k: int = 5) -> Tuple[List[str], float]:
        """
        随机选择k个槽位进行扩展
        """
        if not available_slots or anchor_slot_id not in available_slots:
            return [anchor_slot_id], 0.0
        
        # 确保锚点在结果中
        other_slots = [s for s in available_slots if s != anchor_slot_id]
        
        # 随机选择其他槽位
        import random
        selected_others = random.sample(other_slots, min(k-1, len(other_slots)))
        
        selected_slots = [anchor_slot_id] + selected_others
        
        # 计算随机选择的log概率
        n_total = len(other_slots)
        n_selected = len(selected_others)
        
        if n_total > 0 and n_selected > 0:
            # 组合数的log
            log_prob = -math.log(math.comb(n_total, n_selected)) if n_selected <= n_total else 0.0
        else:
            log_prob = 0.0
        
        return selected_slots, log_prob


def create_expansion_policy(policy_config: Dict[str, Any]) -> ExpansionPolicy:
    """
    根据配置创建扩展策略实例
    
    Args:
        policy_config: 策略配置字典
        
    Returns:
        ExpansionPolicy实例
    """
    policy_type = policy_config.get('type', 'nearest_k')
    temperature = policy_config.get('temperature', 1.0)
    
    if policy_type == 'nearest_k':
        rule = policy_config.get('rule', 'euclidean')
        return NearestKExpansion(temperature=temperature, rule=rule)
    
    elif policy_type == 'cluster':
        cluster_radius = policy_config.get('cluster_radius', 10.0)
        return ClusterExpansion(temperature=temperature, cluster_radius=cluster_radius)
    
    elif policy_type == 'random':
        return RandomExpansion(temperature=temperature)
    
    else:
        raise ValueError(f"Unknown expansion policy type: {policy_type}")


# 测试函数
def test_expansion_policy():
    """测试扩展策略功能"""
    print("Testing ExpansionPolicy...")
    
    # 创建测试状态
    test_state = {
        'slots': {
            's_1': type('Slot', (), {'fx': 0.0, 'fy': 0.0})(),
            's_2': type('Slot', (), {'fx': 1.0, 'fy': 0.0})(),
            's_3': type('Slot', (), {'fx': 0.0, 'fy': 1.0})(),
            's_4': type('Slot', (), {'fx': 2.0, 'fy': 0.0})(),
            's_5': type('Slot', (), {'fx': 0.0, 'fy': 2.0})(),
        }
    }
    
    available_slots = ['s_1', 's_2', 's_3', 's_4', 's_5']
    anchor_slot = 's_1'
    
    # 测试NearestKExpansion
    nearest_policy = NearestKExpansion(temperature=1.0, rule='euclidean')
    selected_slots, log_prob = nearest_policy.expand(test_state, anchor_slot, available_slots, k=3)
    
    print(f"NearestKExpansion: {selected_slots}, log_prob: {log_prob:.4f}")
    
    # 测试ClusterExpansion
    cluster_policy = ClusterExpansion(temperature=1.0, cluster_radius=1.5)
    selected_slots, log_prob = cluster_policy.expand(test_state, anchor_slot, available_slots, k=3)
    
    print(f"ClusterExpansion: {selected_slots}, log_prob: {log_prob:.4f}")
    
    print("ExpansionPolicy test completed!")


if __name__ == "__main__":
    test_expansion_policy()



