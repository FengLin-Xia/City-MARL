"""
候选范围限制中间件

实现基于Hub环带的候选范围限制，控制智能体只能在特定半径内建造。
支持累积模式和固定模式。
"""

from typing import Dict, List, Any, Set, Tuple
import math
from contracts import Sequence, EnvironmentState


class CandidateRangeMiddleware:
    """候选范围限制中间件"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("hubs", {})
        self.enabled = self.config.get("mode") == "explicit"
        self.candidate_mode = self.config.get("candidate_mode", "cumulative")
        self.tolerance = self.config.get("tol", 0.5)
        self.hub_list = self.config.get("list", [])
        
        # 缓存的Hub信息
        self._hub_positions: Dict[str, Tuple[float, float]] = {}
        self._current_radii: Dict[str, float] = {}
    
    def apply(self, seq: Sequence, state: EnvironmentState) -> Sequence:
        """
        应用候选范围限制
        
        Args:
            seq: 动作序列
            state: 环境状态
            
        Returns:
            过滤后的动作序列
        """
        if not self.enabled:
            return seq
            
        # 获取当前月份
        current_month = getattr(state, 'month', 0)
        
        # 计算当前可用的候选范围
        available_slots = self._get_available_slots(current_month, state)
        if not available_slots:
            return seq
            
        # 过滤动作，只保留在候选范围内的动作
        filtered_actions = []
        for action_id in seq.actions:
            if self._is_action_in_range(action_id, available_slots, state):
                filtered_actions.append(action_id)
        
        return Sequence(agent=seq.agent, actions=filtered_actions)
    
    def _get_available_slots(self, month: int, state: EnvironmentState) -> Set[str]:
        """获取当前月份可用的槽位"""
        available_slots = set()
        
        for hub_config in self.hub_list:
            hub_id = hub_config["id"]
            R0 = hub_config["R0"]
            dR = hub_config["dR"]
            
            # 计算当前Hub的半径
            if self.candidate_mode == "cumulative":
                # 累积模式：R = R0 + month * dR
                current_radius = R0 + month * dR
            else:
                # 固定模式：R = R0
                current_radius = R0
                
            # 获取Hub位置
            hub_pos = self._get_hub_position(hub_id, state)
            if not hub_pos:
                continue
                
            # 找到在半径内的槽位
            hub_slots = self._find_slots_in_radius(hub_pos, current_radius, state)
            available_slots.update(hub_slots)
            
        return available_slots
    
    def _get_hub_position(self, hub_id: str, state: EnvironmentState) -> Tuple[float, float]:
        """获取Hub位置"""
        if hub_id in self._hub_positions:
            return self._hub_positions[hub_id]
            
        # 从配置中获取Hub位置
        for hub_config in self.hub_list:
            if hub_config["id"] == hub_id:
                pos = (hub_config["x"], hub_config["y"])
                self._hub_positions[hub_id] = pos
                return pos
                
        return None
    
    def _find_slots_in_radius(self, hub_pos: Tuple[float, float], radius: float, state: EnvironmentState) -> Set[str]:
        """找到在指定半径内的槽位"""
        slots_in_range = set()
        
        if hasattr(state, 'slots') and state.slots:
            for slot_data in state.slots:
                slot_id = slot_data.get('id', 'unknown')
                slot_pos = (slot_data.get('x', 0), slot_data.get('y', 0))
                distance = self._calculate_distance(hub_pos, slot_pos)
                
                if distance <= radius + self.tolerance:
                    slots_in_range.add(slot_id)
                    
        return slots_in_range
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点间距离"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def _is_action_in_range(self, action_id: int, available_slots: Set[str], state: EnvironmentState) -> bool:
        """检查动作是否在候选范围内"""
        # 获取动作对应的槽位
        slot_id = self._get_slot_for_action(action_id, state)
        if not slot_id:
            return True  # 如果无法确定槽位，允许动作
            
        return slot_id in available_slots
    
    def _get_slot_for_action(self, action_id: int, state: EnvironmentState) -> str:
        """获取动作对应的槽位ID"""
        # 这里需要根据action_id和当前状态确定槽位
        # 简化实现：返回None，表示无法确定
        return None
    
    def get_current_radii(self, month: int) -> Dict[str, float]:
        """获取当前月份各Hub的半径"""
        radii = {}
        for hub_config in self.hub_list:
            hub_id = hub_config["id"]
            R0 = hub_config["R0"]
            dR = hub_config["dR"]
            
            if self.candidate_mode == "cumulative":
                radii[hub_id] = R0 + month * dR
            else:
                radii[hub_id] = R0
                
        return radii
    
    def reset(self):
        """重置中间件状态"""
        self._hub_positions.clear()
        self._current_radii.clear()
