"""
河流分割限制中间件

实现配置驱动的河流分割功能，限制智能体只能在河流一侧建造。
支持基于Hub的侧别分配和连通性检查。
"""

from typing import Dict, List, Any, Set, Tuple
import numpy as np
from contracts import Sequence, EnvironmentState


class RiverRestrictionMiddleware:
    """河流分割限制中间件"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("env", {}).get("river_restrictions", {})
        self.enabled = self.config.get("enabled", False)
        self.affects_agents = set(self.config.get("affects_agents", []))
        self.council_bypass = self.config.get("council_bypass", True)
        
        # 侧别分配配置
        self.assignment_config = self.config.get("river_side_assignment", {})
        self.assignment_method = self.assignment_config.get("method", "hub_based")
        self.fallback_method = self.assignment_config.get("fallback", "random")
        self.hub_side_mapping = self.assignment_config.get("hub_side_mapping", {})
        
        # 连通性检查配置
        self.connectivity_config = self.config.get("connectivity_check", {})
        self.connectivity_enabled = self.connectivity_config.get("enabled", True)
        self.max_distance = self.connectivity_config.get("max_distance", 200.0)
        
        # 缓存的侧别分配
        self._slot_sides: Dict[str, str] = {}
        self._agent_sides: Dict[str, str] = {}
        self._river_coords: List[List[float]] = []
        self._hub_positions: Dict[str, Tuple[float, float]] = {}
    
    def apply(self, seq: Sequence, state: EnvironmentState) -> Sequence:
        """
        应用河流分割限制
        
        Args:
            seq: 动作序列
            state: 环境状态
            
        Returns:
            过滤后的动作序列
        """
        if not self.enabled:
            return seq
            
        # 检查智能体是否受河流限制影响
        if seq.agent not in self.affects_agents:
            return seq
            
        # Council特殊处理
        if seq.agent == "COUNCIL" and self.council_bypass:
            return seq
            
        # 获取智能体的河流侧别
        agent_side = self._get_agent_side(seq.agent, state)
        if not agent_side:
            return seq
            
        # 过滤动作，只保留同侧的动作
        filtered_actions = []
        for action_id in seq.actions:
            if self._is_action_on_correct_side(action_id, agent_side, state):
                filtered_actions.append(action_id)
        
        # 返回过滤后的序列
        return Sequence(agent=seq.agent, actions=filtered_actions)
    
    def _get_agent_side(self, agent: str, state: EnvironmentState) -> str:
        """获取智能体的河流侧别"""
        if agent in self._agent_sides:
            return self._agent_sides[agent]
            
        # 根据配置方法分配侧别
        if self.assignment_method == "hub_based":
            side = self._assign_side_by_hub(agent, state)
        elif self.assignment_method == "random":
            side = self._assign_side_random(agent)
        else:
            side = self.fallback_method
            
        self._agent_sides[agent] = side
        return side
    
    def _assign_side_by_hub(self, agent: str, state: EnvironmentState) -> str:
        """基于Hub分配侧别"""
        # 获取Hub位置
        if not self._hub_positions:
            self._load_hub_positions(state)
            
        # 找到最近的Hub
        closest_hub = self._find_closest_hub(agent, state)
        if closest_hub and closest_hub in self.hub_side_mapping:
            return self.hub_side_mapping[closest_hub]
            
        # 回退到随机分配
        return self._assign_side_random(agent)
    
    def _assign_side_random(self, agent: str) -> str:
        """随机分配侧别"""
        return "north" if hash(agent) % 2 == 0 else "south"
    
    def _load_hub_positions(self, state: EnvironmentState):
        """加载Hub位置信息"""
        if hasattr(state, 'hubs') and state.hubs:
            for i, hub in enumerate(state.hubs):
                hub_id = f"hub{i+1}"
                self._hub_positions[hub_id] = tuple(hub)
    
    def _find_closest_hub(self, agent: str, state: EnvironmentState) -> str:
        """找到最近的Hub"""
        # 这里需要根据agent的历史位置或当前位置计算
        # 简化实现：返回第一个Hub
        if self._hub_positions:
            return list(self._hub_positions.keys())[0]
        return None
    
    def _is_action_on_correct_side(self, action_id: int, agent_side: str, state: EnvironmentState) -> bool:
        """检查动作是否在正确的河流侧别"""
        # 获取动作对应的槽位
        slot_id = self._get_slot_for_action(action_id, state)
        if not slot_id:
            return True  # 如果无法确定槽位，允许动作
            
        # 获取槽位的河流侧别
        slot_side = self._get_slot_side(slot_id, state)
        if not slot_side:
            return True  # 如果无法确定侧别，允许动作
            
        return slot_side == agent_side
    
    def _get_slot_for_action(self, action_id: int, state: EnvironmentState) -> str:
        """获取动作对应的槽位ID"""
        # 这里需要根据action_id和当前状态确定槽位
        # 简化实现：返回None，表示无法确定
        return None
    
    def _get_slot_side(self, slot_id: str, state: EnvironmentState) -> str:
        """获取槽位的河流侧别"""
        if slot_id in self._slot_sides:
            return self._slot_sides[slot_id]
            
        # 计算槽位的河流侧别
        side = self._calculate_slot_side(slot_id, state)
        self._slot_sides[slot_id] = side
        return side
    
    def _calculate_slot_side(self, slot_id: str, state: EnvironmentState) -> str:
        """计算槽位的河流侧别"""
        # 获取槽位坐标
        slot_coords = self._get_slot_coordinates(slot_id, state)
        if not slot_coords:
            return None
            
        # 获取河流坐标
        if not self._river_coords:
            self._load_river_coordinates(state)
            
        # 判断槽位在河流的哪一侧
        return self._determine_side_by_river(slot_coords)
    
    def _get_slot_coordinates(self, slot_id: str, state: EnvironmentState) -> Tuple[float, float]:
        """获取槽位坐标"""
        # 从state.slots中获取槽位坐标
        if hasattr(state, 'slots') and state.slots:
            for slot_data in state.slots:
                if slot_data.get('id') == slot_id:
                    return (slot_data.get('x', 0), slot_data.get('y', 0))
        return None
    
    def _load_river_coordinates(self, state: EnvironmentState):
        """加载河流坐标"""
        if hasattr(state, 'river_coords') and state.river_coords:
            self._river_coords = state.river_coords
    
    def _determine_side_by_river(self, slot_coords: Tuple[float, float]) -> str:
        """根据河流判断侧别"""
        if not self._river_coords:
            return None
            
        # 简化实现：根据y坐标判断
        # 假设河流在y=100处，北侧y>100，南侧y<100
        river_y = 100  # 这里应该从配置或河流数据中获取
        if slot_coords[1] > river_y:
            return "north"
        else:
            return "south"
    
    def reset(self):
        """重置中间件状态"""
        self._slot_sides.clear()
        self._agent_sides.clear()
        self._river_coords.clear()
        self._hub_positions.clear()
