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
        """获取当前月份可用的槽位（支持Hub延迟激活）"""
        available_slots = set()
        
        # 获取演化配置
        evolution_config = self.config.get("land_price", {}).get("evolution", {})
        
        for hub_config in self.hub_list:
            hub_id = hub_config["id"]
            
            # 检查Hub是否已激活
            if not self._is_hub_active(hub_id, month, evolution_config):
                continue
            
            # 计算当前Hub的半径
            current_radius = self._compute_radius(hub_config, month)
                
            # 获取Hub位置
            hub_pos = self._get_hub_position(hub_id, state)
            if not hub_pos:
                continue
                
            # 找到在半径内的槽位
            hub_slots = self._find_slots_in_radius(hub_pos, current_radius, state)
            available_slots.update(hub_slots)
            
        return available_slots
    
    def _is_hub_active(self, hub_id: str, current_month: int, evolution_config: Dict) -> bool:
        """检查Hub是否在当前月份激活"""
        # 检查是否有hub特定的激活时间配置
        if hub_id == "hub3":
            hub3_activation_month = evolution_config.get("hub3_activation_month")
            if hub3_activation_month is not None:
                return current_month >= hub3_activation_month
        
        # 对于hub1和hub2，使用默认的hub_activation_month
        hub_activation_month = evolution_config.get("hub_activation_month", 7)
        return current_month >= hub_activation_month
    
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
            radii[hub_id] = self._compute_radius(hub_config, month)
                
        return radii
    
    def reset(self):
        """重置中间件状态"""
        self._hub_positions.clear()
        self._current_radii.clear()

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _compute_radius(self, hub_config: Dict[str, Any], month: int) -> float:
        """根据配置计算当前 Hub 的候选半径（支持减速曲线）"""
        R0 = float(hub_config.get("R0", 0.0))
        if self.candidate_mode != "cumulative":
            return R0

        month = max(0, int(month))
        schedule = hub_config.get("growth_schedule") or []
        default_dR = float(hub_config.get("dR", 0.0))

        if not schedule:
            return R0 + month * default_dR

        radius = R0
        prev_month = 0
        last_dR = default_dR

        for segment in schedule:
            seg_dR = float(segment.get("dR", last_dR))
            until = segment.get("until_month")

            if until is None:
                duration = max(0, month - prev_month)
                radius += duration * seg_dR
                return radius

            until = int(until)
            if month <= prev_month:
                return radius

            duration = max(0, min(month, until) - prev_month)
            if duration > 0:
                radius += duration * seg_dR
                prev_month += duration

            last_dR = seg_dR

            if month <= until:
                return radius

            prev_month = max(prev_month, until)

        if month > prev_month:
            radius += (month - prev_month) * last_dR

        return radius
