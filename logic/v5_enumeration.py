"""
v5.0 动作枚举器

基于契约对象和配置的动作枚举系统。
"""

from typing import Dict, List, Any, Optional, Set
import numpy as np
from dataclasses import dataclass

from contracts import ActionCandidate, Sequence, StepLog
from config_loader import ConfigLoader
from utils.logger_factory import get_logger, topic_enabled, sampling_allows


@dataclass
class SlotInfo:
    """槽位信息"""
    slot_id: str
    x: int
    y: int
    neighbors: List[str]
    building_level: int = 3  # 建筑等级：3=只能建S, 4=可建S/M, 5=可建S/M/L
    occupied: bool = False
    reserved: bool = False


class V5ActionEnumerator:
    """v5.0动作枚举器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化枚举器
        
        Args:
            config: v5.0配置
        """
        self.config = config
        self.action_params = config.get("action_params", {})
        self.agents_config = config.get("agents", {})
        self.slots: Dict[str, SlotInfo] = {}
        self.logger = get_logger("enumeration")
        
    def load_slots(self, slots_data: List[Dict[str, Any]]) -> None:
        """
        加载槽位数据
        
        Args:
            slots_data: 槽位数据列表
        """
        self.slots = {}
        for slot_data in slots_data:
            slot = SlotInfo(
                slot_id=slot_data["id"],
                x=slot_data["x"],
                y=slot_data["y"],
                neighbors=slot_data.get("neighbors", []),
                building_level=slot_data.get("building_level", 3)
            )
            self.slots[slot.slot_id] = slot
    
    def enumerate_actions(self, agent: str, occupied_slots: Set[str], 
                         lp_provider, budget: float, current_month: int = 0) -> List[ActionCandidate]:
        """
        为指定智能体枚举动作
        
        Args:
            agent: 智能体名称
            occupied_slots: 已占用的槽位
            lp_provider: 地价提供函数
            budget: 预算
            
        Returns:
            动作候选列表
        """
        agent_config = self.agents_config.get("defs", {}).get(agent, {})
        action_ids = agent_config.get("action_ids", [])
        
        candidates = []
        
        for action_id in action_ids:
            # 获取动作参数
            action_params = self.action_params.get(str(action_id), {})
            if not action_params:
                continue
                
            # 检查预算
            cost = action_params.get("cost", 0)
            if cost > budget:
                continue
            
            # 枚举该动作的所有可能位置
            positions = self._enumerate_positions(action_id, occupied_slots, lp_provider, current_month, agent)
            
            for pos in positions:
                # 创建特征向量
                features = self._create_features(action_id, pos, lp_provider)
                
                # 创建元数据
                meta = {
                    "agent": agent,
                    "action_id": action_id,
                    "cost": cost,
                    "reward": action_params.get("reward", 0),
                    "prestige": action_params.get("prestige", 0),
                    "slots": pos["slots"],
                    "zone": pos.get("zone"),
                    "lp_norm": pos.get("lp_norm", 0.0)
                }
                
                candidate = ActionCandidate(
                    id=action_id,
                    features=features,
                    meta=meta
                )
                candidates.append(candidate)
        
        # 轻量日志：候选总数
        if topic_enabled("candidates") and sampling_allows(agent, current_month, None):
            self.logger.info(f"candidates_total agent={agent} month={current_month} count={len(candidates)}")
        return candidates
    
    def _enumerate_positions(self, action_id: int, occupied_slots: Set[str], 
                           lp_provider, current_month: int, agent: str) -> List[Dict[str, Any]]:
        """
        枚举动作的可能位置
        
        Args:
            action_id: 动作ID
            occupied_slots: 已占用槽位
            lp_provider: 地价提供函数
            
        Returns:
            位置列表
        """
        action_params = self.action_params.get(str(action_id), {})
        desc = action_params.get("desc", "")
        
        # 根据动作类型确定占地面积
        if "S" in desc:
            footprint_size = 1
        elif "M" in desc:
            footprint_size = 2
        elif "L" in desc:
            footprint_size = 4
        else:
            footprint_size = 1
        
        positions = []
        
        # 获取可用槽位
        available_slots = [sid for sid, slot in self.slots.items() 
                          if sid not in occupied_slots and not slot.occupied and not slot.reserved]
        initial_cnt = len(available_slots)
        
        # 应用候选范围限制
        available_slots = self._apply_candidate_range_filter(available_slots, current_month)
        after_range_cnt = len(available_slots)
        
        # 应用河流限制
        available_slots = self._apply_river_restriction_filter(available_slots, agent)
        after_river_cnt = len(available_slots)

        # 轻量日志：过滤前后数量
        if topic_enabled("candidates") and sampling_allows(agent, current_month, None):
            self.logger.info(
                f"slots_filter agent={agent} month={current_month} initial={initial_cnt} range={after_range_cnt} river={after_river_cnt}")
        
        if footprint_size == 1:
            # 单槽位动作
            for slot_id in available_slots:
                slot = self.slots[slot_id]
                lp_norm = float(lp_provider(slot_id))
                zone = self._calculate_zone(slot_id)
                
                positions.append({
                    "slots": [slot_id],
                    "zone": zone,
                    "lp_norm": lp_norm
                })
        else:
            # 多槽位动作（简化实现）
            for slot_id in available_slots:
                slot = self.slots[slot_id]
                if slot.building_level < footprint_size:
                    continue
                    
                # 尝试找到相邻槽位组成足迹
                footprint = self._find_footprint(slot_id, footprint_size, occupied_slots)
                if footprint:
                    lp_vals = [float(lp_provider(sid)) for sid in footprint]
                    lp_norm = float(sum(lp_vals) / max(1, len(lp_vals)))
                    zone = self._calculate_zone(slot_id)
                    
                    positions.append({
                        "slots": footprint,
                        "zone": zone,
                        "lp_norm": lp_norm
                    })
        
        return positions
    
    def _apply_candidate_range_filter(self, available_slots: List[str], current_month: int) -> List[str]:
        """应用候选范围过滤"""
        # 获取Hub配置
        hubs_config = self.config.get("hubs", {})
        if not hubs_config.get("mode") == "explicit":
            return available_slots
            
        hub_list = hubs_config.get("list", [])
        candidate_mode = hubs_config.get("candidate_mode", "cumulative")
        tolerance = hubs_config.get("tol", 0.5)
        
        filtered_slots = []
        
        for slot_id in available_slots:
            slot = self.slots[slot_id]
            slot_pos = (slot.x, slot.y)
            
            # 检查是否在任何Hub的候选范围内
            in_range = False
            for hub_config in hub_list:
                hub_pos = (hub_config["x"], hub_config["y"])
                R0 = hub_config["R0"]
                dR = hub_config["dR"]
                
                # 计算当前Hub的半径
                if candidate_mode == "cumulative":
                    current_radius = R0 + current_month * dR
                else:
                    current_radius = R0
                
                # 计算距离
                distance = ((slot_pos[0] - hub_pos[0])**2 + (slot_pos[1] - hub_pos[1])**2)**0.5
                
                if distance <= current_radius + tolerance:
                    in_range = True
                    break
            
            if in_range:
                filtered_slots.append(slot_id)
        
        return filtered_slots
    
    def _apply_river_restriction_filter(self, available_slots: List[str], agent: str) -> List[str]:
        """应用河流限制过滤"""
        # 获取河流限制配置
        river_config = self.config.get("env", {}).get("river_restrictions", {})
        if not river_config.get("enabled", False):
            return available_slots
            
        # 检查智能体是否受河流限制影响
        affects_agents = river_config.get("affects_agents", [])
        if agent not in affects_agents:
            return available_slots
            
        # Council特殊处理
        if agent == "COUNCIL" and river_config.get("council_bypass", True):
            return available_slots
            
        # 获取智能体的河流侧别
        agent_side = self._get_agent_river_side(agent)
        if not agent_side:
            return available_slots
            
        # 过滤槽位，只保留同侧的槽位
        filtered_slots = []
        for slot_id in available_slots:
            slot = self.slots[slot_id]
            slot_side = self._get_slot_river_side(slot.x, slot.y)
            
            if slot_side == agent_side:
                filtered_slots.append(slot_id)
        
        return filtered_slots
    
    def _get_agent_river_side(self, agent: str) -> str:
        """获取智能体的河流侧别"""
        # 简化实现：基于智能体类型分配侧别
        if agent == "IND":
            return "north"  # IND在河流北侧
        elif agent == "EDU":
            return "south"  # EDU在河流南侧
        else:
            return "north"  # 默认北侧
    
    def _get_slot_river_side(self, x: float, y: float) -> str:
        """获取槽位的河流侧别"""
        # 简化实现：基于Y坐标判断侧别
        # 假设河流在y=100处，北侧y<100，南侧y>=100
        river_y = 100
        if y < river_y:
            return "north"
        else:
            return "south"
    
    def _find_footprint(self, start_slot_id: str, size: int, occupied_slots: Set[str]) -> Optional[List[str]]:
        """
        寻找足迹槽位组合
        
        Args:
            start_slot_id: 起始槽位
            size: 足迹大小
            occupied_slots: 已占用槽位
            
        Returns:
            足迹槽位列表，如果找不到则返回None
        """
        if size == 1:
            return [start_slot_id]
        
        # 简化实现：只考虑单槽位
        # 实际实现需要根据建筑类型确定足迹形状
        return [start_slot_id]
    
    def _calculate_zone(self, slot_id: str) -> str:
        """
        计算槽位所属区域
        
        Args:
            slot_id: 槽位ID
            
        Returns:
            区域名称
        """
        # 简化实现：基于槽位ID或位置计算区域
        # 实际实现需要根据配置和地理信息计算
        return "default"
    
    def _create_features(self, action_id: int, position: Dict[str, Any], 
                        lp_provider) -> np.ndarray:
        """
        创建特征向量
        
        Args:
            action_id: 动作ID
            position: 位置信息
            lp_provider: 地价提供函数
            
        Returns:
            特征向量
        """
        # 基础特征
        features = [
            action_id,  # 动作ID
            position["lp_norm"],  # 地价强度
            len(position["slots"]),  # 占地面积
        ]
        
        # 添加槽位特征
        for slot_id in position["slots"]:
            slot = self.slots.get(slot_id)
            if slot:
                features.extend([slot.x, slot.y, slot.building_level])
        
        # 填充到固定长度
        target_length = 32  # 可配置
        while len(features) < target_length:
            features.append(0.0)
        
        return np.array(features[:target_length], dtype=np.float32)
    
    def create_sequence(self, agent: str, action_ids: List[int]) -> Sequence:
        """
        创建动作序列
        
        Args:
            agent: 智能体名称
            action_ids: 动作ID列表
            
        Returns:
            动作序列
        """
        return Sequence(
            agent=agent,
            actions=action_ids
        )
    
    def validate_sequence(self, sequence: Sequence, occupied_slots: Set[str]) -> bool:
        """
        验证序列合法性
        
        Args:
            sequence: 动作序列
            occupied_slots: 已占用槽位
            
        Returns:
            是否合法
        """
        # 简化实现：检查动作ID是否在智能体的允许范围内
        agent_config = self.agents_config.get("defs", {}).get(sequence.agent, {})
        allowed_actions = set(agent_config.get("action_ids", []))
        
        return all(action_id in allowed_actions for action_id in sequence.actions)
