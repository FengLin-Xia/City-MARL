"""
v5.0 动作枚举器

基于契约对象和配置的动作枚举系统。
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
import math
from dataclasses import dataclass, field

from contracts import ActionCandidate, Sequence, StepLog, CandidateIndex, AtomicAction
from config_loader import ConfigLoader
from utils.logger_factory import get_logger, topic_enabled, sampling_allows


@dataclass
class SlotInfo:
    """槽位信息"""
    slot_id: str
    x: float
    y: float
    angle: float = 0.0  # 角度信息
    z: int = 0  # z坐标（整数）
    neighbors: List[str] = field(default_factory=list)
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
        
        # 读取建筑等级限制配置
        level_config = config.get("constraints", {}).get("building_level_restriction", {})
        self.building_level_enabled = level_config.get("enabled", True)
        self.level_applies_to = set(level_config.get("apply_to_agents", []))
        
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
                angle=slot_data.get("angle", 0.0),  # 添加角度信息
                z=slot_data.get("z", 0),  # z坐标（整数）
                neighbors=slot_data.get("neighbors", []),
                building_level=slot_data.get("building_level", 3)
            )
            self.slots[slot.slot_id] = slot
    
    def enumerate_actions(self, agent: str, occupied_slots: Set[str], 
                         lp_provider, budget: float, current_month: int = 0, 
                         unlocked_actions: Set[int] = None) -> List[ActionCandidate]:
        """
        为指定智能体枚举动作
        
        Args:
            agent: 智能体名称
            occupied_slots: 已占用的槽位
            lp_provider: 地价提供函数
            budget: 预算
            current_month: 当前月份
            unlocked_actions: 已解锁的动作集合
            
        Returns:
            动作候选列表
        """
        agent_config = self.agents_config.get("defs", {}).get(agent, {})
        action_ids = agent_config.get("action_ids", [])
        
        # 检查解锁状态
        if unlocked_actions is not None:
            # 过滤掉未解锁的动作
            original_count = len(action_ids)
            action_ids = [aid for aid in action_ids if aid in unlocked_actions]
            if topic_enabled("candidates"):
                self.logger.info(f"[ENUM_DEBUG] Agent {agent} unlocked actions: {unlocked_actions}, filtered action_ids: {action_ids} (from {original_count})")
        else:
            if topic_enabled("candidates"):
                self.logger.info(f"[ENUM_DEBUG] Agent {agent} no unlock info, using all actions: {action_ids}")
        
        # 调试：显示已占用槽位
        from utils.logger_factory import get_logger, topic_enabled
        
        # 检查特殊规则：start_after_month
        special_rules = agent_config.get("constraints", {}).get("special_rules", {})
        start_after_month = special_rules.get("start_after_month")
        if start_after_month is not None and current_month < start_after_month:
            if topic_enabled("candidates"):
                self.logger.info(f"[ENUM_DEBUG] Agent {agent} not active until month {start_after_month}, current={current_month}")
            return []
        logger = get_logger("enumeration")
        if topic_enabled("occupied_slots"):
            logger.info(f"[ENUM_DEBUG] enumerate_actions agent={agent}")
            logger.info(f"[ENUM_DEBUG]   occupied_slots: {occupied_slots}")
            logger.info(f"[ENUM_DEBUG]   occupied_slots count: {len(occupied_slots)}")
            logger.info(f"[ENUM_DEBUG]   total_slots: {len(self.slots)}")
            logger.info(f"[ENUM_DEBUG]   agent_config: {agent_config}")
            logger.info(f"[ENUM_DEBUG]   action_ids: {action_ids}")
        
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
        
        # 轻量日志：候选总数；若为0，强制警告输出
        if len(candidates) == 0:
            self.logger.warning(f"[ENUM_NO_CANDIDATES] agent={agent} month={current_month} reason=empty_after_filters")
        elif topic_enabled("candidates") and sampling_allows(agent, current_month, None):
            self.logger.info(f"candidates_total agent={agent} month={current_month} count={len(candidates)}")
        return candidates
    
    def enumerate_with_index(self, agent: str, occupied_slots: Set[str], 
                            lp_provider, budget: float, current_month: int = 0, 
                            unlocked_actions: Set[int] = None) -> Tuple[List[ActionCandidate], CandidateIndex]:
        """
        枚举动作并生成候选索引（v5.1 多动作机制）
        
        Args:
            agent: 智能体名称
            occupied_slots: 已占用的槽位
            lp_provider: 地价提供函数
            budget: 预算
            current_month: 当前月份
            unlocked_actions: 已解锁的动作集合
            
        Returns:
            (candidates, cand_idx) 元组
        """
        # 获取配置（为后续使用）
        agent_config = self.agents_config.get("defs", {}).get(agent, {})
        
        # 检查特殊规则：start_after_month
        # 注意：这个检查只应该影响配置了start_after_month的agent（如COUNCIL）
        special_rules = agent_config.get("constraints", {}).get("special_rules", {})
        start_after_month = special_rules.get("start_after_month")
        
        # 添加调试日志：检查IND是否被这个逻辑意外影响
        if agent == "IND":
            from utils.logger_factory import topic_enabled
            if topic_enabled("candidates"):
                self.logger.info(f"[IND_START_CHECK] Agent={agent} month={current_month} start_after_month={start_after_month} special_rules_keys={list(special_rules.keys())}")
        
        if start_after_month is not None and current_month < start_after_month:
            from utils.logger_factory import topic_enabled
            if topic_enabled("candidates"):
                self.logger.info(f"[ENUM_DEBUG] Agent {agent} not active until month {start_after_month}, current={current_month}")
            # 特别警告：如果IND被这个检查阻塞，这是异常的
            if agent == "IND":
                self.logger.warning(f"[IND_BLOCKED] WARNING: IND被start_after_month检查阻塞！这不应该发生！month={current_month} start_after_month={start_after_month}")
            return [], CandidateIndex(points=[], types_per_point=[], point_to_slots={})
        
        # Step 1: 枚举所有可用点
        available_points = self._enumerate_available_points(occupied_slots, lp_provider, current_month, agent)
        
        if not available_points:
            # 无可用点，强制诊断日志
            self.logger.warning(f"[ENUM_NO_CANDIDATES] agent={agent} month={current_month} reason=no_available_points")
            return [], CandidateIndex(points=[], types_per_point=[], point_to_slots={})
        
        # Step 2: 为每个点枚举可用类型
        point_ids = list(available_points.keys())
        types_per_point = []
        
        for point_id in point_ids:
            valid_types = self._get_valid_types_for_point(
                agent, point_id, available_points[point_id], budget, current_month, unlocked_actions
            )
            types_per_point.append(valid_types)
            if agent == "IND" and 15 <= current_month <= 18:
                self.logger.info(
                    f"[IND_DEBUG] point_types month={current_month} point={point_id} types={valid_types}")
        
        # Step 3: 过滤掉没有可用类型的点
        filtered_points = []
        filtered_types = []
        filtered_point_to_slots = {}
        
        for i, point_id in enumerate(point_ids):
            if len(types_per_point[i]) > 0:
                filtered_points.append(point_id)
                filtered_types.append(types_per_point[i])
                filtered_point_to_slots[point_id] = available_points[point_id]["slots"]
        if not filtered_points:
            # 有点但所有点无可用类型
            self.logger.warning(f"[ENUM_NO_CANDIDATES] agent={agent} month={current_month} reason=no_types_for_all_points points={len(point_ids)}")
        
        # Step 4: 构建候选索引
        cand_idx = CandidateIndex(
            points=filtered_points,
            types_per_point=filtered_types,
            point_to_slots=filtered_point_to_slots,
            meta={"agent": agent, "month": current_month}
        )

        # 追加：类型可用性统计（按动作ID聚合），强制输出（WARNING 级别）
        available_by_type: Dict[int, int] = {}
        for tlist in filtered_types:
            for aid in tlist:
                available_by_type[aid] = available_by_type.get(aid, 0) + 1
        if available_by_type:
            self.logger.warning(
                f"[CANDIDATE_STATS] agent={agent} month={current_month} available_by_type={dict(sorted(available_by_type.items()))}"
            )
        
        # Step 5: 生成候选列表（保持与原有接口兼容）
        candidates = []
        for p_idx, point_id in enumerate(cand_idx.points):
            for t_idx, action_id in enumerate(cand_idx.types_per_point[p_idx]):
                # 获取点信息
                point_info = available_points[point_id]
                
                # 创建特征向量
                features = self._create_features(action_id, point_info, lp_provider)
                
                # 创建元数据（包含点和类型索引）
                action_params = self.action_params.get(str(action_id), {})
                
                # 计算到Hub3的距离（如果是动作9、10、11）
                hub3_distance = float('inf')
                if action_id in [9, 10, 11]:
                    hub3_distance = self._calculate_hub3_distance(point_info["slots"])
                
                meta = {
                    "agent": agent,
                    "action_id": action_id,
                    "point_idx": p_idx,      # 新增：点索引
                    "type_idx": t_idx,        # 新增：类型索引
                    "point_id": point_id,     # 新增：点ID
                    "cost": action_params.get("cost", 0),
                    "reward": action_params.get("reward", 0),
                    "prestige": action_params.get("prestige", 0),
                    "slots": point_info["slots"],
                    "zone": point_info.get("zone"),
                    "lp_norm": point_info.get("lp_norm", 0.0),
                    "hub3_distance": hub3_distance  # 新增：到Hub3的距离
                }
                
                # 日志：记录cand.meta["slots"]的值和对应的point_id
                try:
                    self.logger.warning(f"[CAND_META_SLOTS] agent={agent} action_id={action_id} point_id={point_id} point_idx={p_idx} slots={point_info['slots']}")
                except Exception:
                    pass
                
                candidate = ActionCandidate(
                    id=action_id,
                    features=features,
                    meta=meta
                )
                candidates.append(candidate)
        
        # 轻量日志
        from utils.logger_factory import topic_enabled, sampling_allows
        if topic_enabled("candidates") and sampling_allows(agent, current_month, None):
            self.logger.info(
                f"candidates_indexed agent={agent} month={current_month} "
                f"points={len(cand_idx.points)} total_candidates={len(candidates)}"
            )
        
        return candidates, cand_idx
    
    def _enumerate_available_points(self, occupied_slots: Set[str], lp_provider, 
                                    current_month: int, agent: str) -> Dict[int, Dict[str, Any]]:
        """
        枚举所有可用点（槽位或槽位组）
        
        Args:
            occupied_slots: 已占用槽位
            lp_provider: 地价提供函数
            current_month: 当前月份
            agent: 智能体名称
            
        Returns:
            {point_id: {"slots": [...], "zone": ..., "lp_norm": ...}}
        """
        # 获取可用槽位（初始）
        initial_slots = [sid for sid, slot in self.slots.items() 
                          if sid not in occupied_slots and not slot.occupied and not slot.reserved]
        initial_cnt = len(initial_slots)

        # 应用过滤器（逐步计数）
        after_range = self._apply_candidate_range_filter(initial_slots, current_month)
        after_range_cnt = len(after_range)

        after_river = self._apply_river_restriction_filter(after_range, agent)
        after_river_cnt = len(after_river)

        # 阶段计数汇总（WARNING 强制输出）
        try:
            self.logger.warning(
                f"[ENUM_STAGE_COUNTS] agent={agent} month={current_month} initial={initial_cnt} "
                f"after_range={after_range_cnt} after_river={after_river_cnt}"
            )
        except Exception:
            pass
        
        available_slots = after_river
        
        # 为每个槽位创建一个点（修复：使用slot_id作为point_id，避免哈希冲突）
        available_points = {}
        for slot_id in available_slots:
            slot = self.slots[slot_id]
            point_id = slot_id  # 直接使用slot_id作为point_id，避免哈希冲突
            lp_norm = float(lp_provider(slot_id))
            zone = self._calculate_zone(slot_id)
            
            available_points[point_id] = {
                "slots": [slot_id],
                "zone": zone,
                "lp_norm": lp_norm,
                "slot_id": slot_id  # 保留原始slot_id
            }
        
        return available_points
    
    def _get_valid_types_for_point(self, agent: str, point_id: int, point_info: Dict[str, Any], 
                                   budget: float, current_month: int, unlocked_actions: Set[int] = None) -> List[int]:
        """
        获取指定点上的可用动作类型
        
        Args:
            agent: 智能体名称
            point_id: 点ID
            point_info: 点信息
            budget: 预算
            current_month: 当前月份
            
        Returns:
            可用的action_id列表
        """
        agent_config = self.agents_config.get("defs", {}).get(agent, {})
        action_ids = agent_config.get("action_ids", [])
        
        # 检查解锁状态
        if unlocked_actions is not None:
            # 过滤掉未解锁的动作
            original_count = len(action_ids)
            action_ids = [aid for aid in action_ids if aid in unlocked_actions]
            if topic_enabled("candidates"):
                self.logger.info(f"[ENUM_INDEX_DEBUG] Agent {agent} unlocked actions: {unlocked_actions}, filtered action_ids: {action_ids} (from {original_count})")
        else:
            if topic_enabled("candidates"):
                self.logger.info(f"[ENUM_INDEX_DEBUG] Agent {agent} no unlock info, using all actions: {action_ids}")
        
        valid_types = []
        
        for action_id in action_ids:
            # 获取动作参数
            action_params = self.action_params.get(str(action_id), {})
            if not action_params:
                continue
            
            # 检查预算
            cost = action_params.get("cost", 0)
            if cost > budget:
                continue
            
            # 检查槽位是否支持该动作类型
            desc = action_params.get("desc", "")
            
            # 根据动作类型确定占地面积和建筑等级要求
            if "S" in desc:
                footprint_size = 1
                required_level = 3  # S型建筑需要等级3
            elif "M" in desc:
                footprint_size = 2
                required_level = 4  # M型建筑需要等级4
            elif "L" in desc:
                footprint_size = 4
                required_level = 5  # L型建筑需要等级5
            else:
                footprint_size = 1
                required_level = 3
            
            # 检查建筑等级（根据配置决定是否检查，只对指定智能体生效）
            should_check_level = (self.building_level_enabled and agent in self.level_applies_to)
            if should_check_level:
                slot_id = point_info["slots"][0]
                slot = self.slots[slot_id]
                if slot.building_level < required_level:
                    continue
            
            # 该类型有效
            valid_types.append(action_id)
        
        return valid_types
    
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
        available_slots = self._apply_candidate_range_filter(available_slots, current_month, action_id)
        after_range_cnt = len(available_slots)
        
        # 应用河流限制
        available_slots = self._apply_river_restriction_filter(available_slots, agent)
        after_river_cnt = len(available_slots)

        # 轻量日志：过滤前后数量（对IND在15-18月加细化调试）
        if agent == "IND" and 15 <= current_month <= 18:
            self.logger.info(
                f"[IND_DEBUG] slots_filter action_id={action_id} month={current_month} initial={initial_cnt} after_range={after_range_cnt} after_river={after_river_cnt}")
        elif topic_enabled("candidates") and sampling_allows(agent, current_month, None):
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
                # 检查建筑等级（根据配置决定是否检查，只对指定智能体生效）
                should_check_level = (self.building_level_enabled and agent in self.level_applies_to)
                if should_check_level:
                    # 根据动作类型确定建筑等级要求
                    desc = action_params.get("desc", "")
                    if "S" in desc:
                        required_level = 3
                    elif "M" in desc:
                        required_level = 4
                    elif "L" in desc:
                        required_level = 5
                    else:
                        required_level = 3
                    
                    if slot.building_level < required_level:
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
    
    def _apply_candidate_range_filter(self, available_slots: List[str], current_month: int, action_id: int = None) -> List[str]:
        """应用候选范围过滤（支持Hub延迟激活和Hub3特定动作）"""
        # 获取Hub配置
        hubs_config = self.config.get("hubs", {})
        if not hubs_config.get("mode") == "explicit":
            return available_slots
            
        hub_list = hubs_config.get("list", [])
        candidate_mode = hubs_config.get("candidate_mode", "cumulative")
        tolerance = hubs_config.get("tol", 0.5)
        
        # 获取演化配置
        evolution_config = self.config.get("land_price", {}).get("evolution", {})
        
        # 检查是否是Hub3特定的动作
        is_hub3_only = False
        if action_id is not None:
            action_params = self.action_params.get(str(action_id), {})
            is_hub3_only = action_params.get("hub3_only", False)
        
        # 如果是Hub3特定动作，检查Hub3是否激活
        if is_hub3_only:
            if not self._is_hub_active("hub3", current_month, evolution_config):
                return []  # Hub3未激活，直接返回空列表
        
        filtered_slots = []
        
        for slot_id in available_slots:
            slot = self.slots[slot_id]
            slot_pos = (slot.x, slot.y)
            
            # 检查是否在任何激活Hub的候选范围内
            in_range = False
            for i, hub_config in enumerate(hub_list):
                hub_id = hub_config.get("id", f"hub{i+1}")
                
                # 如果是Hub3特定动作，只检查Hub3
                if is_hub3_only and hub_id != "hub3":
                    continue
                
                # 检查Hub是否已激活（优先使用 hub_config 中的 activation_month）
                activation_month = hub_config.get("activation_month")
                if activation_month is None:
                    # 如果没有配置，使用 evolution_config 中的值
                    if not self._is_hub_active(hub_id, current_month, evolution_config):
                        continue
                else:
                    # 使用 hub_config 中的 activation_month
                    if current_month < activation_month:
                        continue
                
                hub_pos = (hub_config["x"], hub_config["y"])
                
                # 计算当前Hub的半径（支持 growth_schedule，从激活月份开始计算）
                current_radius = self._compute_hub_radius(hub_config, current_month, candidate_mode, activation_month)
                
                # 计算距离
                distance = ((slot_pos[0] - hub_pos[0])**2 + (slot_pos[1] - hub_pos[1])**2)**0.5
                
                if distance <= current_radius + tolerance:
                    in_range = True
                    break
            
            if in_range:
                filtered_slots.append(slot_id)
        
        return filtered_slots
    
    def _compute_hub_radius(self, hub_config: Dict[str, Any], month: int, candidate_mode: str, activation_month: int = None) -> float:
        """根据配置计算当前 Hub 的候选半径（支持 growth_schedule 减速曲线，从激活月份开始计算）"""
        R0 = float(hub_config.get("R0", 0.0))
        if candidate_mode != "cumulative":
            return R0

        month = max(0, int(month))
        
        # 如果 hub 有激活月份，从激活月份开始计算生长时间
        if activation_month is not None:
            growth_months = max(0, month - activation_month)
        else:
            growth_months = month
        
        schedule = hub_config.get("growth_schedule") or []
        default_dR = float(hub_config.get("dR", 0.0))

        if not schedule:
            return R0 + growth_months * default_dR

        radius = R0
        prev_month = 0
        last_dR = default_dR

        for segment in schedule:
            seg_dR = float(segment.get("dR", last_dR))
            until = segment.get("until_month")

            if until is None:
                duration = max(0, growth_months - prev_month)
                radius += duration * seg_dR
                return radius

            until = int(until)
            if growth_months <= prev_month:
                return radius

            duration = max(0, min(growth_months, until) - prev_month)
            if duration > 0:
                radius += duration * seg_dR
                prev_month += duration

            last_dR = seg_dR

            if growth_months <= until:
                return radius

            prev_month = max(prev_month, until)

        if growth_months > prev_month:
            radius += (growth_months - prev_month) * last_dR

        return radius
    
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
        
        # 使用get_legacy_ids()兼容AtomicAction
        legacy_ids = sequence.get_legacy_ids()
        return all(action_id in allowed_actions for action_id in legacy_ids)
    
    def _calculate_hub3_distance(self, slots: List[str]) -> float:
        """
        计算槽位到Hub3的最小距离
        
        Args:
            slots: 槽位ID列表
            
        Returns:
            到Hub3的最小距离（如果Hub3不存在或槽位为空，返回inf）
        """
        # 获取Hub3位置
        hubs_config = self.config.get("hubs", {})
        hub_list = hubs_config.get("list", [])
        hub3_config = None
        for hub in hub_list:
            if hub.get("id") == "hub3":
                hub3_config = hub
                break
        
        if not hub3_config:
            self.logger.warning(f"[Hub3Distance] Hub3配置不存在")
            return float('inf')
        
        hub3_pos = (hub3_config["x"], hub3_config["y"])
        
        # 计算最小距离
        min_distance = float('inf')
        for slot_id in slots:
            slot = self.slots.get(slot_id)
            if slot:
                slot_pos = (slot.x, slot.y)
                distance = math.sqrt(
                    (slot_pos[0] - hub3_pos[0])**2 + 
                    (slot_pos[1] - hub3_pos[1])**2
                )
                min_distance = min(min_distance, distance)
            else:
                self.logger.warning(f"[Hub3Distance] 槽位不存在: slot_id={slot_id}")
        
        # 调试日志
        if min_distance == float('inf'):
            self.logger.warning(f"[Hub3Distance] 距离计算失败: slots={slots}, hub3_pos={hub3_pos}")
        else:
            self.logger.info(f"[Hub3Distance] 计算成功: slots={slots}, hub3_pos={hub3_pos}, distance={min_distance:.2f}")
        
        return min_distance
    
    def _is_hub_active(self, hub_id: str, current_month: int, evolution_config: Dict) -> bool:
        """检查Hub是否在当前月份激活"""
        # 检查是否有hub特定的激活时间配置
        if hub_id == "hub3":
            hub3_activation_month = evolution_config.get("hub3_activation_month")
            if hub3_activation_month is not None:
                is_active = current_month >= hub3_activation_month
                # print(f"[HUB_DEBUG] Hub3激活检查: month={current_month}, threshold={hub3_activation_month}, active={is_active}")
                return is_active
        
        # 对于hub1和hub2，使用默认的hub_activation_month
        hub_activation_month = evolution_config.get("hub_activation_month", 7)
        is_active = current_month >= hub_activation_month
        # print(f"[HUB_DEBUG] {hub_id}激活检查: month={current_month}, threshold={hub_activation_month}, active={is_active}")
        return is_active
