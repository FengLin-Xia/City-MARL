"""
v5.0 核心契约定义

统一系统各模块间的数据结构和接口契约。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import numpy as np


@dataclass
class AtomicAction:
    """原子动作：点×类型组合（v5.1 多动作机制）"""
    point: int                            # 候选点索引（在 CandidateIndex.points 中的位置）
    atype: int                            # 动作类型索引（在 types_per_point[point] 中的位置）
    meta: Dict[str, Any] = field(default_factory=dict)  # 额外信息（action_id, slots等）
    
    def __post_init__(self):
        """验证数据完整性"""
        assert self.point >= 0, f"Point index must be non-negative, got {self.point}"
        assert self.atype >= 0, f"Action type index must be non-negative, got {self.atype}"
        assert isinstance(self.meta, dict), "Meta must be a dictionary"


@dataclass
class CandidateIndex:
    """候选索引：组织点×类型的二级结构（v5.1 多动作机制）"""
    points: List[int]                     # 可用点列表（点ID或槽位组ID）
    types_per_point: List[List[int]]      # 每个点可用的类型列表（action_id）
    point_to_slots: Dict[int, List[str]]  # 点到槽位的映射
    meta: Dict[str, Any] = field(default_factory=dict)  # 额外信息
    
    def __post_init__(self):
        """验证数据完整性"""
        assert len(self.points) == len(self.types_per_point), \
            "Points and types_per_point must have same length"
        assert all(len(types) > 0 for types in self.types_per_point), \
            "Each point must have at least one valid type"
        assert isinstance(self.point_to_slots, dict), "point_to_slots must be a dictionary"


@dataclass(frozen=True)
class ActionCandidate:
    """统一候选动作的评分对象"""
    id: int                   # 动作编号（0..N-1）
    features: np.ndarray      # 给策略网络打分的特征
    meta: Dict[str, Any]      # 例如 slot_id、agent、合法性标记等
    
    def __post_init__(self):
        """验证数据完整性"""
        assert self.id >= 0, f"Action ID must be non-negative, got {self.id}"
        assert len(self.features) > 0, "Features cannot be empty"
        assert isinstance(self.meta, dict), "Meta must be a dictionary"


@dataclass(frozen=False)
class Sequence:
    """统一动作序列（每个 agent 在一个阶段内的选择）
    
    v5.1 扩展: 支持 Union[int, AtomicAction] 以兼容多动作机制
    """
    agent: str                                    # "IND" / "EDU" / "COUNCIL"
    actions: List[Union[int, AtomicAction]]       # 支持旧版int和新版AtomicAction
    
    def __post_init__(self):
        """验证数据完整性并实现兼容层"""
        assert self.agent in ["IND", "EDU", "COUNCIL"], f"Invalid agent: {self.agent}"
        # 允许空动作列表（用于多动作STOP情况）
        # assert len(self.actions) > 0, "Actions list cannot be empty"
        
        # 兼容层：自动转换旧版int为AtomicAction（用于向后兼容）
        # 注意：这里的转换是保守的，point=0 表示使用默认点索引
        converted = []
        for a in self.actions:
            if isinstance(a, int):
                # 旧版：int action_id → AtomicAction(point=0, atype=action_id)
                # meta中保留原始action_id方便后续查找
                converted.append(AtomicAction(point=0, atype=a, meta={'legacy_id': a}))
            elif isinstance(a, AtomicAction):
                # 新版：直接使用
                converted.append(a)
            else:
                raise TypeError(f"Action must be int or AtomicAction, got {type(a)}")
        
        # 更新actions为转换后的列表
        object.__setattr__(self, 'actions', converted)
    
    def get_legacy_ids(self) -> List[int]:
        """获取旧版动作ID列表（用于兼容v5.0日志）"""
        legacy_ids = []
        for a in self.actions:
            if 'legacy_id' in a.meta:
                legacy_ids.append(a.meta['legacy_id'])
            elif 'action_id' in a.meta:
                legacy_ids.append(a.meta['action_id'])
            else:
                # 降级：使用atype作为action_id
                legacy_ids.append(a.atype)
        return legacy_ids


@dataclass
class StepLog:
    """统一一步日志"""
    t: int
    agent: str
    chosen: List[int]                     # 实际执行的动作编号
    reward_terms: Dict[str, float]        # {"revenue":..., "cost":..., ...}
    budget_snapshot: Optional[Dict[str, float]] = None
    slot_positions: Optional[List[Dict[str, Any]]] = None  # 槽位位置信息
    
    def __post_init__(self):
        """验证数据完整性"""
        assert self.t >= 0, f"Time step must be non-negative, got {self.t}"
        assert self.agent in ["IND", "EDU", "COUNCIL"], f"Invalid agent: {self.agent}"
        
        # 验证chosen字段（兼容int和AtomicAction）
        for a in self.chosen:
            if isinstance(a, int):
                assert a >= 0, f"Action ID must be non-negative, got {a}"
            elif isinstance(a, AtomicAction):
                # 如果是AtomicAction，提取action_id进行验证
                action_id = a.meta.get('action_id', a.atype)
                assert action_id >= 0, f"Action ID must be non-negative, got {action_id}"
            else:
                raise TypeError(f"chosen must contain int or AtomicAction, got {type(a)}")
        
        assert isinstance(self.reward_terms, dict), "Reward terms must be a dictionary"
        if self.budget_snapshot:
            assert isinstance(self.budget_snapshot, dict), "Budget snapshot must be a dictionary"


@dataclass
class EnvironmentState:
    """环境状态数据契约"""
    month: int
    land_prices: np.ndarray
    buildings: List[Dict[str, Any]]
    budgets: Dict[str, float]
    slots: List[Dict[str, Any]]
    
    def validate(self) -> bool:
        """验证数据完整性"""
        assert self.month >= 0, "Month must be non-negative"
        assert all(budget >= 0 for budget in self.budgets.values()), "Budgets must be non-negative"
        assert len(self.land_prices.shape) == 2, "Land prices must be 2D array"
        return True


@dataclass
class Action:
    """动作数据契约"""
    agent_id: str
    action_type: str
    target_slot: Optional[str]
    parameters: Dict[str, Any]
    
    def validate(self) -> bool:
        """验证动作合法性"""
        assert self.agent_id in ["EDU", "IND", "COUNCIL"], f"Invalid agent_id: {self.agent_id}"
        assert self.action_type in ["BUILD", "SKIP"], f"Invalid action_type: {self.action_type}"
        return True


@dataclass
class Observation:
    """观察数据契约"""
    agent_id: str
    features: np.ndarray
    metadata: Dict[str, Any]
    
    def validate(self) -> bool:
        """验证观察数据"""
        assert self.agent_id in ["EDU", "IND", "COUNCIL"], f"Invalid agent_id: {self.agent_id}"
        assert len(self.features) > 0, "Features cannot be empty"
        return True


@dataclass
class RewardTerms:
    """奖励分项数据契约"""
    revenue: float = 0.0
    cost: float = 0.0
    prestige: float = 0.0
    proximity: float = 0.0
    diversity: float = 0.0
    other: Dict[str, float] = None
    
    def __post_init__(self):
        if self.other is None:
            object.__setattr__(self, 'other', {})
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        result = {
            'revenue': self.revenue,
            'cost': self.cost,
            'prestige': self.prestige,
            'proximity': self.proximity,
            'diversity': self.diversity
        }
        result.update(self.other)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'RewardTerms':
        """从字典创建"""
        other = {k: v for k, v in data.items() 
                if k not in ['revenue', 'cost', 'prestige', 'proximity', 'diversity']}
        return cls(
            revenue=data.get('revenue', 0.0),
            cost=data.get('cost', 0.0),
            prestige=data.get('prestige', 0.0),
            proximity=data.get('proximity', 0.0),
            diversity=data.get('diversity', 0.0),
            other=other
        )


