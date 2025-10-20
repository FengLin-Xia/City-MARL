"""
v5.0 核心契约定义

统一系统各模块间的数据结构和接口契约。
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


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


@dataclass(frozen=True)
class Sequence:
    """统一动作序列（每个 agent 在一个阶段内的选择）"""
    agent: str                # "IND" / "EDU" / "COUNCIL"
    actions: List[int]        # 例如 [0,3,5]：全是编号
    
    def __post_init__(self):
        """验证数据完整性"""
        assert self.agent in ["IND", "EDU", "COUNCIL"], f"Invalid agent: {self.agent}"
        assert all(a >= 0 for a in self.actions), "All action IDs must be non-negative"
        assert len(self.actions) > 0, "Actions list cannot be empty"


@dataclass
class StepLog:
    """统一一步日志"""
    t: int
    agent: str
    chosen: List[int]                     # 实际执行的动作编号
    reward_terms: Dict[str, float]        # {"revenue":..., "cost":..., ...}
    budget_snapshot: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """验证数据完整性"""
        assert self.t >= 0, f"Time step must be non-negative, got {self.t}"
        assert self.agent in ["IND", "EDU", "COUNCIL"], f"Invalid agent: {self.agent}"
        assert all(a >= 0 for a in self.chosen), "All chosen action IDs must be non-negative"
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
