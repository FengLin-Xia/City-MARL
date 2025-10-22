"""
v5.0 契约层

定义系统各模块间的接口和数据契约，确保松耦合和高内聚。
"""

from .contracts import (
    ActionCandidate,
    Sequence, 
    StepLog,
    EnvironmentState,
    Action,
    Observation,
    RewardTerms,
    AtomicAction,
    CandidateIndex
)

__all__ = [
    'ActionCandidate',
    'Sequence',
    'StepLog', 
    'EnvironmentState',
    'Action',
    'Observation',
    'RewardTerms',
    'AtomicAction',
    'CandidateIndex'
]


