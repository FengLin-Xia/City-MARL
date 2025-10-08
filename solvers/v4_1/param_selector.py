"""
v4.1 参数化选择器（保留原有逻辑，用于对照）
"""

from typing import Dict, List, Tuple, Set, Optional
from logic.v4_enumeration import V4Planner, Action, Sequence


class ParamSelector:
    """参数化选择器 - 使用穷举/束搜索方法"""
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.planner = V4Planner(cfg)
    
    def choose_action_sequence(
        self,
        slots: Dict,
        candidates: Set[str],
        occupied: Set[str],
        lp_provider,
        river_distance_provider=None,
        agent_types: Optional[List[str]] = None,
        sizes: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[Action], Sequence]:
        """选择最优动作序列"""
        actions, sequence = self.planner.plan(
            slots=slots,
            candidates=candidates,
            occupied=occupied,
            lp_provider=lp_provider,
            river_distance_provider=river_distance_provider,
            agent_types=agent_types,
            sizes=sizes,
        )
        return actions, sequence

