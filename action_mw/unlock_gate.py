# action_mw/unlock_gate.py
"""
预算解锁中间件

根据预算条件和时间条件解锁动作，支持sticky和dynamic两种策略。
"""

from typing import Dict, Any, List, Set
from contracts.contracts import Sequence
import logging

logger = logging.getLogger(__name__)


class UnlockGateMW:
    """预算解锁中间件"""
    
    def __init__(self, params: Dict[str, Any] | None = None):
        """
        初始化解锁中间件
        
        Args:
            params: 中间件参数
        """
        self.params = params or {}
        self._unlocked: Dict[str, Set[int]] = {}
        
        logger.info("预算解锁中间件初始化")
    
    def _agent_budget(self, state, agent: str) -> float:
        """
        获取智能体预算
        
        Args:
            state: 环境状态
            agent: 智能体名称
            
        Returns:
            智能体预算
        """
        # 优先使用ledger系统
        if hasattr(state, 'ledger') and hasattr(state.ledger, 'balance'):
            return state.ledger.balance(agent)
        
        # 回退到budgets字典
        if hasattr(state, 'budgets') and agent in state.budgets:
            return state.budgets[agent]
        
        return 0.0
    
    def _total_budget(self, state) -> float:
        """
        获取总预算
        
        Args:
            state: 环境状态
            
        Returns:
            总预算
        """
        # 优先使用ledger系统
        if hasattr(state, 'ledger') and hasattr(state.ledger, 'balance'):
            if hasattr(state, 'agents'):
                return sum(state.ledger.balance(a) for a in state.agents)
        
        # 回退到budgets字典
        if hasattr(state, 'budgets'):
            return sum(state.budgets.values())
        
        return 0.0
    
    def apply(self, seq: Sequence, state) -> Sequence:
        """
        应用解锁逻辑
        
        Args:
            seq: 动作序列
            state: 环境状态
            
        Returns:
            过滤后的动作序列
        """
        # 获取解锁配置
        cfg = getattr(state, 'cfg', {})
        if not cfg:
            logger.warning("状态中未找到配置，跳过解锁检查")
            return seq
        
        action_unlocks = cfg.get("action_unlocks", {})
        rules: List[Dict[str, Any]] = action_unlocks.get("rules", [])
        persistence = action_unlocks.get("persistence", "sticky")
        
        if not rules:
            logger.debug("未配置解锁规则，跳过解锁检查")
            return seq
        
        # 检查当前时间
        current_month = getattr(state, 'month', getattr(state, 't', 0))
        if not hasattr(state, 'month') and not hasattr(state, 't'):
            logger.warning("状态中未找到时间信息，跳过解锁检查")
            return seq
        
        unlocked_now: Set[int] = set()
        candidate_locked: Set[int] = set()
        
        # 处理所有解锁规则
        for rule in rules:
            rule_agent = rule.get("agent")
            # 支持 "ALL" 表示所有智能体
            if rule_agent != "ALL" and rule_agent != seq.agent:
                continue
            
            # 收集需要解锁的动作
            unlock_actions = rule.get("unlock_action_ids", [])
            candidate_locked.update(unlock_actions)
            
            # 检查时间条件
            after_month = rule.get("after_month", 0)
            if current_month < after_month:
                logger.debug(f"规则 {rule.get('id', 'unknown')} 时间未到: {current_month} < {after_month}")
                continue
            
            # 检查预算条件
            passed = True
            
            # 检查智能体预算
            if "budget_threshold" in rule:
                agent_budget = self._agent_budget(state, seq.agent)
                budget_threshold = rule["budget_threshold"]
                passed &= (agent_budget >= budget_threshold)
                logger.debug(f"智能体预算检查: {agent_budget} >= {budget_threshold} = {passed}")
            
            # 检查总预算
            if "total_budget_threshold" in rule:
                total_budget = self._total_budget(state)
                total_threshold = rule["total_budget_threshold"]
                passed &= (total_budget >= total_threshold)
                logger.debug(f"总预算检查: {total_budget} >= {total_threshold} = {passed}")
            
            # 如果条件满足，解锁动作
            if passed:
                unlocked_now.update(unlock_actions)
                logger.info(f"规则 {rule.get('id', 'unknown')} 满足条件，解锁动作: {unlock_actions}")
            else:
                logger.debug(f"规则 {rule.get('id', 'unknown')} 条件不满足")
        
        # 处理解锁状态
        history = self._unlocked.setdefault(seq.agent, set())
        
        if persistence == "sticky":
            # Sticky模式：解锁后永久保持
            newly = unlocked_now - history
            if newly:
                logger.info(f"智能体 {seq.agent} 新解锁动作: {newly}")
                # 触发事件（如果支持）
                if hasattr(state, "events"):
                    try:
                        state.events.emit("ActionUnlocked", {
                            "t": current_month,
                            "agent": seq.agent,
                            "new_actions": list(newly)
                        })
                    except Exception as e:
                        logger.warning(f"触发解锁事件失败: {e}")
            
            history |= unlocked_now
            unlocked_effective = history
        else:
            # Dynamic模式：每次重新检查
            unlocked_effective = unlocked_now
        
        # 过滤动作序列
        filtered_actions = []
        for action in seq.actions:
            # 获取动作ID
            action_id = getattr(action, 'action_id', getattr(action, 'atype', None))
            if action_id is None:
                # 如果无法获取动作ID，保留动作
                filtered_actions.append(action)
                continue
            
            if action_id in unlocked_effective or action_id not in candidate_locked:
                filtered_actions.append(action)
            else:
                logger.debug(f"动作 {action_id} 被锁定，已过滤")
        
        filtered_seq = Sequence(agent=seq.agent, actions=filtered_actions)
        
        if len(filtered_seq.actions) != len(seq.actions):
            logger.info(f"智能体 {seq.agent} 动作过滤: {len(seq.actions)} -> {len(filtered_seq.actions)}")
        
        return filtered_seq
    
    def get_unlocked_actions(self, agent: str) -> Set[int]:
        """
        获取已解锁的动作
        
        Args:
            agent: 智能体名称
            
        Returns:
            已解锁的动作集合
        """
        return self._unlocked.get(agent, set()).copy()
    
    def reset_agent(self, agent: str) -> None:
        """
        重置智能体的解锁状态
        
        Args:
            agent: 智能体名称
        """
        if agent in self._unlocked:
            del self._unlocked[agent]
            logger.info(f"智能体 {agent} 解锁状态已重置")
    
    def reset_all(self) -> None:
        """重置所有解锁状态"""
        self._unlocked.clear()
        logger.info("所有解锁状态已重置")
