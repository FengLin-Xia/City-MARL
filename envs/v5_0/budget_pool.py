#!/usr/bin/env python3
"""
v5.0 预算池管理

实现共享预算池的动态分配和管理。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class BudgetPool:
    """预算池"""
    name: str
    members: List[str]
    total_budget: float
    remaining_budget: float
    allocation_strategy: str = "dynamic"
    
    def can_afford(self, cost: float) -> bool:
        """检查是否有足够预算"""
        return self.remaining_budget >= cost
    
    def deduct(self, cost: float) -> bool:
        """扣除预算"""
        if self.can_afford(cost):
            self.remaining_budget -= cost
            return True
        return False
    
    def get_remaining(self) -> float:
        """获取剩余预算"""
        return self.remaining_budget
    
    def reset(self) -> None:
        """重置预算池"""
        self.remaining_budget = self.total_budget


class BudgetPoolManager:
    """预算池管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化预算池管理器
        
        Args:
            config: 预算池配置
        """
        self.pools: Dict[str, BudgetPool] = {}
        self.agent_pools: Dict[str, str] = {}  # agent -> pool_name
        
        # 初始化预算池
        budget_pools_config = config.get("budget_pools", {})
        for pool_name, pool_config in budget_pools_config.items():
            pool = BudgetPool(
                name=pool_name,
                members=pool_config.get("members", []),
                total_budget=pool_config.get("total_budget", 0),
                remaining_budget=pool_config.get("total_budget", 0),
                allocation_strategy=pool_config.get("allocation_strategy", "dynamic")
            )
            self.pools[pool_name] = pool
            
            # 建立智能体到预算池的映射
            for member in pool.members:
                self.agent_pools[member] = pool_name
    
    def get_pool_for_agent(self, agent: str) -> Optional[BudgetPool]:
        """获取智能体的预算池"""
        pool_name = self.agent_pools.get(agent)
        if pool_name:
            return self.pools.get(pool_name)
        return None
    
    def can_afford(self, agent: str, cost: float) -> bool:
        """检查智能体是否有足够预算"""
        pool = self.get_pool_for_agent(agent)
        if pool:
            return pool.can_afford(cost)
        return False
    
    def deduct(self, agent: str, cost: float) -> bool:
        """为智能体扣除预算"""
        pool = self.get_pool_for_agent(agent)
        if pool:
            return pool.deduct(cost)
        return False
    
    def get_remaining_budget(self, agent: str) -> float:
        """获取智能体的剩余预算"""
        pool = self.get_pool_for_agent(agent)
        if pool:
            return pool.get_remaining()
        return 0.0
    
    def reset_all_pools(self) -> None:
        """重置所有预算池"""
        for pool in self.pools.values():
            pool.reset()
    
    def get_pool_status(self) -> Dict[str, Any]:
        """获取预算池状态"""
        status = {}
        for pool_name, pool in self.pools.items():
            status[pool_name] = {
                "total_budget": pool.total_budget,
                "remaining_budget": pool.remaining_budget,
                "members": pool.members,
                "allocation_strategy": pool.allocation_strategy
            }
        return status
