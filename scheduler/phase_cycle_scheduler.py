"""
v5.0 Phase Cycle 调度器

实现基于阶段的智能体调度，支持并发和顺序执行模式。
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PhaseConfig:
    """阶段配置"""
    agents: List[str]
    mode: str  # "concurrent" 或 "sequential"


class PhaseCycleScheduler:
    """Phase Cycle 调度器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化调度器
        
        Args:
            config: 调度器配置，包含step_unit, period, offset, phases等
        """
        self.step_unit = config.get("step_unit", "month")
        self.period = config.get("period", 2)
        self.offset = config.get("offset", 0)
        self.phases = [PhaseConfig(**phase) for phase in config.get("phases", [])]
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置完整性"""
        assert self.period > 0, f"Period must be positive, got {self.period}"
        assert len(self.phases) > 0, "At least one phase must be defined"
        
        for phase in self.phases:
            assert phase.mode in ["concurrent", "sequential"], f"Invalid mode: {phase.mode}"
            assert len(phase.agents) > 0, "Each phase must have at least one agent"
    
    def get_active_agents(self, step: int) -> List[str]:
        """
        根据当前步骤获取活跃智能体
        
        Args:
            step: 当前步骤（从0开始）
            
        Returns:
            活跃智能体列表
        """
        # 计算当前阶段
        phase_index = self._get_phase_index(step)
        phase = self.phases[phase_index]
        
        return phase.agents.copy()
    
    def get_execution_mode(self, step: int) -> str:
        """
        获取执行模式
        
        Args:
            step: 当前步骤
            
        Returns:
            执行模式："concurrent" 或 "sequential"
        """
        phase_index = self._get_phase_index(step)
        phase = self.phases[phase_index]
        
        return phase.mode
    
    def get_agent_order(self, step: int) -> List[str]:
        """
        获取智能体执行顺序
        
        Args:
            step: 当前步骤
            
        Returns:
            智能体执行顺序列表
        """
        return self.get_active_agents(step)
    
    def _get_phase_index(self, step: int) -> int:
        """
        计算当前步骤对应的阶段索引
        
        Args:
            step: 当前步骤
            
        Returns:
            阶段索引
        """
        # 考虑偏移量
        adjusted_step = step - self.offset
        
        # 计算阶段索引
        phase_index = (adjusted_step // self.period) % len(self.phases)
        
        return phase_index
    
    def get_phase_info(self, step: int) -> Dict[str, Any]:
        """
        获取阶段信息
        
        Args:
            step: 当前步骤
            
        Returns:
            阶段信息字典
        """
        phase_index = self._get_phase_index(step)
        phase = self.phases[phase_index]
        
        return {
            "phase_index": phase_index,
            "agents": phase.agents,
            "mode": phase.mode,
            "step": step,
            "period": self.period,
            "offset": self.offset
        }
    
    def is_agent_active(self, step: int, agent: str) -> bool:
        """
        检查智能体是否在当前步骤活跃
        
        Args:
            step: 当前步骤
            agent: 智能体名称
            
        Returns:
            是否活跃
        """
        active_agents = self.get_active_agents(step)
        return agent in active_agents
    
    def get_next_phase_change(self, current_step: int) -> int:
        """
        获取下一个阶段变更的步骤
        
        Args:
            current_step: 当前步骤
            
        Returns:
            下一个阶段变更的步骤
        """
        # 计算当前周期内的剩余步骤
        adjusted_step = current_step - self.offset
        remaining_in_period = self.period - (adjusted_step % self.period)
        
        return current_step + remaining_in_period
