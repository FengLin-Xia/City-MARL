"""
v5.0 规模奖励计算器

参照v4.1的规模奖励逻辑。
"""

from typing import Dict, Any

from contracts import ActionCandidate, EnvironmentState


class V5SizeCalculator:
    """规模奖励计算 - 参照v4.1的规模奖励逻辑"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化规模计算器
        
        Args:
            config: v5.0配置
        """
        self.config = config
        self.size_config = config.get("reward_terms", {}).get("size_bonus", {})
        
        # 获取配置参数 - 现在直接按动作ID配置
        self.size_bonus_map = self.size_config
    
    def calculate_size_bonus(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """
        计算规模奖励 - 参照v4.1的规模奖励逻辑
        
        Args:
            action: 动作候选
            state: 环境状态
            
        Returns:
            float: 规模奖励
        """
        try:
            # 1. 鼓励建造M/L型建筑
            # 2. 不同规模有不同的奖励系数
            # 3. 支持按COUNCIL 6/7/8缩放 (对应v4.1的EDU A/B/C)
            
            action_id = action.id
            
            # 直接从配置中获取规模奖励
            size_bonus = self.size_bonus_map.get(str(action_id), 0)
            
            return float(size_bonus)
            
        except Exception as e:
            print(f"[SIZE_ERROR] 计算规模奖励失败: {e}")
            return 0.0
    
    def _get_agent_from_action(self, action: ActionCandidate) -> str:
        """从动作ID推断智能体类型"""
        action_id = action.id
        
        if action_id in [0, 1, 2]:
            return "EDU"
        elif action_id in [3, 4, 5]:
            return "IND"
        elif action_id in [6, 7, 8]:
            return "COUNCIL"
        else:
            return "UNKNOWN"
