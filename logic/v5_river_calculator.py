"""
v5.0 河流溢价计算器

参照v4.1的河流溢价逻辑。
"""

from typing import Dict, Any, List, Tuple
import math

from contracts import ActionCandidate, EnvironmentState


class V5RiverCalculator:
    """河流溢价计算 - 参照v4.1的河流溢价逻辑"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化河流计算器
        
        Args:
            config: v5.0配置
        """
        self.config = config
        self.river_config = config.get("reward_terms", {}).get("river", {})
        
        # 获取配置参数
        self.river_half_distance = self.river_config.get("RiverD_half_m", 120.0)
        self.river_max_pct = self.river_config.get("RiverPmax_pct", {"IND": 20.0, "EDU": 15.0, "COUNCIL": 15.0})
        self.river_premium_cap = self.river_config.get("RiverPremiumCap_kGBP", 10000.0)
    
    def calculate_river_premium(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """
        计算河流溢价 - 参照v4.1的河流溢价逻辑
        
        Args:
            action: 动作候选
            state: 环境状态
            
        Returns:
            float: 河流溢价
        """
        try:
            # 1. 计算河流距离
            river_distance = self._calculate_river_distance(action, state)
            
            # 2. 计算衰减: 2^(-d / RiverD_half_m)
            if self.river_half_distance <= 1e-9:
                decay = 0.0
            else:
                decay = 2.0 ** (-(max(0.0, river_distance) / self.river_half_distance))
            
            # 3. 计算溢价: base * rpct * decay
            agent = self._get_agent_from_action(action)
            rpct = self.river_max_pct.get(agent, 0.0) / 100.0
            
            # 获取基础收入
            base_reward = self._get_base_reward_for_agent(agent)
            
            # 计算原始溢价
            raw_premium = base_reward * rpct * decay
            
            # 4. 应用上限和舍入
            premium = max(0.0, min(self.river_premium_cap, raw_premium))
            
            return premium
            
        except Exception as e:
            print(f"[RIVER_ERROR] 计算河流溢价失败: {e}")
            return 0.0
    
    def _calculate_river_distance(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """计算到河流的距离"""
        try:
            # 获取动作的槽位信息
            if hasattr(action, 'meta') and 'slots' in action.meta:
                slots = action.meta['slots']
            else:
                return float('inf')
            
            if not slots:
                return float('inf')
            
            # 获取槽位坐标
            slot_coords = self._get_slot_coordinates(slots)
            if not slot_coords:
                return float('inf')
            
            # 获取河流坐标
            river_coords = self._get_river_coordinates(state)
            if not river_coords:
                return float('inf')
            
            # 计算最小距离
            min_distance = float('inf')
            for slot_coord in slot_coords:
                for river_coord in river_coords:
                    distance = math.sqrt(
                        (slot_coord[0] - river_coord[0])**2 + 
                        (slot_coord[1] - river_coord[1])**2
                    )
                    min_distance = min(min_distance, distance)
            
            return min_distance
            
        except Exception as e:
            print(f"[RIVER_ERROR] 计算河流距离失败: {e}")
            return float('inf')
    
    def _get_slot_coordinates(self, slots: List[str]) -> List[Tuple[float, float]]:
        """获取槽位坐标"""
        try:
            # 这里需要从环境中获取槽位坐标
            # 简化实现：返回随机坐标
            import random
            coords = []
            for slot_id in slots:
                x = random.uniform(0, 200)
                y = random.uniform(0, 200)
                coords.append((x, y))
            return coords
        except Exception:
            return []
    
    def _get_river_coordinates(self, state: EnvironmentState) -> List[Tuple[float, float]]:
        """获取河流坐标"""
        try:
            # 这里需要从环境状态中获取河流坐标
            # 简化实现：返回随机坐标
            import random
            coords = []
            for _ in range(5):  # 假设有5个河流点
                x = random.uniform(0, 200)
                y = random.uniform(0, 200)
                coords.append((x, y))
            return coords
        except Exception:
            return []
    
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
    
    def _get_base_reward_for_agent(self, agent: str) -> float:
        """获取智能体的基础收入"""
        if agent == "EDU":
            return 140.0
        elif agent == "IND":
            return 180.0
        elif agent == "COUNCIL":
            return 160.0
        else:
            return 100.0
