"""
v5.0 邻近性奖励计算器

参照v4.1的邻近性逻辑。
"""

from typing import Dict, Any, List, Tuple
import math

from contracts import ActionCandidate, EnvironmentState


class V5ProximityCalculator:
    """邻近性奖励计算 - 参照v4.1的邻近性逻辑"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化邻近性计算器
        
        Args:
            config: v5.0配置
        """
        self.config = config
        self.proximity_config = config.get("reward_terms", {}).get("proximity", {})
        
        # 获取配置参数
        self.proximity_threshold = self.proximity_config.get("proximity_threshold", 10.0)
        self.proximity_reward = self.proximity_config.get("proximity_reward", 50.0)
        self.distance_penalty_coef = self.proximity_config.get("distance_penalty_coef", 2.0)
        self.proximity_scale_council = self.proximity_config.get("proximity_scale_COUNCIL_by_action", {
            "6": 1.0, "7": 1.2, "8": 1.5
        })
    
    def calculate_proximity_reward(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """
        计算邻近性奖励 - 参照v4.1的邻近性奖励逻辑
        
        Args:
            action: 动作候选
            state: 环境状态
            
        Returns:
            float: 邻近性奖励
        """
        try:
            # 1. 计算到最近建筑的距离
            min_distance = self._calculate_min_distance_to_buildings(action, state)
            
            # 2. 距离近 → 邻近奖励
            # 3. 距离远 → 距离惩罚
            if min_distance <= self.proximity_threshold:
                # 邻近奖励（距离越近，奖励越高）
                proximity_bonus = self.proximity_reward * (1.0 - min_distance / self.proximity_threshold)
            else:
                # 距离惩罚（距离越远，惩罚越大）
                proximity_bonus = -(min_distance - self.proximity_threshold) * self.distance_penalty_coef
            
            # 4. 支持按COUNCIL 6/7/8缩放 (对应v4.1的EDU A/B/C)
            agent = self._get_agent_from_action(action)
            if agent == "COUNCIL":
                action_id = str(action.id)
                scale = self.proximity_scale_council.get(action_id, 1.0)
                proximity_bonus *= scale
            
            return proximity_bonus
            
        except Exception as e:
            print(f"[PROXIMITY_ERROR] 计算邻近性奖励失败: {e}")
            return 0.0
    
    def _calculate_min_distance_to_buildings(self, action: ActionCandidate, state: EnvironmentState) -> float:
        """计算到最近建筑的距离"""
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
            
            # 获取现有建筑坐标
            building_coords = self._get_building_coordinates(state)
            if not building_coords:
                return float('inf')
            
            # 计算最小距离
            min_distance = float('inf')
            for slot_coord in slot_coords:
                for building_coord in building_coords:
                    distance = math.sqrt(
                        (slot_coord[0] - building_coord[0])**2 + 
                        (slot_coord[1] - building_coord[1])**2
                    )
                    min_distance = min(min_distance, distance)
            
            return min_distance
            
        except Exception as e:
            print(f"[PROXIMITY_ERROR] 计算最小距离失败: {e}")
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
    
    def _get_building_coordinates(self, state: EnvironmentState) -> List[Tuple[float, float]]:
        """获取现有建筑坐标"""
        try:
            # 这里需要从环境状态中获取建筑坐标
            # 简化实现：返回随机坐标
            import random
            coords = []
            for _ in range(10):  # 假设有10个现有建筑
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
