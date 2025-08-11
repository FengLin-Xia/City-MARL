"""
多智能体城市环境
整合地形、寻路和地块系统
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import defaultdict

from .terrain_system import TerrainSystem, TerrainType
from .pathfinding import PathfindingSystem
from .land_system import LandSystem, LandType

class CityEnvironment(gym.Env):
    """多智能体城市环境"""
    
    def __init__(self, 
                 width: int = 50, 
                 height: int = 50,
                 num_agents: int = 4,
                 max_steps: int = 1000,
                 render_mode: Optional[str] = None):
        """
        初始化城市环境
        
        Args:
            width: 地图宽度
            height: 地图高度
            num_agents: 智能体数量
            max_steps: 最大步数
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # 初始化子系统
        self.terrain_system = TerrainSystem(width, height)
        self.pathfinding_system = PathfindingSystem(self.terrain_system)
        self.land_system = LandSystem(width, height)
        
        # 智能体状态
        self.agents = {}
        self.agent_positions = {}
        self.agent_resources = {}
        self.agent_goals = {}
        self.agent_paths = {}
        
        # 环境状态
        self.current_step = 0
        self.total_reward = 0.0
        self.episode_rewards = defaultdict(float)
        
        # 定义动作空间
        self.action_space = spaces.Dict({
            'agent_id': spaces.Discrete(num_agents),
            'action_type': spaces.Discrete(6),  # 移动、建设、升级、修复、收集、等待
            'target_x': spaces.Discrete(width),
            'target_y': spaces.Discrete(height),
            'land_type': spaces.Discrete(8)  # 地块类型
        })
        
        # 定义观察空间
        self.observation_space = spaces.Dict({
            'agent_id': spaces.Discrete(num_agents),
            'position': spaces.Tuple((spaces.Discrete(width), spaces.Discrete(height))),
            'resources': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
            'goal': spaces.Tuple((spaces.Discrete(width), spaces.Discrete(height))),
            'terrain_map': spaces.Box(low=0, high=5, shape=(height, width), dtype=np.int32),
            'land_map': spaces.Box(low=0, high=7, shape=(height, width), dtype=np.int32),
            'agent_positions': spaces.Box(low=0, high=max(width, height), 
                                        shape=(num_agents, 2), dtype=np.int32),
            'local_view': spaces.Box(low=0, high=7, shape=(7, 7), dtype=np.int32)
        })
        
        # 奖励设置
        self.reward_config = {
            'movement_cost': -0.1,
            'building_reward': 10.0,
            'upgrade_reward': 5.0,
            'repair_reward': 2.0,
            'resource_collection': 1.0,
            'goal_reached': 50.0,
            'collision_penalty': -10.0,
            'invalid_action_penalty': -5.0
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 重置选项
            
        Returns:
            (observations, info): 观察和额外信息
        """
        super().reset(seed=seed)
        
        # 重置环境状态
        self.current_step = 0
        self.total_reward = 0.0
        self.episode_rewards = defaultdict(float)
        
        # 生成地形
        self.terrain_system.generate_random_terrain(seed)
        
        # 初始化智能体
        self._initialize_agents()
        
        # 获取观察
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def step(self, actions: Dict) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        执行一步
        
        Args:
            actions: 智能体动作字典
            
        Returns:
            (observations, rewards, terminated, truncated, info): 环境反馈
        """
        self.current_step += 1
        
        # 执行所有智能体的动作
        rewards = defaultdict(float)
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                reward = self._execute_action(agent_id, action)
                rewards[agent_id] = reward
                self.episode_rewards[agent_id] += reward
        
        # 更新环境状态
        self._update_environment()
        
        # 检查终止条件
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # 获取观察和信息
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, dict(rewards), terminated, truncated, info
    
    def _initialize_agents(self):
        """初始化智能体"""
        self.agents = {}
        self.agent_positions = {}
        self.agent_resources = {}
        self.agent_goals = {}
        self.agent_paths = {}
        
        # 为每个智能体分配起始位置和目标
        available_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.terrain_system.is_passable(x, y):
                    available_positions.append((x, y))
        
        for agent_id in range(self.num_agents):
            # 随机选择起始位置
            start_pos = random.choice(available_positions)
            available_positions.remove(start_pos)
            
            # 随机选择目标位置
            goal_pos = random.choice(available_positions)
            
            self.agents[agent_id] = {
                'id': agent_id,
                'type': f'agent_{agent_id}',
                'efficiency': random.uniform(0.8, 1.2)
            }
            
            self.agent_positions[agent_id] = start_pos
            self.agent_resources[agent_id] = 100.0  # 初始资源
            self.agent_goals[agent_id] = goal_pos
            
            # 计算到目标的路径
            path = self.pathfinding_system.a_star(start_pos, goal_pos)
            self.agent_paths[agent_id] = path if path else []
    
    def _execute_action(self, agent_id: int, action: Dict) -> float:
        """执行单个智能体的动作"""
        reward = 0.0
        
        try:
            action_type = action['action_type']
            target_x = action['target_x']
            target_y = action['target_y']
            
            current_pos = self.agent_positions[agent_id]
            current_resources = self.agent_resources[agent_id]
            
            if action_type == 0:  # 移动
                reward = self._execute_movement(agent_id, target_x, target_y)
            
            elif action_type == 1:  # 建设
                land_type = LandType(action.get('land_type', 1))
                reward = self._execute_building(agent_id, target_x, target_y, land_type)
            
            elif action_type == 2:  # 升级
                reward = self._execute_upgrade(agent_id, target_x, target_y)
            
            elif action_type == 3:  # 修复
                cost = action.get('cost', 10.0)
                reward = self._execute_repair(agent_id, target_x, target_y, cost)
            
            elif action_type == 4:  # 收集资源
                reward = self._execute_collect(agent_id, target_x, target_y)
            
            elif action_type == 5:  # 等待
                reward = 0.0
            
            else:
                reward = self.reward_config['invalid_action_penalty']
        
        except Exception as e:
            print(f"执行动作时出错: {e}")
            reward = self.reward_config['invalid_action_penalty']
        
        return reward
    
    def _execute_movement(self, agent_id: int, target_x: int, target_y: int) -> float:
        """执行移动动作"""
        current_pos = self.agent_positions[agent_id]
        target_pos = (target_x, target_y)
        
        # 检查目标位置是否可通行
        if not self.terrain_system.is_passable(target_x, target_y):
            return self.reward_config['invalid_action_penalty']
        
        # 检查是否与其他智能体碰撞
        for other_id, other_pos in self.agent_positions.items():
            if other_id != agent_id and other_pos == target_pos:
                return self.reward_config['collision_penalty']
        
        # 计算移动成本
        path = self.pathfinding_system.a_star(current_pos, target_pos)
        if path:
            movement_cost = self.pathfinding_system.get_path_length(path)
            cost = movement_cost * self.reward_config['movement_cost']
            
            # 执行移动
            self.agent_positions[agent_id] = target_pos
            
            # 检查是否到达目标
            if target_pos == self.agent_goals[agent_id]:
                return self.reward_config['goal_reached']
            
            return cost
        else:
            return self.reward_config['invalid_action_penalty']
    
    def _execute_building(self, agent_id: int, x: int, y: int, land_type: LandType) -> float:
        """执行建设动作"""
        current_pos = self.agent_positions[agent_id]
        
        # 检查是否在目标位置附近
        if abs(current_pos[0] - x) > 1 or abs(current_pos[1] - y) > 1:
            return self.reward_config['invalid_action_penalty']
        
        # 检查资源是否足够
        building_cost = 20.0
        if self.agent_resources[agent_id] < building_cost:
            return self.reward_config['invalid_action_penalty']
        
        # 检查目标位置是否为空地
        if self.land_system.get_land_type(x, y) != LandType.EMPTY:
            return self.reward_config['invalid_action_penalty']
        
        # 执行建设
        success = self.land_system.set_land_function(x, y, land_type)
        if success:
            self.agent_resources[agent_id] -= building_cost
            return self.reward_config['building_reward']
        else:
            return self.reward_config['invalid_action_penalty']
    
    def _execute_upgrade(self, agent_id: int, x: int, y: int) -> float:
        """执行升级动作"""
        current_pos = self.agent_positions[agent_id]
        
        # 检查是否在目标位置附近
        if abs(current_pos[0] - x) > 1 or abs(current_pos[1] - y) > 1:
            return self.reward_config['invalid_action_penalty']
        
        # 检查资源是否足够
        upgrade_cost = 15.0
        if self.agent_resources[agent_id] < upgrade_cost:
            return self.reward_config['invalid_action_penalty']
        
        # 执行升级
        success = self.land_system.upgrade_land(x, y)
        if success:
            self.agent_resources[agent_id] -= upgrade_cost
            return self.reward_config['upgrade_reward']
        else:
            return self.reward_config['invalid_action_penalty']
    
    def _execute_repair(self, agent_id: int, x: int, y: int, cost: float) -> float:
        """执行修复动作"""
        current_pos = self.agent_positions[agent_id]
        
        # 检查是否在目标位置附近
        if abs(current_pos[0] - x) > 1 or abs(current_pos[1] - y) > 1:
            return self.reward_config['invalid_action_penalty']
        
        # 检查资源是否足够
        if self.agent_resources[agent_id] < cost:
            return self.reward_config['invalid_action_penalty']
        
        # 执行修复
        actual_cost = self.land_system.repair_land(x, y, cost)
        if actual_cost > 0:
            self.agent_resources[agent_id] -= actual_cost
            return self.reward_config['repair_reward']
        else:
            return self.reward_config['invalid_action_penalty']
    
    def _execute_collect(self, agent_id: int, x: int, y: int) -> float:
        """执行资源收集动作"""
        current_pos = self.agent_positions[agent_id]
        
        # 检查是否在目标位置
        if current_pos != (x, y):
            return self.reward_config['invalid_action_penalty']
        
        # 收集资源
        resource_value = self.terrain_system.get_resource_value(x, y)
        land_revenue = self.land_system.get_land_revenue(x, y)
        
        total_collection = resource_value + land_revenue
        self.agent_resources[agent_id] += total_collection
        
        return total_collection * self.reward_config['resource_collection']
    
    def _update_environment(self):
        """更新环境状态"""
        # 更新地块状态
        self.land_system.update_all_lands(1)
        
        # 更新智能体路径
        for agent_id in range(self.num_agents):
            current_pos = self.agent_positions[agent_id]
            goal_pos = self.agent_goals[agent_id]
            
            if current_pos != goal_pos:
                path = self.pathfinding_system.a_star(current_pos, goal_pos)
                self.agent_paths[agent_id] = path if path else []
    
    def _check_termination(self) -> bool:
        """检查是否终止"""
        # 检查所有智能体是否都到达目标
        all_reached = all(
            self.agent_positions[agent_id] == self.agent_goals[agent_id]
            for agent_id in range(self.num_agents)
        )
        
        return all_reached
    
    def _get_observations(self) -> Dict:
        """获取所有智能体的观察"""
        observations = {}
        
        for agent_id in range(self.num_agents):
            obs = self._get_agent_observation(agent_id)
            observations[agent_id] = obs
        
        return observations
    
    def _get_agent_observation(self, agent_id: int) -> Dict:
        """获取单个智能体的观察"""
        pos = self.agent_positions[agent_id]
        goal = self.agent_goals[agent_id]
        resources = self.agent_resources[agent_id]
        
        # 获取局部视野
        local_view = self._get_local_view(pos, 3)
        
        # 获取其他智能体位置
        other_positions = []
        for other_id in range(self.num_agents):
            if other_id != agent_id:
                other_positions.append(self.agent_positions[other_id])
            else:
                other_positions.append([0, 0])  # 占位符
        
        observation = {
            'agent_id': agent_id,
            'position': pos,
            'resources': np.array([resources], dtype=np.float32),
            'goal': goal,
            'terrain_map': self.terrain_system.terrain,
            'land_map': np.array([[self.land_system.get_land_type(x, y).value 
                                 for x in range(self.width)] 
                                for y in range(self.height)], dtype=np.int32),
            'agent_positions': np.array(other_positions, dtype=np.int32),
            'local_view': local_view
        }
        
        return observation
    
    def _get_local_view(self, pos: Tuple[int, int], radius: int) -> np.ndarray:
        """获取局部视野"""
        x, y = pos
        local_view = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int32)
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # 组合地形和地块信息
                    terrain_value = self.terrain_system.get_terrain_at(nx, ny)
                    land_value = self.land_system.get_land_type(nx, ny).value
                    local_view[dy + radius, dx + radius] = terrain_value + land_value * 10
                else:
                    local_view[dy + radius, dx + radius] = -1  # 边界外
        
        return local_view
    
    def _get_info(self) -> Dict:
        """获取环境信息"""
        info = {
            'step': self.current_step,
            'total_reward': self.total_reward,
            'episode_rewards': dict(self.episode_rewards),
            'terrain_stats': self.terrain_system.get_terrain_stats(),
            'land_stats': self.land_system.get_stats(),
            'agent_positions': dict(self.agent_positions),
            'agent_resources': dict(self.agent_resources)
        }
        return info
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            # 这里可以实现可视化渲染
            pass
    
    def close(self):
        """关闭环境"""
        pass
