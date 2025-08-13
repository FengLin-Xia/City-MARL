#!/usr/bin/env python3
"""
地形道路强化学习环境
支持mesh导入、高程提取和道路规划
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time

class TerrainType(Enum):
    """地形类型"""
    WATER = 0
    GRASS = 1
    FOREST = 2
    MOUNTAIN = 3
    ROAD = 4
    BUILDING = 5

class RoadAction(Enum):
    """道路动作类型"""
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    BUILD_ROAD = 4
    UPGRADE_ROAD = 5
    WAIT = 6

class TerrainRoadEnvironment(gym.Env):
    """
    地形道路强化学习环境
    
    状态空间：
    - 地形高程图 (height_map)
    - 地形类型图 (terrain_map)
    - 道路网络图 (road_map)
    - 智能体位置 (agent_pos)
    - 目标位置 (target_pos)
    - 资源状态 (resources)
    
    动作空间：
    - 移动方向 (4个方向)
    - 建造道路
    - 升级道路
    - 等待
    """
    
    def __init__(self, 
                 mesh_file: Optional[str] = None,
                 grid_size: Tuple[int, int] = (50, 50),
                 max_steps: int = 1000,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        self.height_map = None
        self.terrain_map = None
        self.road_map = None
        self.agent_pos = None
        self.target_pos = None
        self.resources = None
        self.current_step = 0
        
        # 可视化相关
        self.fig = None
        self.ax = None
        self.agent_scatter = None
        self.target_scatter = None
        self.path_line = None
        self.agent_path = []
        
        # 回放相关
        self.episode_history = []
        self.recording = False
        self.current_frame = 0
        
        if mesh_file:
            self.load_mesh(mesh_file)
        else:
            self._generate_random_terrain()
        
        self.observation_space = spaces.Dict({
            'height_map': spaces.Box(low=0, high=100, shape=grid_size, dtype=np.float32),
            'terrain_map': spaces.Box(low=0, high=5, shape=grid_size, dtype=np.int32),
            'road_map': spaces.Box(low=0, high=3, shape=grid_size, dtype=np.int32),
            'agent_pos': spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.int32),
            'target_pos': spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.int32),
            'resources': spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)  # 金钱、材料、时间
        })
        
        self.action_space = spaces.Discrete(7)  # 7个动作
        
        self.directions = [
            (-1, 0),  # 北
            (1, 0),   # 南
            (0, 1),   # 东
            (0, -1),  # 西
        ]
        
        self.costs = {
            'road_build': 10,
            'road_upgrade': 5,
            'movement': 1,
            'terrain_penalty': {
                TerrainType.WATER: 10,
                TerrainType.GRASS: 1,
                TerrainType.FOREST: 3,
                TerrainType.MOUNTAIN: 8,
                TerrainType.ROAD: 0.5,
                TerrainType.BUILDING: 15
            }
        }
    
    def start_recording(self):
        """开始记录episode"""
        self.recording = True
        self.episode_history = []
        print("🎬 开始记录episode...")
    
    def stop_recording(self):
        """停止记录episode"""
        self.recording = False
        print(f"📹 记录完成，共{len(self.episode_history)}帧")
    
    def save_episode(self, filename: str = None):
        """保存episode到文件"""
        if not self.episode_history:
            print("❌ 没有可保存的episode数据")
            return
        
        if filename is None:
            filename = f"episode_{int(time.time())}.json"
        
        # 转换numpy数组为列表
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif hasattr(obj, 'dtype'):  # 处理其他numpy类型
                return obj.item()
            else:
                return obj
        
        episode_data = {
            'grid_size': self.grid_size,
            'height_map': self.height_map.tolist(),
            'terrain_map': self.terrain_map.tolist(),
            'target_pos': self.target_pos.tolist(),
            'frames': convert_numpy(self.episode_history),
            'metadata': {
                'total_steps': len(self.episode_history),
                'final_reward': self.episode_history[-1]['reward'] if self.episode_history else 0,
                'success': self.episode_history[-1]['done'] if self.episode_history else False
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        print(f"💾 Episode已保存到: {filename}")
    
    def load_episode(self, filename: str):
        """从文件加载episode"""
        with open(filename, 'r') as f:
            episode_data = json.load(f)
        
        self.grid_size = tuple(episode_data['grid_size'])
        self.height_map = np.array(episode_data['height_map'])
        self.terrain_map = np.array(episode_data['terrain_map'])
        self.target_pos = np.array(episode_data['target_pos'])
        self.episode_history = episode_data['frames']
        
        print(f"📂 Episode已加载: {filename}")
        print(f"📊 总帧数: {len(self.episode_history)}")
        print(f"🎯 目标位置: {self.target_pos}")
        print(f"✅ 是否成功: {episode_data['metadata']['success']}")
        print(f"🏆 最终奖励: {episode_data['metadata']['final_reward']:.2f}")

    def load_mesh(self, mesh_file: str):
        """从mesh文件加载地形数据"""
        try:
            if mesh_file.endswith('.npy'):
                # 加载NumPy数组格式的mesh
                mesh_data = np.load(mesh_file)
                self._process_mesh_data(mesh_data)
            elif mesh_file.endswith('.json'):
                # 加载JSON格式的mesh
                with open(mesh_file, 'r') as f:
                    mesh_data = json.load(f)
                self._process_mesh_data(mesh_data)
            elif mesh_file.endswith('.txt'):
                # 加载文本格式的mesh
                mesh_data = np.loadtxt(mesh_file)
                self._process_mesh_data(mesh_data)
            else:
                print(f"不支持的文件格式: {mesh_file}")
                self._generate_random_terrain()
                
        except Exception as e:
            print(f"加载mesh文件失败: {e}")
            self._generate_random_terrain()
    
    def _process_mesh_data(self, mesh_data):
        """处理mesh数据，提取高程和地形信息"""
        if isinstance(mesh_data, np.ndarray):
            # 假设mesh_data是3D数组 [height, width, channels]
            # channels: [elevation, terrain_type, ...]
            if len(mesh_data.shape) == 3:
                self.height_map = mesh_data[:, :, 0]  # 第一层是高程
                if mesh_data.shape[2] > 1:
                    self.terrain_map = mesh_data[:, :, 1].astype(np.int32)  # 第二层是地形类型
                else:
                    self._generate_terrain_from_height()
            else:
                # 2D数组，假设是高程图
                self.height_map = mesh_data
                self._generate_terrain_from_height()
        else:
            # 字典格式
            self.height_map = np.array(mesh_data.get('height_map', []))
            self.terrain_map = np.array(mesh_data.get('terrain_map', []))
        
        # 确保尺寸匹配
        if self.height_map.shape != self.grid_size:
            self.height_map = self._resize_array(self.height_map, self.grid_size)
        if self.terrain_map.shape != self.grid_size:
            self.terrain_map = self._resize_array(self.terrain_map, self.grid_size)
        
        # 初始化道路图
        self.road_map = np.zeros(self.grid_size, dtype=np.int32)
        
        # 设置智能体和目标位置
        self._set_agent_and_target()
    
    def _generate_terrain_from_height(self):
        """根据高程生成地形类型"""
        self.terrain_map = np.zeros(self.grid_size, dtype=np.int32)
        
        # 基于高程分类地形
        height_percentiles = np.percentile(self.height_map, [20, 40, 60, 80])
        
        self.terrain_map[self.height_map < height_percentiles[0]] = TerrainType.WATER.value
        self.terrain_map[(self.height_map >= height_percentiles[0]) & 
                        (self.height_map < height_percentiles[1])] = TerrainType.GRASS.value
        self.terrain_map[(self.height_map >= height_percentiles[1]) & 
                        (self.height_map < height_percentiles[2])] = TerrainType.FOREST.value
        self.terrain_map[(self.height_map >= height_percentiles[2]) & 
                        (self.height_map < height_percentiles[3])] = TerrainType.MOUNTAIN.value
        self.terrain_map[self.height_map >= height_percentiles[3]] = TerrainType.BUILDING.value
    
    def _resize_array(self, arr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整数组大小 - 使用简单的最近邻插值"""
        if arr.shape == target_size:
            return arr
        h, w = arr.shape
        new_h, new_w = target_size
        y_coords = np.linspace(0, h-1, new_h, dtype=int)
        x_coords = np.linspace(0, w-1, new_w, dtype=int)
        resized = arr[y_coords[:, None], x_coords]
        return resized.astype(arr.dtype)
    
    def _generate_random_terrain(self):
        """生成随机地形 - 使用简单的平均滤波平滑"""
        self.height_map = np.random.uniform(0, 100, self.grid_size)
        smoothed = np.copy(self.height_map)
        for i in range(1, self.grid_size[0]-1):
            for j in range(1, self.grid_size[1]-1):
                smoothed[i, j] = np.mean(self.height_map[i-1:i+2, j-1:j+2])
        self.height_map = smoothed
        self._generate_terrain_from_height()
        self.road_map = np.zeros(self.grid_size, dtype=np.int32)
        self._set_agent_and_target()
    
    def _set_agent_and_target(self):
        """设置智能体和目标位置"""
        # 找到可通行的位置（非水域、非建筑）
        passable_positions = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.terrain_map[i, j] not in [TerrainType.WATER.value, TerrainType.BUILDING.value]:
                    passable_positions.append((i, j))
        
        if len(passable_positions) < 2:
            raise ValueError("没有足够的可通行位置")
        
        # 随机选择起始和目标位置
        start_idx = np.random.randint(0, len(passable_positions))
        self.agent_pos = np.array(passable_positions[start_idx])
        
        # 确保目标位置与起始位置不同
        remaining_positions = [pos for pos in passable_positions if pos != tuple(self.agent_pos)]
        target_idx = np.random.randint(0, len(remaining_positions))
        self.target_pos = np.array(remaining_positions[target_idx])
        
        # 初始化资源
        self.resources = np.array([100.0, 50.0, 25.0])  # 金钱、材料、时间
        self.agent_path = [tuple(self.agent_pos)]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self._set_agent_and_target()
        self.agent_path = [tuple(self.agent_pos)]
        self.road_map = np.zeros(self.grid_size, dtype=np.int32)
        self.resources = np.array([100.0, 50.0, 0.0])
        
        return self._get_observation(), {}
    
    def step(self, action: int):
        """执行动作"""
        self.current_step += 1
        
        # 记录当前状态（如果正在录制）
        if self.recording:
            # 确保所有数据都是Python原生类型
            frame_data = {
                'step': int(self.current_step),
                'agent_pos': self.agent_pos.tolist(),
                'road_map': self.road_map.tolist(),
                'resources': self.resources.tolist(),
                'agent_path': [(int(pos[0]), int(pos[1])) for pos in self.agent_path],
                'action': int(action)
            }
            self.episode_history.append(frame_data)
        
        # 解析动作
        if action < 4:
            # 移动动作
            reward = self._move_agent(action)
        elif action == 4:
            # 建造道路
            reward = self._build_road()
        elif action == 5:
            # 升级道路
            reward = self._upgrade_road()
        else:
            # 等待
            reward = 0
        
        # 检查是否到达目标
        done = self._check_target_reached()
        
        # 检查是否超时
        if self.current_step >= self.max_steps:
            done = True
            reward -= 50  # 超时惩罚
        
        # 更新资源（时间）
        self.resources[2] += 1
        
        # 更新最后一帧的数据
        if self.recording and self.episode_history:
            self.episode_history[-1].update({
                'reward': float(reward),
                'done': bool(done),
                'resources': self.resources.tolist(),
                'road_map': self.road_map.tolist()
            })
        
        # 渲染
        if self.render_mode == 'human':
            self.render()
        
        return self._get_observation(), reward, done, False, {}
    
    def _move_agent(self, direction: int) -> float:
        """移动智能体"""
        if direction >= len(self.directions):
            return -1.0
        
        new_pos = self.agent_pos + np.array(self.directions[direction])
        
        # 检查边界
        if not (0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]):
            return -5.0  # 边界惩罚
        
        # 检查地形是否可通行
        terrain_type = self.terrain_map[new_pos[0], new_pos[1]]
        if terrain_type == TerrainType.WATER.value or terrain_type == TerrainType.BUILDING.value:
            return -10.0  # 不可通行惩罚
        
        # 计算移动成本
        base_cost = self.costs['movement']
        terrain_penalty = self.costs['terrain_penalty'][TerrainType(terrain_type)]
        
        # 如果有道路，减少移动成本
        if self.road_map[new_pos[0], new_pos[1]] > 0:
            terrain_penalty *= 0.3
        
        total_cost = base_cost + terrain_penalty
        
        # 检查资源是否足够
        if self.resources[0] < total_cost:
            return -5.0  # 资源不足惩罚
        
        # 执行移动
        self.agent_pos = new_pos
        self.agent_path.append(tuple(self.agent_pos))
        self.resources[0] -= total_cost
        
        # 计算奖励
        reward = -total_cost
        
        # 如果移动接近目标，给予奖励
        distance_to_target = np.linalg.norm(self.agent_pos - self.target_pos)
        if distance_to_target < 5:
            reward += 5 - distance_to_target
        
        return reward
    
    def _build_road(self) -> float:
        """建造道路"""
        cost = self.costs['road_build']
        
        if self.resources[0] < cost or self.resources[1] < cost * 0.5:
            return -5.0  # 资源不足惩罚
        
        # 检查当前位置是否已有道路
        if self.road_map[self.agent_pos[0], self.agent_pos[1]] > 0:
            return -2.0  # 已有道路惩罚
        
        # 建造道路
        self.road_map[self.agent_pos[0], self.agent_pos[1]] = 1
        self.resources[0] -= cost
        self.resources[1] -= cost * 0.5
        
        return 5.0  # 建造道路奖励
    
    def _upgrade_road(self) -> float:
        """升级道路"""
        cost = self.costs['road_upgrade']
        
        if self.resources[0] < cost:
            return -5.0  # 资源不足惩罚
        
        # 检查当前位置是否有道路可升级
        current_level = self.road_map[self.agent_pos[0], self.agent_pos[1]]
        if current_level == 0:
            return -2.0  # 没有道路惩罚
        if current_level >= 3:
            return -1.0  # 道路已满级惩罚
        
        # 升级道路
        self.road_map[self.agent_pos[0], self.agent_pos[1]] = current_level + 1
        self.resources[0] -= cost
        
        return 3.0  # 升级道路奖励
    
    def _check_target_reached(self) -> bool:
        """检查是否到达目标"""
        return np.array_equal(self.agent_pos, self.target_pos)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取当前观察"""
        return {
            'height_map': self.height_map.astype(np.float32),
            'terrain_map': self.terrain_map.astype(np.int32),
            'road_map': self.road_map.astype(np.int32),
            'agent_pos': self.agent_pos.astype(np.int32),
            'target_pos': self.target_pos.astype(np.int32),
            'resources': self.resources.astype(np.float32)
        }
    
    def render(self):
        """渲染环境"""
        if self.fig is None:
            plt.ion()  # 开启交互模式
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.fig.suptitle('Terrain Road Pathfinding Visualization', fontsize=16)
        
        self.ax.clear()
        
        # 绘制地形
        terrain_colors = {
            TerrainType.WATER.value: 'blue',
            TerrainType.GRASS.value: 'lightgreen',
            TerrainType.FOREST.value: 'darkgreen',
            TerrainType.MOUNTAIN.value: 'gray',
            TerrainType.ROAD.value: 'yellow',
            TerrainType.BUILDING.value: 'red'
        }
        
        # 创建地形图像
        terrain_img = np.zeros((*self.grid_size, 3))
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                terrain_type = self.terrain_map[i, j]
                if terrain_type == TerrainType.WATER.value:
                    terrain_img[i, j] = [0, 0, 1]  # 蓝色
                elif terrain_type == TerrainType.GRASS.value:
                    terrain_img[i, j] = [0.5, 1, 0.5]  # 浅绿色
                elif terrain_type == TerrainType.FOREST.value:
                    terrain_img[i, j] = [0, 0.5, 0]  # 深绿色
                elif terrain_type == TerrainType.MOUNTAIN.value:
                    terrain_img[i, j] = [0.5, 0.5, 0.5]  # 灰色
                elif terrain_type == TerrainType.ROAD.value:
                    terrain_img[i, j] = [1, 1, 0]  # 黄色
                elif terrain_type == TerrainType.BUILDING.value:
                    terrain_img[i, j] = [1, 0, 0]  # 红色
        
        # 添加道路覆盖
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.road_map[i, j] > 0:
                    road_intensity = min(self.road_map[i, j] / 3, 1.0)
                    terrain_img[i, j] = [1, 1, 0.5 * road_intensity]  # 道路颜色
        
        self.ax.imshow(terrain_img, origin='upper')
        
        # 绘制路径
        if len(self.agent_path) > 1:
            path_x = [pos[1] for pos in self.agent_path]
            path_y = [pos[0] for pos in self.agent_path]
            self.ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.7, label='Agent Path')
        
        # 绘制智能体
        self.ax.scatter(self.agent_pos[1], self.agent_pos[0], 
                       c='red', s=200, marker='o', edgecolors='black', linewidth=2, label='Agent')
        
        # 绘制目标
        self.ax.scatter(self.target_pos[1], self.target_pos[0], 
                       c='green', s=200, marker='*', edgecolors='black', linewidth=2, label='Target')
        
        # 添加信息文本
        info_text = f'Step: {self.current_step}\n'
        info_text += f'Resources: Money={self.resources[0]:.1f}, Materials={self.resources[1]:.1f}, Time={self.resources[2]:.1f}\n'
        info_text += f'Distance to target: {np.linalg.norm(self.agent_pos - self.target_pos):.1f}'
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.ax.set_title(f'Terrain Road Environment - Step {self.current_step}')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 检查窗口是否还存在
        if self.fig and plt.fignum_exists(self.fig.number):
            plt.pause(0.1)  # 控制刷新速度
        else:
            # 窗口被关闭，重置图形对象
            self.fig = None
            self.ax = None
    
    def get_terrain_info(self) -> Dict[str, Any]:
        """获取地形信息"""
        return {
            'grid_size': self.grid_size,
            'height_range': (np.min(self.height_map), np.max(self.height_map)),
            'terrain_distribution': {
                terrain_type.name: np.sum(self.terrain_map == terrain_type.value)
                for terrain_type in TerrainType
            },
            'road_coverage': np.sum(self.road_map > 0) / (self.grid_size[0] * self.grid_size[1])
        }

    def replay_episode(self, speed: float = 1.0, save_video: bool = False):
        """回放episode"""
        if not self.episode_history:
            print("❌ 没有可回放的episode数据")
            return
        
        print(f"🎬 开始回放episode (速度: {speed}x)")
        print(f"📊 总帧数: {len(self.episode_history)}")
        
        # 创建可视化窗口
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            self.fig.suptitle('Episode Replay', fontsize=16)
        
        # 添加控制按钮
        from matplotlib.widgets import Button, Slider
        plt.subplots_adjust(bottom=0.2)
        
        # 速度滑块
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        speed_slider = Slider(ax_slider, 'Speed', 0.1, 5.0, valinit=speed)
        
        # 控制按钮
        ax_play = plt.axes([0.1, 0.02, 0.1, 0.05])
        ax_pause = plt.axes([0.25, 0.02, 0.1, 0.05])
        ax_reset = plt.axes([0.4, 0.02, 0.1, 0.05])
        ax_save = plt.axes([0.55, 0.02, 0.1, 0.05])
        
        play_button = Button(ax_play, 'Play')
        pause_button = Button(ax_pause, 'Pause')
        reset_button = Button(ax_reset, 'Reset')
        save_button = Button(ax_save, 'Save')
        
        # 回放状态
        playing = True
        current_frame = 0
        
        def update_frame(frame_idx):
            """更新帧显示"""
            if frame_idx >= len(self.episode_history):
                return
            
            frame = self.episode_history[frame_idx]
            
            # 更新环境状态
            self.agent_pos = np.array(frame['agent_pos'])
            self.road_map = np.array(frame['road_map'])
            self.resources = np.array(frame['resources'])
            self.agent_path = frame['agent_path'].copy()
            self.current_step = frame['step']
            
            # 渲染
            self.ax.clear()
            
            # 绘制地形
            terrain_img = np.zeros((*self.grid_size, 3))
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    terrain_type = self.terrain_map[i, j]
                    if terrain_type == TerrainType.WATER.value:
                        terrain_img[i, j] = [0, 0, 1]  # 蓝色
                    elif terrain_type == TerrainType.GRASS.value:
                        terrain_img[i, j] = [0.5, 1, 0.5]  # 浅绿色
                    elif terrain_type == TerrainType.FOREST.value:
                        terrain_img[i, j] = [0, 0.5, 0]  # 深绿色
                    elif terrain_type == TerrainType.MOUNTAIN.value:
                        terrain_img[i, j] = [0.5, 0.5, 0.5]  # 灰色
                    elif terrain_type == TerrainType.ROAD.value:
                        terrain_img[i, j] = [1, 1, 0]  # 黄色
                    elif terrain_type == TerrainType.BUILDING.value:
                        terrain_img[i, j] = [1, 0, 0]  # 红色
            
            # 添加道路覆盖
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    if self.road_map[i, j] > 0:
                        road_intensity = min(self.road_map[i, j] / 3, 1.0)
                        terrain_img[i, j] = [1, 1, 0.5 * road_intensity]  # 道路颜色
            
            self.ax.imshow(terrain_img, origin='upper')
            
            # 绘制路径
            if len(self.agent_path) > 1:
                path_x = [pos[1] for pos in self.agent_path]
                path_y = [pos[0] for pos in self.agent_path]
                self.ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.7, label='Agent Path')
            
            # 绘制智能体
            self.ax.scatter(self.agent_pos[1], self.agent_pos[0], 
                           c='red', s=200, marker='o', edgecolors='black', linewidth=2, label='Agent')
            
            # 绘制目标
            self.ax.scatter(self.target_pos[1], self.target_pos[0], 
                           c='green', s=200, marker='*', edgecolors='black', linewidth=2, label='Target')
            
            # 添加信息文本
            info_text = f'Frame: {frame_idx + 1}/{len(self.episode_history)}\n'
            info_text += f'Step: {frame["step"]}\n'
            info_text += f'Action: {frame["action"]}\n'
            info_text += f'Reward: {frame["reward"]:.2f}\n'
            info_text += f'Resources: Money={self.resources[0]:.1f}, Materials={self.resources[1]:.1f}, Time={self.resources[2]:.1f}\n'
            info_text += f'Distance to target: {np.linalg.norm(self.agent_pos - self.target_pos):.1f}'
            
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.ax.set_title(f'Episode Replay - Frame {frame_idx + 1}/{len(self.episode_history)}')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        def on_play(event):
            nonlocal playing
            playing = True
        
        def on_pause(event):
            nonlocal playing
            playing = False
        
        def on_reset(event):
            nonlocal current_frame
            current_frame = 0
            update_frame(current_frame)
        
        def on_save(event):
            # 保存当前帧为图片
            timestamp = int(time.time())
            filename = f"replay_frame_{current_frame}_{timestamp}.png"
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"💾 帧已保存: {filename}")
        
        # 绑定按钮事件
        play_button.on_clicked(on_play)
        pause_button.on_clicked(on_pause)
        reset_button.on_clicked(on_reset)
        save_button.on_clicked(on_save)
        
        # 初始显示
        update_frame(0)
        
        # 回放循环
        try:
            while current_frame < len(self.episode_history):
                if playing:
                    update_frame(current_frame)
                    current_frame += 1
                    plt.pause(1.0 / speed_slider.val)  # 根据滑块调整速度
                else:
                    plt.pause(0.1)
                    
                # 检查窗口是否被关闭
                if not plt.fignum_exists(self.fig.number):
                    print("\n🪟 窗口被关闭，停止回放")
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ 回放被用户中断")
        except Exception as e:
            print(f"\n❌ 回放出错: {e}")
        finally:
            # 清理资源
            try:
                if self.fig and plt.fignum_exists(self.fig.number):
                    plt.close(self.fig)
                self.fig = None
                self.ax = None
            except:
                pass
        
        print("✅ 回放完成!")
