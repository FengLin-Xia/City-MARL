#!/usr/bin/env python3
"""
地形网格导航环境 - 在GridNavEnv基础上添加高度和坡度信息
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
import random


# 动作定义：上右下左
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上右下左，对应 0,1,2,3


def make_terrain_map(H: int, W: int, height_range: Tuple[float, float] = (0.0, 10.0), 
                    smoothness: float = 0.8) -> np.ndarray:
    """生成地形高程图"""
    # 使用简单的噪声生成地形
    terrain = np.random.uniform(height_range[0], height_range[1], (H, W))
    
    # 简单的平滑处理
    if smoothness > 0:
        from scipy.ndimage import gaussian_filter
        terrain = gaussian_filter(terrain, sigma=smoothness)
    
    return terrain.astype(np.float32)


def calculate_slope(terrain: np.ndarray, pos: Tuple[int, int]) -> float:
    """计算指定位置的坡度"""
    H, W = terrain.shape
    x, y = pos
    
    # 计算周围8个方向的坡度
    slopes = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                # 计算高度差和距离
                height_diff = abs(terrain[nx, ny] - terrain[x, y])
                distance = np.sqrt(dx**2 + dy**2)
                slope = height_diff / distance if distance > 0 else 0
                slopes.append(slope)
    
    # 返回最大坡度
    return max(slopes) if slopes else 0.0


def get_local_terrain_features(terrain: np.ndarray, pos: Tuple[int, int], 
                             window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """获取局部地形特征"""
    H, W = terrain.shape
    x, y = pos
    
    # 计算窗口边界
    half = window_size // 2
    x_min = max(0, x - half)
    x_max = min(H, x + half + 1)
    y_min = max(0, y - half)
    y_max = min(W, y + half + 1)
    
    # 提取局部地形
    local_terrain = terrain[x_min:x_max, y_min:y_max]
    
    # 如果边界不足，进行填充
    if local_terrain.shape != (window_size, window_size):
        padded_terrain = np.zeros((window_size, window_size), dtype=np.float32)
        h, w = local_terrain.shape
        padded_terrain[:h, :w] = local_terrain
        local_terrain = padded_terrain
    
    # 计算局部坡度
    local_slope = np.zeros((window_size, window_size), dtype=np.float32)
    h, w = local_terrain.shape
    for i in range(window_size):
        for j in range(window_size):
            if i < h and j < w:
                # 确保索引不越界
                terrain_x = min(max(x_min + i, 0), H - 1)
                terrain_y = min(max(y_min + j, 0), W - 1)
                local_slope[i, j] = calculate_slope(terrain, (terrain_x, terrain_y))
    
    return local_terrain, local_slope


class TerrainGridNavEnv(gym.Env):
    """地形网格导航环境"""
    
    def __init__(self, H: int = 20, W: int = 20, max_steps: int = 200,
                 height_range: Tuple[float, float] = (0.0, 10.0),
                 slope_penalty_weight: float = 0.1,
                 height_penalty_weight: float = 0.05,
                 custom_terrain: Optional[np.ndarray] = None,
                 fixed_start: Optional[Tuple[int, int]] = None,
                 fixed_goal: Optional[Tuple[int, int]] = None):
        super().__init__()
        
        self.H, self.W = H, W
        self.max_steps = max_steps
        self.height_range = height_range
        self.slope_penalty_weight = slope_penalty_weight
        self.height_penalty_weight = height_penalty_weight
        
        # 观测空间 - 增强版
        self.observation_space = spaces.Dict({
            'position': spaces.Box(
                low=0, high=max(H, W), shape=(2,), dtype=np.float32
            ),
            'goal': spaces.Box(
                low=0, high=max(H, W), shape=(2,), dtype=np.float32
            ),
            'distance_to_goal': spaces.Box(
                low=0, high=H+W, shape=(1,), dtype=np.float32
            ),
            'current_height': spaces.Box(
                low=height_range[0], high=height_range[1], shape=(1,), dtype=np.float32
            ),
            'goal_height': spaces.Box(
                low=height_range[0], high=height_range[1], shape=(1,), dtype=np.float32
            ),
            'height_difference': spaces.Box(
                low=-height_range[1], high=height_range[1], shape=(1,), dtype=np.float32
            ),
            'current_slope': spaces.Box(
                low=0, high=10.0, shape=(1,), dtype=np.float32
            ),
            'local_terrain': spaces.Box(
                low=height_range[0], high=height_range[1], shape=(5, 5), dtype=np.float32
            ),
            'local_slope': spaces.Box(
                low=0, high=10.0, shape=(5, 5), dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1, shape=(4,), dtype=np.float32
            )
        })
        
        # 动作空间
        self.action_space = spaces.Discrete(4)
        
        # 环境状态
        self.terrain = None
        self.start = None
        self.goal = None
        self.pos = None
        self.t = 0
        
        # 保存自定义参数
        self.custom_terrain = custom_terrain
        self.fixed_start = fixed_start
        self.fixed_goal = fixed_goal
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 使用自定义地形或生成新地形
        if self.custom_terrain is not None:
            self.terrain = self.custom_terrain.copy()
            # 更新地形尺寸
            self.H, self.W = self.terrain.shape
        else:
            self.terrain = make_terrain_map(self.H, self.W, self.height_range)
        
        # 设置起点和终点
        if self.fixed_start is not None:
            self.start = self.fixed_start
        else:
            self.start = (2, 2)  # 默认起点
            
        if self.fixed_goal is not None:
            self.goal = self.fixed_goal
        else:
            self.goal = (17, 17)  # 默认终点
        
        # 设置初始位置
        self.pos = self.start
        self.t = 0
        
        return self._obs(), {}
    
    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """计算曼哈顿距离"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _legal(self, pos: Tuple[int, int], a: int) -> bool:
        """检查动作是否合法"""
        dx, dy = ACTIONS[a]
        nx, ny = pos[0] + dx, pos[1] + dy
        
        # 检查边界
        if nx < 0 or ny < 0 or nx >= self.H or ny >= self.W:
            return False
        
        return True
    
    def get_action_mask(self) -> np.ndarray:
        """获取动作掩膜"""
        mask = np.zeros(4, dtype=np.int32)
        for a in range(4):
            mask[a] = 1 if self._legal(self.pos, a) else 0
        return mask
    
    def step(self, a: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一步"""
        self.t += 1
        D_t = self._manhattan(self.pos, self.goal)
        
        # 获取当前高度和坡度
        current_height = self.terrain[self.pos[0], self.pos[1]]
        current_slope = calculate_slope(self.terrain, self.pos)
        
        # 检查动作合法性
        if not self._legal(self.pos, a):
            return self._obs(), -0.05, False, False, {"illegal": True}
        
        # 合法移动
        dx, dy = ACTIONS[a]
        next_pos = (self.pos[0] + dx, self.pos[1] + dy)
        
        # 获取新位置的高度和坡度
        next_height = self.terrain[next_pos[0], next_pos[1]]
        next_slope = calculate_slope(self.terrain, next_pos)
        
        # 计算高度变化
        height_change = abs(next_height - current_height)
        
        self.pos = next_pos
        D_tp1 = self._manhattan(self.pos, self.goal)
        
        # 基础奖励：距离减少
        reward = -0.01 + 0.05 * (D_t - D_tp1)
        
        # 地形惩罚：高度变化和坡度（增强版）
        height_penalty = self.height_penalty_weight * height_change
        slope_penalty = self.slope_penalty_weight * next_slope
        
        # 额外的地形惩罚：如果坡度太高，给予额外惩罚
        if next_slope > 5.0:  # 高坡度惩罚
            slope_penalty *= 2.0
        
        # 如果高度变化太大，给予额外惩罚
        if height_change > 3.0:  # 大高度变化惩罚
            height_penalty *= 1.5
        
        reward -= height_penalty + slope_penalty
        
        done = False
        truncated = False
        info = {
            'height_change': height_change,
            'slope': next_slope,
            'height_penalty': height_penalty,
            'slope_penalty': slope_penalty
        }
        
        if self.pos == self.goal:
            reward += 1.0
            done = True
            info["reason"] = "reached_goal"
        elif self.t >= self.max_steps:
            truncated = True
        
        return self._obs(), reward, done, truncated, info
    
    def _obs(self) -> Dict:
        """获取观测"""
        current_height = self.terrain[self.pos[0], self.pos[1]]
        goal_height = self.terrain[self.goal[0], self.goal[1]]
        current_slope = calculate_slope(self.terrain, self.pos)
        
        # 获取局部地形特征
        local_terrain, local_slope = get_local_terrain_features(self.terrain, self.pos)
        
        return {
            "position": np.array(self.pos, dtype=np.float32),
            "goal": np.array(self.goal, dtype=np.float32),
            "distance_to_goal": np.array([self._manhattan(self.pos, self.goal)], dtype=np.float32),
            "current_height": np.array([current_height], dtype=np.float32),
            "goal_height": np.array([goal_height], dtype=np.float32),
            "height_difference": np.array([goal_height - current_height], dtype=np.float32),
            "current_slope": np.array([current_slope], dtype=np.float32),
            "local_terrain": local_terrain,
            "local_slope": local_slope,
            "action_mask": self.get_action_mask().astype(np.float32),
        }
    
    def render(self):
        """渲染环境"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制地形
        im1 = ax1.imshow(self.terrain, cmap='terrain', origin='lower')
        ax1.set_title('地形高程图')
        plt.colorbar(im1, ax=ax1)
        
        # 绘制起点和终点
        ax1.plot(self.start[1], self.start[0], 'go', markersize=15, label='Start', 
                markeredgecolor='black', markeredgewidth=2)
        ax1.plot(self.goal[1], self.goal[0], 'ro', markersize=15, label='Goal', 
                markeredgecolor='black', markeredgewidth=2)
        ax1.plot(self.pos[1], self.pos[0], 'bo', markersize=10, label='Current')
        
        # 绘制坡度图
        slope_map = np.zeros_like(self.terrain)
        for i in range(self.H):
            for j in range(self.W):
                slope_map[i, j] = calculate_slope(self.terrain, (i, j))
        
        im2 = ax2.imshow(slope_map, cmap='hot', origin='lower')
        ax2.set_title('坡度图')
        plt.colorbar(im2, ax=ax2)
        
        # 在坡度图上也标记位置
        ax2.plot(self.start[1], self.start[0], 'go', markersize=15, label='Start', 
                markeredgecolor='black', markeredgewidth=2)
        ax2.plot(self.goal[1], self.goal[0], 'ro', markersize=15, label='Goal', 
                markeredgecolor='black', markeredgewidth=2)
        ax2.plot(self.pos[1], self.pos[0], 'bo', markersize=10, label='Current')
        
        # 显示当前信息
        current_height = self.terrain[self.pos[0], self.pos[1]]
        current_slope = calculate_slope(self.terrain, self.pos)
        distance = self._manhattan(self.pos, self.goal)
        
        fig.suptitle(f'Step {self.t}, Distance: {distance}, Height: {current_height:.1f}, Slope: {current_slope:.2f}')
        ax1.legend()
        ax2.legend()
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """关闭环境"""
        plt.close('all')


if __name__ == "__main__":
    # 测试环境
    env = TerrainGridNavEnv()
    obs, _ = env.reset()
    
    print("地形网格导航环境测试:")
    print(f"网格尺寸: {env.H}x{env.W}")
    print(f"起点: {obs['position']}")
    print(f"终点: {obs['goal']}")
    print(f"距离: {obs['distance_to_goal'][0]:.1f}")
    print(f"当前高度: {obs['current_height'][0]:.1f}")
    print(f"目标高度: {obs['goal_height'][0]:.1f}")
    print(f"高度差: {obs['height_difference'][0]:.1f}")
    print(f"当前坡度: {obs['current_slope'][0]:.2f}")
    print(f"动作掩膜: {obs['action_mask']}")
    
    # 运行几步测试
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"步骤 {i+1}: 动作={action}, 奖励={reward:.3f}, 距离={obs['distance_to_goal'][0]:.1f}")
        print(f"  高度变化: {info.get('height_change', 0):.2f}, 坡度: {info.get('slope', 0):.2f}")
        
        if done:
            break
    
    env.render()
    env.close()
