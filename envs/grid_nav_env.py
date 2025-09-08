#!/usr/bin/env python3
"""
网格导航环境 - 使用曼哈顿距离和动作掩膜
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
import random


# 动作定义：上右下左
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上右下左，对应 0,1,2,3


def make_map(H: int, W: int, obstacle_ratio: float = 0.0) -> np.ndarray:
    """生成地图"""
    grid = np.zeros((H, W), dtype=np.int32)
    
    if obstacle_ratio > 0:
        # 随机放置障碍物
        num_obstacles = int(H * W * obstacle_ratio)
        for _ in range(num_obstacles):
            x, y = random.randint(0, H-1), random.randint(0, W-1)
            grid[x, y] = 1
    
    return grid


def sample_free(grid: np.ndarray) -> Tuple[int, int]:
    """在空闲位置采样"""
    H, W = grid.shape
    while True:
        x, y = random.randint(0, H-1), random.randint(0, W-1)
        if grid[x, y] == 0:
            return (x, y)


def sample_far_free(grid: np.ndarray, start: Tuple[int, int], min_dist: int = 10) -> Tuple[int, int]:
    """在远离起点的空闲位置采样"""
    H, W = grid.shape
    attempts = 0
    while attempts < 100:
        x, y = random.randint(0, H-1), random.randint(0, W-1)
        if grid[x, y] == 0:
            dist = abs(x - start[0]) + abs(y - start[1])  # 曼哈顿距离
            if dist >= min_dist:
                return (x, y)
        attempts += 1
    
    # 如果找不到足够远的，就返回任意空闲位置
    return sample_free(grid)


def crop_patch(grid: np.ndarray, center: Tuple[int, int], size: int) -> np.ndarray:
    """裁剪以center为中心的patch"""
    H, W = grid.shape
    x, y = center
    
    # 计算边界
    half = size // 2
    x_min = max(0, x - half)
    x_max = min(H, x + half + 1)
    y_min = max(0, y - half)
    y_max = min(W, y + half + 1)
    
    # 提取patch
    patch = grid[x_min:x_max, y_min:y_max]
    
    # 如果边界不足，进行填充
    if patch.shape != (size, size):
        padded_patch = np.zeros((size, size), dtype=np.int32)
        h, w = patch.shape
        padded_patch[:h, :w] = patch
        return padded_patch
    
    return patch


class GridNavEnv(gym.Env):
    """网格导航环境"""
    
    def __init__(self, H: int = 20, W: int = 20, max_steps: int = 200):
        super().__init__()
        
        self.H, self.W = H, W
        self.max_steps = max_steps
        
        # 观测空间
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
            'local_grid': spaces.Box(
                low=0, high=1, shape=(11, 11), dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1, shape=(4,), dtype=np.float32
            )
        })
        
        # 动作空间
        self.action_space = spaces.Discrete(4)
        
        # 环境状态
        self.grid = None
        self.start = None
        self.goal = None
        self.pos = None
        self.t = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 生成地图
        self.grid = make_map(self.H, self.W, obstacle_ratio=0.0)  # Stage0：无障碍
        
        # 固定起点和终点
        self.start = (2, 2)  # 固定起点
        self.goal = (17, 17)  # 固定终点
        
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
        
        # 检查障碍物
        if self.grid[nx, ny] == 1:
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
        
        # 检查动作合法性
        if not self._legal(self.pos, a):
            return self._obs(), -0.05, False, False, {"illegal": True}
        
        # 合法移动
        dx, dy = ACTIONS[a]
        next_pos = (self.pos[0] + dx, self.pos[1] + dy)
        self.pos = next_pos
        
        D_tp1 = self._manhattan(self.pos, self.goal)
        reward = -0.01 + 0.05 * (D_t - D_tp1)  # shaping：更近加分
        
        done = False
        truncated = False
        info = {}
        
        if self.pos == self.goal:
            reward += 1.0
            done = True
            info["reason"] = "reached_goal"
        elif self.t >= self.max_steps:
            truncated = True
        
        return self._obs(), reward, done, truncated, info
    
    def _obs(self) -> Dict:
        """获取观测"""
        patch = crop_patch(self.grid, center=self.pos, size=11)
        
        return {
            "position": np.array(self.pos, dtype=np.float32),
            "goal": np.array(self.goal, dtype=np.float32),
            "distance_to_goal": np.array([self._manhattan(self.pos, self.goal)], dtype=np.float32),
            "local_grid": patch.astype(np.float32),
            "action_mask": self.get_action_mask().astype(np.float32),
        }
    
    def render(self):
        """渲染环境"""
        plt.figure(figsize=(10, 10))
        
        # 绘制网格
        plt.imshow(self.grid, cmap='gray', origin='lower')
        
        # 绘制起点和终点
        plt.plot(self.start[1], self.start[0], 'go', markersize=15, label='Start', markeredgecolor='black', markeredgewidth=2)
        plt.plot(self.goal[1], self.goal[0], 'ro', markersize=15, label='Goal', markeredgecolor='black', markeredgewidth=2)
        
        # 绘制当前位置
        plt.plot(self.pos[1], self.pos[0], 'bo', markersize=10, label='Current')
        
        # 绘制动作掩膜
        mask = self.get_action_mask()
        for i, (dx, dy) in enumerate(ACTIONS):
            if mask[i] == 1:
                nx, ny = self.pos[0] + dx, self.pos[1] + dy
                plt.plot(ny, nx, 'cyan', marker='o', markersize=8, alpha=0.5)
        
        plt.title(f'Step {self.t}, Distance: {self._manhattan(self.pos, self.goal)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def close(self):
        """关闭环境"""
        plt.close('all')


if __name__ == "__main__":
    # 测试环境
    env = GridNavEnv()
    obs, _ = env.reset()
    
    print("网格导航环境测试:")
    print(f"网格尺寸: {env.H}x{env.W}")
    print(f"起点: {obs['position']}")
    print(f"终点: {obs['goal']}")
    print(f"距离: {obs['distance_to_goal'][0]:.1f}")
    print(f"动作掩膜: {obs['action_mask']}")
    
    # 运行几步测试
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"步骤 {i+1}: 动作={action}, 奖励={reward:.2f}, 距离={obs['distance_to_goal'][0]:.1f}")
        
        if done:
            break
    
    env.render()
    env.close()
