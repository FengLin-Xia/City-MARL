#!/usr/bin/env python3
"""
测试地形效果 - 验证地形惩罚是否起作用
"""

import numpy as np
import matplotlib.pyplot as plt
from envs.terrain_grid_nav_env import TerrainGridNavEnv

def test_terrain_effect():
    """测试地形效果"""
    print("测试地形效果...")
    
    # 创建环境
    env = TerrainGridNavEnv(
        H=20, W=20,
        max_steps=100,
        height_range=(0.0, 12.0),
        slope_penalty_weight=0.3,
        height_penalty_weight=0.2
    )
    
    obs, _ = env.reset()
    
    print(f"起点: {obs['position']}")
    print(f"终点: {obs['goal']}")
    print(f"起点高度: {obs['current_height'][0]:.2f}")
    print(f"终点高度: {obs['goal_height'][0]:.2f}")
    print(f"起点坡度: {obs['current_slope'][0]:.2f}")
    
    # 测试不同路径的奖励
    print("\n测试不同路径的奖励:")
    
    # 路径1：直线路径（可能经过高地形）
    print("\n路径1: 直线路径")
    total_reward1 = 0
    path1 = []
    env.reset()
    
    for step in range(50):  # 最多50步
        # 计算到目标的方向
        dx = env.goal[0] - env.pos[0]
        dy = env.goal[1] - env.pos[1]
        
        # 选择主要方向
        if abs(dx) > abs(dy):
            action = 1 if dx > 0 else 3  # 右或左
        else:
            action = 0 if dy > 0 else 2  # 上或下
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward1 += reward
        path1.append(list(env.pos))
        
        print(f"  步骤{step+1}: 位置{env.pos}, 高度{obs['current_height'][0]:.2f}, "
              f"坡度{obs['current_slope'][0]:.2f}, 奖励{reward:.3f}")
        
        if done:
            break
    
    print(f"路径1总奖励: {total_reward1:.3f}")
    
    # 路径2：尝试避开高地形
    print("\n路径2: 尝试避开高地形")
    total_reward2 = 0
    path2 = []
    env.reset()
    
    for step in range(50):
        # 获取当前位置的地形信息
        current_height = obs['current_height'][0]
        current_slope = obs['current_slope'][0]
        
        # 计算到目标的方向
        dx = env.goal[0] - env.pos[0]
        dy = env.goal[1] - env.pos[1]
        
        # 如果当前坡度很高，尝试选择其他方向
        if current_slope > 3.0:
            # 尝试其他方向
            possible_actions = []
            for a in range(4):
                if env._legal(env.pos, a):
                    next_pos = (env.pos[0] + env.ACTIONS[a][0], env.pos[1] + env.ACTIONS[a][1])
                    next_slope = env.calculate_slope(env.terrain, next_pos)
                    if next_slope < current_slope:
                        possible_actions.append((a, next_slope))
            
            if possible_actions:
                # 选择坡度最小的动作
                action = min(possible_actions, key=lambda x: x[1])[0]
            else:
                # 如果所有方向坡度都很高，选择主要方向
                action = 1 if abs(dx) > abs(dy) else 0
        else:
            # 正常选择方向
            if abs(dx) > abs(dy):
                action = 1 if dx > 0 else 3
            else:
                action = 0 if dy > 0 else 2
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward2 += reward
        path2.append(list(env.pos))
        
        print(f"  步骤{step+1}: 位置{env.pos}, 高度{obs['current_height'][0]:.2f}, "
              f"坡度{obs['current_slope'][0]:.2f}, 奖励{reward:.3f}")
        
        if done:
            break
    
    print(f"路径2总奖励: {total_reward2:.3f}")
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 地形图
    plt.subplot(1, 3, 1)
    plt.imshow(env.terrain, cmap='terrain', origin='lower')
    plt.title('地形高程图')
    plt.colorbar()
    
    # 路径1
    plt.subplot(1, 3, 2)
    plt.imshow(env.terrain, cmap='terrain', origin='lower')
    path1 = np.array(path1)
    plt.plot(path1[:, 1], path1[:, 0], 'b-', linewidth=2, label='路径1')
    plt.plot(env.start[1], env.start[0], 'go', markersize=10, label='起点')
    plt.plot(env.goal[1], env.goal[0], 'ro', markersize=10, label='终点')
    plt.title(f'路径1 (奖励: {total_reward1:.3f})')
    plt.legend()
    
    # 路径2
    plt.subplot(1, 3, 3)
    plt.imshow(env.terrain, cmap='terrain', origin='lower')
    path2 = np.array(path2)
    plt.plot(path2[:, 1], path2[:, 0], 'r-', linewidth=2, label='路径2')
    plt.plot(env.start[1], env.start[0], 'go', markersize=10, label='起点')
    plt.plot(env.goal[1], env.goal[0], 'ro', markersize=10, label='终点')
    plt.title(f'路径2 (奖励: {total_reward2:.3f})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n结论:")
    print(f"路径1总奖励: {total_reward1:.3f}")
    print(f"路径2总奖励: {total_reward2:.3f}")
    print(f"差异: {abs(total_reward1 - total_reward2):.3f}")
    
    if abs(total_reward1 - total_reward2) > 0.1:
        print("✓ 地形惩罚起作用！不同路径的奖励有明显差异")
    else:
        print("✗ 地形惩罚可能太弱，需要进一步调整")

if __name__ == "__main__":
    test_terrain_effect()
