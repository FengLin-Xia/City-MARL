#!/usr/bin/env python3
"""
快速回放最新的episode
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.terrain_road_env import TerrainRoadEnvironment

def replay_latest_episode():
    """回放最新的episode"""
    print("🎬 开始回放最新episode...")
    
    # 查找最新的episode文件
    episodes_dir = Path("data/episodes")
    episode_files = list(episodes_dir.glob("episode_*.json"))
    
    if not episode_files:
        print("❌ 未找到episode文件")
        return
    
    # 按修改时间排序，获取最新的
    latest_file = max(episode_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 回放文件: {latest_file}")
    
    # 加载episode数据
    with open(latest_file, 'r', encoding='utf-8') as f:
        episode_data = json.load(f)
    
    # 检查episode数据是否有效
    if episode_data is None:
        print("❌ Episode数据为空")
        return
    
    print(f"📊 Episode信息:")
    if 'frames' in episode_data:
        print(f"   总步数: {len(episode_data['frames'])}")
    if 'metadata' in episode_data:
        print(f"   总奖励: {episode_data['metadata']['final_reward']:.2f}")
        print(f"   是否到达目标: {episode_data['metadata']['success']}")
    else:
        print("   数据格式不完整")
    
    # 创建环境
    env = TerrainRoadEnvironment()
    
    # 回放episode
    print("🎮 开始回放...")
    env.replay_episode(episode_data)
    
    print("✅ 回放完成!")

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    replay_latest_episode()
