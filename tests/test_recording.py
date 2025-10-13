#!/usr/bin/env python3
"""
测试录制功能
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.terrain_road_env import TerrainRoadEnvironment

def test_recording():
    """测试录制功能"""
    print("🧪 测试录制功能...")
    
    # 创建环境
    env = TerrainRoadEnvironment()
    
    # 开始录制
    env.start_recording()
    
    # 重置环境
    obs, _ = env.reset()
    
    # 执行几个动作
    for i in range(10):
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, truncated, info = env.step(action)
        print(f"步骤 {i+1}: 动作={action}, 奖励={reward:.2f}, 完成={done}")
        
        if done:
            break
    
    # 停止录制
    episode_data = env.stop_recording()
    
    print(f"📊 录制结果:")
    print(f"   Episode数据: {episode_data is not None}")
    if episode_data:
        print(f"   帧数: {len(episode_data['frames'])}")
        print(f"   网格尺寸: {episode_data['grid_size']}")
        print(f"   元数据: {episode_data['metadata']}")
        
        # 保存测试episode
        test_file = Path("data/episodes/test_recording.json")
        test_file.parent.mkdir(exist_ok=True)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, ensure_ascii=False, indent=2)
        
        print(f"   💾 测试episode已保存到: {test_file}")
    else:
        print("   ❌ 录制失败")

if __name__ == "__main__":
    test_recording()

