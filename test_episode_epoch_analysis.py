#!/usr/bin/env python3
"""
分析一个episode包含几个epoch

检查episode和epoch的关系
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def analyze_episode_epoch():
    """分析episode和epoch的关系"""
    print("=" * 80)
    print("分析一个episode包含几个epoch")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 检查时间配置
        print(f"\n   时间配置:")
        time_model = env.config.get("env", {}).get("time_model", {})
        print(f"   - 时间单位: {time_model.get('step_unit')}")
        print(f"   - 总步数: {time_model.get('total_steps')}")
        
        # 检查PPO配置
        print(f"\n   PPO配置:")
        ppo_config = env.config.get("mappo", {}).get("ppo", {})
        rollout_config = env.config.get("mappo", {}).get("rollout", {})
        
        print(f"   - 更新次数/迭代: {rollout_config.get('updates_per_iter')}")
        print(f"   - 最大更新次数: {rollout_config.get('max_updates')}")
        print(f"   - 批次大小: {rollout_config.get('minibatch_size')}")
        print(f"   - 时间范围: {rollout_config.get('horizon')}")
        
        # 分析episode和epoch的关系
        print(f"\n   Episode和Epoch关系分析:")
        
        total_steps = time_model.get('total_steps', 30)
        updates_per_iter = rollout_config.get('updates_per_iter', 8)
        max_updates = rollout_config.get('max_updates', 10)
        
        print(f"   - 一个Episode包含: {total_steps} 步")
        print(f"   - 每次迭代更新: {updates_per_iter} 次")
        print(f"   - 最大更新次数: {max_updates} 次")
        
        # 计算epoch数量
        if max_updates > 0:
            epochs_per_episode = min(updates_per_iter, max_updates)
            print(f"   - 一个Episode的Epoch数: {epochs_per_episode}")
        else:
            epochs_per_episode = updates_per_iter
            print(f"   - 一个Episode的Epoch数: {epochs_per_episode}")
        
        # 分析训练流程
        print(f"\n   训练流程分析:")
        print(f"   1. 收集经验: {total_steps} 步")
        print(f"   2. 训练更新: {epochs_per_episode} 个epoch")
        print(f"   3. 每个epoch: 使用收集的经验进行梯度更新")
        
        # 检查调度器配置
        print(f"\n   调度器配置:")
        scheduler_config = env.config.get("scheduler", {})
        scheduler_params = scheduler_config.get("params", {})
        print(f"   - 调度器类型: {scheduler_config.get('name')}")
        print(f"   - 周期: {scheduler_params.get('period')}")
        print(f"   - 阶段数: {len(scheduler_params.get('phases', []))}")
        
        # 分析智能体执行
        print(f"\n   智能体执行分析:")
        phases = scheduler_params.get('phases', [])
        for i, phase in enumerate(phases):
            agents = phase.get('agents', [])
            mode = phase.get('mode', 'sequential')
            print(f"   - 阶段 {i+1}: {agents} ({mode})")
        
        # 计算每个episode的智能体执行次数
        total_agent_executions = 0
        for step in range(total_steps):
            active_agents = env.scheduler.get_active_agents(step)
            total_agent_executions += len(active_agents)
        
        print(f"   - 每个Episode智能体执行次数: {total_agent_executions}")
        print(f"   - 平均每步智能体数: {total_agent_executions / total_steps:.1f}")
        
    except Exception as e:
        print(f"   [FAIL] 分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    analyze_episode_epoch()
