#!/usr/bin/env python3
"""
测试v5.0训练步数问题

验证为什么只导出了3个月的数据而不是30个月
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainers.v5_0.ppo_trainer import V5PPOTrainer
from envs.v5_0.city_env import V5CityEnvironment


def test_training_steps():
    """测试训练步数"""
    print("=" * 60)
    print("测试v5.0训练步数问题")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("\n1. 配置检查:")
    total_steps = config.get('env', {}).get('time_model', {}).get('total_steps', 0)
    print(f"   配置的total_steps: {total_steps}")
    
    print("\n2. 环境测试:")
    try:
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态月份: {state.month}")
        
        # 检查环境的最大步数
        max_steps = getattr(env, 'max_steps', None)
        if max_steps:
            print(f"   环境max_steps: {max_steps}")
        else:
            print("   环境max_steps: 未设置")
        
        # 手动运行几步
        print("\n3. 手动运行测试:")
        step_count = 0
        max_test_steps = 5
        
        while step_count < max_test_steps:
            current_agent = env.current_agent
            candidates = env.get_action_candidates(current_agent)
            
            if candidates:
                # 选择第一个候选动作
                selected_sequence = candidates[0]
                next_state, reward, done, info = env.step(selected_sequence, selected_sequence)
                print(f"   步骤 {step_count}: 智能体 {current_agent}, 月份 {next_state.month}, 奖励 {reward:.2f}")
            else:
                print(f"   步骤 {step_count}: 智能体 {current_agent}, 无可用动作")
                env._update_state()
            
            step_count += 1
            
            if done:
                print(f"   环境结束于步骤 {step_count}")
                break
        
        print(f"   最终月份: {env.current_state.month}")
        
    except Exception as e:
        print(f"   [FAIL] 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n4. 训练器测试:")
    try:
        trainer = V5PPOTrainer('configs/city_config_v5_0.json')
        print("   [PASS] 训练器初始化成功")
        
        # 测试收集经验
        print("   测试收集经验...")
        experiences = trainer.collect_experience(5)  # 收集5步
        print(f"   收集到 {len(experiences)} 个经验")
        
        if experiences:
            for i, exp in enumerate(experiences[:3]):  # 只显示前3个
                step_log = exp.get('step_log')
                if step_log:
                    print(f"   经验 {i}: 智能体 {step_log.agent}, 月份 {step_log.month}")
        
    except Exception as e:
        print(f"   [FAIL] 训练器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n5. 问题分析:")
    print("   可能的原因:")
    print("   1. 训练管道硬编码了收集步数 (20步)")
    print("   2. 环境可能提前结束")
    print("   3. 导出系统只处理了部分数据")
    print("   4. 配置中的total_steps没有被正确使用")
    
    print("\n6. 建议修复:")
    print("   1. 修改训练管道使用配置中的total_steps")
    print("   2. 检查环境是否正确使用total_steps")
    print("   3. 验证导出系统处理所有月份的数据")
    print("   4. 确保训练循环运行完整的30个月")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        test_training_steps()
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
