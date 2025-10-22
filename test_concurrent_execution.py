#!/usr/bin/env python3
"""
测试并发执行功能

验证EDU和COUNCIL能同时执行
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def test_concurrent_execution():
    """测试并发执行功能"""
    print("=" * 80)
    print("测试并发执行功能")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 测试10步的并发执行
        print(f"\n   测试10步并发执行:")
        print(f"   {'步骤':<4} {'月份':<4} {'Phase':<6} {'智能体':<20} {'执行模式':<10}")
        print(f"   {'-'*4} {'-'*4} {'-'*6} {'-'*20} {'-'*10}")
        
        for step in range(10):
            # 获取当前phase信息
            phase_agents = env.get_phase_agents()
            execution_mode = env.get_phase_execution_mode()
            
            print(f"   {step:<4} {state.month:<4} {step%2:<6} {str(phase_agents):<20} {execution_mode:<10}")
            
            # 模拟phase执行
            phase_sequences = {}
            for agent in phase_agents:
                candidates = env.get_action_candidates(agent)
                if candidates:
                    # 选择第一个候选动作
                    selected_candidate = candidates[0]
                    sequence = Sequence(
                        agent=agent,
                        actions=[selected_candidate.id]
                    )
                    phase_sequences[agent] = sequence
                else:
                    phase_sequences[agent] = None
            
            # 执行phase
            try:
                next_state, phase_rewards, done, info = env.step_phase(phase_agents, phase_sequences)
                
                # 显示执行结果
                print(f"     执行结果: {phase_rewards}")
                print(f"     日志数量: {len(info.get('phase_logs', []))}")
                
                state = next_state
                
                if done:
                    print(f"     环境结束于步骤 {step}")
                    break
                    
            except Exception as e:
                print(f"     执行失败: {e}")
                break
        
        # 分析结果
        print(f"\n   分析结果:")
        print(f"   - 最终月份: {state.month}")
        print(f"   - 最终步骤: {env.current_step}")
        print(f"   - 步骤日志数量: {len(env.step_logs)}")
        
        # 检查月份分布
        months = set()
        for log in env.step_logs:
            months.add(log.t)
        
        print(f"   - 月份分布: {sorted(months)}")
        print(f"   - 月份数量: {len(months)}")
        
        # 验证并发执行
        print(f"\n   验证并发执行:")
        if len(months) >= 5:
            print(f"   [PASS] 月份数量充足: {len(months)}个月")
        else:
            print(f"   [FAIL] 月份数量不足: {len(months)}个月")
        
        if env.current_step >= 5:
            print(f"   [PASS] 步骤数量充足: {env.current_step}步")
        else:
            print(f"   [FAIL] 步骤数量不足: {env.current_step}步")
        
        # 检查phase执行
        print(f"\n   检查phase执行:")
        for step in range(min(10, env.current_step)):
            phase_agents = env.scheduler.get_active_agents(step)
            execution_mode = env.scheduler.get_execution_mode(step)
            
            if step % 2 == 0:
                expected_agents = ["IND"]
                expected_mode = "sequential"
            else:
                expected_agents = ["EDU", "COUNCIL"]
                expected_mode = "concurrent"
            
            if phase_agents == expected_agents:
                print(f"   [PASS] 步骤{step}: 智能体正确 {phase_agents}")
            else:
                print(f"   [FAIL] 步骤{step}: 智能体错误 {phase_agents}, 期望{expected_agents}")
            
            if execution_mode == expected_mode:
                print(f"   [PASS] 步骤{step}: 执行模式正确 {execution_mode}")
            else:
                print(f"   [FAIL] 步骤{step}: 执行模式错误 {execution_mode}, 期望{expected_mode}")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_concurrent_execution()

