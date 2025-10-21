#!/usr/bin/env python3
"""
测试月份导出问题

验证30步应该导出30个月的数据
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def test_month_export_issue():
    """测试月份导出问题"""
    print("=" * 80)
    print("测试月份导出问题")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 测试30步的月份切换
        print(f"\n   测试30步月份切换:")
        print(f"   {'步骤':<4} {'月份':<4} {'智能体':<20} {'动作数':<6}")
        print(f"   {'-'*4} {'-'*4} {'-'*20} {'-'*6}")
        
        months_data = {}
        
        for step in range(30):
            # 获取当前phase信息
            phase_agents = env.get_phase_agents()
            execution_mode = env.get_phase_execution_mode()
            
            # 统计动作数量
            total_actions = 0
            for agent in phase_agents:
                candidates = env.get_action_candidates(agent)
                if candidates:
                    total_actions += 1
            
            print(f"   {step:<4} {state.month:<4} {str(phase_agents):<20} {total_actions:<6}")
            
            # 记录月份数据
            if state.month not in months_data:
                months_data[state.month] = {
                    'step': step,
                    'agents': phase_agents,
                    'actions': total_actions
                }
            
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
        
        # 检查月份数据
        print(f"\n   检查月份数据:")
        for month in sorted(months_data.keys()):
            data = months_data[month]
            print(f"   月份{month}: 步骤{data['step']}, 智能体{data['agents']}, 动作{data['actions']}")
        
        # 验证问题
        print(f"\n   验证问题:")
        if len(months) == 30:
            print(f"   [PASS] 月份数量正确: {len(months)}个月")
        else:
            print(f"   [FAIL] 月份数量错误: {len(months)}个月，期望30个月")
        
        if env.current_step == 30:
            print(f"   [PASS] 步骤数量正确: {env.current_step}步")
        else:
            print(f"   [FAIL] 步骤数量错误: {env.current_step}步，期望30步")
        
        # 检查缺失的月份
        expected_months = set(range(30))
        actual_months = set(months)
        missing_months = expected_months - actual_months
        
        if missing_months:
            print(f"   [FAIL] 缺失月份: {sorted(missing_months)}")
        else:
            print(f"   [PASS] 所有月份都存在")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_month_export_issue()
