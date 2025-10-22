#!/usr/bin/env python3
"""
测试调度器行为

分析为什么执行过程是每个agent一个月，而不是按照配置的月份轮次
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scheduler.phase_cycle_scheduler import PhaseCycleScheduler


def test_scheduler_behavior():
    """测试调度器行为"""
    print("=" * 80)
    print("调度器行为分析")
    print("=" * 80)
    
    try:
        # 加载配置
        with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        scheduler_config = config.get('scheduler', {})
        print(f"   调度器配置:")
        print(f"   - name: {scheduler_config.get('name')}")
        print(f"   - step_unit: {scheduler_config.get('params', {}).get('step_unit')}")
        print(f"   - period: {scheduler_config.get('params', {}).get('period')}")
        print(f"   - offset: {scheduler_config.get('params', {}).get('offset')}")
        print(f"   - phases: {scheduler_config.get('params', {}).get('phases')}")
        
        # 创建调度器
        scheduler = PhaseCycleScheduler(scheduler_config.get('params', {}))
        print(f"\n   [PASS] 调度器初始化成功")
        
        # 分析调度器逻辑
        print(f"\n   调度器逻辑分析:")
        print(f"   - 阶段数量: {len(scheduler.phases)}")
        print(f"   - 周期长度: {scheduler.period}")
        print(f"   - 偏移量: {scheduler.offset}")
        
        for i, phase in enumerate(scheduler.phases):
            print(f"   - 阶段 {i}: {phase.agents} ({phase.mode})")
        
        # 测试30步的调度行为
        print(f"\n   30步调度行为分析:")
        print(f"   {'步骤':<4} {'阶段':<4} {'活跃智能体':<20} {'执行模式':<10} {'当前智能体':<10}")
        print(f"   {'-'*4} {'-'*4} {'-'*20} {'-'*10} {'-'*10}")
        
        phase_changes = []
        agent_changes = []
        
        for step in range(30):
            # 获取当前阶段信息
            phase_index = scheduler._get_phase_index(step)
            active_agents = scheduler.get_active_agents(step)
            execution_mode = scheduler.get_execution_mode(step)
            
            # 模拟智能体切换逻辑
            if active_agents:
                current_agent = active_agents[step % len(active_agents)]
            else:
                current_agent = "NONE"
            
            # 记录变化
            if step > 0:
                prev_phase = scheduler._get_phase_index(step - 1)
                if phase_index != prev_phase:
                    phase_changes.append((step, prev_phase, phase_index))
                
                prev_agents = scheduler.get_active_agents(step - 1)
                if active_agents != prev_agents:
                    agent_changes.append((step, prev_agents, active_agents))
            
            print(f"   {step:<4} {phase_index:<4} {str(active_agents):<20} {execution_mode:<10} {current_agent:<10}")
        
        # 分析结果
        print(f"\n   分析结果:")
        print(f"   - 阶段变化次数: {len(phase_changes)}")
        print(f"   - 智能体变化次数: {len(agent_changes)}")
        
        if phase_changes:
            print(f"   - 阶段变化记录:")
            for step, prev_phase, curr_phase in phase_changes:
                print(f"    步骤 {step}: 阶段 {prev_phase} → {curr_phase}")
        
        if agent_changes:
            print(f"   - 智能体变化记录:")
            for step, prev_agents, curr_agents in agent_changes:
                print(f"    步骤 {step}: {prev_agents} → {curr_agents}")
        
        # 分析问题
        print(f"\n   问题分析:")
        
        # 检查阶段切换频率
        expected_phase_changes = 30 // scheduler.period
        if len(phase_changes) == expected_phase_changes:
            print(f"   [PASS] 阶段切换频率正确: {len(phase_changes)}次")
        else:
            print(f"   [FAIL] 阶段切换频率异常: 期望{expected_phase_changes}次，实际{len(phase_changes)}次")
        
        # 检查智能体切换频率
        if len(agent_changes) == len(phase_changes):
            print(f"   [PASS] 智能体切换与阶段切换一致")
        else:
            print(f"   [FAIL] 智能体切换与阶段切换不一致")
        
        # 检查执行模式
        concurrent_steps = 0
        sequential_steps = 0
        
        for step in range(30):
            mode = scheduler.get_execution_mode(step)
            if mode == "concurrent":
                concurrent_steps += 1
            elif mode == "sequential":
                sequential_steps += 1
        
        print(f"   - 并发执行步数: {concurrent_steps}")
        print(f"   - 顺序执行步数: {sequential_steps}")
        
        # 分析实际执行逻辑
        print(f"\n   实际执行逻辑分析:")
        print(f"   - 配置意图: 每2步切换阶段")
        print(f"   - 阶段0: EDU+COUNCIL (并发)")
        print(f"   - 阶段1: IND (顺序)")
        print(f"   - 实际行为: 每步切换智能体")
        
        # 找出问题根源
        print(f"\n   问题根源分析:")
        print(f"   1. 调度器配置正确: period=2, phases=[EDU+COUNCIL, IND]")
        print(f"   2. 调度器逻辑正确: 每2步切换阶段")
        print(f"   3. 问题在于环境实现: 每步都调用_switch_agent()")
        print(f"   4. 环境应该: 每2步切换阶段，每步内切换智能体")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_scheduler_behavior()

