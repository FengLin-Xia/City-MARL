#!/usr/bin/env python3
"""
分析v5.0时间单位执行流程

详细分析经验收集、环境步数、月份切换的关系
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def analyze_time_flow():
    """分析时间流程"""
    print("=" * 80)
    print("v5.0 时间单位执行流程分析")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 获取配置
        total_steps = env.config.get('env', {}).get('time_model', {}).get('total_steps', 30)
        print(f"\n   配置分析:")
        print(f"   - total_steps: {total_steps}")
        print(f"   - 预期月份范围: 0 到 {total_steps}")
        
        # 重置环境
        state = env.reset()
        print(f"\n   初始状态:")
        print(f"   - current_step: {env.current_step}")
        print(f"   - current_month: {env.current_month}")
        print(f"   - state.month: {state.month}")
        
        # 手动执行30步
        print(f"\n   执行流程分析:")
        print(f"   {'步骤':<4} {'环境步数':<6} {'环境月份':<6} {'状态月份':<6} {'智能体':<8} {'动作':<4}")
        print(f"   {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*4}")
        
        month_changes = []
        step_logs = []
        
        for step in range(30):
            # 记录执行前状态
            pre_step = env.current_step
            pre_month = env.current_month
            pre_state_month = state.month
            
            # 获取当前智能体
            current_agent = env.current_agent
            candidates = env.get_action_candidates(current_agent)
            
            # 执行动作
            if candidates:
                selected_candidate = candidates[0]
                selected_sequence = Sequence(
                    agent=current_agent,
                    actions=[selected_candidate.id]
                )
                next_state, reward, done, info = env.step(current_agent, selected_sequence)
                
                # 记录步骤日志
                step_log = info.get('step_log')
                if step_log:
                    step_logs.append(step_log)
                
                action_info = f"ID{selected_candidate.id}"
            else:
                env._update_state()
                next_state = env._get_current_state()
                action_info = "无动作"
            
            # 记录执行后状态
            post_step = env.current_step
            post_month = env.current_month
            post_state_month = next_state.month
            
            # 检查月份变化
            if post_month != pre_month:
                month_changes.append((step, pre_month, post_month))
            
            # 打印步骤信息
            print(f"   {step:<4} {post_step:<6} {post_month:<6} {post_state_month:<6} {current_agent:<8} {action_info:<4}")
            
            state = next_state
            
            if done:
                print(f"   环境结束于步骤 {step}")
                break
        
        # 分析结果
        print(f"\n   执行结果分析:")
        print(f"   - 总执行步数: {env.current_step}")
        print(f"   - 最终月份: {env.current_month}")
        print(f"   - 最终状态月份: {state.month}")
        print(f"   - 月份变化次数: {len(month_changes)}")
        print(f"   - 步骤日志数量: {len(step_logs)}")
        
        # 分析月份变化
        print(f"\n   月份变化记录:")
        for step, pre_month, post_month in month_changes:
            print(f"   - 步骤 {step}: {pre_month} → {post_month}")
        
        # 分析步骤日志
        print(f"\n   步骤日志分析:")
        if step_logs:
            months_in_logs = set()
            for log in step_logs:
                # 从环境状态获取月份
                months_in_logs.add(log.t)  # 使用时间步作为月份标识
            
            print(f"   - 日志中的月份: {sorted(months_in_logs)}")
            print(f"   - 日志月份数量: {len(months_in_logs)}")
        
        # 逻辑一致性检查
        print(f"\n   逻辑一致性检查:")
        
        # 检查1: 步数与月份关系
        if env.current_step == env.current_month:
            print(f"   [PASS] 步数与月份一致: {env.current_step}步 = {env.current_month}月")
        else:
            print(f"   [FAIL] 步数与月份不一致: {env.current_step}步 ≠ {env.current_month}月")
        
        # 检查2: 配置与实现关系
        if env.current_step == total_steps:
            print(f"   [PASS] 执行步数与配置一致: {env.current_step}步 = {total_steps}步")
        else:
            print(f"   [FAIL] 执行步数与配置不一致: {env.current_step}步 ≠ {total_steps}步")
        
        # 检查3: 月份范围
        if env.current_month == total_steps:
            print(f"   [PASS] 月份范围与配置一致: 0-{env.current_month}月 = {total_steps}个月")
        else:
            print(f"   [FAIL] 月份范围与配置不一致: 0-{env.current_month}月 ≠ {total_steps}个月")
        
        # 检查4: 数据完整性
        if len(step_logs) > 0:
            print(f"   [PASS] 生成了步骤日志: {len(step_logs)}个")
        else:
            print(f"   [FAIL] 没有生成步骤日志")
        
        # 总结
        print(f"\n   总结:")
        print(f"   - 时间单位关系: 1步 = 1月")
        print(f"   - 执行逻辑: 30步 → 30个月")
        print(f"   - 数据生成: {len(step_logs)}个步骤日志")
        print(f"   - 月份覆盖: 0-{env.current_month}月")
        
        if len(month_changes) == total_steps:
            print(f"   [PASS] 时间单位逻辑清晰正确")
        else:
            print(f"   [PARTIAL] 时间单位逻辑需要进一步分析")
        
    except Exception as e:
        print(f"   [FAIL] 分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    analyze_time_flow()
