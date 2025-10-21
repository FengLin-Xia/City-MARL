#!/usr/bin/env python3
"""
测试月份修复效果

验证修复后是否能正确生成30个月的数据
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment


def test_month_fix():
    """测试月份修复"""
    print("=" * 60)
    print("测试月份修复效果")
    print("=" * 60)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始月份: {state.month}")
        
        # 手动运行30步
        print("\n   手动运行30步:")
        month_changes = []
        
        for step in range(30):
            # 获取当前智能体
            current_agent = env.current_agent
            candidates = env.get_action_candidates(current_agent)
            
            if candidates:
                # 选择第一个候选动作并创建Sequence
                from contracts import Sequence
                selected_candidate = candidates[0]
                selected_sequence = Sequence(
                    agent=current_agent,
                    actions=[selected_candidate.id]
                )
                next_state, reward, done, info = env.step(current_agent, selected_sequence)
                
                # 记录月份变化
                if next_state.month != state.month:
                    month_changes.append((step, next_state.month))
                    print(f"   步骤 {step}: 月份 {state.month} → {next_state.month}")
                
                state = next_state
            else:
                print(f"   步骤 {step}: 无可用动作")
                env._update_state()
                state = env._get_current_state()
            
            if done:
                print(f"   环境结束于步骤 {step}")
                break
        
        print(f"\n   最终月份: {state.month}")
        print(f"   月份变化次数: {len(month_changes)}")
        print(f"   月份变化记录: {month_changes}")
        
        # 分析结果
        print("\n   分析结果:")
        if len(month_changes) >= 29:  # 应该有29次月份变化 (0→1, 1→2, ..., 28→29)
            print("   [PASS] 月份切换正常，应该能导出30个月数据")
        elif len(month_changes) >= 2:
            print("   [PARTIAL] 月份切换部分正常，但可能不够30个月")
        else:
            print("   [FAIL] 月份切换异常，可能无法导出足够月份数据")
        
        # 检查环境配置
        total_steps = env.config.get('env', {}).get('time_model', {}).get('total_steps', 0)
        print(f"\n   配置检查:")
        print(f"   total_steps: {total_steps}")
        print(f"   实际运行步数: {step + 1}")
        print(f"   最终月份: {state.month}")
        
        if state.month >= total_steps - 1:
            print("   [PASS] 月份数量符合配置")
        else:
            print("   [FAIL] 月份数量不符合配置")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_month_fix()
