#!/usr/bin/env python3
"""
简单测试v5.0训练步数

验证为什么只导出了3个月的数据
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainers.v5_0.ppo_trainer import V5PPOTrainer


def test_simple_training():
    """简单测试训练"""
    print("=" * 60)
    print("简单测试v5.0训练步数")
    print("=" * 60)
    
    try:
        # 创建训练器
        trainer = V5PPOTrainer('configs/city_config_v5_0.json')
        print("   [PASS] 训练器初始化成功")
        
        # 测试收集不同步数的经验
        test_steps = [5, 10, 20, 30]
        
        for steps in test_steps:
            print(f"\n   测试收集 {steps} 步经验:")
            try:
                experiences = trainer.collect_experience(steps)
                print(f"   收集到 {len(experiences)} 个经验")
                
                if experiences:
                    # 统计时间步分布
                    time_steps = {}
                    for exp in experiences:
                        step_log = exp.get('step_log')
                        if step_log:
                            t = step_log.t
                            time_steps[t] = time_steps.get(t, 0) + 1
                    
                    print(f"   时间步分布: {dict(sorted(time_steps.items()))}")
                    print(f"   覆盖时间步: {min(time_steps.keys()) if time_steps else 0} - {max(time_steps.keys()) if time_steps else 0}")
                
            except Exception as e:
                print(f"   [FAIL] 收集 {steps} 步失败: {e}")
        
        print("\n   分析:")
        print("   1. 如果收集步数少于配置，可能是环境提前结束")
        print("   2. 如果月份分布不连续，可能是智能体切换问题")
        print("   3. 如果完全没有经验，可能是环境初始化问题")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_simple_training()
