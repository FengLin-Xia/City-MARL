#!/usr/bin/env python3
"""
调试训练器收集经验问题

找出为什么collect_experience失败
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainers.v5_0.ppo_trainer import V5PPOTrainer


def test_trainer_debug():
    """调试训练器"""
    print("=" * 60)
    print("调试训练器收集经验问题")
    print("=" * 60)
    
    try:
        # 创建训练器
        trainer = V5PPOTrainer('configs/city_config_v5_0.json')
        print("   [PASS] 训练器初始化成功")
        
        # 测试收集少量经验
        print("\n   测试收集5步经验:")
        try:
            experiences = trainer.collect_experience(5)
            print(f"   [PASS] 收集到 {len(experiences)} 个经验")
        except Exception as e:
            print(f"   [FAIL] 收集5步失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试收集10步经验
        print("\n   测试收集10步经验:")
        try:
            experiences = trainer.collect_experience(10)
            print(f"   [PASS] 收集到 {len(experiences)} 个经验")
        except Exception as e:
            print(f"   [FAIL] 收集10步失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试收集30步经验
        print("\n   测试收集30步经验:")
        try:
            experiences = trainer.collect_experience(30)
            print(f"   [PASS] 收集到 {len(experiences)} 个经验")
            
            # 分析经验数据
            if experiences:
                print(f"   经验分析:")
                print(f"   - 总经验数: {len(experiences)}")
                
                # 统计月份分布
                months = {}
                for exp in experiences:
                    step_log = exp.get('step_log')
                    if step_log:
                        # 从环境状态获取月份
                        next_state = exp.get('next_state')
                        if next_state:
                            month = next_state.month
                            months[month] = months.get(month, 0) + 1
                
                print(f"   - 月份分布: {dict(sorted(months.items()))}")
                print(f"   - 覆盖月份: {min(months.keys()) if months else 0} - {max(months.keys()) if months else 0}")
                print(f"   - 月份数量: {len(months)}")
                
                if len(months) >= 25:
                    print("   [PASS] 月份数量充足，应该能导出足够数据")
                elif len(months) >= 10:
                    print("   [PARTIAL] 月份数量中等，可能导出部分数据")
                else:
                    print("   [FAIL] 月份数量不足，可能无法导出足够数据")
            
        except Exception as e:
            print(f"   [FAIL] 收集30步失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"   [FAIL] 训练器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("调试完成!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_trainer_debug()
