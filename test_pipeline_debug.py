#!/usr/bin/env python3
"""
调试训练管道问题

找出为什么训练管道在collect_experience步骤失败
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integration.v5_0.training_pipeline import V5TrainingPipeline


def test_pipeline_debug():
    """调试训练管道"""
    print("=" * 60)
    print("调试训练管道问题")
    print("=" * 60)
    
    try:
        # 创建训练管道
        pipeline = V5TrainingPipeline('configs/city_config_v5_0.json')
        print("   [PASS] 训练管道初始化成功")
        
        # 测试初始化组件
        print("\n   测试初始化组件:")
        try:
            data = pipeline._initialize_components({})
            print("   [PASS] 组件初始化成功")
        except Exception as e:
            print(f"   [FAIL] 组件初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试重置环境
        print("\n   测试重置环境:")
        try:
            data = pipeline._reset_environment(data)
            print("   [PASS] 环境重置成功")
        except Exception as e:
            print(f"   [FAIL] 环境重置失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试收集经验
        print("\n   测试收集经验:")
        try:
            data = pipeline._collect_experience(data)
            print("   [PASS] 经验收集成功")
            
            # 分析收集到的数据
            step_logs = data.get("step_logs", [])
            env_states = data.get("env_states", [])
            
            print(f"   收集到的数据:")
            print(f"   - step_logs: {len(step_logs)}")
            print(f"   - env_states: {len(env_states)}")
            
            if step_logs and env_states:
                # 统计月份分布
                months = {}
                for state in env_states:
                    month = state.month
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
            print(f"   [FAIL] 经验收集失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"   [FAIL] 训练管道初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("调试完成!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_pipeline_debug()

