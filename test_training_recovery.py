#!/usr/bin/env python3
"""
验证修复后的训练效果
检查 entropy_sum 是否恢复，IND 是否开始选择动作4/5
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_city_simulation_v5_0 import run_training_mode
import argparse

def test_training_recovery():
    """测试训练恢复效果"""
    print("=== 验证修复后的训练效果 ===")
    
    # 创建参数
    args = argparse.Namespace()
    args.config = "configs/city_config_v5_0.json"
    args.episodes = 1  # 只运行1个episode进行快速验证
    args.export = True
    args.verbose = True
    args.output_dir = "outputs"  # 添加缺失的参数
    
    print("开始短期训练验证...")
    print("预期效果：")
    print("- entropy_sum > 0，恢复探索能力")
    print("- IND 开始选择动作4/5，利用size bonus")
    print("- 不再出现策略塌缩")
    
    try:
        # 运行训练
        result = run_training_mode(args)
        
        if result:
            print("\n[SUCCESS] 训练完成！")
            print("请检查日志文件 logs/v5_0.log 中的以下内容：")
            print("1. entropy_sum 是否大于 0")
            print("2. IND 是否开始选择动作4/5")
            print("3. action_reward 是否在正常范围内")
        else:
            print("\n[FAILED] 训练失败！")
            
    except Exception as e:
        print(f"\n[ERROR] 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_recovery()
