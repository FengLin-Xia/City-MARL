"""
持续奖励系统训练测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_city_simulation_v5_0 import run_training_mode
import argparse


def test_continuous_reward_training():
    """测试持续奖励系统的训练"""
    print("开始测试持续奖励系统...")
    
    # 创建参数对象
    args = argparse.Namespace()
    args.config = "configs/city_config_v5_0.json"
    args.months = 10  # 短期测试
    args.episodes = 1  # 添加episodes参数
    args.output_dir = "outputs"
    args.log_level = "INFO"
    
    try:
        # 运行训练
        result = run_training_mode(args)
        
        if result:
            print("持续奖励系统训练测试成功！")
            print(f"训练结果: {result}")
        else:
            print("持续奖励系统训练测试失败！")
            
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_continuous_reward_training()
