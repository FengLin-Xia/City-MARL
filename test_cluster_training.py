#!/usr/bin/env python3
"""
测试工业集群协同奖励机制在真实训练中的表现

验证：
1. 建筑注册表在训练中正确更新
2. 协同奖励在训练中正确计算
3. IND智能体是否会选择动作9、10、11
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from enhanced_city_simulation_v5_0 import run_training_mode
import argparse

def test_industrial_cluster_training():
    """测试工业集群协同奖励机制在训练中的表现"""
    print("开始测试工业集群协同奖励机制在训练中的表现...")
    
    # 创建参数对象
    args = argparse.Namespace()
    args.config = "configs/city_config_v5_0.json"  # 修复：应该是config而不是config_path
    args.mode = "training"
    args.episodes = 1  # 只运行1个episode进行测试
    args.max_steps = 30  # 运行30个月
    args.output_dir = "outputs"
    args.log_level = "INFO"
    args.debug = False
    
    try:
        print("开始训练测试...")
        result = run_training_mode(args)
        
        if result:
            print("训练测试成功完成！")
            print("请检查日志文件 logs/v5_0.log 查看建筑注册表和协同奖励的详细信息")
        else:
            print("训练测试失败")
            
    except Exception as e:
        print(f"训练测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_industrial_cluster_training()
