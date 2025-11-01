#!/usr/bin/env python3
"""
测试修复后的COUNCIL延迟机制在实际训练中的效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_city_simulation_v5_0 import run_training_mode
import argparse

def test_council_delay_in_training():
    """测试修复后的COUNCIL延迟机制在实际训练中的效果"""
    
    print("=== 测试修复后的COUNCIL延迟机制 ===")
    print("运行短期训练，检查COUNCIL是否在第6月前被正确延迟...")
    print()
    
    # 创建参数对象
    args = argparse.Namespace()
    args.config = "configs/city_config_v5_0.json"
    args.output_dir = "outputs"
    args.episodes = 1
    args.max_updates = 10  # 只运行10个更新，覆盖前几个月
    
    try:
        # 运行训练
        result = run_training_mode(args)
        
        if result:
            print("训练成功完成！")
            print("现在检查export文件，确认COUNCIL是否在第6月前被延迟...")
            
            # 检查前几个月的export文件
            for month in range(1, 8):
                export_file = f"outputs/export_month_{month:02d}.txt"
                if os.path.exists(export_file):
                    with open(export_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    
                    if content:
                        lines = content.split('\n')
                        print(f"第{month}月:")
                        for i, line in enumerate(lines):
                            if line.strip():
                                # 检查是否有COUNCIL的动作（动作6,7,8）
                                actions = line.split(', ')
                                council_actions = [action for action in actions if action.startswith(('6(', '7(', '8('))]
                                if council_actions:
                                    print(f"  第{i+1}行: 发现COUNCIL动作 - {council_actions}")
                                else:
                                    print(f"  第{i+1}行: 无COUNCIL动作")
                    else:
                        print(f"第{month}月: 无动作")
                else:
                    print(f"第{month}月: 文件不存在")
            
        else:
            print("训练失败！")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_council_delay_in_training()

