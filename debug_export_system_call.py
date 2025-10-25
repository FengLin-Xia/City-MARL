#!/usr/bin/env python3
"""
调试导出系统调用

检查导出系统是否正确传递了StepLog，以及StepLog中是否包含槽位位置信息
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence
from exporters.v5_0.export_system import V5ExportSystem


def debug_export_system_call():
    """调试导出系统调用"""
    print("=" * 80)
    print("调试导出系统调用")
    print("=" * 80)
    
    try:
        # 创建环境
        env = V5CityEnvironment('configs/city_config_v5_0.json')
        print("   [PASS] 环境初始化成功")
        
        # 重置环境
        state = env.reset()
        print(f"   初始状态: 月份={state.month}, 步骤={env.current_step}")
        
        # 模拟训练过程
        print(f"\n   模拟训练过程:")
        
        step_logs = []
        env_states = []
        
        for step in range(3):
            print(f"\n   步骤 {step}:")
            
            # 获取当前智能体
            current_agent = env.current_agent
            print(f"   - 当前智能体: {current_agent}")
            
            # 获取候选动作
            candidates = env.get_action_candidates(current_agent)
            if candidates:
                # 选择第一个候选
                selected_candidate = candidates[0]
                selected_slots = selected_candidate.meta.get("slots", [])
                
                if selected_slots:
                    slot_id = selected_slots[0]
                    print(f"   - 选择槽位: {slot_id}")
                    
                    # 执行动作
                    sequence = Sequence(
                        agent=current_agent,
                        actions=[selected_candidate.id]
                    )
                    
                    next_state, reward, done, info = env.step(current_agent, sequence)
                    
                    # 记录步骤日志和环境状态
                    if env.step_logs:
                        step_logs.append(env.step_logs[-1])
                        env_states.append(next_state)
                        
                        # 检查StepLog中的槽位位置信息
                        log = env.step_logs[-1]
                        print(f"   - StepLog: {log}")
                        
                        if hasattr(log, 'slot_positions') and log.slot_positions:
                            print(f"   - 槽位位置信息: {log.slot_positions}")
                        else:
                            print(f"   - 无槽位位置信息")
            else:
                print(f"   - 无候选动作")
        
        # 检查收集的数据
        print(f"\n   收集的数据检查:")
        print(f"   - StepLog数量: {len(step_logs)}")
        print(f"   - 环境状态数量: {len(env_states)}")
        
        for i, log in enumerate(step_logs):
            print(f"   - StepLog {i+1}: {log}")
            if hasattr(log, 'slot_positions') and log.slot_positions:
                print(f"     - 槽位位置: {log.slot_positions}")
            else:
                print(f"     - 无槽位位置信息")
        
        # 测试导出系统
        print(f"\n   测试导出系统:")
        
        # 创建导出系统
        export_system = V5ExportSystem('configs/city_config_v5_0.json')
        print(f"   - 导出系统创建成功")
        
        # 测试导出
        output_dir = "debug_output"
        try:
            results = export_system.export_all(step_logs, env_states, output_dir)
            print(f"   - 导出成功: {results}")
            
            # 检查导出文件
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                print(f"   - 导出文件: {files}")
                
                # 检查TXT文件内容
                txt_files = [f for f in files if f.endswith('.txt')]
                if txt_files:
                    for file in txt_files:
                        file_path = os.path.join(output_dir, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                print(f"   - {file}: {content}")
                        except Exception as e:
                            print(f"   - {file}: 读取失败 - {e}")
        except Exception as e:
            print(f"   - 导出失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试TXT导出器直接调用
        print(f"\n   测试TXT导出器直接调用:")
        
        from exporters.v5_0.txt_exporter import V5TXTExporter
        
        txt_exporter = V5TXTExporter('configs/city_config_v5_0.json')
        print(f"   - TXT导出器创建成功")
        
        # 测试坐标获取
        if step_logs:
            log = step_logs[0]
            print(f"   - 测试StepLog: {log}")
            
            if hasattr(log, 'slot_positions') and log.slot_positions:
                print(f"   - 槽位位置信息: {log.slot_positions}")
                
                # 测试坐标获取方法
                coordinates = txt_exporter._get_coordinates_from_env(log, env_states[0])
                print(f"   - 获取的坐标: {coordinates}")
            else:
                print(f"   - 无槽位位置信息")
        
    except Exception as e:
        print(f"   [FAIL] 调试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("调试完成!")
    print("=" * 80)


if __name__ == "__main__":
    debug_export_system_call()

