#!/usr/bin/env python3
"""
测试导出重复问题

检查导出系统中是否有重复建筑问题
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.v5_0.city_env import V5CityEnvironment
from contracts import Sequence


def test_export_duplicates():
    """测试导出重复问题"""
    print("=" * 80)
    print("测试导出重复问题")
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
        
        for step in range(5):
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
                    print(f"   - 执行后已占用槽位: {len(env.occupied_slots)}")
            else:
                print(f"   - 无候选动作")
        
        # 检查建筑记录
        print(f"\n   建筑记录检查:")
        if env.buildings:
            print(f"   - 建筑记录数量: {len(env.buildings)}")
            for i, building in enumerate(env.buildings):
                print(f"   - 建筑 {i+1}: {building}")
        else:
            print(f"   - 无建筑记录")
        
        # 检查步骤日志
        print(f"\n   步骤日志检查:")
        if env.step_logs:
            print(f"   - 步骤日志数量: {len(env.step_logs)}")
            for i, log in enumerate(env.step_logs):
                print(f"   - 日志 {i+1}: {log}")
        else:
            print(f"   - 无步骤日志")
        
        # 检查槽位占用记录
        print(f"\n   槽位占用记录检查:")
        print(f"   - 已占用槽位: {len(env.occupied_slots)}")
        print(f"   - 已占用槽位列表: {list(env.occupied_slots)}")
        
        # 检查槽位状态
        print(f"\n   槽位状态检查:")
        occupied_slot_count = 0
        for slot_id, slot in env.slots.items():
            if hasattr(slot, 'occupied') and slot.occupied:
                occupied_slot_count += 1
                print(f"   - 槽位 {slot_id}: 已占用, 位置 ({slot.x:.1f}, {slot.y:.1f})")
            elif isinstance(slot, dict) and slot.get('occupied', False):
                occupied_slot_count += 1
                print(f"   - 槽位 {slot_id}: 已占用, 位置 ({slot.get('x', 0):.1f}, {slot.get('y', 0):.1f})")
        
        print(f"   - 总占用槽位数: {occupied_slot_count}")
        
        # 检查导出数据
        print(f"\n   导出数据检查:")
        
        # 检查环境状态
        current_state = env._get_current_state()
        print(f"   - 当前月份: {current_state.month}")
        print(f"   - 建筑数量: {len(current_state.buildings)}")
        print(f"   - 槽位数量: {len(current_state.slots)}")
        
        # 检查槽位数据
        slot_positions = []
        for slot in current_state.slots:
            if hasattr(slot, 'x') and hasattr(slot, 'y'):
                slot_positions.append((slot.x, slot.y))
            elif isinstance(slot, dict):
                slot_positions.append((slot.get('x', 0), slot.get('y', 0)))
        
        if slot_positions:
            x_coords = [pos[0] for pos in slot_positions]
            y_coords = [pos[1] for pos in slot_positions]
            print(f"   - 槽位X坐标范围: {min(x_coords):.1f} - {max(x_coords):.1f}")
            print(f"   - 槽位Y坐标范围: {min(y_coords):.1f} - {max(y_coords):.1f}")
            
            # 检查重复坐标
            position_counts = {}
            for pos in slot_positions:
                if pos in position_counts:
                    position_counts[pos] += 1
                else:
                    position_counts[pos] = 1
            
            duplicate_positions = {pos: count for pos, count in position_counts.items() if count > 1}
            if duplicate_positions:
                print(f"   - [WARNING] 发现重复坐标: {len(duplicate_positions)} 个")
                for pos, count in duplicate_positions.items():
                    print(f"     - 坐标 {pos}: {count} 次")
            else:
                print(f"   - [PASS] 无重复坐标")
        
        # 检查导出文件
        print(f"\n   导出文件检查:")
        
        # 检查输出目录
        output_dir = "outputs"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"   - 输出目录: {output_dir}")
            print(f"   - 文件数量: {len(files)}")
            
            # 检查v4兼容文件
            v4_files = [f for f in files if f.startswith("v4_compatible_month_")]
            if v4_files:
                print(f"   - v4兼容文件: {len(v4_files)} 个")
                for file in sorted(v4_files)[:3]:  # 只显示前3个
                    file_path = os.path.join(output_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            print(f"     - {file}: {len(content)} 字符")
                            if content:
                                lines = content.strip().split('\n')
                                print(f"       - 行数: {len(lines)}")
                                if lines:
                                    print(f"       - 第一行: {lines[0]}")
                    except Exception as e:
                        print(f"     - {file}: 读取失败 - {e}")
            else:
                print(f"   - 无v4兼容文件")
        else:
            print(f"   - 输出目录不存在")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_export_duplicates()
