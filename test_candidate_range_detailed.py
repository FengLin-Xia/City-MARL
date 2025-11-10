#!/usr/bin/env python3
"""
候选范围功能详细测试

展示候选范围功能的具体工作原理
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts import Sequence, EnvironmentState
from action_mw.candidate_range import CandidateRangeMiddleware


def test_candidate_range_workflow():
    """测试候选范围工作流程"""
    print("=" * 60)
    print("候选范围功能详细测试")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建候选范围中间件
    range_mw = CandidateRangeMiddleware(config)
    
    print("\n1. 配置信息:")
    print(f"   模式: {range_mw.config.get('mode', 'unknown')}")
    print(f"   候选模式: {range_mw.candidate_mode}")
    print(f"   容差: {range_mw.tolerance}")
    print(f"   Hub数量: {len(range_mw.hub_list)}")
    
    # 显示Hub配置
    for i, hub in enumerate(range_mw.hub_list):
        print(f"   Hub{i+1}: {hub['id']} 位置({hub['x']}, {hub['y']}) R0={hub['R0']} dR={hub['dR']}")
    
    print("\n2. 半径计算测试:")
    test_months = [0, 5, 10, 15, 20]
    for month in test_months:
        radii = range_mw.get_current_radii(month)
        print(f"   第{month}个月:")
        for hub_id, radius in radii.items():
            print(f"     {hub_id}: 半径 {radius:.1f}")
    
    print("\n3. 槽位过滤测试:")
    
    # 创建测试槽位
    test_slots = [
        {"id": "slot_001", "x": 120, "y": 80, "type": "near_hub1"},    # 接近hub1
        {"id": "slot_002", "x": 130, "y": 90, "type": "near_hub1"},    # 接近hub1
        {"id": "slot_003", "x": 200, "y": 200, "type": "far_away"},    # 远离所有hub
        {"id": "slot_004", "x": 110, "y": 120, "type": "near_hub2"},   # 接近hub2
        {"id": "slot_005", "x": 150, "y": 150, "type": "middle"},      # 中间位置
    ]
    
    # 创建环境状态
    mock_state = EnvironmentState(
        month=10,  # 第10个月
        land_prices=np.zeros((200, 200)),
        buildings=[],
        budgets={"IND": 15000, "EDU": 10000, "COUNCIL": 0},
        slots=test_slots
    )
    
    # 测试不同月份的可用槽位
    for month in [0, 5, 10, 15, 20]:
        print(f"\n   第{month}个月可用槽位:")
        available_slots = range_mw._get_available_slots(month, mock_state)
        print(f"     可用槽位数量: {len(available_slots)}")
        
        for slot in test_slots:
            slot_id = slot["id"]
            if slot_id in available_slots:
                print(f"     [可用] {slot_id} ({slot['x']}, {slot['y']}) - {slot['type']}")
            else:
                print(f"     [不可用] {slot_id} ({slot['x']}, {slot['y']}) - {slot['type']}")
    
    print("\n4. 动作序列过滤测试:")
    
    # 创建测试序列
    test_sequences = [
        Sequence(agent="IND", actions=[3, 4, 5]),
        Sequence(agent="EDU", actions=[0, 1, 2]),
        Sequence(agent="COUNCIL", actions=[6, 7, 8])
    ]
    
    for seq in test_sequences:
        print(f"\n   {seq.agent}智能体序列: {seq.actions}")
        
        # 应用候选范围过滤
        filtered_seq = range_mw.apply(seq, mock_state)
        print(f"     过滤后: {filtered_seq.actions}")
        print(f"     过滤率: {len(filtered_seq.actions)}/{len(seq.actions)} = {len(filtered_seq.actions)/len(seq.actions)*100:.1f}%")
    
    print("\n5. 累积模式 vs 固定模式对比:")
    
    # 测试累积模式
    print("   累积模式 (cumulative):")
    for month in [0, 5, 10]:
        radii = range_mw.get_current_radii(month)
        print(f"     第{month}个月: {radii}")
    
    # 模拟固定模式
    print("   固定模式 (fixed):")
    for month in [0, 5, 10]:
        fixed_radii = {}
        for hub_config in range_mw.hub_list:
            hub_id = hub_config["id"]
            R0 = hub_config["R0"]
            fixed_radii[hub_id] = R0
        print(f"     第{month}个月: {fixed_radii}")
    
    print("\n6. 距离计算验证:")
    
    # 验证距离计算
    hub1_pos = (122, 80)
    test_positions = [
        (120, 80, "非常接近hub1"),
        (130, 90, "接近hub1"),
        (150, 100, "中等距离"),
        (200, 200, "远离hub1")
    ]
    
    for x, y, desc in test_positions:
        distance = range_mw._calculate_distance(hub1_pos, (x, y))
        print(f"   位置({x}, {y}) - {desc}: 距离={distance:.1f}")
    
    print("\n" + "=" * 60)
    print("候选范围功能测试完成!")
    print("=" * 60)
    
    return True


def test_candidate_range_edge_cases():
    """测试候选范围边界情况"""
    print("\n" + "=" * 60)
    print("候选范围边界情况测试")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    range_mw = CandidateRangeMiddleware(config)
    
    print("\n1. 空槽位测试:")
    empty_state = EnvironmentState(
        month=5,
        land_prices=np.zeros((200, 200)),
        buildings=[],
        budgets={"IND": 15000, "EDU": 10000, "COUNCIL": 0},
        slots=[]
    )
    
    available_slots = range_mw._get_available_slots(5, empty_state)
    print(f"   空槽位环境可用槽位: {len(available_slots)}")
    
    print("\n2. 边界槽位测试:")
    boundary_slots = [
        {"id": "boundary_1", "x": 122, "y": 80, "type": "exact_hub1"},      # 正好在hub1位置
        {"id": "boundary_2", "x": 127, "y": 80, "type": "edge_hub1"},      # 在hub1边界
        {"id": "boundary_3", "x": 128, "y": 80, "type": "outside_hub1"},   # 刚好在hub1外面
    ]
    
    boundary_state = EnvironmentState(
        month=5,
        land_prices=np.zeros((200, 200)),
        buildings=[],
        budgets={"IND": 15000, "EDU": 10000, "COUNCIL": 0},
        slots=boundary_slots
    )
    
    available_slots = range_mw._get_available_slots(5, boundary_state)
    print(f"   边界槽位环境可用槽位: {len(available_slots)}")
    
    for slot in boundary_slots:
        slot_id = slot["id"]
        if slot_id in available_slots:
            print(f"     [可用] {slot_id} - {slot['type']}")
        else:
            print(f"     [不可用] {slot_id} - {slot['type']}")
    
    print("\n3. 容差测试:")
    print(f"   当前容差: {range_mw.tolerance}")
    
    # 测试不同容差的影响
    for tolerance in [0.0, 0.5, 1.0, 2.0]:
        range_mw.tolerance = tolerance
        available_slots = range_mw._get_available_slots(5, boundary_state)
        print(f"   容差 {tolerance}: 可用槽位 {len(available_slots)}")
    
    # 恢复原始容差
    range_mw.tolerance = 0.5
    
    print("\n边界情况测试完成!")


if __name__ == "__main__":
    try:
        # 测试候选范围工作流程
        test_candidate_range_workflow()
        
        # 测试边界情况
        test_candidate_range_edge_cases()
        
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

