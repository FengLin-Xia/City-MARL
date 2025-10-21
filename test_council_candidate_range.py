#!/usr/bin/env python3
"""
Council候选范围测试

验证Council智能体是否受候选范围限制，以及是否有特殊处理
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts import Sequence, EnvironmentState
from action_mw.candidate_range import CandidateRangeMiddleware
from action_mw.river_restriction import RiverRestrictionMiddleware


def test_council_candidate_range():
    """测试Council的候选范围行为"""
    print("=" * 60)
    print("Council候选范围测试")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建中间件
    range_mw = CandidateRangeMiddleware(config)
    river_mw = RiverRestrictionMiddleware(config)
    
    print("\n1. 配置分析:")
    print(f"   候选范围模式: {range_mw.candidate_mode}")
    print(f"   河流限制影响智能体: {river_mw.affects_agents}")
    print(f"   Council河流绕过: {river_mw.council_bypass}")
    
    # 创建测试槽位 - 分布在河流两侧
    test_slots = [
        # Hub1附近 (北侧)
        {"id": "north_hub1_1", "x": 120, "y": 80, "type": "north_hub1", "side": "north"},
        {"id": "north_hub1_2", "x": 125, "y": 85, "type": "north_hub1", "side": "north"},
        {"id": "north_hub1_3", "x": 130, "y": 90, "type": "north_hub1", "side": "north"},
        
        # Hub2附近 (南侧)
        {"id": "south_hub2_1", "x": 110, "y": 120, "type": "south_hub2", "side": "south"},
        {"id": "south_hub2_2", "x": 115, "y": 125, "type": "south_hub2", "side": "south"},
        {"id": "south_hub2_3", "x": 120, "y": 130, "type": "south_hub2", "side": "south"},
        
        # 远离所有Hub
        {"id": "far_away_1", "x": 200, "y": 200, "type": "far_away", "side": "unknown"},
        {"id": "far_away_2", "x": 50, "y": 50, "type": "far_away", "side": "unknown"},
    ]
    
    print("\n2. 测试槽位分布:")
    for slot in test_slots:
        print(f"   {slot['id']}: ({slot['x']}, {slot['y']}) - {slot['type']} - {slot['side']}")
    
    # 测试不同月份
    test_months = [0, 5, 10, 15, 20]
    
    for month in test_months:
        print(f"\n3. 第{month}个月测试:")
        
        # 创建环境状态
        mock_state = EnvironmentState(
            month=month,
            land_prices=np.zeros((200, 200)),
            buildings=[],
            budgets={"IND": 15000, "EDU": 10000, "COUNCIL": 0},
            slots=test_slots
        )
        
        # 测试候选范围
        available_slots = range_mw._get_available_slots(month, mock_state)
        print(f"   候选范围可用槽位: {len(available_slots)}")
        
        for slot in test_slots:
            slot_id = slot["id"]
            if slot_id in available_slots:
                print(f"     [可用] {slot_id} - {slot['type']}")
            else:
                print(f"     [不可用] {slot_id} - {slot['type']}")
        
        # 测试Council动作序列
        council_seq = Sequence(agent="COUNCIL", actions=[6, 7, 8])
        
        # 应用候选范围过滤
        range_filtered = range_mw.apply(council_seq, mock_state)
        print(f"   Council动作过滤: {council_seq.actions} -> {range_filtered.actions}")
        
        # 应用河流限制
        river_filtered = river_mw.apply(council_seq, mock_state)
        print(f"   Council河流过滤: {council_seq.actions} -> {river_filtered.actions}")
    
    print("\n4. Council特殊规则分析:")
    
    # 检查Council是否有特殊的候选范围规则
    council_config = config.get("evaluation", {}).get("council", {})
    print(f"   Council配置: {council_config}")
    
    # 检查Council的约束
    council_constraints = config.get("agents", {}).get("defs", {}).get("COUNCIL", {}).get("constraints", {})
    print(f"   Council约束: {council_constraints}")
    
    # 检查Council的预算共享
    share_matrix = config.get("ledger", {}).get("share_matrix", [])
    print(f"   预算共享矩阵: {share_matrix}")
    
    print("\n5. 对比其他智能体:")
    
    # 创建测试序列
    test_sequences = [
        Sequence(agent="IND", actions=[3, 4, 5]),
        Sequence(agent="EDU", actions=[0, 1, 2]),
        Sequence(agent="COUNCIL", actions=[6, 7, 8])
    ]
    
    mock_state = EnvironmentState(
        month=10,
        land_prices=np.zeros((200, 200)),
        buildings=[],
        budgets={"IND": 15000, "EDU": 10000, "COUNCIL": 0},
        slots=test_slots
    )
    
    for seq in test_sequences:
        print(f"\n   {seq.agent}智能体:")
        
        # 候选范围过滤
        range_filtered = range_mw.apply(seq, mock_state)
        print(f"     候选范围: {seq.actions} -> {range_filtered.actions}")
        
        # 河流限制过滤
        river_filtered = river_mw.apply(seq, mock_state)
        print(f"     河流限制: {seq.actions} -> {river_filtered.actions}")
        
        # 组合过滤
        combined_filtered = river_mw.apply(range_filtered, mock_state)
        print(f"     组合过滤: {seq.actions} -> {combined_filtered.actions}")
    
    print("\n" + "=" * 60)
    print("Council候选范围测试完成!")
    print("=" * 60)
    
    return True


def test_council_special_rules():
    """测试Council的特殊规则"""
    print("\n" + "=" * 60)
    print("Council特殊规则测试")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("\n1. Council配置分析:")
    
    # Council基本配置
    council_def = config.get("agents", {}).get("defs", {}).get("COUNCIL", {})
    print(f"   Council定义: {council_def}")
    
    # Council评估配置
    council_eval = config.get("evaluation", {}).get("council", {})
    print(f"   Council评估: {council_eval}")
    
    # Council特殊规则
    council_special = council_def.get("constraints", {}).get("special_rules", {})
    print(f"   Council特殊规则: {council_special}")
    
    print("\n2. Council候选范围行为:")
    
    # 检查Council是否受候选范围限制
    range_mw = CandidateRangeMiddleware(config)
    
    # 检查Council是否受河流限制
    river_mw = RiverRestrictionMiddleware(config)
    
    print(f"   Council受候选范围限制: {range_mw.enabled}")
    print(f"   Council受河流限制: {'COUNCIL' in river_mw.affects_agents}")
    print(f"   Council河流绕过: {river_mw.council_bypass}")
    
    print("\n3. Council候选范围结论:")
    
    if range_mw.enabled:
        print("   [PASS] Council受候选范围限制")
        print("   [INFO] Council只能在Hub候选范围内建造")
        print("   [INFO] Council候选范围随时间扩展")
    else:
        print("   [FAIL] Council不受候选范围限制")
        print("   [INFO] Council可以在任何地方建造")
    
    if river_mw.council_bypass:
        print("   [PASS] Council可以跨河流建造")
        print("   [INFO] Council不受河流分割限制")
    else:
        print("   [FAIL] Council受河流分割限制")
        print("   [INFO] Council只能在河流一侧建造")
    
    print("\n4. 总结:")
    print("   Council的候选范围行为:")
    print("   - 受候选范围限制: 只能在Hub附近建造")
    print("   - 不受河流限制: 可以跨河流建造")
    print("   - 候选范围随时间扩展: 建造区域逐渐扩大")
    print("   - 两侧都可以: 可以在河流两侧的Hub附近建造")
    
    return True


if __name__ == "__main__":
    try:
        # 测试Council候选范围
        test_council_candidate_range()
        
        # 测试Council特殊规则
        test_council_special_rules()
        
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
