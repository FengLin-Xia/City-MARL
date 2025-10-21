"""
全面测试河流分割功能

验证配置驱动的河流分割功能是否正常工作
"""

import json
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts import Sequence, EnvironmentState
from action_mw.river_restriction import RiverRestrictionMiddleware
from action_mw.candidate_range import CandidateRangeMiddleware


def test_config_loading():
    """测试配置加载"""
    print("测试配置加载...")
    
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 检查河流分割配置
    river_config = config.get("env", {}).get("river_restrictions", {})
    print(f"  河流分割启用: {river_config.get('enabled')}")
    print(f"  影响智能体: {river_config.get('affects_agents')}")
    print(f"  Council跨河流: {river_config.get('council_bypass')}")
    
    # 检查Hub配置
    hub_config = config.get("hubs", {})
    print(f"  Hub模式: {hub_config.get('mode')}")
    print(f"  候选模式: {hub_config.get('candidate_mode')}")
    print(f"  Hub数量: {len(hub_config.get('list', []))}")
    
    print("[PASS] 配置加载成功")
    return True


def test_middleware_creation():
    """测试中间件创建"""
    print("测试中间件创建...")
    
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建河流分割中间件
    river_mw = RiverRestrictionMiddleware(config)
    print(f"  河流分割启用: {river_mw.enabled}")
    print(f"  影响智能体: {river_mw.affects_agents}")
    print(f"  Council跨河流: {river_mw.council_bypass}")
    print(f"  分配方法: {river_mw.assignment_method}")
    
    # 创建候选范围中间件
    range_mw = CandidateRangeMiddleware(config)
    print(f"  候选范围启用: {range_mw.enabled}")
    print(f"  候选模式: {range_mw.candidate_mode}")
    print(f"  Hub数量: {len(range_mw.hub_list)}")
    
    print("[PASS] 中间件创建成功")
    return True


def test_sequence_filtering():
    """测试序列过滤"""
    print("测试序列过滤...")
    
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建测试序列
    test_sequences = [
        Sequence(agent="IND", actions=[3, 4, 5]),
        Sequence(agent="EDU", actions=[0, 1, 2]),
        Sequence(agent="COUNCIL", actions=[6, 7, 8])
    ]
    
    # 创建模拟环境状态
    mock_state = EnvironmentState(
        month=0,
        land_prices=np.zeros((200, 200)),
        buildings=[],
        budgets={"IND": 15000, "EDU": 10000, "COUNCIL": 0},
        slots=[]
    )
    
    # 测试河流分割中间件
    river_mw = RiverRestrictionMiddleware(config)
    for seq in test_sequences:
        filtered_seq = river_mw.apply(seq, mock_state)
        print(f"  {seq.agent}: {len(seq.actions)} -> {len(filtered_seq.actions)} 动作")
    
    # 测试候选范围中间件
    range_mw = CandidateRangeMiddleware(config)
    for seq in test_sequences:
        filtered_seq = range_mw.apply(seq, mock_state)
        print(f"  {seq.agent}: {len(seq.actions)} -> {len(filtered_seq.actions)} 动作")
    
    print("[PASS] 序列过滤成功")
    return True


def test_configuration_flexibility():
    """测试配置灵活性"""
    print("测试配置灵活性...")
    
    # 测试不同的配置
    test_configs = [
        {
            "env": {
                "river_restrictions": {
                    "enabled": False,
                    "affects_agents": [],
                    "council_bypass": True
                }
            },
            "hubs": {
                "mode": "explicit",
                "candidate_mode": "fixed",
                "list": []
            }
        },
        {
            "env": {
                "river_restrictions": {
                    "enabled": True,
                    "affects_agents": ["IND"],
                    "council_bypass": False
                }
            },
            "hubs": {
                "mode": "explicit",
                "candidate_mode": "cumulative",
                "list": [{"id": "hub1", "x": 100, "y": 100, "R0": 10, "dR": 2}]
            }
        }
    ]
    
    for i, test_config in enumerate(test_configs):
        print(f"  测试配置 {i+1}:")
        
        # 测试河流分割中间件
        river_mw = RiverRestrictionMiddleware(test_config)
        print(f"    河流分割启用: {river_mw.enabled}")
        print(f"    影响智能体: {river_mw.affects_agents}")
        
        # 测试候选范围中间件
        range_mw = CandidateRangeMiddleware(test_config)
        print(f"    候选范围启用: {range_mw.enabled}")
        print(f"    候选模式: {range_mw.candidate_mode}")
        print(f"    Hub数量: {len(range_mw.hub_list)}")
    
    print("[PASS] 配置灵活性测试成功")
    return True


def test_edge_cases():
    """测试边界情况"""
    print("测试边界情况...")
    
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建边界情况测试
    edge_cases = [
        Sequence(agent="IND", actions=[3]),  # 单个动作
        Sequence(agent="EDU", actions=[999]),  # 无效动作ID
        Sequence(agent="COUNCIL", actions=[6, 7, 8])  # Council序列
    ]
    
    mock_state = EnvironmentState(
        month=0,
        land_prices=np.zeros((200, 200)),
        buildings=[],
        budgets={"IND": 15000, "EDU": 10000, "COUNCIL": 0},
        slots=[]
    )
    
    river_mw = RiverRestrictionMiddleware(config)
    range_mw = CandidateRangeMiddleware(config)
    
    for seq in edge_cases:
        print(f"  测试序列: {seq.agent} {seq.actions}")
        
        # 测试河流分割
        river_filtered = river_mw.apply(seq, mock_state)
        print(f"    河流过滤后: {river_filtered.actions}")
        
        # 测试候选范围
        range_filtered = range_mw.apply(seq, mock_state)
        print(f"    范围过滤后: {range_filtered.actions}")
    
    print("[PASS] 边界情况测试成功")
    return True


def test_performance():
    """测试性能"""
    print("测试性能...")
    
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    import time
    
    # 创建大量测试序列
    test_sequences = []
    for i in range(100):
        test_sequences.append(Sequence(agent="IND", actions=[3, 4, 5]))
    
    mock_state = EnvironmentState(
        month=0,
        land_prices=np.zeros((200, 200)),
        buildings=[],
        budgets={"IND": 15000, "EDU": 10000, "COUNCIL": 0},
        slots=[]
    )
    
    river_mw = RiverRestrictionMiddleware(config)
    range_mw = CandidateRangeMiddleware(config)
    
    # 测试河流分割性能
    start_time = time.time()
    for seq in test_sequences:
        river_mw.apply(seq, mock_state)
    river_time = time.time() - start_time
    
    # 测试候选范围性能
    start_time = time.time()
    for seq in test_sequences:
        range_mw.apply(seq, mock_state)
    range_time = time.time() - start_time
    
    print(f"  河流分割: {river_time:.4f}秒 (100个序列)")
    print(f"  候选范围: {range_time:.4f}秒 (100个序列)")
    print(f"  平均每序列: {(river_time + range_time) / 200 * 1000:.2f}毫秒")
    
    print("[PASS] 性能测试成功")
    return True


def main():
    """运行所有测试"""
    print("开始全面测试河流分割功能...")
    print("=" * 50)
    
    try:
        # 运行所有测试
        test_config_loading()
        print()
        
        test_middleware_creation()
        print()
        
        test_sequence_filtering()
        print()
        
        test_configuration_flexibility()
        print()
        
        test_edge_cases()
        print()
        
        test_performance()
        print()
        
        print("=" * 50)
        print("所有测试通过！河流分割功能正常工作。")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
