"""
测试河流分割功能

验证配置驱动的河流分割功能是否正常工作
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contracts import Sequence, EnvironmentState
from action_mw.river_restriction import RiverRestrictionMiddleware
from action_mw.candidate_range import CandidateRangeMiddleware


def test_river_restriction_config():
    """测试河流分割配置"""
    print("测试河流分割配置...")
    
    # 加载v5.0配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 检查河流分割配置
    river_config = config.get("env", {}).get("river_restrictions", {})
    
    assert river_config.get("enabled") == True, "河流分割应该启用"
    assert "IND" in river_config.get("affects_agents", []), "IND应该受河流限制"
    assert "EDU" in river_config.get("affects_agents", []), "EDU应该受河流限制"
    assert river_config.get("council_bypass") == True, "Council应该可以跨河流"
    
    print("[PASS] 河流分割配置正确")
    return True


def test_candidate_range_config():
    """测试候选范围配置"""
    print("测试候选范围配置...")
    
    # 加载v5.0配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 检查Hub配置
    hub_config = config.get("hubs", {})
    
    assert hub_config.get("mode") == "explicit", "Hub模式应该是explicit"
    assert hub_config.get("candidate_mode") == "cumulative", "候选模式应该是cumulative"
    assert len(hub_config.get("list", [])) == 2, "应该有2个Hub"
    
    # 检查Hub参数
    hub_list = hub_config.get("list", [])
    for hub in hub_list:
        assert "id" in hub, "Hub应该有ID"
        assert "x" in hub and "y" in hub, "Hub应该有坐标"
        assert "R0" in hub and "dR" in hub, "Hub应该有半径参数"
    
    print("[PASS] 候选范围配置正确")
    return True


def test_middleware_initialization():
    """测试中间件初始化"""
    print("测试中间件初始化...")
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 测试河流分割中间件
    river_mw = RiverRestrictionMiddleware(config)
    assert river_mw.enabled == True, "河流分割中间件应该启用"
    assert "IND" in river_mw.affects_agents, "IND应该受河流限制"
    assert "EDU" in river_mw.affects_agents, "EDU应该受河流限制"
    assert river_mw.council_bypass == True, "Council应该可以跨河流"
    
    # 测试候选范围中间件
    range_mw = CandidateRangeMiddleware(config)
    assert range_mw.enabled == True, "候选范围中间件应该启用"
    assert range_mw.candidate_mode == "cumulative", "候选模式应该是cumulative"
    assert len(range_mw.hub_list) == 2, "应该有2个Hub"
    
    print("[PASS] 中间件初始化成功")
    return True


def test_middleware_application():
    """测试中间件应用"""
    print("测试中间件应用...")
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建测试序列
    test_seq = Sequence(agent="IND", actions=[3, 4, 5])
    
    # 创建模拟环境状态
    import numpy as np
    mock_state = EnvironmentState(
        month=0,
        land_prices=np.zeros((200, 200)),
        buildings=[],
        budgets={"IND": 15000, "EDU": 10000, "COUNCIL": 0},
        slots=[]
    )
    
    # 测试河流分割中间件
    river_mw = RiverRestrictionMiddleware(config)
    filtered_seq = river_mw.apply(test_seq, mock_state)
    
    # 测试候选范围中间件
    range_mw = CandidateRangeMiddleware(config)
    filtered_seq = range_mw.apply(test_seq, mock_state)
    
    print("[PASS] 中间件应用成功")
    return True


def main():
    """运行所有测试"""
    print("开始测试河流分割功能...")
    
    try:
        # 测试配置
        test_river_restriction_config()
        test_candidate_range_config()
        
        # 测试中间件
        test_middleware_initialization()
        test_middleware_application()
        
        print("\n所有测试通过！河流分割功能配置正确。")
        return True
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
