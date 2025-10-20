#!/usr/bin/env python3
"""
测试Council智能体配置
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

def test_council_config():
    """测试Council智能体配置"""
    print("=== 测试Council智能体配置 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 检查智能体列表
    agents = cfg['solver']['rl']['agents']
    print(f"智能体列表: {agents}")
    assert 'Council' in agents, "Council智能体未添加到agents列表"
    assert len(agents) == 3, f"智能体数量应为3，实际为{len(agents)}"
    
    # 检查预算配置
    budgets = cfg['budget_system']['initial_budgets']
    print(f"预算配置: {budgets}")
    assert 'Council' in budgets, "Council预算未配置"
    assert budgets['Council'] == 8000, f"Council预算应为8000，实际为{budgets['Council']}"
    
    # 检查Council专用配置
    council_config = cfg['growth_v4_1']['evaluation']['council']
    print(f"Council配置: {council_config}")
    assert council_config['enabled'] == True, "Council未启用"
    assert council_config['river_bypass'] == True, "Council河流绕过未启用"
    assert council_config['skip_if_no_candidates'] == True, "Council跳过无候选槽位未启用"
    
    # 检查尺寸奖励
    size_bonus = council_config['size_bonus']
    print(f"Council尺寸奖励: {size_bonus}")
    assert size_bonus['A'] == 2550, f"A型奖励应为2550，实际为{size_bonus['A']}"
    assert size_bonus['B'] == 3550, f"B型奖励应为3550，实际为{size_bonus['B']}"
    assert size_bonus['C'] == 4050, f"C型奖励应为4050，实际为{size_bonus['C']}"
    
    # 检查对岸奖励
    other_side_bonus = council_config['other_side_bonus']
    print(f"Council对岸奖励: {other_side_bonus}")
    assert other_side_bonus['A'] == 70.0, f"A型对岸奖励应为70.0，实际为{other_side_bonus['A']}"
    assert other_side_bonus['B'] == 90.0, f"B型对岸奖励应为90.0，实际为{other_side_bonus['B']}"
    assert other_side_bonus['C'] == 110.0, f"C型对岸奖励应为110.0，实际为{other_side_bonus['C']}"
    
    # 检查邻近性缩放
    proximity_scale = council_config['proximity_scale']
    print(f"Council邻近性缩放: {proximity_scale}")
    assert proximity_scale['A'] == 0.12, f"A型邻近性缩放应为0.12，实际为{proximity_scale['A']}"
    assert proximity_scale['B'] == 0.10, f"B型邻近性缩放应为0.10，实际为{proximity_scale['B']}"
    assert proximity_scale['C'] == 0.10, f"C型邻近性缩放应为0.10，实际为{proximity_scale['C']}"
    
    # 检查动作枚举配置
    caps = cfg['growth_v4_1']['enumeration']['caps']['top_slots_per_agent_size']
    print(f"动作枚举配置: {caps}")
    assert 'Council' in caps, "Council动作枚举配置未添加"
    assert caps['Council']['A'] == 120, f"Council A型槽位上限应为120，实际为{caps['Council']['A']}"
    assert caps['Council']['B'] == 120, f"Council B型槽位上限应为120，实际为{caps['Council']['B']}"
    assert caps['Council']['C'] == 120, f"Council C型槽位上限应为120，实际为{caps['Council']['C']}"
    
    # 检查EDU配置清理
    edu_caps = caps['EDU']
    print(f"EDU动作枚举配置: {edu_caps}")
    assert 'A' not in edu_caps, "EDU不应包含A型配置"
    assert 'B' not in edu_caps, "EDU不应包含B型配置"
    assert 'C' not in edu_caps, "EDU不应包含C型配置"
    assert 'S' in edu_caps, "EDU应包含S型配置"
    assert 'M' in edu_caps, "EDU应包含M型配置"
    assert 'L' in edu_caps, "EDU应包含L型配置"
    
    # 检查EDU尺寸奖励清理
    edu_size_bonus = cfg['growth_v4_1']['evaluation']['size_bonus']
    print(f"EDU尺寸奖励: {edu_size_bonus}")
    assert 'A' not in edu_size_bonus, "EDU不应包含A型奖励"
    assert 'B' not in edu_size_bonus, "EDU不应包含B型奖励"
    assert 'C' not in edu_size_bonus, "EDU不应包含C型奖励"
    
    print("OK 所有配置检查通过！")
    print("=== 测试完成 ===")

if __name__ == "__main__":
    test_council_config()
