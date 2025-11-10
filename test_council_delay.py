#!/usr/bin/env python3
"""
测试COUNCIL的start_after_month延迟机制
"""

import json
from logic.v5_enumeration import V5ActionEnumerator

def test_council_delay():
    """测试COUNCIL的延迟机制"""
    
    # 加载配置
    with open("configs/city_config_v5_0.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 创建枚举器
    enumerator = V5ActionEnumerator(config)
    
    # 测试不同月份的COUNCIL动作枚举
    test_months = [0, 1, 2, 5, 6, 7]
    
    print("=== COUNCIL延迟机制测试 ===")
    print(f"配置中的start_after_month: {config['agents']['defs']['COUNCIL']['constraints']['special_rules']['start_after_month']}")
    print()
    
    for month in test_months:
        print(f"--- 第{month}月 ---")
        
        # 获取COUNCIL配置
        agent_config = config["agents"]["defs"]["COUNCIL"]
        print(f"Agent配置: {agent_config}")
        
        # 检查special_rules
        special_rules = agent_config.get("constraints", {}).get("special_rules", {})
        start_after_month = special_rules.get("start_after_month")
        print(f"Special rules: {special_rules}")
        print(f"Start after month: {start_after_month}")
        
        # 模拟延迟检查逻辑
        if start_after_month is not None and month < start_after_month:
            print(f"[DELAYED] COUNCIL应该被延迟到第{start_after_month}月，当前是第{month}月")
            print(f"   应该返回空候选列表")
        else:
            print(f"[ACTIVE] COUNCIL可以执行动作，当前是第{month}月")
        
        print()

if __name__ == "__main__":
    test_council_delay()
