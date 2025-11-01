#!/usr/bin/env python3
"""
测试修复后的COUNCIL延迟机制
"""

import json
from logic.v5_enumeration import V5ActionEnumerator

def test_council_delay_fix():
    """测试修复后的COUNCIL延迟机制"""
    
    # 加载配置
    with open("configs/city_config_v5_0.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 创建枚举器
    enumerator = V5ActionEnumerator(config)
    
    # 测试不同月份的COUNCIL动作枚举
    test_months = [0, 1, 2, 5, 6, 7]
    
    print("=== 修复后的COUNCIL延迟机制测试 ===")
    print(f"配置中的start_after_month: {config['agents']['defs']['COUNCIL']['constraints']['special_rules']['start_after_month']}")
    print()
    
    for month in test_months:
        print(f"--- 第{month}月 ---")
        
        # 测试enumerate_with_index方法（多动作模式使用的方法）
        try:
            candidates, cand_idx = enumerator.enumerate_with_index(
                agent="COUNCIL",
                occupied_slots=set(),
                lp_provider=lambda x: 100.0,  # 简单的地价提供函数
                budget=20000.0,
                current_month=month,
                unlocked_actions=None
            )
            
            if len(candidates) == 0:
                print(f"[DELAYED] COUNCIL被正确延迟，返回空候选列表")
            else:
                print(f"[ACTIVE] COUNCIL可以执行动作，返回{len(candidates)}个候选")
                
        except Exception as e:
            print(f"[ERROR] 测试失败: {e}")
        
        print()

if __name__ == "__main__":
    test_council_delay_fix()

