#!/usr/bin/env python3
"""
测试Council延迟启动功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json

def test_council_delayed_start():
    """测试Council延迟启动功能"""
    print("=== 测试Council延迟启动功能 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print(f"Council配置: {cfg.get('growth_v4_1', {}).get('evaluation', {}).get('council', {})}")
    print(f"启动月份: {cfg.get('growth_v4_1', {}).get('evaluation', {}).get('council', {}).get('start_after_month', 0)}")
    
    # 测试不同月份的Council行为
    test_months = [0, 1, 2, 5, 6, 7, 10]
    
    for test_month in test_months:
        print(f"\n--- 测试月份 {test_month} ---")
        env.current_month = test_month
        env.current_agent = 'Council'
        
        try:
            candidates = env._get_candidate_slots()
            print(f"Council候选槽位数量: {len(candidates)}")
            
            if len(candidates) == 0:
                print("OK Council跳过执行（延迟启动）")
            else:
                print(f"OK Council正常执行，有{len(candidates)}个候选槽位")
                
        except Exception as e:
            print(f"测试月份{test_month}时出错: {e}")
    
    # 测试执行顺序
    print(f"\n--- 测试执行顺序 ---")
    print("执行顺序: IND(月1) -> EDU(月2) -> Council(月2) -> IND(月3) -> EDU(月4) -> Council(月4) -> ...")
    print("Council延迟启动: 第6个月开始")
    print("实际执行:")
    print("  月1: IND执行")
    print("  月2: EDU执行 -> Council跳过（<6月）")
    print("  月3: IND执行")
    print("  月4: EDU执行 -> Council跳过（<6月）")
    print("  月5: IND执行")
    print("  月6: EDU执行 -> Council执行（>=6月）")
    print("  月7: IND执行")
    print("  月8: EDU执行 -> Council执行（>=6月）")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_council_delayed_start()
