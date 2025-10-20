#!/usr/bin/env python3
"""
测试智能体执行顺序
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json

def test_agent_execution_order():
    """测试智能体执行顺序"""
    print("=== 测试智能体执行顺序 ===")
    
    # 加载配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print(f"智能体列表: {env.rl_cfg['agents']}")
    print(f"turn_based模式: {env.v4_cfg.get('enumeration', {}).get('turn_based', False)}")
    print(f"自定义执行顺序: {env.v4_cfg.get('enumeration', {}).get('custom_execution_order', {})}")
    print(f"初始月份: {env.current_month}")
    print(f"初始智能体: {env.current_agent}")
    
    # 模拟多轮执行
    print("\n--- 模拟多轮执行 ---")
    for round_num in range(1, 8):  # 模拟7轮执行
        print(f"\n第{round_num}轮:")
        print(f"  当前月份: {env.current_month}")
        print(f"  当前智能体: {env.current_agent}")
        
        # 模拟智能体执行动作
        if env.current_agent == 'IND':
            print(f"  IND智能体执行动作...")
        elif env.current_agent == 'EDU':
            print(f"  EDU智能体执行动作...")
        elif env.current_agent == 'Council':
            print(f"  Council智能体执行动作...")
        
        # 模拟智能体轮换（不实际执行step）
        try:
            # 手动触发智能体轮换逻辑
            turn_based = env.v4_cfg.get('enumeration', {}).get('turn_based', False)
            custom_order = env.v4_cfg.get('enumeration', {}).get('custom_execution_order', {})
            
            if turn_based and custom_order.get('enabled', False):
                # 自定义执行顺序模式：IND单月，EDU+Council双月
                pattern = custom_order.get('pattern', 'IND_single_EDU_Council_pair')
                
                if pattern == 'IND_single_EDU_Council_pair':
                    # 模式：IND单月，EDU+Council双月
                    if env.current_agent == 'IND':
                        # IND执行后，进入下个月，切换到EDU
                        env.current_month += 1
                        env.current_agent = 'EDU'
                        env.agent_turn = 1  # EDU的索引
                        env._council_execution_phase = 'EDU'  # 标记当前是EDU阶段
                    elif env.current_agent == 'EDU':
                        # EDU执行后，同月切换到Council
                        env.current_agent = 'Council'
                        env.agent_turn = 2  # Council的索引
                        env._council_execution_phase = 'Council'  # 标记当前是Council阶段
                    elif env.current_agent == 'Council':
                        # Council执行后，进入下个月，切换到IND
                        env.current_month += 1
                        env.current_agent = 'IND'
                        env.agent_turn = 0  # IND的索引
                        env._council_execution_phase = None  # 重置阶段标记
                else:
                    # 默认turn-based模式
                    env.current_month += 1
                    env.agent_turn = (env.agent_turn + 1) % len(env.rl_cfg['agents'])
                    env.current_agent = env.rl_cfg['agents'][env.agent_turn]
            elif turn_based:
                # 标准Turn-Based模式：每月一个agent，轮流行动
                # 先进入下个月
                env.current_month += 1
                
                # 再轮换到下一个agent
                env.agent_turn = (env.agent_turn + 1) % len(env.rl_cfg['agents'])
                env.current_agent = env.rl_cfg['agents'][env.agent_turn]
            else:
                # Multi-Agent模式：每月两个agent依次行动
                # 先轮换agent
                env.agent_turn = (env.agent_turn + 1) % len(env.rl_cfg['agents'])
                env.current_agent = env.rl_cfg['agents'][env.agent_turn]
                
                # 如果轮换回第一个智能体，进入下个月
                if env.agent_turn == 0:
                    env.current_month += 1
            
            print(f"  轮换后月份: {env.current_month}")
            print(f"  轮换后智能体: {env.current_agent}")
            print(f"  执行阶段: {getattr(env, '_council_execution_phase', 'None')}")
            
            # 检查是否完成
            done = env.current_month >= env.total_months
            if done:
                print("  模拟结束")
                break
                
        except Exception as e:
            print(f"  轮换出错: {e}")
            break
    
    print("\n--- 执行顺序分析 ---")
    custom_order = env.v4_cfg.get('enumeration', {}).get('custom_execution_order', {})
    if custom_order.get('enabled', False):
        print("根据自定义执行顺序配置:")
        print("1. IND单月执行")
        print("2. EDU和Council双月同时执行")
        print("3. 执行顺序: IND(月1) -> EDU(月2) -> Council(月2) -> IND(月3) -> EDU(月4) -> Council(月4) -> ...")
        print("4. EDU和Council在同一个月执行")
    else:
        print("根据turn_based=true的配置:")
        print("1. 每月只有一个智能体执行")
        print("2. 智能体按顺序轮流: IND -> EDU -> Council -> IND -> ...")
        print("3. 每个智能体在不同月份执行")
        print("4. Council和EDU不在同一个月执行")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_agent_execution_order()
