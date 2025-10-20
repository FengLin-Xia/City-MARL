#!/usr/bin/env python3
"""
测试对岸检测功能
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.v4_1.city_env import CityEnvironment
from solvers.v4_1.rl_selector import RLPolicySelector
from logic.v4_enumeration import Action

def test_other_side_detection():
    """测试对岸检测功能"""
    print("=== 测试对岸检测功能 ===")
    
    # 加载配置
    config_path = "configs/city_config_v4_1.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建环境
    env = CityEnvironment(config)
    obs = env.reset()
    print(f"环境初始化完成")
    
    # 创建RL选择器
    selector = RLPolicySelector(config, slots=env.slots)
    print(f"RL选择器初始化完成")
    print(f"self.cfg是否为None: {selector.cfg is None}")
    print(f"self.rl_cfg是否为None: {selector.rl_cfg is None}")
    
    # 获取EDU动作池
    print("\n=== 获取EDU动作池 ===")
    actions, _, _ = env.get_action_pool('EDU')
    print(f"EDU动作池大小: {len(actions)}")
    
    # 找到A/B/C动作
    abc_actions = [a for a in actions if a.size in ['A', 'B', 'C']]
    print(f"A/B/C动作数量: {len(abc_actions)}")
    
    if abc_actions:
        # 测试前5个A/B/C动作的对岸检测
        print("\n=== 测试对岸检测 ===")
        for i, action in enumerate(abc_actions[:5]):
            print(f"\n测试动作 {i+1}: {action.size}型, 槽位={action.footprint_slots[0] if action.footprint_slots else 'N/A'}")
            try:
                is_other_side = selector._is_other_side_action(action)
                print(f"对岸检测结果: {is_other_side}")
            except Exception as e:
                print(f"对岸检测异常: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("没有找到A/B/C动作")

if __name__ == "__main__":
    test_other_side_detection()
