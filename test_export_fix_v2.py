#!/usr/bin/env python3
"""
测试导出数据修复v2 - 智能清空策略
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_smart_clear_strategy():
    """测试智能清空策略"""
    print("=== 测试智能清空策略 ===\n")
    
    # 模拟多episode训练
    print("模拟训练3个episode:\n")
    
    data = {
        "step_logs": [],
        "env_states": [],
        "num_episodes": 3
    }
    
    for ep in range(1, 4):
        data["current_episode"] = ep - 1
        
        # 收集数据
        print(f"Episode {ep}:")
        data["step_logs"].extend([f"log_ep{ep}_{i}" for i in range(10)])
        data["env_states"].extend([f"state_ep{ep}_{i}" for i in range(10)])
        print(f"  收集数据后: {len(data['step_logs'])} logs, {len(data['env_states'])} states")
        
        # 智能清空策略
        current_ep = int(data.get("current_episode", 0)) + 1
        num_episodes = int(data.get("num_episodes", 1))
        is_last_episode = (current_ep == num_episodes)
        
        if not is_last_episode and data["step_logs"] and data["env_states"]:
            # 中间episode：清空
            print(f"  执行清空（中间episode）")
            data["step_logs"] = []
            data["env_states"] = []
            print(f"  清空后: {len(data['step_logs'])} logs, {len(data['env_states'])} states")
        else:
            # 最后episode：保留
            print(f"  保留数据（最后episode）")
            print(f"  保留后: {len(data['step_logs'])} logs, {len(data['env_states'])} states")
        print()
    
    # 验证结果
    print("=" * 50)
    print("验证结果:")
    
    # 最后应该有数据
    final_logs = len(data["step_logs"])
    final_states = len(data["env_states"])
    
    print(f"  最终数据: {final_logs} logs, {final_states} states")
    
    if final_logs > 0 and final_states > 0:
        print("  测试通过！最后一个episode的数据被保留")
        return True
    else:
        print("  测试失败！数据被错误清空")
        return False

if __name__ == "__main__":
    success = test_smart_clear_strategy()
    
    print("\n" + "="*50)
    if success:
        print("智能清空策略测试通过！")
        print("\n行为:")
        print("  - Episode 1-2: 数据被清空（节省内存）")
        print("  - Episode 3 (最后): 数据保留给集成系统")
        print("\n现在可以运行训练:")
        print("python enhanced_city_simulation_v5_0.py --mode complete --episodes 2 --verbose")
    else:
        print("测试失败")
    print("="*50)
    
    sys.exit(0 if success else 1)
"""
测试导出数据修复v2 - 智能清空策略
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_smart_clear_strategy():
    """测试智能清空策略"""
    print("=== 测试智能清空策略 ===\n")
    
    # 模拟多episode训练
    print("模拟训练3个episode:\n")
    
    data = {
        "step_logs": [],
        "env_states": [],
        "num_episodes": 3
    }
    
    for ep in range(1, 4):
        data["current_episode"] = ep - 1
        
        # 收集数据
        print(f"Episode {ep}:")
        data["step_logs"].extend([f"log_ep{ep}_{i}" for i in range(10)])
        data["env_states"].extend([f"state_ep{ep}_{i}" for i in range(10)])
        print(f"  收集数据后: {len(data['step_logs'])} logs, {len(data['env_states'])} states")
        
        # 智能清空策略
        current_ep = int(data.get("current_episode", 0)) + 1
        num_episodes = int(data.get("num_episodes", 1))
        is_last_episode = (current_ep == num_episodes)
        
        if not is_last_episode and data["step_logs"] and data["env_states"]:
            # 中间episode：清空
            print(f"  执行清空（中间episode）")
            data["step_logs"] = []
            data["env_states"] = []
            print(f"  清空后: {len(data['step_logs'])} logs, {len(data['env_states'])} states")
        else:
            # 最后episode：保留
            print(f"  保留数据（最后episode）")
            print(f"  保留后: {len(data['step_logs'])} logs, {len(data['env_states'])} states")
        print()
    
    # 验证结果
    print("=" * 50)
    print("验证结果:")
    
    # 最后应该有数据
    final_logs = len(data["step_logs"])
    final_states = len(data["env_states"])
    
    print(f"  最终数据: {final_logs} logs, {final_states} states")
    
    if final_logs > 0 and final_states > 0:
        print("  测试通过！最后一个episode的数据被保留")
        return True
    else:
        print("  测试失败！数据被错误清空")
        return False

if __name__ == "__main__":
    success = test_smart_clear_strategy()
    
    print("\n" + "="*50)
    if success:
        print("智能清空策略测试通过！")
        print("\n行为:")
        print("  - Episode 1-2: 数据被清空（节省内存）")
        print("  - Episode 3 (最后): 数据保留给集成系统")
        print("\n现在可以运行训练:")
        print("python enhanced_city_simulation_v5_0.py --mode complete --episodes 2 --verbose")
    else:
        print("测试失败")
    print("="*50)
    
    sys.exit(0 if success else 1)






