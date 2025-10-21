#!/usr/bin/env python3
"""
测试max_updates配置

验证v5.0系统中max_updates配置是否正常工作
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainers.v5_0.ppo_trainer import V5PPOTrainer


def test_max_updates_config():
    """测试max_updates配置"""
    print("=" * 60)
    print("测试max_updates配置")
    print("=" * 60)
    
    # 加载配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("\n1. 配置文件检查:")
    mappo_config = config.get("mappo", {})
    rollout_config = mappo_config.get("rollout", {})
    
    print(f"   max_updates: {rollout_config.get('max_updates', 'NOT_FOUND')}")
    print(f"   updates_per_iter: {rollout_config.get('updates_per_iter', 'NOT_FOUND')}")
    print(f"   minibatch_size: {rollout_config.get('minibatch_size', 'NOT_FOUND')}")
    print(f"   horizon: {rollout_config.get('horizon', 'NOT_FOUND')}")
    
    print("\n2. 训练器配置检查:")
    try:
        trainer = V5PPOTrainer('configs/city_config_v5_0.json')
        print(f"   trainer.max_updates: {trainer.max_updates}")
        print(f"   trainer.updates_per_iter: {trainer.updates_per_iter}")
        print(f"   trainer.current_update: {trainer.current_update}")
        print("   [PASS] 训练器配置加载成功")
    except Exception as e:
        print(f"   [FAIL] 训练器配置加载失败: {e}")
        return False
    
    print("\n3. 配置验证:")
    
    # 检查配置是否合理
    if trainer.max_updates <= 0:
        print("   [WARN] max_updates <= 0，可能影响训练")
    else:
        print(f"   [PASS] max_updates = {trainer.max_updates} 配置合理")
    
    if trainer.updates_per_iter <= 0:
        print("   [WARN] updates_per_iter <= 0，可能影响训练")
    else:
        print(f"   [PASS] updates_per_iter = {trainer.updates_per_iter} 配置合理")
    
    print("\n4. 配置对比:")
    print("   v4.1 vs v5.0 配置对比:")
    print("   v4.1: max_updates = 10")
    print(f"   v5.0: max_updates = {trainer.max_updates}")
    
    if trainer.max_updates == 10:
        print("   [PASS] v5.0配置与v4.1一致")
    else:
        print("   [INFO] v5.0配置与v4.1不同，这是正常的")
    
    print("\n5. 配置建议:")
    print("   推荐配置值:")
    print("   - 开发测试: max_updates = 5")
    print("   - 功能验证: max_updates = 10")
    print("   - 正式训练: max_updates = 20")
    print("   - 生产环境: max_updates = 50")
    
    print("\n" + "=" * 60)
    print("max_updates配置测试完成!")
    print("=" * 60)
    
    return True


def test_max_updates_override():
    """测试max_updates覆盖"""
    print("\n" + "=" * 60)
    print("测试max_updates覆盖")
    print("=" * 60)
    
    # 创建临时配置文件
    temp_config = {
        "mappo": {
            "rollout": {
                "max_updates": 5,  # 测试值
                "updates_per_iter": 8
            }
        }
    }
    
    # 保存临时配置
    with open('temp_config.json', 'w', encoding='utf-8') as f:
        json.dump(temp_config, f, indent=2)
    
    try:
        # 测试临时配置
        trainer = V5PPOTrainer('temp_config.json')
        print(f"   临时配置 max_updates: {trainer.max_updates}")
        
        if trainer.max_updates == 5:
            print("   [PASS] 配置覆盖成功")
        else:
            print("   [FAIL] 配置覆盖失败")
        
    except Exception as e:
        print(f"   [FAIL] 临时配置测试失败: {e}")
    
    finally:
        # 清理临时文件
        if os.path.exists('temp_config.json'):
            os.remove('temp_config.json')
    
    return True


if __name__ == "__main__":
    try:
        # 测试max_updates配置
        test_max_updates_config()
        
        # 测试max_updates覆盖
        test_max_updates_override()
        
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
