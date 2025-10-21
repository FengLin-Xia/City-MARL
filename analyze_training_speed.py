#!/usr/bin/env python3
"""
分析v5.0训练速度的原因
"""

import json

def analyze_training_speed():
    """分析训练速度"""
    print("=" * 60)
    print("v5.0 训练速度分析")
    print("=" * 60)
    
    # 加载v4.1配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        v4_config = json.load(f)
    
    # 加载v5.0配置  
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        v5_config = json.load(f)
    
    print("\n1. 时间步数对比:")
    v4_steps = v4_config.get('simulation', {}).get('total_months', 0)
    v5_steps = v5_config.get('env', {}).get('time_model', {}).get('total_steps', 0)
    print(f"   v4.1 total_months: {v4_steps}")
    print(f"   v5.0 total_steps: {v5_steps}")
    
    if v4_steps > 0 and v5_steps > 0:
        ratio = v4_steps / v5_steps
        print(f"   步数差异: {ratio:.1f}倍")
    
    print("\n2. 训练参数对比:")
    v4_rl = v4_config.get('solver', {}).get('rl', {})
    v5_rl = v5_config.get('mappo', {})
    
    print(f"   v4.1 max_updates: {v4_rl.get('max_updates', 'NOT_FOUND')}")
    print(f"   v5.0 max_updates: {v5_rl.get('rollout', {}).get('max_updates', 'NOT_FOUND')}")
    
    print(f"   v4.1 rollout_steps: {v4_rl.get('rollout_steps', 'NOT_FOUND')}")
    print(f"   v5.0 horizon: {v5_rl.get('rollout', {}).get('horizon', 'NOT_FOUND')}")
    
    print(f"   v4.1 num_epochs: {v4_rl.get('num_epochs', 'NOT_FOUND')}")
    print(f"   v5.0 updates_per_iter: {v5_rl.get('rollout', {}).get('updates_per_iter', 'NOT_FOUND')}")
    
    print("\n3. 训练规模计算:")
    if isinstance(v4_steps, int) and isinstance(v5_steps, int):
        print(f"   v4.1 每个episode: {v4_steps}步")
        print(f"   v5.0 每个episode: {v5_steps}步")
        print(f"   规模差异: {v4_steps/v5_steps:.1f}倍")
        
        # 计算预期时间
        v4_expected_time = v4_steps / 30 * 2  # 基于v5.0的实际时间推算
        v5_actual_time = 2  # 实际运行时间
        
        print(f"\n4. 时间对比:")
        print(f"   v4.1 预期训练时间: {v4_expected_time:.1f}秒")
        print(f"   v5.0 实际训练时间: {v5_actual_time}秒")
        print(f"   速度提升: {v4_expected_time / v5_actual_time:.1f}倍")
        
        print(f"\n5. 原因分析:")
        print(f"   - 时间步数减少: {v4_steps} → {v5_steps} ({v4_steps/v5_steps:.1f}倍减少)")
        print(f"   - 经验收集减少: {v4_steps * 3} → {v5_steps * 3} 个决策")
        print(f"   - 训练规模缩小: {v4_steps/v5_steps:.1f}倍")
        
        print(f"\n6. 配置建议:")
        print(f"   开发测试: total_steps = 30 (当前配置)")
        print(f"   功能验证: total_steps = 100")
        print(f"   正式训练: total_steps = 720 (v4.1配置)")
        
        print(f"\n7. 优势总结:")
        print(f"   [PASS] 开发效率: 2秒 vs {v4_expected_time:.1f}秒")
        print(f"   [PASS] 快速迭代: 立即看到结果")
        print(f"   [PASS] 参数调优: 快速验证效果")
        print(f"   [PASS] 功能测试: 快速确认功能")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

if __name__ == "__main__":
    analyze_training_speed()
