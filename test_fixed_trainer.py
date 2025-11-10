#!/usr/bin/env python3
"""
测试修复后的训练器

验证30步应该导出30个月的数据
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainers.v5_0.ppo_trainer_fixed import V5PPOTrainerFixed


def test_fixed_trainer():
    """测试修复后的训练器"""
    print("=" * 80)
    print("测试修复后的训练器")
    print("=" * 80)
    
    try:
        # 创建训练器
        trainer = V5PPOTrainerFixed('configs/city_config_v5_0.json')
        print("   [PASS] 训练器初始化成功")
        
        # 收集30步经验
        print(f"\n   收集30步经验...")
        experiences = trainer.collect_experience(30)
        print(f"   - 收集到 {len(experiences)} 条经验")
        
        # 分析经验数据
        print(f"\n   分析经验数据:")
        
        # 统计智能体
        agents = set()
        for exp in experiences:
            agents.add(exp['agent'])
        print(f"   - 智能体: {sorted(agents)}")
        
        # 统计月份
        months = set()
        for exp in experiences:
            if 'step_log' in exp and exp['step_log']:
                months.add(exp['step_log'].t)
        print(f"   - 月份分布: {sorted(months)}")
        print(f"   - 月份数量: {len(months)}")
        
        # 统计奖励
        rewards = [exp['reward'] for exp in experiences]
        print(f"   - 奖励范围: {min(rewards):.2f} ~ {max(rewards):.2f}")
        print(f"   - 平均奖励: {sum(rewards)/len(rewards):.2f}")
        
        # 验证问题
        print(f"\n   验证问题:")
        if len(months) == 30:
            print(f"   [PASS] 月份数量正确: {len(months)}个月")
        else:
            print(f"   [FAIL] 月份数量错误: {len(months)}个月，期望30个月")
        
        if len(experiences) >= 30:
            print(f"   [PASS] 经验数量充足: {len(experiences)}条")
        else:
            print(f"   [FAIL] 经验数量不足: {len(experiences)}条，期望至少30条")
        
        # 检查缺失的月份
        expected_months = set(range(30))
        actual_months = set(months)
        missing_months = expected_months - actual_months
        
        if missing_months:
            print(f"   [FAIL] 缺失月份: {sorted(missing_months)}")
        else:
            print(f"   [PASS] 所有月份都存在")
        
        # 测试训练步骤
        print(f"\n   测试训练步骤:")
        training_stats = trainer.train_step(experiences)
        print(f"   - 训练损失: {training_stats}")
        
        # 获取训练统计
        stats = trainer.get_training_stats()
        print(f"   - 当前更新: {stats['current_update']}/{stats['max_updates']}")
        
    except Exception as e:
        print(f"   [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_fixed_trainer()

