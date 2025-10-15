#!/usr/bin/env python3
"""
KL散度修复调试脚本 - 按照1013-7.md的优先级执行修复
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Dict, List

def fix_kl_divergence():
    """按优先级修复KL散度问题"""
    
    print("开始KL散度修复调试...")
    
    # 1. 关熵 5 个更新
    print("\n1. 关闭熵奖励 (ent_coef = 0)")
    fix_entropy_coef()
    
    # 2. 让logits先尖一点（训练期临时）
    print("\n2. 应用温度缩放 (tau = 0.5)")
    # 这个已经在ppo_trainer.py中实现了
    
    # 3. 检查并拉高logits的起伏
    print("\n3. 重置最后一层初始化")
    reset_actor_last_layer()
    
    # 4. 增大有效batch
    print("\n4. 增大有效batch")
    increase_batch_size()
    
    # 5. 让KL自调
    print("\n5. 实现自适应KL调整")
    implement_adaptive_kl()
    
    print("\n所有修复完成！现在运行测试...")

def fix_entropy_coef():
    """关闭熵奖励"""
    config_path = "configs/city_config_v4_1.json"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 关闭熵奖励
    config['solver']['rl']['ent_coef'] = 0.0
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"  已设置 ent_coef = 0.0")

def reset_actor_last_layer():
    """重置actor网络最后一层初始化"""
    
    # 读取现有的actor网络定义
    selector_path = "solvers/v4_1/rl_selector.py"
    
    with open(selector_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换初始化代码
    old_init = """        # 按照1013-6.md建议：重初始化最后一层（提高gain到0.5）
        torch.nn.init.orthogonal_(self.network[-1].weight, gain=0.5)
        torch.nn.init.zeros_(self.network[-1].bias)"""
    
    new_init = """        # 按照1013-7.md建议：重初始化最后一层（正交+小增益）
        torch.nn.init.orthogonal_(self.network[-1].weight, gain=0.1)
        torch.nn.init.zeros_(self.network[-1].bias)"""
    
    content = content.replace(old_init, new_init)
    
    with open(selector_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  已重置actor最后一层初始化 (gain=0.1)")

def increase_batch_size():
    """增大有效batch"""
    config_path = "configs/city_config_v4_1.json"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 增大batch size
    config['solver']['rl']['mini_batch_size'] = 32  # 从10增加到32
    config['solver']['rl']['rollout_steps'] = 20    # 从10增加到20
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"  已增大batch size: mini_batch_size=32, rollout_steps=20")

def implement_adaptive_kl():
    """实现自适应KL调整"""
    
    trainer_path = "trainers/v4_1/ppo_trainer.py"
    
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 在PPOTrainer类中添加自适应KL调整方法
    adaptive_kl_method = '''
    def _adaptive_kl_adjustment(self, kl_after: float):
        """自适应KL调整（按照1013-7.md建议）"""
        target_kl = 0.02
        
        if kl_after < 0.2 * target_kl:  # 太保守
            # 增大学习率
            for agent in self.selector.actor_optimizers.keys():
                optimizer = self.selector.actor_optimizers[agent]
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 1.5
            print(f"[adaptive] KL too low ({kl_after:.4f} < {0.2 * target_kl:.4f}), increased lr")
            
        elif kl_after > 2.0 * target_kl:  # 太猛
            # 减小学习率
            for agent in self.selector.actor_optimizers.keys():
                optimizer = self.selector.actor_optimizers[agent]
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            print(f"[adaptive] KL too high ({kl_after:.4f} > {2.0 * target_kl:.4f}), decreased lr")
'''
    
    # 找到类定义的结束位置并插入方法
    if "_adaptive_kl_adjustment" not in content:
        # 在set_seed方法后插入
        insert_pos = content.find("def set_seed(self, seed: int):")
        if insert_pos != -1:
            # 找到set_seed方法的结束位置
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "def set_seed(self, seed: int):" in line:
                    # 找到下一个方法的开始
                    for j in range(i+1, len(lines)):
                        if lines[j].strip().startswith("def ") and not lines[j].strip().startswith("    "):
                            # 在set_seed方法后插入新方法
                            lines.insert(j, adaptive_kl_method)
                            break
                    break
            
            content = '\n'.join(lines)
            
            with open(trainer_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  已添加自适应KL调整方法")
    
    # 在AFTER测量部分调用自适应调整
    if "self._adaptive_kl_adjustment(kl_after)" not in content:
        # 找到KL_after打印的位置
        kl_after_line = "print(f\"[probe] Δloc_L2={dL2:.4g} | KL_before={kl_before:.3g} | KL_after={kl_after:.3g}\")"
        replacement = f"{kl_after_line}\n                    \n                    # 自适应KL调整\n                    self._adaptive_kl_adjustment(kl_after)"
        
        content = content.replace(kl_after_line, replacement)
        
        with open(trainer_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  已集成自适应KL调整到训练循环")

def create_causality_test():
    """创建因果性测试（如果KL还是0的话）"""
    
    test_code = '''#!/usr/bin/env python3
"""
因果性测试 - 检查动作是否真的影响回报
"""

import torch
import numpy as np
from enhanced_city_simulation_v4_1 import CityEnvironment
from solvers.v4_1.rl_selector import RLPolicySelector
import json

def test_action_causality():
    """测试同一状态下不同动作的回报差异"""
    print("🔍 开始因果性测试...")
    
    # 加载配置
    with open("configs/city_config_v4_1.json", 'r') as f:
        cfg = json.load(f)
    
    # 创建环境和选择器
    env = CityEnvironment(cfg)
    selector = RLPolicySelector(cfg)
    
    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 获取一个状态
    state = env.get_state()
    actions = selector.enumerate_actions(state)
    
    if len(actions) < 5:
        print(f"❌ 可用动作太少: {len(actions)}")
        return
    
    # 选择前5个动作进行测试
    test_actions = actions[:5]
    returns = []
    
    for i, action in enumerate(test_actions):
        print(f"测试动作 {i+1}: {action}")
        
        # 重置环境到相同状态
        env.reset()
        env.set_state(state)
        
        # 执行单个动作
        reward = env.step(action)
        
        # 运行几步看回报
        total_reward = reward
        for _ in range(5):  # 运行5步
            if env.is_done():
                break
            step_reward = env.step_random()  # 随机动作
            total_reward += step_reward
        
        returns.append(total_reward)
        print(f"  总回报: {total_reward:.3f}")
    
    # 计算回报差异
    min_return = min(returns)
    max_return = max(returns)
    delta_return = max_return - min_return
    
    print(f"\\n📊 因果性测试结果:")
    print(f"  回报范围: [{min_return:.3f}, {max_return:.3f}]")
    print(f"  差异 Δreturn: {delta_return:.3f}")
    
    if delta_return < 0.01:
        print("⚠️  动作对回报影响很小，可能需要shaping奖励")
        print("建议: 添加即时shaping奖励 α*(score - λ*cost + β*rent_gain)")
    else:
        print("✅ 动作对回报有明显影响，问题可能在梯度传播")
    
    return delta_return

if __name__ == "__main__":
    test_action_causality()
'''
    
    with open("test_causality.py", 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print(f"  已创建因果性测试文件: test_causality.py")

def run_test():
    """运行测试"""
    print("\n运行修复后的测试...")
    
    # 运行主程序
    os.system("python enhanced_city_simulation_v4_1.py --mode rl")
    
    print("\n观察指标:")
    print("  - KL应该从 ~1e-6 升到 1e-3~1e-2")
    print("  - clip_fraction 应该 > 0")
    print("  - entropy 应该开始缓慢下降")
    print("  - loc.std 应该从 ~3e-05 升到 ~1e-3~1e-1")

if __name__ == "__main__":
    fix_kl_divergence()
    create_causality_test()
    run_test()
