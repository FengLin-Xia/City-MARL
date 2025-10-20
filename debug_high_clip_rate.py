#!/usr/bin/env python3
"""
调试高clip率问题
"""

import json
import torch
import numpy as np

print("="*80)
print("调试高Clip率问题")
print("="*80)

# 加载配置
with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

rl_cfg = cfg['solver']['rl']

print(f"\n当前训练参数:")
print(f"  clip_eps: {rl_cfg['clip_eps']}")
print(f"  lr: {rl_cfg['lr']}")
print(f"  rollout_steps: {rl_cfg['rollout_steps']}")
print(f"  K_epochs: {rl_cfg['K_epochs']}")

print(f"\n问题分析:")
print(f"  当前clip率: 95.63%")
print(f"  正常范围: 10-30%")

print(f"\n95%的clip率意味着:")
print(f"  ratio = new_prob / old_prob")
print(f"  95%的样本的ratio在[0.8, 1.2]之外")

print(f"\n可能的原因:")
print(f"  [1] 学习率还是太高（当前3e-4）")
print(f"  [2] K_epochs=2，每次训练2轮，累积变化大")
print(f"  [3] Reward方差还是太大")
print(f"  [4] 样本量太少（rollout_steps=10）")
print(f"  [5] 初始化问题（网络参数随机初始化）")

print(f"\n测试：模拟ratio分布")
print("-"*80)

# 模拟一些概率变化
old_probs = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
new_probs_normal = np.array([0.11, 0.19, 0.16, 0.29, 0.25])  # 正常变化
new_probs_large = np.array([0.05, 0.35, 0.10, 0.40, 0.10])   # 大变化

ratio_normal = new_probs_normal / old_probs
ratio_large = new_probs_large / old_probs

clip_eps = rl_cfg['clip_eps']
clip_normal = np.mean((np.abs(ratio_normal - 1.0) > clip_eps))
clip_large = np.mean((np.abs(ratio_large - 1.0) > clip_eps))

print(f"  正常策略更新: clip率={clip_normal*100:.1f}%")
print(f"  激进策略更新: clip率={clip_large*100:.1f}%")

print(f"\n  你的clip率: 95.63%")
print(f"  -> 说明策略变化非常激进，接近'大变化'水平")

print(f"\n解决方案分析:")
print("-"*80)

solutions = [
    ("降低学习率", "3e-4 -> 1e-4", "减小每次更新的步长", "★★★★★"),
    ("减少K_epochs", "2 -> 1", "减少每批数据的训练轮数", "★★★★"),
    ("增加rollout_steps", "10 -> 20", "收集更多样本，平滑梯度", "★★★"),
    ("增加mini_batch_size", "10 -> 20", "增大batch size，稳定梯度", "★★★"),
    ("降低clip_eps", "0.2 -> 0.1", "更严格的裁剪（治标不治本）", "★★"),
]

for i, (method, change, reason, priority) in enumerate(solutions, 1):
    print(f"  [{i}] {method:20s} {change:20s} {reason:30s} {priority}")

print(f"\n推荐组合方案:")
print(f"  方案A（激进）: lr=1e-4 + K_epochs=1")
print(f"             预期clip率: 30-50%")
print(f"  ")
print(f"  方案B（稳健）: lr=1e-4 + rollout_steps=20 + K_epochs=1")
print(f"             预期clip率: 20-30%")
print(f"  ")
print(f"  方案C（保守）: lr=5e-5 + rollout_steps=20")
print(f"             预期clip率: 10-20%")

print(f"\n当前状态:")
print(f"  num_actions bug: 已修复 ✓")
print(f"  Value loss: 已优化 ✓")
print(f"  KL散度: 已修正 ✓")
print(f"  Clip率: 还需要调整学习率/K_epochs")

print("\n" + "="*80)
print("建议:")
print("="*80)
print("\n先尝试方案A (最简单):")
print('  修改配置: "lr": 1e-4, "K_epochs": 1')
print("  然后重新训练")
print("\n如果clip率还是>50%:")
print("  再用方案B或C")
print("="*80)




