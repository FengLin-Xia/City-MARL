#!/usr/bin/env python3
"""
分析训练趋势
"""

import matplotlib.pyplot as plt
import numpy as np

# 你的训练数据
updates = [1, 2, 3]
policy_losses = [3.47, 3.53, 1.90]
value_losses = [10924.66, 3203.81, 5834.63]
clip_fractions = [0.9806, 0.9563, 0.9398]
kl_divergences = [-1.88, 3.52, 2.44]

print("="*80)
print("训练趋势分析")
print("="*80)

print(f"\nUpdate-by-Update分析:")
print(f"{'Update':<8} {'Policy':>10} {'Value':>12} {'Clip%':>8} {'KL':>10}")
print("-"*60)
for i in range(len(updates)):
    print(f"{updates[i]:<8} {policy_losses[i]:>10.2f} {value_losses[i]:>12.2f} {clip_fractions[i]*100:>7.1f}% {kl_divergences[i]:>10.2f}")

print(f"\n趋势判断:")
print("-"*80)

# Policy loss趋势
policy_trend = "下降" if policy_losses[-1] < policy_losses[0] else "上升"
print(f"Policy Loss: {policy_losses[0]:.2f} -> {policy_losses[-1]:.2f} ({policy_trend})")
if policy_trend == "下降":
    print(f"  [OK] 策略损失在下降，训练有进展")
else:
    print(f"  [WARN] 策略损失在上升")

# Value loss趋势
print(f"\nValue Loss: {value_losses[0]:.0f} -> {value_losses[-1]:.0f}")
print(f"  Update 1->2: {value_losses[0]:.0f} -> {value_losses[1]:.0f} (降低{(1-value_losses[1]/value_losses[0])*100:.1f}%)")
print(f"  Update 2->3: {value_losses[1]:.0f} -> {value_losses[2]:.0f} (上升{(value_losses[2]/value_losses[1]-1)*100:.1f}%)")
print(f"  [CRITICAL] Value Loss在震荡！第2次更新后反而恶化")

# Clip fraction趋势
clip_trend = clip_fractions[-1] - clip_fractions[0]
print(f"\nClip Fraction: {clip_fractions[0]*100:.1f}% -> {clip_fractions[-1]*100:.1f}%")
print(f"  变化: {clip_trend*100:+.1f}%")
if abs(clip_trend) < 0.05:
    print(f"  [WARN] Clip率几乎不变，在95%左右徘徊")
else:
    print(f"  [OK] Clip率在缓慢下降")

# KL趋势
print(f"\nKL Divergence: {kl_divergences[0]:.2f} -> {kl_divergences[-1]:.2f}")
print(f"  [OK] Update 1的KL=-1.88已修复为正数")
print(f"  [WARN] KL=2.4还是偏大（正常<0.1）")

print(f"\n" + "="*80)
print("核心问题诊断")
print("="*80)

print(f"\nValue Loss震荡原因:")
print(f"  Update 1: 10925 (极高，网络乱预测)")
print(f"  Update 2: 3204 (降低70%，网络开始学习)")
print(f"  Update 3: 5835 (反弹82%！网络被新样本打乱)")
print(f"\n  -> 这说明：")
print(f"     1. Reward方差太大，Value网络无法稳定学习")
print(f"     2. 学习率可能太高，过拟合到每批样本")
print(f"     3. 样本量太少（10 episodes），梯度噪声大")

print(f"\nClip率95%的原因:")
print(f"  不只是bug，还有:")
print(f"  - 学习率3e-4在reward方差大时太激进")
print(f"  - K_epochs=2，每批数据训练2轮，累积变化")
print(f"  - 初期策略不稳定")

print(f"\n" + "="*80)
print("解决方案")
print("="*80)

print(f"\n根本问题：Reward方差太大")
print(f"\n方案1（推荐）：降低学习率+减少K_epochs")
print(f"  lr: 3e-4 -> 1e-4")
print(f"  K_epochs: 2 -> 1")
print(f"  预期：clip率降到50-70%，value loss震荡减小")

print(f"\n方案2（更激进）：进一步降低Budget惩罚")
print(f"  debt_penalty_coef: 0.3 -> 0.1-0.2")
print(f"  initial_budgets: 10000 -> 15000")
print(f"  预期：减少负债概率，reward方差变小")

print(f"\n方案3（最保险）：Reward归一化（运行时）")
print(f"  不用固定缩放 /200")
print(f"  而是收集一批reward，动态标准化")
print(f"  mean=0, std=1")
print(f"  预期：Value loss稳定在10-50")

print(f"\n当前建议：")
print(f"  先试方案1（最简单）")
print(f"  如果value loss还是震荡，再试方案2或3")

print("="*80)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Value loss
ax1 = axes[0, 0]
ax1.plot(updates, value_losses, 'b-o', linewidth=2, markersize=8)
ax1.set_xlabel('Update')
ax1.set_ylabel('Value Loss')
ax1.set_title('Value Loss Trend (Unstable!)')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=500, color='r', linestyle='--', label='Target <500')
ax1.legend()

# Clip fraction
ax2 = axes[0, 1]
ax2.plot(updates, np.array(clip_fractions)*100, 'r-s', linewidth=2, markersize=8)
ax2.set_xlabel('Update')
ax2.set_ylabel('Clip Fraction (%)')
ax2.set_title('Clip Fraction Trend (Too High!)')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=30, color='g', linestyle='--', label='Target <30%')
ax2.legend()
ax2.set_ylim([0, 105])

# KL divergence
ax3 = axes[1, 0]
ax3.plot(updates, kl_divergences, 'g-^', linewidth=2, markersize=8)
ax3.set_xlabel('Update')
ax3.set_ylabel('KL Divergence')
ax3.set_title('KL Divergence (Fixed to Positive!)')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='r', linestyle='--', label='Zero Line')
ax3.axhline(y=0.1, color='g', linestyle='--', label='Target <0.1')
ax3.legend()

# Policy loss
ax4 = axes[1, 1]
ax4.plot(updates, policy_losses, 'm-d', linewidth=2, markersize=8)
ax4.set_xlabel('Update')
ax4.set_ylabel('Policy Loss')
ax4.set_title('Policy Loss Trend')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_trend_analysis.png', dpi=150)
print(f"\n可视化已保存: training_trend_analysis.png")
plt.show()

