#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证mask对logprob计算的影响（即使选择的点可选）
"""
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("验证mask对logprob计算的影响（即使选择的点可选）")
print("=" * 80)

print("\n1. 关键问题")
print("-" * 80)
print("""
问题：即使选择的点在实际收集时可选（mask=1），mask处理是否仍然会导致logprob不一致？

关键点：
- 经验收集时：mask后归一化，只包含可选点的概率
- 训练时：不mask，对所有点归一化
- 即使选择的点可选，归一化分母不同，logprob也会不同！
""")

print("\n2. 数学验证")
print("-" * 80)

# 模拟场景
print("\n场景：10个点，选择点5（可选），但只有前3个点可选")
print("-" * 80)

# 模拟网络输出（10个点的logits）
np.random.seed(42)
point_logits = np.random.randn(10) * 2  # 10个点的logits
print(f"\n点logits（10个点）: {point_logits[:5]}... (前5个)")

# 计算softmax（所有点）
point_probs_all = np.exp(point_logits) / np.sum(np.exp(point_logits))
print(f"\n所有点的概率: {point_probs_all[:5]}... (前5个)")
print(f"选择点5的概率（无mask）: {point_probs_all[5]:.6f}")
print(f"选择点5的logprob（无mask）: {np.log(point_probs_all[5] + 1e-8):.6f}")

# 应用mask（只有前3个点可选）
point_mask = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # 前3个点可选
point_probs_masked = point_probs_all * point_mask
point_probs_masked_normalized = point_probs_masked / (point_probs_masked.sum() + 1e-8)
print(f"\n应用mask后的概率（前3个点）: {point_probs_masked_normalized[:3]}")
print(f"选择点5的概率（有mask，但点5不可选）: {point_probs_masked_normalized[5]:.6f}")
print(f"选择点5的logprob（有mask，但点5不可选）: {np.log(point_probs_masked_normalized[5] + 1e-8):.6f}")

print("\n结论：如果选择的点不可选（mask=0），logprob会很小（接近-inf）")

# 重新模拟：选择点2（可选）
print("\n场景：10个点，选择点2（可选），只有前3个点可选")
print("-" * 80)

print(f"\n选择点2的概率（无mask）: {point_probs_all[2]:.6f}")
print(f"选择点2的logprob（无mask）: {np.log(point_probs_all[2] + 1e-8):.6f}")

print(f"\n选择点2的概率（有mask，归一化后）: {point_probs_masked_normalized[2]:.6f}")
print(f"选择点2的logprob（有mask，归一化后）: {np.log(point_probs_masked_normalized[2] + 1e-8):.6f}")

diff = np.log(point_probs_all[2] + 1e-8) - np.log(point_probs_masked_normalized[2] + 1e-8)
print(f"\n差异: {diff:.6f}")

print("\n结论：即使选择的点可选，mask后归一化也会导致logprob不同！")
print("     差异大小取决于：可选点的数量、概率分布")

print("\n3. 实际影响")
print("-" * 80)
print("""
实际影响：
1. 如果可选点很少（例如只有3个点可选），归一化后每个点的概率会变大
2. 如果可选点很多（例如10个点可选），归一化后每个点的概率会变小
3. 训练时不使用mask，假设所有点都可用，归一化分母不同
4. 这会导致logprob计算不一致，ratio异常

关键发现：
- 即使选择的点在实际收集时可选，mask后归一化也会导致logprob不同
- 训练时不使用mask，归一化分母不同，logprob也会不同
- 这确实是mask处理不一致导致的问题！
""")

print("\n4. 验证结论")
print("-" * 80)
print("""
验证结果：
1. ✅ mask处理不一致确实会导致logprob计算不一致
2. ✅ 即使选择的点可选，mask后归一化也会导致logprob不同
3. ✅ 训练时不使用mask，归一化分母不同，logprob也会不同
4. ✅ 这确实是mask处理不一致导致的问题！

修复方案：
- 在训练时也应用mask（与经验收集时一致）
- 或者统一处理方式（都不使用mask）
""")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)



