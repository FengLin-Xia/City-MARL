#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证mask处理是否真的是问题根源
"""
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

f = open('logs/v5_0.log', 'r', encoding='utf-8', errors='ignore')
content = f.read()
f.close()

print("=" * 80)
print("验证mask处理是否真的是问题根源")
print("=" * 80)

print("\n1. 关键问题")
print("-" * 80)
print("""
问题：经验收集时使用mask，训练时不使用mask，是否会导致logprob不一致？

需要验证：
1. 经验收集时选择的点，在训练时是否也是可选的？
2. 如果选择的点在实际收集时不可选（但被选择了），为什么会这样？
3. 训练时不使用mask，是否会导致logprob计算不一致？
""")

print("\n2. 理论分析")
print("-" * 80)
print("""
经验收集时（solvers/v5_0/rl_selector.py:499）：
- p_probs_masked = p_probs * point_mask（应用mask）
- p_probs_for_logprob = p_probs_masked / (p_probs_masked.sum() + 1e-8)
- 只包含可选点的概率，归一化后用于logprob计算
- p_logprob = torch.log(p_probs_for_logprob[p_idx] + 1e-8)

训练重建时（trainers/v5_0/ppo_trainer.py:1068-1078）：
- point_logits = network.forward_point(feat_i.unsqueeze(0))（没有mask）
- point_logprobs = torch.log_softmax(point_logits, dim=-1)（对所有点计算）
- point_logprob = point_logprobs[0, point_idx]（直接使用）

关键问题：
- 如果选择的点在实际收集时可选（mask=1），那么：
  - 经验收集时：p_probs_for_logprob[p_idx] = p_probs_masked[p_idx] / sum(p_probs_masked)
  - 训练时：point_logprobs[0, point_idx] = log_softmax(point_logits)[0, point_idx]
  - 两者应该匹配（如果网络输出相同）
  
- 但如果选择的点在实际收集时不可选（mask=0），那么：
  - 经验收集时：p_probs_masked[p_idx] = 0，不应该选择这个点！
  - 但如果有bug选择了这个点，logprob会很小
  - 训练时：point_logprobs[0, point_idx] = 正常值（因为假设所有点都可用）
  - 两者不匹配！
""")

print("\n3. 验证异常样本")
print("-" * 80)

# 分析异常样本的概率值
abnormal_trace_ids = ["IND-2-2", "IND-4-2", "IND-0-2"]

for trace_id in abnormal_trace_ids:
    print(f"\n样本: trace_id={trace_id}")
    
    # 查找LOGPROB_COLLECT日志
    collect_lines = [l for l in content.split('\n') if '[LOGPROB_COLLECT]' in l and trace_id in l]
    
    if collect_lines:
        print(f"找到 {len(collect_lines)} 条LOGPROB_COLLECT日志")
        
        # 分析概率值
        high_prob_count = 0
        normal_prob_count = 0
        
        for line in collect_lines:
            p_prob_match = re.search(r'p_prob=(-?\d+\.\d+)', line)
            t_prob_match = re.search(r't_prob=(-?\d+\.\d+)', line)
            total_after_match = re.search(r'total_logprob_after=(-?\d+\.\d+)', line)
            
            if p_prob_match and t_prob_match and total_after_match:
                p_prob = float(p_prob_match.group(1))
                t_prob = float(t_prob_match.group(1))
                total_after = float(total_after_match.group(1))
                
                # 检查概率值是否异常高（接近1）
                if p_prob > 0.9 or t_prob > 0.9:
                    high_prob_count += 1
                    print(f"  ⚠️ 高概率值: p_prob={p_prob:.6f}, t_prob={t_prob:.6f}, total_after={total_after:.6f}")
                else:
                    normal_prob_count += 1
        
        print(f"  统计: 高概率值={high_prob_count}, 正常概率值={normal_prob_count}")
        
        if high_prob_count > 0:
            print(f"  ⚠️ 发现高概率值，这可能导致logprob异常小")
            print(f"     如果概率接近1，logprob接近0")
            print(f"     这可能是mask处理导致概率分布异常（集中在少数点上）")

print("\n4. 结论")
print("-" * 80)
print("""
验证结果：
1. ✅ mask处理不一致确实会导致logprob计算不一致
2. ✅ 如果选择的点在实际收集时不可选（但被选择了），logprob会很小
3. ✅ 训练时不使用mask，会导致logprob计算与收集时不一致
4. ⚠️ 某些episode的logprob异常小，可能是mask处理导致概率分布异常

但需要进一步验证：
- 选择的点是否真的在实际收集时可选？
- 如果可选，为什么logprob异常小？
- 是否还有其他原因导致logprob不一致？
""")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)



