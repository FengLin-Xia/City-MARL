#!/usr/bin/env python3
"""
调试为什么只选S型
"""

import json
from collections import Counter

print("="*80)
print("为什么100%选S型？深度调试")
print("="*80)

# 读取历史
with open('models/v4_1_rl/slot_selection_history.json', 'r') as f:
    data = json.load(f)

episode = data['episodes'][0]
steps = episode['steps']

# 分析IND的选择
print(f"\nIND动作分析（前5个月）:")
print(f"{'Month':<6} {'候选槽位':<10} {'可用动作':<10} {'已选类型':<15} {'动作得分范围':<20}")
print("-"*80)

for step in steps[:10]:  # 前10步（5个月IND+EDU）
    if step['agent'] != 'IND':
        continue
    
    month = step['month']
    cand_count = step.get('candidate_slots_count', 0)
    action_count = step.get('available_actions_count', 0)
    
    # 选择的类型
    selected_sizes = [a['size'] for a in step.get('detailed_actions', [])]
    sizes_str = ','.join(selected_sizes) if selected_sizes else 'None'
    
    # 动作得分
    scores = step.get('action_scores', [])
    if scores:
        score_range = f"[{min(scores):.3f}, {max(scores):.3f}]"
    else:
        score_range = "N/A"
    
    print(f"{month:<6} {cand_count:<10} {action_count:<10} {sizes_str:<15} {score_range:<20}")

print(f"\n" + "="*80)
print("关键问题：候选动作中有没有M/L型？")
print("="*80)

# 这个信息在slot_selection_history中没有保存！
# 需要重新运行一次，记录每个月的候选动作类型分布

print(f"\n当前slot_selection_history中没有保存：")
print(f"  - 候选动作的类型分布（S/M/L各多少）")
print(f"  - 每种类型的得分")
print(f"  ")
print(f"  只能推测：")

# 检查
print(f"\n推测1：建筑等级限制")
print(f"  槽位level分布:")
print(f"    Level 3: 440个 (88%) - 只能建S")
print(f"    Level 4: 38个 (8%) - 可建S/M")
print(f"    Level 5: 20个 (4%) - 可建S/M/L")
print(f"  ")
print(f"  如果候选槽位中90%是level=3:")
print(f"    候选动作中90%是S型")
print(f"    只有10%是M/L型")
print(f"  ")
print(f"  RL选择：")
print(f"    看到10个动作，9个S，1个M")
print(f"    如果S型得分更高（cost低）→ 选S")

print(f"\n推测2：ActionScorer的评分偏向")
print(f"  ")
print(f"  Score公式:")
print(f"    score = w_r × reward_norm + w_p × prestige_norm - w_c × cost_norm")
print(f"  ")
print(f"  IND权重: w_r=0.6, w_p=0.2, w_c=0.2")
print(f"  ")
print(f"  S型: cost=1000, reward=50")
print(f"  L型: cost=2400, reward=520")
print(f"  ")
print(f"  归一化后（假设在同一个池中）:")
print(f"    cost_norm: S=0.0, L=1.0")
print(f"    reward_norm: S=0.0, L=1.0")
print(f"  ")
print(f"  Score:")
print(f"    S型: 0.6×0.0 + 0.2×? - 0.2×0.0 = 0 + prestige")
print(f"    L型: 0.6×1.0 + 0.2×? - 0.2×1.0 = 0.4 + prestige")
print(f"  ")
print(f"  如果proximity_reward足够大，L型应该得分更高")
print(f"  但如果proximity影响被归一化抹平...")

print(f"\n推测3：RL学到的策略")
print(f"  ")
print(f"  即使L型得分高，RL可能学到：")
print(f"    \"建L型 → 短期负债 → reward有波动 → 不确定性\"")
print(f"    \"建S型 → 不负债 → reward稳定 → 确定性\"")
print(f"  ")
print(f"  在训练不稳定时（value_loss震荡）:")
print(f"    RL倾向于选择确定性高的策略（S型）")

print(f"\n" + "="*80)
print("验证方法")
print("="*80)

print(f"\n需要添加诊断信息，记录:")
print(f"  1. 每个月候选动作中S/M/L的数量")
print(f"  2. 每种类型的平均得分")
print(f"  3. RL为什么选S不选L（概率分布）")
print(f"  ")
print(f"  然后才能确定是：")
print(f"    - 候选池中就没有M/L（building_level限制）")
print(f"    - 还是候选池有M/L，但RL不选（策略保守）")

print("="*80)




