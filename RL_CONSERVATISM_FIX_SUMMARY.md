# RL保守行为修复总结

## 问题描述
v4.1的RL模型在10个episode的评估中**100%建造S型建筑**，完全不建M/L型，表现极度保守。

## 根本原因分析

### 1. **Proximity Reward淹没Size差异**
- `proximity_reward=1500`太大，产生的bonus约1000
- S/M/L的base revenue差异只有100-150
- **Proximity占reward的70-85%**，Size的内在价值被淹没

### 2. **ROI严重失衡**
修复前的ROI：
- S型: **11.62** (压倒性优势)
- M型: 4.70
- L型: 2.96

**S型的ROI是L型的4倍！** RL自然学到"只建S型"的保守策略。

### 3. **Size Bonus不足**
之前的size_bonus (M:+200, L:+500)无法补偿高成本和低ROI。

### 4. **Debt Penalty强化保守**
`debt_penalty_coef=0.3`较高，惩罚了高成本策略，进一步强化了保守行为。

## 修复方案

### 修改1: 降低Proximity Reward
```json
"proximity_reward": 1500.0 → 400.0  // 降低75%
```
**理由**：让Size的内在价值差异得以体现，Proximity变为bonus而非主导因素。

### 修改2: 科学计算Size Bonus
```json
"size_bonus": {"S": 0, "M": 200, "L": 500} → {"S": 0, "M": 1000, "L": 2750}
```
**计算依据**：
- 目标：M型ROI=5.0，L型ROI=5.5，均略高于S型(4.3)
- M型bonus: 1500 (目标总reward) - 477 (base+proximity) = 1023 ≈ 1000
- L型bonus: 3300 (目标总reward) - 543 (base+proximity) = 2757 ≈ 2750

### 修改3: 降低Debt Penalty
```json
"debt_penalty_coef": 0.3 → 0.1  // 降低70%
```
**理由**：鼓励探索高成本策略，减少过度规避风险的行为。

## 修复效果

### ROI平衡
| Size | 修复前ROI | 修复后ROI | 改善 |
|------|----------|----------|------|
| S型  | 11.62    | 4.29     | -63% |
| M型  | 4.70     | 4.92     | +5%  |
| L型  | 2.96     | 5.49     | +85% |

**L型ROI从最低变为最高！**

### 净收益对比
| Size | 修复前Net | 修复后Net | 提升 |
|------|----------|----------|------|
| S型  | 1062     | 329      | -69% |
| M型  | 1110     | 1177     | +6%  |
| L型  | 1176     | 2693     | +129% |

**L型净收益是S型的8.2倍！**

### Reward组成
修复后（proximity_reward=400时）：
```
Size    Base    Proximity    SizeBonus    Total    ROI
S       162     267          0            429      4.29
M       210     267          1000         1477     4.92
L       276     267          2750         3293     5.49
```

- Proximity占比降至18-62%（原85%），不再主导
- Size bonus充分补偿了高成本
- L型获得最高ROI和净收益

## 预期RL行为

修复后，RL应该学到：

1. **M/L型更有价值**
   - M型ROI略高于S型
   - L型ROI最高，净收益最大
   - 不再过度偏好S型

2. **根据槽位level智能选择**
   - Level 3槽位 → S型
   - Level 4槽位 → M型（ROI=4.92）
   - Level 5槽位 → L型（ROI=5.49，最优）

3. **预期Size分布**
   - S型: 30-40%（Level 3槽位占88%）
   - M型: 5-10%（Level 4槽位稀缺，仅7个）
   - L型: 50-60%（Level 5槽位占8.4%，但ROI最高）

## 文件修改

1. `configs/city_config_v4_1.json`:
   - `proximity_reward`: 1500 → 400
   - `size_bonus`: {S:0, M:200, L:500} → {S:0, M:1000, L:2750}
   - `debt_penalty_coef`: 0.3 → 0.1

2. `logic/v4_enumeration.py`:
   - 添加size_bonus应用逻辑（第544-548行）

3. 移除临时调试代码，恢复`building_level`约束

## 下一步

1. **重新训练模型**：使用新的reward配置
2. **监控指标**：
   - Size分布（期望：M/L型占比50%+）
   - Entropy（保持>0.5以维持exploration）
   - Value Loss（期望稳定在合理范围）
3. **评估效果**：10 episodes后检查Size分布是否多样化

## 技术要点

### 为什么Proximity Reward会淹没Size差异？

当某个reward组件的数值远大于其他组件时，RL优化器会主要关注该组件，忽略其他细微差异。

**修复前**：
- Proximity: 1000 (占85%)
- Size差异: 100-150 (占10-15%)
- 结果：RL只"看到"proximity，Size差异被噪声淹没

**修复后**：
- Proximity: 267 (占18-62%)
- Size差异: 0-2750 (占0-83%)
- 结果：RL能清晰识别Size的价值差异

### 为什么用ROI而不是绝对Net？

RL在序列决策中会考虑"性价比"（reward/cost），而不仅是绝对收益。高ROI意味着：
1. 单位成本产生更多reward
2. 资源利用效率更高
3. 风险收益比更优

因此，平衡ROI是鼓励RL探索所有Size的关键。

---

**修复完成时间**: 2025-10-10  
**修复版本**: v4.1  
**状态**: 待训练验证



