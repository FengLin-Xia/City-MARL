# IND临近性优化方案

**日期：** 2025-10-09  
**问题：** v4.1中IND侧建筑分布不够集中，临近性效果不理想

---

## 🔍 问题诊断

### 诊断工具
运行 `diagnose_ind_proximity_v4_1.py` 发现以下问题：

### 问题清单

| 问题 | 当前值 | 影响 |
|-----|--------|------|
| **候选槽位太少** | 平均5.2个（最少2个） | IND可选范围极小，难以扩展 |
| **初始半径太小** | R0=5, dR=1.5 | 早期候选范围不足 |
| **max_distance小** | 10.0 | 新建筑必须距已有建筑<10px，过于严格 |
| **proximity_reward弱** | 50.0 | 临近奖励不足，RL不够重视临近性 |
| **动作得分差异小** | std=0.054 | RL难以区分好坏动作 |
| **apply_after_month早** | 1 | 第1个月就启用，限制初始扩展 |

---

## ✅ 优化方案

### 1. 扩大候选范围（核心优化）

**修改：**
```json
"list": [
  {"id": "hub1", "x": 122, "y": 80, "R0": 8, "dR": 2.0, "weight": 0.5},  // IND
  {"id": "hub2", "x": 112, "y": 121, "R0": 8, "dR": 2.0, "weight": 0.5}  // EDU
]
```

**效果：**
- R0: 5 → 8 （初始半径+60%）
- dR: 1.5 → 2.0 （增长速度+33%）
- 预计候选槽位：5.2 → 15-20个

**月度候选范围预测：**
| 月份 | 旧R值 (R0=5, dR=1.5) | 新R值 (R0=8, dR=2.0) | 增幅 |
|-----|---------------------|---------------------|------|
| 0 | 5.0 | 8.0 | +60% |
| 3 | 9.5 | 14.0 | +47% |
| 6 | 14.0 | 20.0 | +43% |
| 12 | 23.0 | 32.0 | +39% |

### 2. 放宽临近性约束

**修改：**
```json
"proximity_constraint": {
  "enabled": true,
  "max_distance": 18.0,        // 10.0 → 18.0 (+80%)
  "apply_after_month": 2,      // 1 → 2 (延后启用)
  "min_candidates": 5
}
```

**效果：**
- 允许在距离已有建筑18px内选择（原来10px）
- 前2个月不启用约束，让IND自由建立初始集群
- 减少过度限制导致的候选槽位被过滤

### 3. 增强临近性奖励

**修改：**
```json
"evaluation": {
  "proximity_threshold": 15.0,        // 10.0 → 15.0
  "proximity_reward": 150.0,          // 50.0 → 150.0 (3倍)
  "distance_penalty_coef": 1.5        // 2.0 → 1.5 (减轻惩罚)
}
```

**效果：**
- 临近奖励增强3倍（50 → 150），RL更重视临近性
- 临近阈值放宽（10 → 15），更多动作能获得奖励
- 距离惩罚减轻（2.0 → 1.5），不会过度惩罚稍远的建筑

---

## 📊 预期效果

### 候选槽位数量
- **Month 0-2:** 2个 → 8-12个 （**+300%**）
- **Month 3-5:** 6-10个 → 20-30个 （**+200%**）
- **Month 6+:** 持续增长，保持充足候选

### 动作得分差异
- 临近奖励增强3倍，动作得分标准差预计从0.054 → 0.15+
- RL能够更清晰地区分"好位置"vs"坏位置"

### 建筑分布
- 前2个月：自由选择，建立2-3个初始集群点
- Month 2后：开始临近性约束，但范围更宽松（18px）
- Month 6+：形成连续的工业区，但不会过度密集

---

## 🧪 验证方法

### 1. 重新训练
```bash
python enhanced_city_simulation_v4_1.py --mode rl
```

### 2. 运行诊断
```bash
python diagnose_ind_proximity_v4_1.py
```

**期望结果：**
- 平均候选槽位数: 5.2 → **15+**
- 平均可用动作数: 7.9 → **20+**
- 动作得分标准差: 0.054 → **0.12+**

### 3. 可视化检查
查看生成的建筑分布图，检查IND建筑是否：
- ✅ 形成连续的工业区
- ✅ 围绕Hub1 (122, 80)分布
- ✅ 不会跳跃式发展

---

## 🎯 调优策略

### 如果还是太分散
1. 进一步增加 `proximity_reward`: 150 → 200-300
2. 减小 `max_distance`: 18 → 15
3. 提前启用约束: `apply_after_month`: 2 → 1

### 如果过度集中
1. 减小 `proximity_reward`: 150 → 100
2. 增大 `max_distance`: 18 → 20-25
3. 延后启用约束: `apply_after_month`: 2 → 3

### 如果候选不足
1. 继续增加R0: 8 → 10
2. 增加dR: 2.0 → 2.5
3. 降低 `tol`: 0.5 → 0.3

---

## 📐 参数对比表

| 参数 | 旧值 | 新值 | 变化 | 原因 |
|-----|------|------|------|------|
| **hub1.R0** | 5 | 8 | +60% | 扩大初始候选范围 |
| **hub1.dR** | 1.5 | 2.0 | +33% | 加快候选范围增长 |
| **max_distance** | 10.0 | 18.0 | +80% | 放宽临近性过滤 |
| **apply_after_month** | 1 | 2 | +1 | 允许初期自由扩展 |
| **proximity_threshold** | 10.0 | 15.0 | +50% | 扩大奖励范围 |
| **proximity_reward** | 50.0 | 150.0 | +200% | 强化临近性激励 |
| **distance_penalty_coef** | 2.0 | 1.5 | -25% | 减轻距离惩罚 |

---

## 🔬 技术原理

### 临近性系统工作流程

```
1. 候选槽位生成 (ring_candidates)
   ↓
   半径R(m) = R0 + m × dR
   
2. 河流连通域过滤
   ↓
   IND → 南岸（连通域0）
   
3. 临近性约束过滤 (if month >= apply_after_month)
   ↓
   只保留距离已有IND建筑 <= max_distance 的槽位
   
4. 动作枚举 (ActionEnumerator)
   ↓
   基于building_level生成S/M/L动作
   
5. 动作打分 (ActionScorer)
   ↓
   base_score + proximity_bonus/penalty
   
   if dist <= proximity_threshold:
       bonus = proximity_reward × (1 - dist/threshold)
   else:
       penalty = (dist - threshold) × distance_penalty_coef
```

### 关键公式

**临近奖励：**
```python
if min_dist <= proximity_threshold:
    proximity_bonus = proximity_reward × (1 - min_dist / proximity_threshold)
    # 距离0时：bonus = 150
    # 距离15时：bonus = 0
```

**距离惩罚：**
```python
else:
    distance_penalty = (min_dist - proximity_threshold) × distance_penalty_coef
    # 距离20时：penalty = (20-15) × 1.5 = 7.5
    # 距离30时：penalty = (30-15) × 1.5 = 22.5
```

---

## ⚠️ 注意事项

### 1. Budget系统影响
由于Budget系统已启用，IND初始预算为10000（已调整）：
- L型建筑成本2400，前4个月可以建3-4个L型
- 需要平衡临近性和预算健康度

### 2. Building Level限制
当前槽位的building_level分布：
- Level 3（仅S）: 大部分槽位
- Level 4（S/M）: 少量槽位
- Level 5（S/M/L）: 极少槽位

如果L型候选不足，需要检查槽位数据。

### 3. RL训练时间
由于候选范围扩大，每个episode的动作数量会增加：
- 旧配置：平均7.9个动作/step
- 新配置：预计15-20个动作/step
- 训练时间可能增加20-30%

---

## 📝 变更记录

**v1.0** (2025-10-09)
- 诊断IND临近性问题
- 优化R0/dR/max_distance/proximity_reward参数
- 创建诊断工具和本文档

---

**维护者：** AI Assistant  
**审核者：** Fenglin  
**最后更新：** 2025-10-09




