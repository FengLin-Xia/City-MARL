# V4.1 Reward与Cost机制分析

## 目录
1. [系统概览](#系统概览)
2. [Cost计算详解](#cost计算详解)
3. [Reward计算详解](#reward计算详解)
4. [RL训练中的Reward组成](#rl训练中的reward组成)
5. [潜在问题](#潜在问题)
6. [具体案例分析](#具体案例分析)

---

## 系统概览

V4.1采用**双层设计**：
1. **ActionScorer层**：计算每个动作的cost/reward/prestige/score
2. **Environment层**：基于action的属性计算RL训练用的total_reward

```
┌─────────────────────────────────────────────────────────┐
│                    ActionScorer                          │
│  计算原始经济指标 (cost/reward/prestige/score)           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│                    Environment                           │
│  基于action属性 → 计算RL的total_reward                    │
└─────────────────────────────────────────────────────────┘
```

---

## Cost计算详解

### 公式（在`logic/v4_enumeration.py`中）

```python
# IND的建造成本
cost = BaseCost_IND[size] + ZC[zone] + LP_value

其中：
- BaseCost_IND = {'S': 900, 'M': 1500, 'L': 2400}  # 基础建造成本（kGBP）
- ZC = {'near': 200, 'mid': 100, 'far': 0}         # 区位附加成本（kGBP）
- LP_value = LP_idx * LandPriceBase                 # 地价成本（kGBP）
  - LP_idx = 10~100 (基于LP_norm = 0~1线性映射)
  - LandPriceBase = 11.0 (默认值)
  - LP_value范围：110~1100
```

### 典型成本范围

| Size | BaseCost | Zone影响 | LP影响 | 总成本范围 |
|------|----------|---------|--------|-----------|
| S型  | 900      | +0~200  | +110~1100 | **1010~2200 kGBP** |
| M型  | 1500     | +0~200  | +110~1100 | **1610~2800 kGBP** |
| L型  | 2400     | +0~200  | +110~1100 | **2510~3700 kGBP** |

**注意**：cost是**一次性**建设投入。

---

## Reward计算详解

### 公式（在`logic/v4_enumeration.py`中）

```python
# IND的月度收益
reward_base = RevBase_IND[size] + ZR[zone] + Adj_bonus*is_adj - OPEX_IND[size] - Rent[size] + river_premium

# LP增益
reward = reward_base * (1 + RewardLP_k * LP_norm)

其中：
- RevBase_IND = {'S': 180, 'M': 320, 'L': 520}     # 基础月收入（kGBP/月）
- ZR = {'near': 80, 'mid': 40, 'far': 0}           # 区位收入加成（kGBP/月）
- OPEX_IND = {'S': 100, 'M': 180, 'L': 300}        # 运营成本（kGBP/月）
- Rent = {'S': 25, 'M': 45, 'L': 70}               # 租金支出（kGBP/月）
- RewardLP_k = 0.25 (IND的LP增益系数)
- Adj_bonus = 0 (默认)
- river_premium：河流距离溢价（0~几百）
```

### 典型收益计算示例

**S型建筑（zone=near, LP_norm=0.8）**：
```
reward_base = 180 + 80 + 0 - 100 - 25 + 0 = 135
reward = 135 * (1 + 0.25 * 0.8) = 135 * 1.2 = 162 kGBP/月
```

**M型建筑（zone=near, LP_norm=0.8）**：
```
reward_base = 320 + 80 + 0 - 180 - 45 + 0 = 175
reward = 175 * (1 + 0.25 * 0.8) = 175 * 1.2 = 210 kGBP/月
```

**L型建筑（zone=near, LP_norm=0.8）**：
```
reward_base = 520 + 80 + 0 - 300 - 70 + 0 = 230
reward = 230 * (1 + 0.25 * 0.8) = 230 * 1.2 = 276 kGBP/月
```

### 典型收益范围

| Size | 基础收益 | LP增益后 | 月收益范围 |
|------|---------|---------|-----------|
| S型  | 55~135  | 55~168  | **55~170 kGBP/月** |
| M型  | 95~175  | 95~218  | **95~220 kGBP/月** |
| L型  | 150~230 | 150~287 | **150~290 kGBP/月** |

**注意**：reward是**每月持续**的运营收益。

---

## RL训练中的Reward组成

### Environment._calculate_reward()的完整流程

```python
# 位置：envs/v4_1/city_env.py

def _calculate_reward(self, agent: str, action: Action) -> float:
    # ===== 第1部分：基础奖励 =====
    base_reward = action.reward  # 使用月度收益（55~290）
    
    # ===== 第2部分：质量奖励 =====
    quality_reward = 0
    quality_reward += action.reward * 0.01      # reward再次计入（0.55~2.9）
    quality_reward += action.prestige * 10.0    # 声望放大10倍（-1~10）
    quality_reward -= action.cost * 0.001       # cost惩罚（-1~-3.7）
    # 质量奖励范围：-4.7 ~ 9.2
    
    # ===== 第3部分：进度奖励 =====
    progress_reward = len(buildings) * 0.1      # 建筑数量奖励（0~4）
    
    # ===== 第4部分：协作奖励 =====
    cooperation_bonus = 0
    if cooperation_lambda > 0:  # = 0.2
        # 功能互补：其他agent建筑数 * 0.05
        cooperation_bonus += other_buildings_count * 0.05
        # 空间协调：距离5-20单位 → +0.02
        cooperation_bonus += spatial_coordination * 0.02
    cooperation_bonus *= 0.2  # 最终：0.01~0.2
    
    # ===== 第5部分：Budget惩罚 =====
    budget_penalty = 0
    budget -= action.cost           # 支付建造成本（-1010~-3700）
    budget += action.reward         # 获得一个月收益（+55~290）
    
    if budget < 0:
        budget_penalty = abs(budget) * debt_penalty_coef  # 0.1
    
    if budget < bankruptcy_threshold:  # -5000
        budget_penalty += 100.0
    
    # ===== 总奖励 =====
    total_reward = base_reward + quality_reward + progress_reward + cooperation_bonus - budget_penalty
    
    # ===== 缩放（送给PPO） =====
    scaled_reward = total_reward / 200.0
    scaled_reward = np.clip(scaled_reward, -10.0, 10.0)
    
    return scaled_reward
```

### 各部分贡献度

| 组件 | 数值范围 | 占比 | 说明 |
|------|---------|------|------|
| **base_reward** | 55~290 | **70-95%** | 主导因素 |
| **quality_reward** | -5~+9 | 2-5% | 影响很小 |
| **progress_reward** | 0~4 | 1-2% | 影响很小 |
| **cooperation_bonus** | 0~0.2 | <0.1% | **几乎无影响** |
| **budget_penalty** | 0~几千 | **可能压倒一切** | 破产后主导 |

### Total Reward典型值

**正常情况（budget充足）**：
```
total_reward = 162 + 2 + 2 + 0.1 - 0 = 166
scaled_reward = 166 / 200 = 0.83
```

**破产情况（budget=-10000）**：
```
budget_penalty = 10000 * 0.1 + 100 = 1100
total_reward = 162 + 2 + 2 + 0.1 - 1100 = -934
scaled_reward = -934 / 200 = -4.67
```

---

## 潜在问题

### 问题1：Cost和Reward的时间维度不一致 ⚠️

**问题描述**：
- `cost`是一次性投入（1010~3700 kGBP）
- `reward`是月度收益（55~290 kGBP/月）
- Budget系统将两者直接相减：`budget = budget - cost + reward`

**问题所在**：
```python
# 第1个月
budget = 15000 - 2000(cost) + 162(reward) = 13162  # 净支出1838

# 如果只看一个月，所有建筑都是"亏损"的！
# 实际上需要 cost/reward = 2000/162 ≈ 12.3个月才能回本
```

**经济学角度**：
- 应该用NPV（净现值）评估：`NPV = -cost + Σ(reward / (1+r)^t)`
- 或者只在第一个月扣cost，后续每月加reward
- 当前实现：**每次建筑都扣cost，但只加一次reward**

**影响**：
- 导致训练初期必然出现负return
- RL误以为"建筑=亏钱"
- 导致过度保守策略

---

### 问题2：Reward的重复计算 ⚠️

**问题描述**：
`action.reward`在total_reward中被用了**3次**：

```python
base_reward = action.reward           # 第1次：作为主要奖励（162）
quality_reward += action.reward * 0.01  # 第2次：质量加成（+1.62）
budget += action.reward                # 第3次：影响budget_penalty（间接）
```

**影响**：
- 放大了reward的影响（虽然第2次只有1%）
- 设计不清晰，难以调试
- 如果未来修改reward计算，需要改3处

**建议**：
只使用一次，明确语义：
```python
# 方案A：只用base_reward
total_reward = action.reward + progress + cooperation - budget_penalty

# 方案B：分离经济和策略
economic_value = action.reward - action.cost / expected_lifetime
strategic_value = progress + cooperation
total_reward = economic_value + strategic_value
```

---

### 问题3：Cost的双重使用 ⚠️

**问题描述**：
`action.cost`也被用了2次：

```python
quality_reward -= action.cost * 0.001  # 第1次：质量惩罚（-2）
budget -= action.cost                  # 第2次：预算扣除（-2000）
```

**问题所在**：
- 第1次的权重（0.001）太小，几乎无影响（-2 vs 162的base_reward）
- 第2次的影响巨大（直接扣除2000）
- 两次使用的语义不同：一个是"质量权衡"，一个是"财务约束"

**影响**：
- `quality_reward`中的cost惩罚形同虚设
- Budget系统主导了cost的影响

---

### 问题4：ActionScorer的score被完全忽略 ⚠️

**问题描述**：
ActionScorer精心设计的综合评分系统被绕过：

```python
# ActionScorer计算（logic/v4_enumeration.py）
score = 0.5 * norm(reward) + 0.3 * norm(prestige) - 0.2 * norm(cost)
# 这个score在[0, 1]区间，平衡了三个维度

# 但Environment完全不用这个score！
base_reward = action.reward  # 直接用原始reward，忽略了prestige和cost的平衡
```

**ActionScorer的设计意图**：
- 归一化确保各维度公平比较
- 权重体现设计者的价值观（reward最重要0.5，prestige次之0.3，cost最轻0.2）
- 适用于确定性选择（v4.0参数化模式）

**Environment的实际做法**：
- 用原始值（未归一化）
- 自己重新设计了权重（但不清晰）
- 适用于RL探索学习

**矛盾**：
两套系统共存但不协调，造成confusion。

---

### 问题5：Cooperation机制形同虚设 ⚠️

**问题描述**：
```python
cooperation_bonus = 0.2 * (other_buildings * 0.05 + spatial * 0.02)
                  = 0.01 ~ 0.2  # 最大值0.2

base_reward = 162  # 数量级：百
```

**占比分析**：
- cooperation_bonus / base_reward = 0.2 / 162 = **0.12%**
- 在total_reward的noise范围内，几乎不可学习

**设计vs现实**：
| 参数 | 设计意图 | 实际效果 |
|------|---------|---------|
| `cooperation_lambda=0.2` | 协作占20%权重 | 实际<0.2% |
| 功能互补奖励 | 鼓励两agent协作 | 信号太弱 |
| 空间协调奖励 | 鼓励合理布局 | 信号太弱 |

**影响**：
- RL无法学习到协作策略
- 两个agent实际上是独立决策
- Multi-Agent的优势未体现

---

### 问题6：租金机制的设计缺陷 ⚠️⚠️

**问题描述**：
代码中存在租金设计，但实现有误：

```python
# EDU的reward
rev_land = Rent[size]  # EDU获得租金收入（25/45/70）

# IND的reward
rev = ... - Rent[size]  # IND支付租金（-25/-45/-70）
```

**看起来像**：IND付租金 → EDU收租金

**实际上**：
- EDU的`Rent[size]`是**EDU自己建筑的地租收入**
- IND的`-Rent[size]`是**IND自己建筑的地租支出**
- **两者没有任何联系！**

**正确的租金机制应该是**：
```python
# EDU应该从所有IND建筑收租
total_ind_rent = sum([Rent[b.size] for b in industrial_buildings])
edu_reward += total_ind_rent

# IND支付租金给EDU
ind_reward -= Rent[size]
```

**当前的误导**：
- 看起来有agent间经济关系
- 实际上两个agent完全独立
- 这是"假的Multi-Agent关系"

---

### 问题7：Budget系统的经济逻辑问题 ⚠️

**问题场景**：
```
初始预算：15000
建造10个S型建筑：

Month 0: 
  - cost: -2000, reward: +162 → budget = 13162 (净-1838)
Month 1:
  - cost: -2000, reward: +162 → budget = 11324 (净-1838)
...
Month 8:
  - cost: -2000, reward: +162 → budget = -290 (破产！)
```

**问题分析**：

1. **只计入一次reward不合理**
   - 现实：建筑建成后，**每个月都产生收益**
   - 当前：建筑只在建造月产生一次reward
   - 导致：无法回本，必然破产

2. **月度收益应该累积**
   ```python
   # 应该是
   month_1_income = building_1.reward
   month_2_income = building_1.reward + building_2.reward
   month_3_income = building_1.reward + building_2.reward + building_3.reward
   ...
   ```

3. **Budget变成了"现金流"而非"净资产"**
   - Budget系统模拟的是现金流（每月收支）
   - 但没有考虑资产价值（已建建筑的价值）
   - 实际上应该：`net_worth = budget + Σ(building_value)`

---

## 具体案例分析

### 案例1：S型 vs L型的选择

**场景**：zone=near, LP_norm=0.8, budget=15000

| 指标 | S型建筑 | L型建筑 |
|------|---------|---------|
| **Cost** | 1010 | 2510 |
| **Reward** | 162 | 276 |
| **回本周期** | 6.2个月 | 9.1个月 |
| **20个月总收益** | 162*20 - 1010 = 2230 | 276*20 - 2510 = 3010 |
| **RL的total_reward** | 162 + 2 + ... = 166 | 276 + 2 + ... = 280 |
| **Budget影响** | -1010 + 162 = -848 | -2510 + 276 = -2234 |

**分析**：
- **长期来看**：L型更优（总收益3010 > 2230）
- **RL视角**：L型略优（280 > 166）
- **Budget视角**：S型优（-848 > -2234，消耗更少）
- **破产风险**：L型更高（需要更多初始资本）

**RL会学到什么**：
- 在budget充足时：可能选L型（reward更高）
- 在budget紧张时：被迫选S型（避免破产惩罚）
- 实际观察：**100% S型** → 说明budget惩罚主导了决策

---

### 案例2：最近一次训练的失败分析

**训练结果**：
```
Episode Returns: -82 到 -89（全负！）
IND Budget: 15000 → -10332
EDU Budget: 10000 → -12168
```

**原因推断**：

1. **Size Bonus设置错误**
   ```
   我之前错误计算：
   - 假设cost = 100/300/600（错！实际是900/1500/2400）
   - 设置size_bonus = {M: 1000, L: 2750}
   - 导致：即使加了bonus，reward仍然远小于cost
   ```

2. **每个动作都亏损**
   ```
   S型：reward=162+0, cost=1010 → net=-848
   M型：reward=210+1000=1210, cost=1500 → net=-290
   L型：reward=276+2750=3026, cost=2400 → net=+626
   
   但实际上还有proximity_reward=1500降到400：
   实际total_reward ≈ 60-90（远低于预期）
   ```

3. **连锁崩溃**
   ```
   每个动作 → budget减少 → budget_penalty增加 → total_reward变负
   → PPO学到"建筑=惩罚" → 下次更保守 → 恶性循环
   ```

---

## 总结

### 当前系统的实际表现

| 方面 | 设计目标 | 实际效果 | 符合度 |
|------|---------|---------|-------|
| **Cost-Reward平衡** | 长期收益权衡 | Budget主导，短期思维 | ⚠️ 20% |
| **Multi-Agent协作** | 两agent互依互补 | 完全独立决策 | ⚠️ 5% |
| **租金机制** | IND向EDU交租 | 两者无关联 | ❌ 0% |
| **Size多样性** | S/M/L都有价值 | 100% S型 | ❌ 0% |
| **经济可持续性** | 盈利可持续 | 必然破产 | ❌ 0% |

### 核心问题优先级

1. **🔴 紧急**：Budget的经济逻辑错误
   - 导致training必然失败
   - 需要立即修复

2. **🟠 重要**：Cost/Reward时间维度不一致
   - 误导RL学习
   - 影响策略质量

3. **🟡 中等**：Cooperation机制太弱
   - 无法体现Multi-Agent优势
   - 但不影响基本训练

4. **🟢 次要**：租金机制设计缺陷
   - 当前影响有限
   - 可以作为future work

---

## 建议的修复方向

### 短期修复（让训练能跑起来）

1. **回滚Budget配置**
   - 恢复`proximity_reward=1500`（从400回到1500）
   - 减小或移除`size_bonus`（当前值过大）
   - 确保大部分动作的`total_reward > 0`

2. **简化Reward计算**
   - 只用`base_reward = action.reward`
   - 移除重复的`quality_reward`
   - 保持Budget作为软约束

### 中期优化（提升训练质量）

1. **修正Budget逻辑**
   - 将cost视为资产投资，不是纯支出
   - 每月累积所有建筑的reward
   - Budget = 初始资金 + Σ(每月总收益) - Σ(历史总投资)

2. **强化Cooperation**
   - 提高`cooperation_lambda`到1.0-2.0
   - 实现真正的租金转移机制
   - 增加agent间的reward传递

### 长期重构（理想设计）

1. **统一Cost/Reward体系**
   - 用NPV统一评估
   - 或者分离"经济价值"和"策略价值"

2. **真正的Multi-Agent**
   - 独立的Actor/Critic网络
   - 通信机制
   - 联合目标函数

---

**文档版本**：v1.0  
**创建时间**：2025-10-11  
**状态**：待讨论

