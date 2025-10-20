# 月度收益机制实施总结

## ✅ 完成状态
**月度收益机制已完成并测试通过！**

---

## 📝 实施内容（基于PRD，不含租金）

### 核心机制

**原来的问题**：
```python
建造时：budget -= cost, budget += reward (一次性)
结果：每次建造都亏损，必然破产
```

**新的机制**：
```python
建造时：budget -= cost, 记录到active_assets
每月开始：budget += sum(所有在营建筑的monthly_income)
结果：建筑持续产生收益，可以回本
```

---

## 🔧 代码修改

### 1. 数据结构扩展（envs/v4_1/city_env.py）

```python
# 添加第76-77行
self.active_assets = {agent: [] for agent in self.rl_cfg['agents']}
self.monthly_income_history = {agent: [] for agent in self.rl_cfg['agents']}

# Asset结构：
asset = {
    'size': 'S/M/L',
    'monthly_income': 162.0,  # 月度运营收益
    'cost': 1010.0,           # 建造成本
    'built_month': 0,         # 建造月份
    'building_id': 0          # 唯一ID
}
```

### 2. 新增方法：_calculate_monthly_income()（第473-479行）

```python
def _calculate_monthly_income(self, agent: str) -> float:
    """计算agent的月度收益（所有在营建筑的累加）"""
    total_income = sum([asset['monthly_income'] for asset in self.active_assets[agent]])
    return float(total_income)
```

### 3. 修改_place_building()（第513-521行）

```python
# 建造时记录到active_assets
asset = {
    'size': action.size,
    'monthly_income': action.reward,  # 持续月度收益
    'cost': action.cost,
    'built_month': self.current_month,
    'building_id': len(self.active_assets[agent])
}
self.active_assets[agent].append(asset)
```

### 4. 重写_calculate_reward()（第364-416行）

```python
# 新的reward计算
def _calculate_reward(self, agent, action):
    # 1. 月度收益（所有在营建筑）
    monthly_income = self._calculate_monthly_income(agent)
    
    # 2. 建造成本
    build_cost = action.cost
    
    # 3. 进度奖励
    progress_reward = len(buildings) * 0.5
    
    # 4. Budget惩罚（软约束）
    budget_after = budget + monthly_income - build_cost
    if budget_after < 0:
        budget_penalty = abs(budget_after) * 0.1
    
    # 5. 总奖励
    total_reward = monthly_income - build_cost + progress_reward - budget_penalty
    
    # 6. 缩放到[-1, 1]
    scaled_reward = total_reward / 500.0
    scaled_reward = clip(scaled_reward, -5.0, 5.0)
    
    return scaled_reward
```

**关键变化**：
- ❌ 移除了重复的`quality_reward`（reward被计入3次）
- ✅ 使用`monthly_income`替代单次`action.reward`
- ✅ 清晰的语义：收入 - 支出 = 净收益

### 5. 修改_advance_turn()（第514-521行）

```python
# 每月开始时，为所有agent累加月度收益
for ag in ['IND', 'EDU']:
    monthly_income = self._calculate_monthly_income(ag)
    self.budgets[ag] += monthly_income
    self.monthly_income_history[ag].append(monthly_income)
```

**效果**：
- ✅ 即使agent不建造，budget也会因已有建筑的收益增长
- ✅ 两个agent都受益（虽然只有行动agent拿reward）

### 6. 修改step()中的budget更新（第215-218行）

```python
# 建造时扣除成本
if self.budgets is not None:
    build_cost = action.cost
    self.budgets[agent] -= build_cost
```

---

## 📊 测试结果

### 测试场景
连续5个月，前3个月建造，后2个月不建造

### 验证通过
```
Month 0 (IND建造):
  Assets: 1 → 2 (+1)
  Monthly income: 162 → 324 (+162) ✓
  Budget: 15000 → 14314 (-686, cost抵消部分收益)

Month 1 (EDU建造):
  Monthly income: 0 → 162 (+162) ✓
  Budget: 10000 → 9152 (-848)

Month 2 (IND建造):
  Assets: 2 → 3 (+1)
  Monthly income: 324 → 486 (+162) ✓
  Budget: 14638 → 14114 (收益开始覆盖支出！)

Month 3 (EDU不建造):
  Assets: 1 (不变)
  Monthly income: 162 (不变)
  Budget: 9314 → 9476 (+162, 纯收益！) ✓

Month 4 (IND不建造):
  Assets: 3 (不变)
  Monthly income: 486 (不变)
  Budget: 14600 → 15086 (+486, 纯收益！) ✓
```

**关键发现**：
- ✅ 月度收益正确累积（每建一个，monthly_income +162）
- ✅ 不建造时budget仍增长（持续收益）
- ✅ 建筑越多，收益越高（累积效应）

---

## 🎯 经济逻辑验证

### 回本周期计算

**S型建筑**：
- Cost: 1010
- Monthly income: 162
- 回本周期: 1010 / 162 ≈ 6.2个月

**验证**：
```
Month 0: 建造，budget -1010 + 162 = -848
Month 1: +162, 累计 -686
Month 2: +162, 累计 -524
Month 3: +162, 累计 -362
Month 4: +162, 累计 -200
Month 5: +162, 累计 -38
Month 6: +162, 累计 +124 ✓ 回本了！
```

**M型建筑**（假设）：
- Cost: 1500
- Monthly income: 210
- 回本周期: 1500 / 210 ≈ 7.1个月

**L型建筑**（假设）：
- Cost: 2400
- Monthly income: 276
- 回本周期: 2400 / 276 ≈ 8.7个月

**20个月episode内**：
- ✅ 所有Size都能回本
- ✅ L型虽然回本慢，但长期总收益最高
- ✅ RL能学到："先投入，后回报"

---

## 💰 Reward结构变化

### 修改前
```python
reward = action.reward - action.cost (一次性)
      = 162 - 1010 = -848 (必亏)
```

### 修改后
```python
# 第一次建造
reward = monthly_income - build_cost + progress
      = 0 - 1010 + 0.5 = -1009.5

# 第二次建造（已有1个建筑）
reward = 162 - 1010 + 1.0 = -847

# 第五次建造（已有4个建筑）
reward = 648 - 1010 + 2.5 = -359.5

# 第七次建造（已有6个建筑）
reward = 972 - 1010 + 3.5 = -34.5

# 第八次建造（已有7个建筑）
reward = 1134 - 1010 + 4.0 = +128 (终于转正！)
```

**关键洞察**：
- ✅ 建筑越多，monthly_income越高
- ✅ 最终会达到"建造=盈利"的状态
- ✅ 鼓励"先苦后甜"的长期策略

---

## 📈 预期RL行为

### 1. 建筑数量增长曲线

```
Episode 1-10: 探索阶段
  - RL体验"建造→亏损→积累→盈利"的过程
  - 学习到建筑的累积价值

Episode 10-20: 加速阶段
  - RL发现：多建→高monthly_income→后续建造变容易
  - 建筑数量快速增长

Episode 20+: 稳定阶段
  - 收敛到最优建造策略
  - 预期：每个agent 20-30个建筑
```

### 2. Size选择多样化

**原来**：
```
100% S型（因为看起来成本最低）
```

**现在**：
```
# 累积效应下
当monthly_income=486时（3个S型）:
  - 建S型: reward = 486 - 1010 = -524
  - 建M型: reward = 486 - 1500 + size_bonus = 486-1500+300 = -714
  - 建L型: reward = 486 - 2400 + size_bonus = 486-2400+800 = -1114

# 但考虑未来收益（RL会学到）
S型未来总收益: 162 * 15个月 = 2430, NPV = 2430-1010 = 1420
M型未来总收益: 210 * 15个月 = 3150, NPV = 3150-1500 = 1650 ✓
L型未来总收益: 276 * 15个月 = 4140, NPV = 4140-2400 = 1740 ✓

→ RL应该学到：L型>M型>S型（长期价值）
```

**预期分布**：
- S型: 30-40%（早期建造）
- M型: 20-30%（中期）
- L型: 30-40%（后期，monthly_income足够时）

---

## 🔄 与旧系统的对比

| 方面 | 旧系统 | 新系统（月度收益） |
|------|--------|------------------|
| **Reward语义** | 单次收益-成本 | 累积收益-新成本 |
| **Budget逻辑** | 混乱 | 清晰现金流 |
| **回本机制** | 不存在 | 6-9个月 |
| **长期规划** | 无 | 有（累积效应） |
| **训练稳定性** | Value Loss 5000+ | 预期<100 |
| **Size多样性** | 100% S | 预期三种都有 |

---

## ⚙️ 配置参数

新增到`configs/city_config_v4_1.json`：

```json
{
  "solver": {
    "rl": {
      "reward_scale": 500.0,   // Reward缩放因子
      "reward_clip": 5.0,      // Reward裁剪上下限
      // ...
    }
  },
  "budget_system": {
    "debt_penalty_coef": 0.1,  // 已降低
    // ...
  }
}
```

---

## 🚀 下一步

1. **立即可以训练**：
   ```bash
   python enhanced_city_simulation_v4_1.py --mode rl
   ```

2. **监控指标**：
   - Episode Return：期望从-89变为正数
   - Value Loss：期望从5000+降到<100
   - KL Divergence：期望<0.1
   - 建筑Size分布：期望M/L型出现

3. **明天考虑**：
   - Turn-Based MAPPO（让非行动agent也拿reward）
   - 租金转移机制

---

**实施日期**: 2025-10-11  
**状态**: ✅ 完成并测试通过  
**版本**: v4.1 + Monthly Income  
**未实施**: 租金转移、Turn-Based MAPPO



