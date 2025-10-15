# 固定NPV Reward机制实施总结

## ✅ 实施完成

**核心改造**：将reward从"月度收益累积"改为"固定NPV"

---

## 🎯 解决的问题

### 问题：Episode 2就固定在90.0

**根本原因**：
```python
# 旧机制
reward = monthly_income - build_cost

后期：monthly_income=1500, build_cost=1010
  建造: 1500 - 1010 = 490
  不建造: 1500 - 0 = 1500 (更高！)
  
→ RL学到："建10个后躺平"
→ Return固定90.0
```

**新机制**：
```python
# 固定NPV
if 建造:
    reward = (action.reward * 12) - action.cost
else:
    reward = 0

建造: 162*12 - 1010 = 934
不建造: 0

→ RL学到："必须持续建造"
→ Return应该持续提升
```

---

## 🔧 代码修改

### 1. 配置参数（configs/city_config_v4_1.json）

```json
{
  "solver": {
    "rl": {
      "reward_scale": 3000.0,      // 从2000提高到3000
      "expected_lifetime": 12,     // 新增：固定回报期
      "reward_clip": 1.0,
      // ...
    }
  }
}
```

### 2. Reward计算（envs/v4_1/city_env.py）

**核心逻辑**：
```python
def _calculate_reward(self, agent, action):
    build_cost = action.cost
    
    if build_cost > 0:  # 建造
        # 固定回报期的未来收益
        expected_lifetime = 12
        future_income = action.reward * expected_lifetime
        
        # NPV
        npv = future_income - build_cost
        
        # Total
        reward = npv + progress - budget_penalty
    else:  # 不建造
        reward = 0
    
    return reward / 3000.0
```

**关键变化**：
- ❌ 不再使用`monthly_income`（被动收益）
- ✅ 只用`action.reward * lifetime`（主动价值）
- ✅ 不建造 = 0（不鼓励躺平）

### 3. Budget累积（保持不变）

```python
# 在_advance_turn()中
for agent in agents:
    monthly_income = self._calculate_monthly_income(agent)
    budget += monthly_income  # ✅ 仍然累积

# Budget反映真实现金流（不变）
```

---

## 📊 Reward结构对比

### 各Size的Reward（progress=5）

| Size | Cost | Monthly | Future(12月) | NPV | Bonus | Total | Scaled |
|------|------|---------|-------------|-----|-------|-------|--------|
| **S** | 1010 | 162 | 1944 | 934 | 0 | 939 | **0.31** |
| **M** | 1500 | 210 | 2520 | 1020 | 1000 | 2025 | **0.68** |
| **L** | 2400 | 276 | 3312 | 912 | 2000 | 2917 | **0.97** |

**Skip**: reward = 0

**关键洞察**：
- ✅ L型reward最高（0.97）
- ✅ 所有建造都优于skip（>0）
- ✅ Scaled在[-1, 1]范围内

---

## 🔄 机制完整流程

### Episode执行示例

```
Month 0 (IND):
  1. 计算reward: NPV(S型) = 162*12-1010 = 934
  2. Scaled: 934/3000 = 0.31
  3. 建造 → active_assets += 1
  4. Budget -= 1010

Month 1 (_advance_turn):
  - Budget += monthly_income(IND) = 162  ✅ 真实收益
  - Budget += monthly_income(EDU) = 0
  
Month 1 (EDU):
  1. 计算reward: NPV(S型) = 162*12-700 = 1244
  2. Scaled: 1244/3000 = 0.41
  3. 建造 → active_assets += 1
  4. Budget -= 700

Month 2 (_advance_turn):
  - Budget(IND) += 162  ✅ 持续收益
  - Budget(EDU) += 162  ✅ 持续收益
  
...

Month 10 (IND已有5个建筑):
  - Budget很健康（因为monthly_income累积）
  - 建造reward = 934 (固定！不变！)
  - 不建造reward = 0
  → 仍然鼓励建造

Month 29 (接近结束):
  - 建造reward = 934 (仍然固定！)
  - Budget仍然增长
  → 仍然鼓励建造
```

---

## ✅ 优势

### 1. **解决"躺平"问题** ✅
```
旧: 后期不建造reward=1500 → 躺平最优
新: 不建造reward=0, 建造reward=934 → 建造最优
```

### 2. **鼓励M/L型** ✅
```
L型scaled reward = 0.97（最高）
M型scaled reward = 0.68
S型scaled reward = 0.31
```

### 3. **Reward稳定** ✅
```
无论哪个月建造，reward都一样
→ Value Function容易学习
→ Value Loss应该<1000
```

### 4. **Budget真实** ✅
```
Budget仍然每月累积收益
→ 经济可持续
→ 不会破产
```

---

## 📈 预期训练效果

### Episode 1-10
```
Return: 100-200（波动，不固定）
建筑数量: 40-60个（更多）
Size: 开始探索M/L型
Value Loss: 500-1000
KL: 0.5-2.0
```

### Episode 20-30
```
Return: 150-250
建筑数量: 60-90个
Size: S 40-50%, M 20-30%, L 20-30%
Value Loss: 100-300
KL: 0.1-0.5
```

---

## 🚀 测试计划

### **先测试5 episodes**

```bash
python enhanced_city_simulation_v4_1.py --mode rl
```

**观察**：
1. Return是否>100（不是固定90）
2. 建筑数量是否>40
3. 是否出现M/L型
4. Value Loss是否<1000

**如果通过** → 继续训练到30
**如果不通过** → 再分析调整

---

## 📝 关键点总结

**固定NPV机制 = 摊销法的进化版**

- ✅ 保留了月度收益累积（budget真实）
- ✅ 修复了"躺平"问题（不建造=0）
- ✅ 鼓励大建筑（L型reward最高）
- ✅ Reward稳定（容易学习）

**vs 纯月度收益**：
- Reward不包含"被动收益"
- 只评估"这次建造"的价值
- 更符合"action value"的语义

---

**实施日期**: 2025-10-12  
**状态**: ✅ 代码已修改，配置已更新  
**下一步**: 测试训练5-10 episodes



