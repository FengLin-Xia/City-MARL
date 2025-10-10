# 建筑放置逻辑修改影响分析

## 📋 概述

**核心问题：** 如果改变建筑放置逻辑，是否只需要重新训练RL模型？

**简短答案：** 
- ✅ **大部分情况：是的**，只需重新训练
- ⚠️ **少数情况：不是**，需要修改代码

---

## 🏗️ 建筑放置逻辑的层次结构

### **第1层：槽位系统（Slot System）**
```
槽位文件 (slots_with_angle.txt)
    ↓
SlotNode 数据结构
    ↓
槽位邻接关系 (4-neighbor)
```

**位置：** `logic/v4_enumeration.py` (SlotNode类)

**作用：**
- 定义可建造的位置
- 定义槽位之间的邻接关系
- 存储槽位属性（坐标、角度、地形掩码等）

---

### **第2层：候选槽位筛选（Candidate Selection）**
```
所有槽位
    ↓
ring_candidates() - 基于R(m)半径筛选
    ↓
过滤已占用槽位
    ↓
候选槽位集合
```

**位置：** `enhanced_city_simulation_v4_0.py` (ring_candidates函数)

**作用：**
- 根据当前月份计算候选半径 R(m)
- 筛选在半径范围内的槽位
- 过滤已被占用的槽位

**配置参数：**
```json
"hubs": {
  "R0": 2,      // 初始半径
  "dR": 1.5,    // 每月增长
  "tol": 0.5    // 容差
}
```

---

### **第3层：动作枚举（Action Enumeration）**
```
候选槽位
    ↓
ActionEnumerator.enumerate_actions()
    ↓
生成所有合法动作（EDU/IND × S/M/L）
    ↓
动作列表
```

**位置：** `logic/v4_enumeration.py` (ActionEnumerator类)

**作用：**
- 为每个agent（EDU/IND）枚举所有size（S/M/L）
- **EDU**: S/M/L都是单槽位
- **IND**: S=单槽位, M=1×2相邻对, L=2×2区块
- 计算每个动作的zone（near/mid/far）

**关键方法：**
```python
def enumerate_actions(candidates, occupied, agent_types, sizes):
    # EDU/IND的S/M/L
    for agent in agent_types:
        for size in sizes[agent]:
            if agent == 'IND' and size in ('M', 'L'):
                footprints = _enumerate_ind_footprints(size, free_ids)
            else:
                footprints = _enumerate_single_slots(free_ids)
```

---

### **第4层：动作评分（Action Scoring）**
```
动作列表
    ↓
ActionScorer.score_actions()
    ↓
计算 cost/reward/prestige/score
    ↓
评分后的动作列表
```

**位置：** `logic/v4_enumeration.py` (ActionScorer类)

**作用：**
- 计算每个动作的建造成本（cost）
- 计算每个动作的月度收益（reward）
- 计算每个动作的声望值（prestige）
- 综合计算得分（score）

**评分公式：**
```python
# 归一化
cost_norm = normalize(cost)
reward_norm = normalize(reward)
prestige_norm = normalize(prestige)

# 加权求和
score = w_r × reward_norm + w_p × prestige_norm - w_c × cost_norm
```

---

### **第5层：序列选择（Sequence Selection）**

#### **5A：参数化模式（V4.0）**
```
评分后的动作
    ↓
SequenceSelector.choose_best_sequence()
    ↓
穷举/束搜索长度≤5的序列
    ↓
选择得分最高的序列
```

**位置：** `logic/v4_enumeration.py` (SequenceSelector类)

#### **5B：RL模式（V4.1）**
```
评分后的动作
    ↓
RLPolicySelector.choose_action_sequence()
    ↓
神经网络选择锚点
    ↓
ExpansionPolicy扩展为序列
    ↓
返回选中的序列
```

**位置：** `solvers/v4_1/rl_selector.py` (RLPolicySelector类)

**关键：RL模型在这一层做决策！**

---

### **第6层：环境执行（Environment Execution）**
```
选中的序列
    ↓
CityEnvironment.step()
    ↓
更新槽位占用状态
    ↓
计算环境奖励
    ↓
返回 (obs, reward, done, info)
```

**位置：** `envs/v4_1/city_env.py` (CityEnvironment类)

**作用：**
- 执行选中的动作序列
- 更新城市状态（槽位占用、建筑列表）
- 计算环境奖励（用于RL训练）

---

## 🔍 不同类型的修改及其影响

### **类型1：修改槽位文件**

**示例：**
- 增加/减少槽位数量
- 改变槽位位置
- 修改槽位角度

**影响范围：**
- ✅ 只影响**动作空间**（可选择的槽位变化）
- ✅ RL模型的**观察空间维度不变**（基于地价场、hub距离等）
- ✅ **只需重新训练**

**原因：**
- 槽位文件只是数据输入
- RL模型学习的是"如何在给定槽位中选择"
- 槽位变化后，模型需要重新学习新的槽位分布

**重新训练建议：**
- 从头开始训练（不要加载旧模型）
- 可能需要更多训练轮次（如果槽位数量大幅增加）

---

### **类型2：修改候选半径参数**

**示例：**
```json
"hubs": {
  "R0": 2 → 5,      // 增加初始半径
  "dR": 1.5 → 2.0   // 加快扩张速度
}
```

**影响范围：**
- ✅ 只影响**每月可用的候选槽位数量**
- ✅ 不改变动作枚举逻辑
- ✅ **只需重新训练**

**原因：**
- 候选半径只是筛选条件
- RL模型看到的是"当前可用的动作集合"
- 半径变化 → 可用动作变化 → 需要重新学习

**注意事项：**
- 半径太小：可用动作少，episode可能提前终止
- 半径太大：可用动作多，训练时间增加

---

### **类型3：修改建筑规模定义**

**示例：**
```python
# 当前：IND M = 1×2相邻对
# 修改为：IND M = 单槽位（像EDU一样）

# 或者
# 当前：IND L = 2×2区块
# 修改为：IND L = 3×3区块
```

**影响范围：**
- ⚠️ **需要修改代码**：`logic/v4_enumeration.py` 中的 `_enumerate_ind_footprints()`
- ⚠️ 改变**动作空间结构**
- ⚠️ 可能需要修改**观察空间**（如果编码了建筑规模信息）
- ✅ 修改后**需要重新训练**

**代码修改位置：**
```python
# logic/v4_enumeration.py 第246-252行
def _enumerate_ind_footprints(self, size: str, free_ids: Set[str]) -> List[List[str]]:
    """IND M=1×2 相邻对；IND L=2×2 区块。"""
    if size == 'M':
        return self._enumerate_adjacent_pairs(free_ids)  # ← 修改这里
    if size == 'L':
        return self._enumerate_2x2_blocks(free_ids)      # ← 修改这里
    return self._enumerate_single_slots(free_ids)
```

---

### **类型4：修改评分公式**

**示例：**
```python
# 当前公式
cost = BaseCost + ZoneAdd + LP_value
reward = RevBase + ZR + Adj_bonus - OPEX + river_premium

# 修改为：增加距离惩罚
reward = RevBase + ZR + Adj_bonus - OPEX + river_premium - distance_penalty
```

**影响范围：**
- ⚠️ **需要修改代码**：`logic/v4_enumeration.py` 中的 `ActionScorer._calc_crp()`
- ⚠️ 改变**动作的cost/reward/prestige值**
- ⚠️ 影响**参数化模式**的序列选择（基于score）
- ⚠️ 影响**RL模式**的环境奖励（如果奖励函数使用了reward）
- ✅ 修改后**需要重新训练**

**代码修改位置：**
```python
# logic/v4_enumeration.py 第337-463行
def _calc_crp(self, a: Action, river_distance_provider=None) -> None:
    # ... 现有计算 ...
    
    # 添加新的计算逻辑
    distance_penalty = calculate_distance_penalty(a)
    reward = reward - distance_penalty
```

---

### **类型5：修改环境奖励函数**

**示例：**
```python
# 当前
reward = (action.reward + cooperation_reward) / 100.0

# 修改为：加入score奖励
reward = (action.reward + action.score * 100 + cooperation_reward) / 100.0
```

**影响范围：**
- ⚠️ **需要修改代码**：`envs/v4_1/city_env.py` 中的 `_calculate_reward()`
- ⚠️ 改变**RL训练的优化目标**
- ⚠️ **不影响参数化模式**（V4.0）
- ✅ 修改后**需要重新训练**

**代码修改位置：**
```python
# envs/v4_1/city_env.py 第320-362行
def _calculate_reward(self, action: Action, agent_id: str) -> float:
    base_reward = action.reward
    cooperation_reward = self._calculate_cooperation_reward(agent_id)
    
    # 添加新的奖励项
    score_bonus = action.score * 100
    
    total_reward = (base_reward + score_bonus + cooperation_reward) / 100.0
    return total_reward
```

---

### **类型6：修改动作空间结构**

**示例：**
- 增加新的agent类型（如RES住宅）
- 增加新的建筑size（如XL超大型）
- 改变动作表示方式（从槽位ID改为坐标）

**影响范围：**
- ❌ **需要大量代码修改**
- ❌ 需要修改**动作枚举**、**动作评分**、**序列选择**、**环境执行**
- ❌ 需要修改**RL网络结构**（输入/输出维度）
- ❌ 需要修改**观察空间编码**
- ✅ 修改后**需要重新训练**（从头开始）

**不推荐轻易修改！**

---

## 📊 修改类型总结表

| 修改类型 | 需要改代码 | 影响范围 | 重新训练 | 难度 |
|---------|-----------|---------|---------|------|
| **槽位文件** | ❌ | 动作空间 | ✅ 是 | ⭐ 简单 |
| **候选半径** | ❌ | 候选槽位数 | ✅ 是 | ⭐ 简单 |
| **配置参数** | ❌ | 评分参数 | ✅ 是 | ⭐ 简单 |
| **建筑规模** | ✅ | 动作枚举 | ✅ 是 | ⭐⭐ 中等 |
| **评分公式** | ✅ | 动作评分 | ✅ 是 | ⭐⭐ 中等 |
| **环境奖励** | ✅ | RL优化目标 | ✅ 是 | ⭐⭐ 中等 |
| **动作空间** | ✅ | 整个系统 | ✅ 是 | ⭐⭐⭐⭐⭐ 困难 |

---

## 🎯 常见场景分析

### **场景1：想让建筑更密集**

**方法A：增加槽位数量**
- 修改槽位文件，增加更多槽位
- ✅ 只需重新训练
- ⭐ 难度：简单

**方法B：减小候选半径增长速度**
- 修改 `dR` 参数，让半径增长更慢
- ✅ 只需重新训练
- ⭐ 难度：简单

---

### **场景2：想让RL选择L型建筑**

**方法A：修改环境奖励函数**
```python
# 加入score奖励，鼓励高分动作
reward = (action.reward + action.score * 100) / 100.0
```
- ✅ 需要修改代码（1行）
- ✅ 需要重新训练
- ⭐⭐ 难度：中等

**方法B：实现Budget系统**
- 限制早期建造数量，鼓励投资L型
- ✅ 需要修改代码（见BUDGET_SYSTEM_PRD.md）
- ✅ 需要重新训练
- ⭐⭐⭐ 难度：中等偏高

---

### **场景3：想让建筑沿河流分布**

**方法A：修改评分公式**
```python
# 增加河流距离奖励
river_bonus = calculate_river_proximity_bonus(a)
reward = reward + river_bonus
```
- ✅ 需要修改代码（`ActionScorer._calc_crp()`）
- ✅ 需要重新训练
- ⭐⭐ 难度：中等

**方法B：修改槽位文件**
- 只保留河流附近的槽位
- ❌ 不需要修改代码
- ✅ 需要重新训练
- ⭐ 难度：简单

---

### **场景4：想让IND建筑成片分布**

**方法A：修改邻接奖励**
```python
# 增加邻接奖励权重
Adj_bonus = 50  # 从0增加到50
```
- ❌ 不需要修改代码（只改配置）
- ✅ 需要重新训练
- ⭐ 难度：简单

**方法B：修改IND的M/L定义**
```python
# 让IND M = 2×2区块（强制成片）
if size == 'M':
    return self._enumerate_2x2_blocks(free_ids)
```
- ✅ 需要修改代码（`ActionEnumerator`）
- ✅ 需要重新训练
- ⭐⭐ 难度：中等

---

## ⚠️ 重要注意事项

### **1. 观察空间一致性**

如果修改影响了观察空间（如增加新特征），需要：
- 修改 `CityEnvironment._get_observation()`
- 修改 RL网络的输入维度
- **从头开始训练**（不能加载旧模型）

### **2. 动作空间一致性**

如果修改影响了动作空间（如增加新agent），需要：
- 修改 `CityEnvironment.get_available_actions()`
- 修改 RL网络的输出维度
- **从头开始训练**（不能加载旧模型）

### **3. 配置文件同步**

修改后确保：
- `city_config_v4_1.json` 与代码一致
- 训练和评估使用相同的配置
- 保存配置文件的副本（用于复现）

### **4. 训练策略**

修改后的训练建议：
- **小改动**（槽位、参数）：训练10-20个updates
- **中改动**（评分、奖励）：训练50-100个updates
- **大改动**（动作空间）：训练100+个updates

---

## 📋 决策流程图

```
修改建筑放置逻辑
    ↓
是否改变动作空间结构？
    ├─ 是 → 需要修改代码（动作枚举/网络结构）→ 重新训练
    └─ 否 ↓
是否改变评分/奖励公式？
    ├─ 是 → 需要修改代码（ActionScorer/CityEnv）→ 重新训练
    └─ 否 ↓
是否只改配置参数？
    ├─ 是 → 不需要修改代码 → 重新训练
    └─ 否 ↓
是否只改槽位文件？
    ├─ 是 → 不需要修改代码 → 重新训练
    └─ 否 → 可能不需要训练（如可视化调整）
```

---

## ✅ 结论

### **只需重新训练的情况（80%）：**
- ✅ 修改槽位文件
- ✅ 修改配置参数（R0, dR, 权重等）
- ✅ 修改地价场参数（河流、hub等）
- ✅ 修改候选范围

### **需要修改代码 + 重新训练的情况（15%）：**
- ⚠️ 修改建筑规模定义
- ⚠️ 修改评分公式
- ⚠️ 修改环境奖励函数
- ⚠️ 增加新的约束条件

### **需要大量修改的情况（5%）：**
- ❌ 改变动作空间结构
- ❌ 增加新的agent类型
- ❌ 改变观察空间编码

---

## 🚀 推荐工作流

1. **明确修改目标**：想要什么样的建筑分布？
2. **选择修改方式**：优先选择"只需重新训练"的方式
3. **小步迭代**：先改小参数，看效果，再逐步调整
4. **保存检查点**：每次修改前保存当前配置和模型
5. **对比评估**：修改后与之前的结果对比

---

**文档维护者：** AI Assistant  
**最后更新：** 2025-10-09

