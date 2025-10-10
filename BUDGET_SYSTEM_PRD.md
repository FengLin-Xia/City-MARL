# Budget预算系统 PRD

## 📋 概述

**版本：** v1.0  
**日期：** 2025-10-09  
**状态：** 待实现

### 目标
为城市模拟系统添加预算约束机制，使强化学习智能体能够：
- 学习**时序投资策略**（早期投资、后期收获）
- 进行**风险收益权衡**（高成本高回报 vs 低成本低回报）
- 增强**探索动机**（敢于尝试激进策略）
- 产生**多样化策略**（激进型、稳健型、混合型）

---

## 🎯 核心问题

### 当前系统的局限性
1. **过度保守**：智能体害怕高cost，即使高收益也不敢选
2. **缺乏长期规划**：没有"先投资后回报"的概念
3. **策略单一**：倾向于选小建筑（S型），因为cost低
4. **探索不足**：不敢尝试高风险高回报的策略
5. **收敛过快**：训练50个episode后策略固化，缺乏多样性

### 预期改进
- ✅ 鼓励早期大胆投资L型建筑（cost=2400, reward=520/月）
- ✅ 引入投资回报周期概念（前5月投资，后15月收获）
- ✅ 增加策略空间维度（预算管理 + 槽位选择）
- ✅ 自然产生多种策略（激进/稳健/混合）

---

## 🏗️ 系统设计

### 方案选择：软约束预算（推荐）

#### 为什么选择软约束？
- ✅ 允许负债但有惩罚，不会因破产提前终止episode
- ✅ 训练更稳定，不会因为一次失误导致episode失败
- ✅ 鼓励冒险但有代价，平衡探索与利用
- ✅ 更符合真实城市规划（政府可以举债）

#### 核心机制
```python
# 预算更新
budget[agent] -= action.cost        # 扣除建造成本
budget[agent] += action.reward      # 增加月度收益

# 负债惩罚
if budget[agent] < 0:
    debt_penalty = abs(budget[agent]) * debt_penalty_coef
    total_reward -= debt_penalty
```

---

## 📐 技术规格

### 1. 配置参数

#### 新增配置节：`budget_system`
```json
{
  "budget_system": {
    "enabled": true,
    "mode": "soft_constraint",
    
    "initial_budgets": {
      "IND": 5000,
      "EDU": 4000
    },
    
    "debt_penalty_coef": 0.5,
    "max_debt": -2000,
    "monthly_grant": 200,
    
    "bankruptcy_threshold": -5000,
    "bankruptcy_penalty": -100.0,
    
    "budget_in_observation": true,
    "normalize_budget": true,
    "budget_norm_range": [-2000, 10000]
  }
}
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `enabled` | bool | false | 是否启用预算系统 |
| `mode` | str | "soft_constraint" | 约束模式：soft_constraint / hard_constraint / shared_pool |
| `initial_budgets` | dict | {IND:5000, EDU:4000} | 各智能体初始预算（kGBP） |
| `debt_penalty_coef` | float | 0.5 | 负债惩罚系数（每kGBP负债的惩罚） |
| `max_debt` | float | -2000 | 最大允许负债（kGBP） |
| `monthly_grant` | float | 200 | 每月政府拨款（kGBP，可选） |
| `bankruptcy_threshold` | float | -5000 | 破产阈值（触发episode终止） |
| `bankruptcy_penalty` | float | -100.0 | 破产惩罚（一次性） |
| `budget_in_observation` | bool | true | 是否将预算加入观察空间 |
| `normalize_budget` | bool | true | 是否归一化预算值 |
| `budget_norm_range` | list | [-2000, 10000] | 预算归一化范围 |

---

### 2. 代码实现位置

#### A. 环境类修改：`envs/v4_1/city_env.py`

**新增属性：**
```python
class CityEnvironment:
    def __init__(self, cfg):
        # ... 现有代码 ...
        
        # Budget系统
        self.budget_cfg = cfg.get('budget_system', {'enabled': False})
        if self.budget_cfg.get('enabled', False):
            self.budgets = dict(self.budget_cfg.get('initial_budgets', {}))
            self.budget_history = {agent: [] for agent in self.rl_cfg['agents']}
        else:
            self.budgets = None
```

**修改方法：**
1. `reset()` - 重置预算
2. `step()` - 更新预算、计算负债惩罚
3. `_calculate_reward()` - 加入预算惩罚项
4. `_get_observation()` - 添加预算状态
5. `get_available_actions()` - 可选：过滤超预算动作（硬约束模式）

---

#### B. 奖励函数修改

**当前奖励函数：**
```python
def _calculate_reward(self, action: Action, agent_id: str) -> float:
    base_reward = action.reward
    cooperation_reward = self._calculate_cooperation_reward(agent_id)
    total_reward = (base_reward + cooperation_reward) / 100.0
    return total_reward
```

**修改后：**
```python
def _calculate_reward(self, action: Action, agent_id: str) -> float:
    base_reward = action.reward
    cooperation_reward = self._calculate_cooperation_reward(agent_id)
    
    # Budget惩罚
    budget_penalty = 0.0
    if self.budgets is not None:
        # 更新预算
        self.budgets[agent_id] -= action.cost
        self.budgets[agent_id] += action.reward
        
        # 负债惩罚
        if self.budgets[agent_id] < 0:
            debt_penalty_coef = self.budget_cfg.get('debt_penalty_coef', 0.5)
            budget_penalty = abs(self.budgets[agent_id]) * debt_penalty_coef
        
        # 记录历史
        self.budget_history[agent_id].append(self.budgets[agent_id])
    
    total_reward = (base_reward + cooperation_reward - budget_penalty) / 100.0
    return total_reward
```

---

#### C. 观察空间扩展

**添加预算状态到observation：**
```python
def _get_observation(self, agent_id: str) -> np.ndarray:
    # ... 现有特征 ...
    
    # 添加预算特征
    if self.budgets is not None and self.budget_cfg.get('budget_in_observation', True):
        budget = self.budgets[agent_id]
        
        # 归一化
        if self.budget_cfg.get('normalize_budget', True):
            norm_range = self.budget_cfg.get('budget_norm_range', [-2000, 10000])
            budget_norm = (budget - norm_range[0]) / (norm_range[1] - norm_range[0])
            budget_norm = np.clip(budget_norm, 0.0, 1.0)
        else:
            budget_norm = budget / 10000.0  # 简单缩放
        
        obs = np.concatenate([obs, [budget_norm]])
    
    return obs
```

---

### 3. 三种模式对比

#### 模式A：软约束（推荐）
```python
# 允许负债，但有惩罚
if budget < 0:
    penalty = abs(budget) * 0.5
    reward -= penalty

# 破产保护（可选）
if budget < -5000:
    done = True
    reward -= 100
```

**优点：** 稳定、鼓励探索、不会提前终止  
**缺点：** 需要调整惩罚系数

---

#### 模式B：硬约束
```python
# 过滤超预算动作
available_actions = [a for a in actions if a.cost <= budget]

# 无负债
if budget < action.cost:
    action = None  # 跳过本轮
```

**优点：** 严格符合预算约束  
**缺点：** 可能导致无可用动作，训练不稳定

---

#### 模式C：共享预算池
```python
# 两个agent共享城市预算
city_budget = 8000

# 协作奖励
if ind_builds_first and edu_builds_near:
    reward += cooperation_bonus
```

**优点：** 强制协作，策略更复杂  
**缺点：** 训练难度大，需要更多样本

---

## 📊 预期效果

### 训练指标对比

| 指标 | 无Budget | 软约束Budget | 硬约束Budget |
|-----|---------|-------------|-------------|
| **策略多样性** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **探索程度** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **训练稳定性** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **最终收益** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **episode长度** | 31步 | 31步 | 15-25步（可能提前终止） |
| **收敛速度** | 快（50 ep） | 中（100 ep） | 慢（150 ep） |

### 预期策略类型

**策略1：激进型**
```
Month 0-2:  建3个L型工业（-7200预算，+1560收益/月）
Month 3-5:  负债期，不建或建S型
Month 6-20: 预算恢复，持续建L型
总收益：   高（但风险大）
```

**策略2：稳健型**
```
Month 0-10: 每月建1个M型（-1500，+320/月）
Month 11-20: 预算充足，建L型
总收益：    中（稳定）
```

**策略3：混合型**
```
IND：激进（早期建L型）
EDU：稳健（建S/M型）
总收益：高（平衡风险）
```

---

## 🧪 测试计划

### 1. 单元测试
- [ ] 预算初始化正确
- [ ] 预算更新逻辑正确（扣cost、加reward）
- [ ] 负债惩罚计算正确
- [ ] 破产检测正确
- [ ] 观察空间维度正确

### 2. 集成测试
- [ ] 训练10个episode不崩溃
- [ ] 预算历史记录正确
- [ ] 不同模式切换正常
- [ ] 配置文件解析正确

### 3. 性能测试
- [ ] 训练速度对比（vs 无Budget）
- [ ] 收敛速度对比
- [ ] 最终收益对比
- [ ] 策略多样性对比

---

## 🚀 实施步骤

### Phase 1：基础实现（1-2小时）
1. 在 `city_config_v4_1.json` 添加 `budget_system` 配置节
2. 修改 `envs/v4_1/city_env.py`：
   - 添加预算追踪属性
   - 修改 `reset()` 方法
   - 修改 `_calculate_reward()` 方法
3. 创建测试脚本 `test_budget_system.py`

### Phase 2：观察空间扩展（30分钟）
1. 修改 `_get_observation()` 方法
2. 更新网络输入维度（如果需要）
3. 测试观察空间正确性

### Phase 3：训练验证（2-3小时）
1. 运行小规模训练（10 episodes）
2. 检查预算历史曲线
3. 对比有/无Budget的训练效果
4. 调整 `debt_penalty_coef` 参数

### Phase 4：可视化分析（1小时）
1. 创建预算历史可视化脚本
2. 绘制预算曲线图
3. 分析策略类型分布
4. 生成对比报告

---

## 📈 成功指标

### 定量指标
- [ ] 策略多样性提升 > 50%（通过动作熵衡量）
- [ ] L型建筑占比提升 > 30%
- [ ] 最终总收益提升 > 20%
- [ ] Episode平均长度保持 ≥ 25步

### 定性指标
- [ ] 出现明显的"早期投资、后期收获"模式
- [ ] 不同episode采用不同策略（激进/稳健）
- [ ] 智能体敢于在早期建造L型建筑
- [ ] 预算曲线呈现合理的"先降后升"趋势

---

## ⚠️ 风险与缓解

### 风险1：训练不稳定
**现象：** 负债惩罚过大，智能体过度保守  
**缓解：** 降低 `debt_penalty_coef` 从 0.5 → 0.2

### 风险2：破产率过高
**现象：** 大量episode因破产提前终止  
**缓解：** 提高 `initial_budgets` 或增加 `monthly_grant`

### 风险3：收敛速度变慢
**现象：** 需要更多episode才能收敛  
**缓解：** 增加 `max_updates` 从 50 → 100，或调整学习率

### 风险4：观察空间维度不匹配
**现象：** 网络输入维度错误  
**缓解：** 自动检测观察空间维度，动态调整网络结构

---

## 📚 参考资料

### 相关论文
1. **Budget-Constrained RL**: "Resource-Constrained Reinforcement Learning" (ICML 2020)
2. **Multi-Agent Budget Games**: "Cooperative Multi-Agent RL with Budget Constraints" (NeurIPS 2021)

### 类似实现
- OpenAI Gym: `gym.spaces.Box` for continuous budget space
- RLlib: `custom_model` with budget features
- Stable-Baselines3: `VecNormalize` for budget normalization

---

## 🔄 未来扩展

### v1.1：动态预算
- 根据城市发展阶段调整初始预算
- 引入"政策红利"（特定条件下额外拨款）

### v1.2：预算预测
- 智能体预测未来N个月的预算变化
- 基于预测进行长期规划

### v1.3：多目标优化
- 同时优化：总收益、预算健康度、建筑多样性
- Pareto前沿分析

---

## ✅ 验收标准

### 必须满足
- [x] 配置文件正确解析
- [x] 预算更新逻辑无bug
- [x] 训练过程不崩溃
- [x] 预算历史正确记录

### 应该满足
- [ ] 策略多样性提升 > 30%
- [ ] L型建筑占比提升 > 20%
- [ ] 最终收益提升 > 10%

### 可以满足
- [ ] 出现3种以上明显不同的策略
- [ ] 预算曲线符合经济学直觉
- [ ] 训练速度下降 < 20%

---

## 📝 变更日志

**v1.0** (2025-10-09)
- 初始PRD文档
- 定义软约束预算方案
- 明确实施步骤和成功指标

---

**文档维护者：** AI Assistant  
**审核者：** Fenglin  
**最后更新：** 2025-10-09

