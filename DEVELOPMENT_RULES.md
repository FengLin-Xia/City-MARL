# Rain's RL City v5 开发规则

## 🎯 核心原则

**配置驱动 + 模块化 + 契约约束**

## 📋 分层规则

### 1. 配置层 (Config)
- ✅ **所有参数通过 `city_config_v5.json` 配置**
- ❌ **禁止硬编码任何业务参数**

```python
# ❌ 错误
reward = 100 * land_price

# ✅ 正确  
reward = config['reward']['base'] * land_price
```

### 2. 契约层 (Contracts)
- ✅ **只使用三个核心类：`ActionCandidate`, `Sequence`, `StepLog`**
- ✅ **额外信息使用 `meta` 字段**
- ❌ **禁止修改契约类结构**

```python
# ❌ 错误
@dataclass
class ActionCandidate:
    new_field: str  # 不要添加新字段

# ✅ 正确
ActionCandidate(meta={'new_field': value})
```

### 3. 控制层 (Control)
- ✅ **只负责调度：pipeline, scheduler, env, ledger**
- ❌ **禁止包含业务逻辑**

```python
# ❌ 错误
if agent == 'IND':
    reward *= 1.2

# ✅ 正确
reward = self.reward_modules['ind_bonus'].compute(state)
```

### 4. 模块层 (Modules)
- ✅ **新功能创建独立模块文件**
- ✅ **使用 `@register("kind","name")` 注册**
- ✅ **从配置读取参数**

## 🔧 标准接口

### 中间件模块
```python
def apply(seq: Sequence, state: EnvironmentState) -> Sequence
```

### 奖励模块
```python
def compute(prev_state: EnvironmentState, state: EnvironmentState, action_id: int) -> float
```

### 调度模块
```python
def who_acts(t: int, state: EnvironmentState) -> List[str]
```

## 🚀 开发流程

1. **先查配置** - 是否可通过配置实现？
2. **再用契约** - 使用现有契约类传递数据
3. **新功能建模块** - 创建独立模块文件并注册
4. **控制层只调度** - 在控制层调用注册模块
5. **不动基石** - 不修改核心契约类

## 📝 代码审查清单

- [ ] 是否通过配置文件控制行为？
- [ ] 是否使用了正确的契约类？
- [ ] 新模块是否正确注册？
- [ ] 控制层是否只负责调度？
- [ ] 接口是否符合标准？
- [ ] 错误处理是否可配置？
- [ ] 是否保持了向后兼容性？

## 🏗️ 架构模式

- **模块协作**: Pipeline（流水线）
- **状态管理**: 混合式（核心集中，agent 分散）
- **错误处理**: 可配置（WARN / FAIL_FAST）
- **性能目标**: 可扩展
- **兼容性**: 与 v4.1 核心接口一致

## 📚 示例代码

### 创建新奖励模块
```python
# reward_terms/river_premium.py
@register("reward", "river_premium")
class RiverPremiumReward:
    def compute(self, prev_state, state, action_id):
        weight = self.config['reward']['river_premium']['weight']
        return self._calculate_premium(state) * weight
```

### 创建新中间件
```python
# action_mw/budget_filter.py
@register("middleware", "budget_filter")
class BudgetMiddleware:
    def apply(self, seq, state):
        max_cost = self.config['budget']['max_cost']
        return self._filter_by_budget(seq, max_cost)
```

### 配置启用模块
```json
{
  "enabled_modules": {
    "reward": ["river_premium", "proximity_bonus"],
    "middleware": ["budget_filter", "conflict_resolver"]
  }
}
```

---

**记住：先查配置 → 再用契约 → 新功能建模块 → 控制层只调度 → 不动基石**

