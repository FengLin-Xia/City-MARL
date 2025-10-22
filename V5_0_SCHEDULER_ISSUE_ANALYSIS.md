# v5.0 调度器问题分析

## 🎯 问题描述

**你的问题**: "我的月份轮次是一个月为IND，下个月则应该是COUNCIL和EDU同时执行，我有印象当时做了这个配置，为什么现在的执行过程又是每个agent一个月呢？"

**答案**: 调度器配置是正确的，但环境实现有问题！

## 📊 配置分析

### **调度器配置**
```json
"scheduler": {
  "name": "phase_cycle",
  "params": {
    "step_unit": "month",
    "period": 2,
    "offset": 0,
    "phases": [
      { "agents": ["EDU","COUNCIL"], "mode": "concurrent" },
      { "agents": ["IND"],           "mode": "sequential" }
    ]
  }
}
```

**配置含义**:
- **period: 2** → 每2步切换阶段
- **阶段0**: EDU+COUNCIL (并发执行)
- **阶段1**: IND (顺序执行)
- **预期行为**: 每2步切换阶段，每步内切换智能体

## 🔍 调度器实际行为分析

### **30步调度行为**
```
步骤0: 阶段0, 活跃智能体=[EDU,COUNCIL], 执行模式=concurrent, 当前智能体=EDU
步骤1: 阶段0, 活跃智能体=[EDU,COUNCIL], 执行模式=concurrent, 当前智能体=COUNCIL
步骤2: 阶段1, 活跃智能体=[IND],        执行模式=sequential, 当前智能体=IND
步骤3: 阶段1, 活跃智能体=[IND],        执行模式=sequential, 当前智能体=IND
步骤4: 阶段0, 活跃智能体=[EDU,COUNCIL], 执行模式=concurrent, 当前智能体=EDU
步骤5: 阶段0, 活跃智能体=[EDU,COUNCIL], 执行模式=concurrent, 当前智能体=COUNCIL
...
```

### **阶段切换模式**
- **步骤0-1**: 阶段0 (EDU+COUNCIL并发)
- **步骤2-3**: 阶段1 (IND顺序)
- **步骤4-5**: 阶段0 (EDU+COUNCIL并发)
- **步骤6-7**: 阶段1 (IND顺序)
- **...**: 每2步切换一次

## 🚨 问题根源分析

### **1. 调度器配置正确**
- ✅ **period: 2** → 每2步切换阶段
- ✅ **phases**: [EDU+COUNCIL, IND] → 正确的阶段配置
- ✅ **mode**: concurrent/sequential → 正确的执行模式

### **2. 调度器逻辑正确**
- ✅ **阶段切换**: 每2步切换一次 (步骤2,4,6,8...)
- ✅ **智能体切换**: 每步切换智能体 (EDU→COUNCIL→IND→IND→EDU→COUNCIL...)
- ✅ **执行模式**: 正确的并发/顺序模式

### **3. 环境实现有问题**
```python
def _should_switch_agent(self) -> bool:
    """检查是否需要切换智能体"""
    # 简化实现：每个智能体执行一步后切换
    return True  # ❌ 问题在这里！
```

**问题**: 环境每步都切换智能体，没有考虑调度器的阶段逻辑！

## 🔧 问题详细分析

### **当前环境行为**
```
步骤0: 智能体=EDU (阶段0)
步骤1: 智能体=COUNCIL (阶段0) ← 正确
步骤2: 智能体=IND (阶段1) ← 正确
步骤3: 智能体=IND (阶段1) ← 正确
步骤4: 智能体=EDU (阶段0) ← 正确
步骤5: 智能体=COUNCIL (阶段0) ← 正确
```

### **你期望的行为**
```
月份0: IND执行
月份1: EDU+COUNCIL同时执行
月份2: IND执行
月份3: EDU+COUNCIL同时执行
...
```

### **实际发生的行为**
```
步骤0: EDU执行
步骤1: COUNCIL执行
步骤2: IND执行
步骤3: IND执行
步骤4: EDU执行
步骤5: COUNCIL执行
...
```

## 🎯 问题总结

### **配置是正确的**
- ✅ 调度器配置: period=2, phases=[EDU+COUNCIL, IND]
- ✅ 阶段逻辑: 每2步切换阶段
- ✅ 执行模式: 并发/顺序正确

### **环境实现有问题**
- ❌ **每步切换智能体**: 没有考虑阶段逻辑
- ❌ **智能体轮换**: 每步都调用`_switch_agent()`
- ❌ **阶段忽略**: 没有使用调度器的阶段信息

### **实际执行逻辑**
```
环境每步都切换智能体 → 每个智能体执行一步 → 看起来像"每个agent一个月"
```

## 💡 解决方案

### **方案1: 修复环境智能体切换逻辑**
```python
def _should_switch_agent(self) -> bool:
    """检查是否需要切换智能体"""
    # 获取当前阶段的活跃智能体
    active_agents = self.scheduler.get_active_agents(self.current_step)
    
    # 如果当前智能体不在活跃列表中，需要切换
    if self.current_agent not in active_agents:
        return True
    
    # 如果当前阶段有多个智能体，需要轮换
    if len(active_agents) > 1:
        return True
    
    # 否则不需要切换
    return False
```

### **方案2: 修复智能体切换逻辑**
```python
def _switch_agent(self):
    """切换智能体"""
    # 获取当前阶段的活跃智能体
    active_agents = self.scheduler.get_active_agents(self.current_step)
    
    if not active_agents:
        return
    
    # 如果当前智能体不在活跃列表中，选择第一个
    if self.current_agent not in active_agents:
        self.current_agent = active_agents[0]
        return
    
    # 轮换到下一个活跃智能体
    current_index = active_agents.index(self.current_agent)
    next_index = (current_index + 1) % len(active_agents)
    self.current_agent = active_agents[next_index]
```

## 📝 总结

**你的配置是正确的**，问题在于环境实现：

1. **调度器配置**: ✅ 正确 (period=2, phases=[EDU+COUNCIL, IND])
2. **调度器逻辑**: ✅ 正确 (每2步切换阶段)
3. **环境实现**: ❌ 有问题 (每步都切换智能体)

**解决方案**: 修复环境的智能体切换逻辑，使其遵循调度器的阶段逻辑。

**你的印象是对的**，配置确实是按照月份轮次设计的！ 🎯

