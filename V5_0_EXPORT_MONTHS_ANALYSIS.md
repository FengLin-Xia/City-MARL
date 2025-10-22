# v5.0 导出月份数据问题分析

## 🔍 问题描述

**现象**: v5.0训练后只导出了3个月的数据，而不是配置的30个月。

**配置**: `total_steps: 30` (30个月)

**实际导出**: 只有 month_00, month_01, month_02 三个文件

## 📊 问题分析

### **1. 时间步与月份的关系**

#### **环境配置**
```json
"time_model": { "step_unit": "month", "total_steps": 30 }
```

#### **环境实现逻辑**
```python
def _should_switch_month(self) -> bool:
    """检查是否需要切换月份"""
    # 简化实现：每12步切换一个月
    return self.current_step % 12 == 0
```

#### **问题根源**
- **配置**: 30个月 = 30个时间单位
- **实现**: 每12步 = 1个月
- **结果**: 30步 ÷ 12 = 2.5个月 (实际只有2-3个月)

### **2. 训练器收集经验**

#### **训练器逻辑**
```python
def collect_experience(self, num_steps: int) -> List[Dict]:
    # 收集30步经验
    experiences = self.trainer.collect_experience(30)
```

#### **环境响应**
- **步骤0-11**: 月份0
- **步骤12-23**: 月份1  
- **步骤24-35**: 月份2
- **步骤36+**: 月份3+ (但只收集到30步)

### **3. 导出系统处理**

#### **分组逻辑**
```python
def _group_by_month(self, step_logs, env_states):
    for log, state in zip(step_logs, env_states):
        month = state.month  # 使用环境状态的月份
        # 只有月份0, 1, 2的数据
```

## 🎯 解决方案

### **方案1: 修正环境月份切换逻辑**

#### **当前实现**
```python
def _should_switch_month(self) -> bool:
    return self.current_step % 12 == 0  # 每12步一个月
```

#### **修正实现**
```python
def _should_switch_month(self) -> bool:
    # 每1步一个月 (符合配置)
    return True
```

### **方案2: 修正配置理解**

#### **当前理解**
- `total_steps: 30` = 30个月
- 每12步 = 1个月
- 30步 = 2.5个月

#### **正确理解**
- `total_steps: 30` = 30步
- 每1步 = 1个月  
- 30步 = 30个月

### **方案3: 配置驱动月份切换**

#### **配置添加**
```json
"env": {
  "time_model": { 
    "step_unit": "month", 
    "total_steps": 30,
    "steps_per_month": 1  // 新增：每步一个月
  }
}
```

#### **环境实现**
```python
def _should_switch_month(self) -> bool:
    steps_per_month = self.config.get("env", {}).get("time_model", {}).get("steps_per_month", 1)
    return self.current_step % steps_per_month == 0
```

## 🔧 推荐修复

### **立即修复 (方案1)**

修改 `envs/v5_0/city_env.py`:

```python
def _should_switch_month(self) -> bool:
    """检查是否需要切换月份"""
    # 每1步切换一个月 (符合total_steps配置)
    return True
```

### **长期优化 (方案3)**

1. **添加配置参数**:
   ```json
   "env": {
     "time_model": { 
       "step_unit": "month", 
       "total_steps": 30,
       "steps_per_month": 1
     }
   }
   ```

2. **修改环境逻辑**:
   ```python
   def _should_switch_month(self) -> bool:
       steps_per_month = self.config.get("env", {}).get("time_model", {}).get("steps_per_month", 1)
       return self.current_step % steps_per_month == 0
   ```

## 📈 预期结果

### **修复后**
- **30步经验** → **30个月数据**
- **导出文件**: month_00 到 month_29
- **完全符合配置**: `total_steps: 30`

### **验证方法**
```bash
python enhanced_city_simulation_v5_0.py --episodes 1 --verbose
# 应该看到: Exported month 0 到 month 29
```

## 🎯 总结

**问题根源**: 环境月份切换逻辑与配置不匹配
- **配置**: 30个月
- **实现**: 每12步1个月
- **结果**: 只有2-3个月数据

**解决方案**: 修改环境月份切换逻辑，使其与配置一致
- **每1步**: 1个月
- **30步**: 30个月
- **完全符合**: `total_steps: 30` 配置

**这是配置与实现不一致的典型问题！** 🎯

