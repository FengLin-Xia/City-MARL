# 硬编码技术债务记录 (Hardcode Technical Debt)

## 概述
本文档记录了在添加新建筑类型(A/B/C)过程中发现的硬编码问题，以及相应的修复方案。这些硬编码问题会导致系统难以扩展，需要系统性的重构来解决。

## 问题分类

### 1. 配置文件硬编码
**问题**: 配置文件中的建筑类型定义不完整
**影响文件**: `configs/city_config_v4_1.json`
**具体问题**:
- `caps.top_slots_per_agent_size.IND` 缺少新建筑类型
- `caps.max_actions_per_agent_size.IND` 缺少新建筑类型
- 经济参数配置不完整

**修复方案**:
```json
"top_slots_per_agent_size": {
  "EDU": {"S": 150, "M": 150, "L": 150},
  "IND": {"L": 60, "A": 60, "B": 60, "C": 60}
}
```

### 2. 动作枚举器硬编码
**问题**: 动作枚举器中的建筑类型硬编码
**影响文件**: `logic/v4_enumeration.py`
**具体问题**:
- `V4Planner.plan` 方法中硬编码 `sizes` 参数
- 足迹规则处理不完整

**修复方案**:
```python
sizes = sizes or {'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L', 'A', 'B', 'C']}
```

### 3. RL选择器硬编码
**问题**: RL选择器中的建筑类型处理不完整
**影响文件**: `solvers/v4_1/rl_selector.py`
**具体问题**:
- `choose_action_sequence` 方法中硬编码 `sizes` 参数
- `_prioritize_high_level_slots` 方法中 `size_priority` 字典缺少新类型
- `_limit_s_size_actions` 方法只处理 S/M/L，忽略新类型

**修复方案**:
```python
# 1. 更新 sizes 参数
sizes=sizes or {'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L', 'A', 'B', 'C']}

# 2. 更新 size_priority 字典
size_priority = {'L': 3, 'M': 2, 'S': 1, 'A': 1, 'B': 2, 'C': 3}.get(action.size, 1)

# 3. 更新 _limit_s_size_actions 方法
a_actions = [a for a in actions if a.size == 'A']
b_actions = [a for a in actions if a.size == 'B']
c_actions = [a for a in actions if a.size == 'C']
balanced_actions = s_actions + m_actions + l_actions + a_actions + b_actions + c_actions
```

### 4. 训练器硬编码
**问题**: PPO训练器中的建筑类型硬编码
**影响文件**: `trainers/v4_1/ppo_trainer.py`
**具体问题**:
- `collect_experience` 方法中硬编码 `sizes` 参数

**修复方案**:
```python
sizes={'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L', 'A', 'B', 'C']}
```

### 5. 环境硬编码
**问题**: 城市环境中的建筑类型硬编码
**影响文件**: `envs/v4_1/city_env.py`
**具体问题**:
- `get_action_pool` 方法中硬编码 `sizes` 参数

**修复方案**:
```python
sizes={'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L', 'A', 'B', 'C']}
```

### 6. 主程序硬编码
**问题**: 主程序中的建筑类型硬编码
**影响文件**: `enhanced_city_simulation_v4_1.py`
**具体问题**:
- `run_rl_mode` 函数中硬编码 `sizes` 参数

**修复方案**:
```python
sizes={'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L', 'A', 'B', 'C']}
```

### 7. 可视化脚本硬编码
**问题**: 可视化脚本中的建筑类型硬编码
**影响文件**: `visualize_best_results.py`, `visualize_city_layout.py`
**具体问题**:
- 硬编码 `sizes` 参数

**修复方案**:
```python
sizes={'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L', 'A', 'B', 'C']}
```

## 根本原因分析

### 1. 缺乏统一的配置管理
- 建筑类型定义分散在各个文件中
- 没有统一的配置管理机制
- 新增建筑类型需要手动修改多个文件

### 2. 硬编码参数传递
- 大量 `sizes` 参数硬编码在方法调用中
- 缺乏动态参数读取机制
- 参数传递链路过长，容易遗漏

### 3. 方法设计不够通用
- `_limit_s_size_actions` 等方法只处理特定建筑类型
- 缺乏通用的建筑类型处理逻辑
- 新建筑类型需要修改核心逻辑

## 技术债务影响

### 1. 开发效率低下
- 每次添加新建筑类型需要修改多个文件
- 容易遗漏某些硬编码位置
- 调试成本高

### 2. 维护困难
- 硬编码分散，难以统一管理
- 修改风险高，容易引入bug
- 代码可读性差

### 3. 扩展性差
- 难以快速添加新建筑类型
- 系统架构不够灵活
- 未来扩展成本高

## 重构建议

### 1. 统一配置管理
```python
# 建议创建统一的配置管理器
class BuildingTypeConfig:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
    
    def get_building_types(self, agent_type):
        return self.config['building_types'][agent_type]
    
    def get_caps(self, agent_type):
        return self.config['caps'][agent_type]
```

### 2. 动态参数传递
```python
# 建议使用配置驱动的参数传递
def enumerate_actions(self, config_manager, agent_type):
    sizes = config_manager.get_building_types(agent_type)
    caps = config_manager.get_caps(agent_type)
    # 使用动态参数而不是硬编码
```

### 3. 通用化处理逻辑
```python
# 建议创建通用的建筑类型处理逻辑
class BuildingTypeProcessor:
    def __init__(self, building_types):
        self.building_types = building_types
    
    def process_actions(self, actions):
        # 通用处理逻辑，自动适应所有建筑类型
        pass
```

### 4. 测试覆盖
```python
# 建议添加单元测试确保新建筑类型正常工作
def test_new_building_types():
    # 测试新建筑类型是否能正常枚举
    # 测试新建筑类型是否能正常计算得分
    # 测试新建筑类型是否能正常选择
    pass
```

## 修复优先级

### 高优先级
1. 统一配置管理 - 解决根本问题
2. 动态参数传递 - 减少硬编码
3. 通用化处理逻辑 - 提高扩展性

### 中优先级
1. 添加单元测试 - 确保质量
2. 重构核心方法 - 提高可维护性
3. 文档更新 - 提高可读性

### 低优先级
1. 性能优化 - 提高效率
2. 代码风格统一 - 提高可读性
3. 注释完善 - 提高可维护性

## 总结

硬编码问题是一个系统性的技术债务，需要从架构层面进行重构。建议：

1. **短期**: 修复当前硬编码问题，确保系统正常运行
2. **中期**: 实施统一配置管理和动态参数传递
3. **长期**: 重构核心架构，提高系统的可扩展性和可维护性

通过系统性的重构，可以显著提高开发效率，降低维护成本，为未来的功能扩展奠定良好基础。
