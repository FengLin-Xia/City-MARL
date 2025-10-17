# 1016 待办事项 (A/B/C建筑类型功能扩展)

## 概述
基于当前A/B/C建筑类型已经成功集成到IND智能体的基础上，需要进一步扩展功能以提升系统多样性和灵活性。

## 待办事项

### 1. 让A/B/C建筑类型可以在河流两侧同时存在

**目标**: 允许A/B/C建筑类型不受河流连通域限制，可以在河流两侧自由放置

#### 1.1 影响范围
- **核心文件**: 2个
- **修改量**: ~30行代码
- **复杂度**: 低
- **风险等级**: 低

#### 1.2 具体修改点

**文件1: `envs/v4_1/city_env.py`**
- **位置**: 第303-321行
- **修改**: 在候选槽位过滤时检查建筑类型
- **代码量**: ~15行

```python
# 当前逻辑：IND只能在南岸
expected_comp = self.hub_components[agent_idx]

# 新逻辑：A/B/C可以在两岸，S/M/L仍受限制
if self.current_agent == 'IND':
    if self.current_size in ['A', 'B', 'C']:
        # A/B/C不受河流限制
        expected_comp = None  # 不限制连通域
    else:
        # S/M/L仍在南岸
        expected_comp = self.hub_components[1]
```

**文件2: `enhanced_city_simulation_v4_0.py`**
- **位置**: 第606-617行
- **修改**: 同步修改连通域过滤逻辑
- **代码量**: ~5行

```python
if turn_based:
    if active_agent == 'IND' and current_size in ['A', 'B', 'C']:
        target_comp = None  # A/B/C不受限制
    else:
        target_comp = hub1_comp if active_agent == 'IND' else hub2_comp
```

**辅助修改**: ~10行
- 添加配置参数支持
- 添加调试信息
- 更新注释

#### 1.3 实现方案
- **方案1**: 简单条件判断修改 (推荐)
- **方案2**: 配置驱动的灵活控制

#### 1.4 验收标准
- [ ] A/B/C建筑类型可以在河流两侧放置
- [ ] S/M/L建筑类型仍受河流限制
- [ ] 调试输出显示正确的连通域过滤信息
- [ ] 不影响现有训练结果

---

### 2. 将A/B/C建筑类型从IND转移到EDU

**目标**: 将A/B/C建筑类型从IND智能体转移到EDU智能体，实现建筑类型的重新分配

#### 2.1 影响范围
- **配置文件**: 1个
- **代码文件**: 9个
- **修改量**: ~100行代码
- **复杂度**: 中等
- **风险等级**: 中等

#### 2.2 具体修改点

**文件1: `configs/city_config_v4_1.json`**
- **修改**: 移动A/B/C参数从IND到EDU
- **代码量**: ~50行

```json
// 当前状态
"top_slots_per_agent_size": {
  "EDU": {"S": 150, "M": 150, "L": 150},
  "IND": {"L": 60, "A": 60, "B": 60, "C": 60}
}

// 目标状态
"top_slots_per_agent_size": {
  "EDU": {"S": 150, "M": 150, "L": 150, "A": 60, "B": 60, "C": 60},
  "IND": {"L": 60}
}
```

**需要迁移的经济参数**:
- `Base_IND` → `Base_EDU`
- `Add_IND` → `Add_EDU`
- `Capacity_IND` → `Capacity_EDU`
- `GFA_k` → `GFA_EDU`
- `PrestigeBase_IND` → `PrestigeBase_EDU`
- `Pollution_IND` → `Pollution_EDU`
- `OPEX_IND` → `OPEX_EDU`
- `RewardLP_k_IND_by_size` → `RewardLP_k_EDU_by_size`
- `RiverPmax_pct_IND_by_size` → `RiverPmax_pct_EDU_by_size`

**文件2-9: 硬编码参数修改 (8个文件)**
- **修改**: 更新所有硬编码的sizes参数
- **代码量**: ~40行

```python
# 从
{'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L', 'A', 'B', 'C']}
# 改为
{'EDU': ['S', 'M', 'L', 'A', 'B', 'C'], 'IND': ['L']}
```

**涉及文件**:
- `enhanced_city_simulation_v4_1.py`
- `solvers/v4_1/rl_selector.py`
- `trainers/v4_1/ppo_trainer.py`
- `logic/v4_enumeration.py`
- `envs/v4_1/city_env.py`
- `visualize_best_results.py`
- `visualize_city_layout.py`
- `enhanced_city_simulation_v4_0.py`

**文件10: `export_v4_1_rl_sequences_txt.py`**
- **修改**: 重新分配编码映射
- **代码量**: ~10行

```python
AGENT_SIZE_CODE: Dict[Tuple[str, str], int] = {
    ('EDU', 'S'): 0, ('EDU', 'M'): 1, ('EDU', 'L'): 2,
    ('EDU', 'A'): 3, ('EDU', 'B'): 4, ('EDU', 'C'): 5,  # 新增
    ('IND', 'L'): 6,  # 重新分配
}
```

#### 2.3 实施步骤
1. **备份当前配置**
2. **迁移经济参数配置**
3. **更新caps配置**
4. **修改硬编码参数**
5. **更新导出脚本**
6. **运行测试验证**

#### 2.4 验收标准
- [ ] EDU智能体可以选择A/B/C建筑类型
- [ ] IND智能体只能选择L建筑类型
- [ ] A/B/C建筑类型的经济参数正确应用到EDU
- [ ] 导出脚本正确显示新的编码映射
- [ ] 训练过程正常运行
- [ ] 可视化工具正确显示建筑类型

---

## 优先级排序

### 高优先级
1. **河流两侧同时存在** - 修改量小，风险低，收益明显

### 中优先级  
2. **A/B/C转移到EDU** - 修改量大，需要仔细处理配置迁移

## 风险评估

### 河流两侧同时存在
- **技术风险**: 低 - 主要是条件判断修改
- **业务风险**: 低 - 不影响现有功能
- **测试风险**: 低 - 容易验证

### A/B/C转移到EDU
- **技术风险**: 中等 - 配置迁移容易出错
- **业务风险**: 中等 - 可能影响训练结果
- **测试风险**: 中等 - 需要全面测试

## 时间估算

### 河流两侧同时存在
- **开发时间**: 0.5天
- **测试时间**: 0.5天
- **总计**: 1天

### A/B/C转移到EDU
- **开发时间**: 2天
- **测试时间**: 1天
- **总计**: 3天

## 依赖关系

- 两个任务可以并行开发
- 建议先完成河流两侧同时存在，再处理A/B/C转移
- A/B/C转移完成后，河流两侧同时存在功能仍然有效

## 备注

- 两个功能都是基于当前A/B/C成功集成的基础
- 建议在开发前先备份当前工作状态
- 每个任务完成后都要进行充分测试
- 可以考虑分阶段实施，降低风险
