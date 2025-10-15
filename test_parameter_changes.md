# PPO测试参数修改记录

## 概述
在调试PPO KL散度为0的问题过程中，我们对多个参数进行了硬修改以进行测试。本文档记录这些修改。

## 修改的参数

### 1. 奖励缩放相关
**文件**: `configs/city_config_v4_1.json`

#### 原始值 → 修改值
- `reward_scale`: `3000.0` → `30000.0`
  - **原因**: 原始缩放因子太小，导致所有动作的奖励都被clip到最大值1.0，无法体现差异
  - **效果**: 修复了奖励差异问题，不同动作类型现在有显著奖励差异
    - IND_S: 0.025, IND_M: 0.357, IND_L: 0.851
    - EDU_S: 0.056, EDU_M: 0.508, EDU_L: 0.977

### 2. 学习率相关
**文件**: `configs/city_config_v4_1.json`

#### Actor学习率
- `actor_lr`: `5e-5` → `1e-3`
  - **原因**: 尝试"点燃"训练，让策略更快学习
  - **效果**: 无显著改善

#### Critic学习率
- `critic_lr`: `3e-4` → `0.0` → `0.0001`
  - **原因**: 最初设置为0是为了禁用Critic学习，避免"完美预测"问题
  - **效果**: 禁用Critic学习后发现问题不在Critic，而在于advantages标准化
  - **恢复**: 现在已恢复为0.0001，让Critic能正常学习提供更好的价值估计

### 3. GAE参数
**文件**: `configs/city_config_v4_1.json`

- `gae_lambda`: `0.95` → `0.8`
  - **原因**: 尝试调整GAE参数，让advantages更有"惊喜感"
  - **效果**: 无显著改善

### 4. 训练相关参数
**文件**: `configs/city_config_v4_1.json`

- `ent_coef`: `0.04` → `0.0`
  - **原因**: 暂时关闭熵奖励，专注于策略学习
  - **效果**: 无显著改善

- `num_epochs`: `3` → `8`
  - **原因**: 增加训练轮数，让策略有更多更新机会
  - **效果**: 无显著改善

- `rollout_steps`: `10` → `20`
  - **原因**: 增加rollout步数，收集更多经验
  - **效果**: 无显著改善

- `mini_batch_size`: `10` → `32`
  - **原因**: 增加批次大小，提高训练稳定性
  - **效果**: 无显著改善

- `vf_coef`: `0.5` → `0.1`
  - **原因**: 降低value loss权重，专注于策略学习
  - **效果**: 无显著改善

### 5. 动作序列相关
**文件**: `configs/city_config_v4_1.json`

- `length_max`: `5` → `1`
  - **原因**: 禁用"一次盖多个"功能，简化动作空间
  - **效果**: 无显著改善

## 当前状态

### 已修复的问题
1. ✅ **奖励差异问题** - 通过调整reward_scale，不同动作类型现在有显著奖励差异
2. ✅ **动作因果性** - 验证了动作确实影响回报
3. ✅ **PPO算法实现** - 修复了mask ratio consistency等问题
4. ✅ **KL散度为0问题** - 发现并修复了advantages标准化导致的KL散度为0问题

### 仍存在的问题
1. ✅ **KL散度为0** - 已修复！通过注释掉advantages标准化解决
2. ✅ **advantages接近0** - 已修复！advantages.mean()现在为9.18
3. ✅ **policy_loss接近0** - 已修复！policy_loss现在为-9.74

### 关键发现
- **真正的问题**: advantages标准化 `(advantages - advantages.mean()) / (advantages.std() + 1e-8)` 强制将均值设为0
- **修复方法**: 注释掉advantages标准化，保留GAE计算的原始advantages
- **修复结果**: 
  - `advantages.mean()` 从 ~3.97e-09 → 9.18
  - `KL散度` 从 0.0000 → 0.0008
  - `policy_loss` 从 ~0.0000 → -9.74

## 下一步计划

1. **检查GAE具体实现** - 查看advantages计算的代码逻辑
2. **尝试直接使用returns** - 绕过GAE计算，直接使用returns作为advantages
3. **考虑简化方法** - 使用更简单的策略梯度方法

## 文件修改清单

### 主要修改的文件
- `configs/city_config_v4_1.json` - 配置文件参数修改
- `trainers/v4_1/ppo_trainer.py` - PPO训练逻辑修复
- `solvers/v4_1/rl_selector.py` - RL选择器修复
- `test_causality.py` - 因果性测试脚本
- `test_different_action_types.py` - 动作类型差异测试脚本

### 新增的测试文件
- `test_parameter_changes.md` - 本文档
- `test_causality.py` - 因果性测试
- `test_different_action_types.py` - 动作类型差异测试
- `test_ppo_rewards.py` - PPO奖励流测试

## 注意事项

1. **这些修改都是为了调试目的**，生产环境需要恢复原始参数
2. **reward_scale的修改是有效的**，应该保留
3. **其他参数修改的效果有限**，可能需要恢复
4. **下一步重点检查GAE实现**，这可能是根本问题所在
