# v5.0训练指标记录功能增强

## 概述

为v5.0的PPO训练器添加了详细的强化学习训练参数记录功能，参考v4.1版本的成熟实现。

## 新增的训练指标

### 1. 基础训练指标
- **total_loss**: 总损失
- **actor_loss**: Actor网络损失（策略损失）
- **critic_loss**: Critic网络损失（价值损失）
- **entropy_loss**: 熵损失

### 2. 详细训练指标
- **kl_divergence**: KL散度（策略变化程度）
- **clip_fraction**: 裁剪比例（PPO裁剪激活比例）
- **entropy**: 策略熵值（探索程度）
- **ratio_mean**: 策略比率均值
- **temperature**: 当前温度（探索参数）
- **total_steps**: 总训练步数
- **current_update**: 当前更新次数
- **num_agents**: 参与训练的智能体数量
- **total_experiences**: 总经验数量

### 3. 智能体级别指标
每个智能体都会记录：
- 独立的KL散度、裁剪比例、熵值
- 独立的Actor和Critic损失
- 策略比率统计

## 实现细节

### 1. 训练指标计算
```python
# KL散度计算
kl_div = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean()

# 裁剪比例计算
clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean()

# 熵值计算
entropy = -(torch.softmax(masked_logits, dim=-1) * logp_all).sum(dim=-1).mean()
```

### 2. 日志记录
- 使用`topic_enabled("training_step")`控制日志输出
- 记录智能体级别的详细指标
- 记录训练统计摘要

### 3. 训练进度显示
增强的进度显示包含：
- 所有损失指标
- 熵值和温度
- 更新次数
- 训练统计摘要

## 使用方法

### 1. 运行训练
```python
from trainers.v5_0.ppo_trainer import V5PPOTrainer

trainer = V5PPOTrainer("configs/city_config_v5_0.json")
result = trainer.train(num_episodes=100)
```

### 2. 查看训练指标
```python
# 获取训练历史
training_history = result['training_history']

# 查看最新训练指标
latest_stats = training_history[-1]
print(f"KL散度: {latest_stats['kl_divergence']:.4f}")
print(f"裁剪比例: {latest_stats['clip_fraction']:.4f}")
print(f"熵值: {latest_stats['entropy']:.4f}")
```

### 3. 测试功能
```bash
python test_training_metrics.py
```

## 监控建议

### 1. 健康训练指标
- **KL散度**: 0.01 - 0.1 (策略变化适中)
- **裁剪比例**: 10% - 50% (PPO裁剪正常工作)
- **熵值**: 0.5 - 2.0 (保持适当探索)
- **温度**: 逐渐从高到低 (探索到利用)

### 2. 异常指标
- **KL散度 > 0.2**: 策略变化过大，可能不稳定
- **裁剪比例 > 80%**: PPO裁剪过度激活
- **熵值 < 0.1**: 策略过度确定性，可能过早收敛
- **熵值 > 3.0**: 策略过于随机

## 与v4.1的对比

| 指标 | v4.1 | v5.0 | 说明 |
|------|------|------|------|
| KL散度 | ✅ | ✅ | 完全一致的计算方法 |
| 裁剪比例 | ✅ | ✅ | 完全一致的计算方法 |
| 熵值 | ✅ | ✅ | 完全一致的计算方法 |
| 优势统计 | ✅ | ✅ | 均值和标准差 |
| 比率统计 | ✅ | ✅ | 均值和标准差 |
| 智能体级别 | ✅ | ✅ | 分智能体记录 |
| 温度退火 | ❌ | ✅ | v5.0新增功能 |

## 下一步计划

1. **统一奖励计算**: 使用V5RewardCalculator替代V5ActionScorer
2. **参数调整**: 根据新的训练指标调整超参数
3. **稳定性监控**: 基于KL散度和熵值监控训练稳定性
4. **性能优化**: 根据训练指标优化学习率和熵系数

## 相关文件

- `trainers/v5_0/ppo_trainer.py`: 主要实现文件
- `test_training_metrics.py`: 测试脚本
- `configs/city_config_v5_0.json`: 配置文件
- `utils/logger_factory.py`: 日志系统
