# V4.1 最终训练配置总结

## 📋 配置修改总览

基于50 episodes训练结果和专业建议，综合优化配置。

---

## 🎯 解决的问题

### 问题1：Value Loss过高（14000+）✅ 已解决
- **原因**：reward_scale太小（500）
- **修复**：reward_scale → 2000，critic_lr → 5e-4
- **效果**：Value Loss降到630（降95%）

### 问题2：过早收敛（Episode 2就固定90.0）⚠️ 待解决
- **原因**：Entropy太低，exploration不足
- **修复**：提高ent_coef, entropy_coef, temperature
- **预期**：策略持续探索，return有波动

### 问题3：100% S型建筑⚠️ 待解决
- **原因**：Size bonus不足
- **修复**：size_bonus大幅提高（M:1000, L:2000）
- **预期**：出现M/L型建筑

---

## ⚙️ 完整配置参数

### 核心RL参数

```json
{
  "solver": {
    "rl": {
      // === 学习率（稳定更新）===
      "actor_lr": 5e-5,          // 降低50%，更温和
      "critic_lr": 5e-4,         // 保持，Value需要快学
      
      // === PPO裁剪（收紧）===
      "clip_eps": 0.15,          // 从0.2降到0.15
      
      // === Exploration（增强）===
      "ent_coef": 0.15,          // 从0.05提高到0.15（3倍）
      "entropy_coef": 0.08,      // 从0.02提高到0.08（4倍）
      "temperature": 3.5,        // 从2.0提高到3.5
      
      // === 训练设置 ===
      "num_epochs": 3,           // 从2改到3（整除batch）
      "max_updates": 30,         // 从50降到30
      "rollout_steps": 10,
      "mini_batch_size": 10,
      
      // === Reward缩放 ===
      "reward_scale": 2000.0,    // 从500提高到2000
      "reward_clip": 1.0,        // 严格限制[-1,1]
      
      // === 其他 ===
      "gamma": 0.99,
      "gae_lambda": 0.95,
      "vf_coef": 0.5,
      "max_grad_norm": 0.5,
      "cooperation_lambda": 0.0,
    }
  }
}
```

### Size激励参数

```json
{
  "growth_v4_1": {
    "evaluation": {
      "size_bonus": {"S": 0, "M": 1000, "L": 2000},
      "proximity_reward": 1500.0,
    }
  }
}
```

---

## 📊 参数修改对比

| 参数 | 旧值 | 新值 | 变化 | 目的 |
|------|------|------|------|------|
| **actor_lr** | 1e-4 | 5e-5 | -50% | 稳定训练 |
| **critic_lr** | 1e-4 | 5e-4 | +400% | 加速Value学习 |
| **clip_eps** | 0.2 | 0.15 | -25% | 收紧更新 |
| **ent_coef** | 0.05 | 0.15 | +200% | 增强探索 |
| **entropy_coef** | 0.02 | 0.08 | +300% | 增强探索 |
| **temperature** | 2.0 | 3.5 | +75% | 增加随机性 |
| **num_epochs** | 2 | 3 | +50% | 整除优化 |
| **max_updates** | 50 | 30 | -40% | 适度训练 |
| **reward_scale** | 500 | 2000 | +300% | 降低Value Loss |
| **reward_clip** | 5.0 | 1.0 | -80% | 严格限制 |
| **size_bonus M** | 300 | 1000 | +233% | 鼓励M型 |
| **size_bonus L** | 800 | 2000 | +150% | 鼓励L型 |

---

## 🎯 预期训练效果

### Episode 1-10（探索阶段）
```
Return: 60-120（波动）
Value Loss: 500-1000
KL: 1.0-3.0
Clip: 60-90%
Entropy: 1.2-1.6
Size分布: 主要S，偶尔M
```

### Episode 10-20（学习阶段）
```
Return: 100-150（上升）
Value Loss: 200-500
KL: 0.5-1.5
Clip: 40-70%
Entropy: 1.0-1.4
Size分布: S为主，M增加，L开始出现
```

### Episode 20-30（收敛阶段）
```
Return: 150-200（稳定）
Value Loss: 100-300
KL: 0.1-0.5
Clip: 20-50%
Entropy: 0.8-1.2
Size分布: S 50-60%, M 20-30%, L 10-20%
```

---

## 🔬 关键改进点

### 1. **Value Loss修复** ✅
- reward_scale: 500 → 2000
- critic_lr: 1e-4 → 5e-4
- **效果**：Value Loss从14000降到630

### 2. **Exploration增强** 🆕
- ent_coef: 0.05 → 0.15
- entropy_coef: 0.02 → 0.08
- temperature: 2.0 → 3.5
- **目标**：打破Episode 2就固定的僵局

### 3. **更新稳定性** 🆕
- actor_lr: 1e-4 → 5e-5
- clip_eps: 0.2 → 0.15
- **目标**：降低KL波动

### 4. **Size多样性** 🆕
- size_bonus: {M:300, L:800} → {M:1000, L:2000}
- **目标**：鼓励建造M/L型

### 5. **训练效率** 🆕
- num_epochs: 2 → 3
- max_updates: 50 → 30
- **目标**：batch整除优化，适度训练量

---

## 🚀 训练命令

```bash
# 删除旧模型（重新训练）
rm models/v4_1_rl/*.pth

# 或备份
mv models/v4_1_rl models/v4_1_rl_backup_50ep

# 重新训练
python enhanced_city_simulation_v4_1.py --mode rl
```

---

## 📈 监控指标

训练时重点观察：

### 关键指标
1. **Value Loss**: 应该<1000，逐渐降到<300
2. **KL**: 应该<2.0，逐渐降到<0.5
3. **Clip**: 应该<80%，逐渐降到<40%
4. **Entropy**: 应该>1.0，保持探索
5. **Return**: 应该波动（60-200），不应该固定

### 成功标志
- ✅ Value Loss稳定在100-300
- ✅ Return在150-200范围
- ✅ 出现M/L型建筑（>10%）
- ✅ 建筑总数>50个

### 失败信号
- 🔴 Return再次固定（如都是120.0）
- 🔴 Value Loss>1000不降
- 🔴 仍然100% S型

---

## 📝 配置文件已更新

所有修改已应用到`configs/city_config_v4_1.json`

**准备好重新训练了！**

---

**配置版本**: v4.1_optimized  
**修改日期**: 2025-10-12  
**状态**: ✅ 已应用，待训练验证

