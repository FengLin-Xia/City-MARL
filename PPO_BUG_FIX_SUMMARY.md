# PPO训练Bug修复总结

**日期：** 2025-10-09  
**状态：** ✅ 已完成并测试  

---

## 🐛 **发现的Bug**

### **Bug #1: num_actions不一致** 🔥🔥🔥🔥🔥

**位置：** `trainers/v4_1/ppo_trainer.py`

**问题：**
```python
# 收集经验时（第198行）：
num_actions = min(5, max_actions)  # 硬编码5

# 更新策略时（第346行）：
num_actions = min(len(exp['available_actions']), max_actions)  # 使用真实数量
```

**影响：**
- 收集时用5个动作计算log_prob
- 更新时用15个动作重新计算
- 概率分布完全不同
- **导致clip率99%！**

**修复：**
```python
# 1. 修改函数签名，接受num_actions参数
def _get_action_log_prob(self, sequence, state, num_actions):
    num_actions = min(num_actions, self.selector.max_actions)
    valid_logits = logits[0, :num_actions]
    # ...

# 2. 调用时传入真实数量
old_log_prob = self._get_action_log_prob(selected_seq, state, len(actions))

# 3. 保存到经验中
experience['num_actions'] = len(actions)

# 4. 更新时使用保存的数量
num_actions = exp.get('num_actions', ...)
```

---

### **Bug #2: Value Loss爆炸** 🔥🔥🔥🔥

**位置：** `envs/v4_1/city_env.py`

**问题：**
```python
# Budget惩罚导致reward波动巨大：
正常: reward = +775, scaled = +7.75
负债-2000: reward = -225, scaled = -2.25
负债-5000: reward = -2080, scaled = -20.8  ← 爆炸！

# Value网络无法预测这种波动
# value_loss = 3203（正常应该<100）
```

**修复：**
```python
# 1. 增强缩放
scaled_reward = total_reward / 200.0  # 从100改为200

# 2. 添加clipping
scaled_reward = np.clip(scaled_reward, -10.0, 10.0)

# 3. 降低Budget惩罚系数
debt_penalty_coef: 0.5 → 0.3
```

---

### **Bug #3: KL散度为负** 🔥

**位置：** `trainers/v4_1/ppo_trainer.py`

**问题：**
```python
# 当前公式：
kl_div = (old_log_probs - current_log_probs).mean()

# 这不是真正的KL散度，只是简化近似
# 当new_prob > old_prob时，会出现负数
```

**修复：**
```python
# 使用正确的KL近似公式：
# KL(old||new) ≈ E[(ratio - 1) - log(ratio)]
kl_div = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean()

# 这个公式保证KL ≥ 0
```

---

## ✅ **修复验证**

### **测试结果：**
```
[PASS] num_actions一致性修复成功
  - 收集时: num_actions=7
  - 更新时: num_actions=7
  - 完全一致！

[PASS] Reward范围控制
  - 最极端情况: -3.625
  - 在[-10, +10]范围内

[PASS] KL散度为正数
  - 新公式计算: 0.0169
  - 符合理论预期
```

---

## 📊 **预期改进**

| 指标 | 修复前 | 修复后（预期） | 改善 |
|-----|--------|--------------|------|
| **value_loss** | 3203.81 | <500 | -85%+ |
| **clip_fraction** | 0.9936 (99%) | <0.3 (30%) | -70%+ |
| **KL散度** | -1.88 (负数) | >0 (正数) | ✓ |
| **训练稳定性** | 差 | 好 | ✓✓✓ |

---

## 🔧 **修改的文件**

### **1. trainers/v4_1/ppo_trainer.py**

**修改1:** 函数签名（第179行）
```python
- def _get_action_log_prob(self, sequence, state):
+ def _get_action_log_prob(self, sequence, state, num_actions):
```

**修改2:** 移除硬编码（第204行）
```python
- num_actions = min(5, self.selector.max_actions)
+ num_actions = min(num_actions, self.selector.max_actions)
```

**修改3:** 调用处传入真实数量（第149行）
```python
- old_log_prob = self._get_action_log_prob(selected_sequence, state)
+ old_log_prob = self._get_action_log_prob(selected_sequence, state, len(actions))
```

**修改4:** 保存num_actions（第158行）
```python
experience = {
    ...
+   'num_actions': len(actions)
}
```

**修改5:** 更新时使用保存的值（第352行）
```python
- num_actions = min(len(exp.get('available_actions', [])), ...)
+ num_actions = exp.get('num_actions', len(exp.get('available_actions', [])))
```

**修改6:** KL散度计算（第419行）
```python
- kl_div = (old_log_probs - current_log_probs).mean()
+ kl_div = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean()
```

**修改7:** 设备一致性（第210行）
```python
- log_prob = dist.log_prob(torch.tensor(valid_action_idx))
+ log_prob = dist.log_prob(torch.tensor(valid_action_idx).to(self.device))
```

### **2. envs/v4_1/city_env.py**

**修改1:** Reward clipping（第415行）
```python
scaled_reward = total_reward / 200.0
+ scaled_reward = np.clip(scaled_reward, -10.0, 10.0)
```

### **3. configs/city_config_v4_1.json**

**修改1:** 降低学习率
```python
- "lr": 1e-3
+ "lr": 3e-4
```

**修改2:** 降低Budget惩罚
```python
- "debt_penalty_coef": 0.5
+ "debt_penalty_coef": 0.3
```

---

## 🚀 **下一步：重新训练**

### **清理旧模型：**
```bash
rm -rf models/v4_1_rl/*.pth
rm -rf models/v4_1_rl/training_results_*.json
```

### **启动新训练：**
```bash
python enhanced_city_simulation_v4_1.py --mode rl
```

### **观察指标：**
```
第1个update应该看到:
  ✓ value_loss < 500 (而不是3203)
  ✓ clip_fraction < 0.5 (而不是0.99)
  ✓ KL > 0 (而不是负数)

如果看到这些改进 → bug修复成功！
```

---

## 📋 **修复总结**

✅ **num_actions一致性** - 收集和更新使用相同数量  
✅ **Reward clipping** - 防止极端值  
✅ **KL散度修正** - 使用正确公式  
✅ **设备一致性** - tensor放在正确的device  
✅ **测试验证** - 所有测试通过  

**修复完成！可以开始训练了！** 🎉

---

**修复者：** AI Assistant  
**测试者：** Fenglin  
**最后更新：** 2025-10-09




