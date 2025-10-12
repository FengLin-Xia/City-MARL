# Entropy参数调整

## 🎯 目标
降低Entropy，减少KL散度，稳定训练

---

## 📊 问题诊断

### Episode 10的状态
```
KL = 4.93 (太高，应该<2.0)
Clip = 100% (完全裁剪)
Entropy = 1.61 (过高)
Value Loss = 567 (还行，但KL是瓶颈)
```

**判断**：
- Entropy太高 → Policy过于随机
- Policy变化大 → KL高
- 需要降低Entropy

---

## 🔧 调整内容

| 参数 | 旧值 | 新值 | 变化 |
|------|------|------|------|
| **ent_coef** | 0.15 | **0.08** | 降47% |
| **entropy_coef** | 0.08 | **0.04** | 降50% |
| **temperature** | 3.5 | **2.5** | 降29% |

**理由**：
- 之前为了解决"过早收敛"，把Entropy提得太高了
- 现在需要找平衡点：保持探索但不过度

---

## 📈 预期效果

### Entropy预期
```
当前: Entropy = 1.61
调整后: Entropy = 1.0-1.3

→ 仍保持探索，但不过度
```

### KL预期
```
当前: KL = 4.93
调整后: 
  - Episode 15: KL = 2.0-3.0
  - Episode 20: KL = 1.0-2.0
  - Episode 30: KL = 0.5-1.0
```

### Clip预期
```
当前: Clip = 100%
调整后:
  - Episode 15: Clip = 70-90%
  - Episode 20: Clip = 50-70%
  - Episode 30: Clip = 30-50%
```

---

## ⚖️ 风险评估

### 风险：会不会又"过早收敛"？

**上次问题**：
```
ent_coef=0.05 → Entropy低 → Episode 2固定90.0
```

**这次**：
```
ent_coef=0.08（vs 上次0.05，高60%）

应该：
  - 不会像上次那样过早收敛
  - 但也不会像0.15那样KL爆炸
  - 是一个中间值
```

**如果仍然固定**：
- 那说明不是Entropy的问题
- 是共享网络的问题
- 需要MAPPO

---

## 🚀 测试计划

### **重新训练10 episodes**

```bash
# 删除当前模型
rm models/v4_1_rl/*.pth

# 重新训练
python enhanced_city_simulation_v4_1.py --mode rl
```

### 观察指标（Episode 10）

**成功标准**：
- ✅ KL < 3.0
- ✅ Clip < 90%
- ✅ Return有波动（不固定）
- ✅ Entropy = 1.0-1.3

**失败信号**：
- 🔴 KL > 4.0
- 🔴 Return又固定
- 🔴 Entropy < 0.5（过早收敛）

**如果失败** → 确认需要MAPPO

---

## 📝 下一步

**如果这次仍然KL>3.0（Episode 10时）**：

**那就实施MAPPO**：
- 独立的Actor/Critic for IND和EDU
- 预期：KL降到<1.0
- 训练稳定

**准备好MAPPO实施方案在todos中了。**

---

**调整日期**: 2025-10-12  
**参数版本**: v4.1_entropy_balanced  
**状态**: ✅ 已应用，待测试

