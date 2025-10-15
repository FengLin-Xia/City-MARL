# Actor LR快速验证测试

## 🎯 测试目标
验证KL高是否是因为actor_lr太大

---

## 🔧 修改
```json
"actor_lr": 5e-5 → 1e-5  // 降低80%
```

**只改这一个参数，其他全部保持**

---

## 📊 预期结果

### 如果actor_lr是问题：
```
Episode 5:
  KL: 1.0-2.0 (vs 之前4.8)  ← 关键改善
  Clip: 60-80% (vs 之前100%)
  Value Loss: 300-500 (保持或略好)
```

### 如果actor_lr不是问题：
```
Episode 5:
  KL: 仍然>4.0
  Clip: 仍然>90%
  
→ 需要找其他原因
```

---

## 🚀 测试计划

**训练5 episodes**：
```bash
# 删除旧模型
rm models/v4_1_rl/*.pth

# 训练
python enhanced_city_simulation_v4_1.py --mode rl
```

**观察Episode 3-5的KL**

---

**测试时间**: 2025-10-12  
**状态**: actor_lr已改为1e-5，准备测试



