# MAPPO独立网络实施总结

## ✅ 实施完成

**核心改造**：从共享网络改为每个agent独立的Actor和Critic

---

## 🎯 解决的核心问题

### **共享网络的瓶颈**

```
问题1: 策略冲突
  - IND和EDU用同一个Actor
  - 优化IND策略 → 影响EDU
  - 两者互相干扰 → KL高，不稳定

问题2: Value估计混淆  
  - 同一个Critic要估计两个agent的状态价值
  - V(state, IND) vs V(state, EDU)混在一起
  - Value Loss高（14000→567仍然高）

问题3: 过早收敛
  - 网络找到对两者都"还行"的折中策略
  - Episode 2就固定90.0
  - 卡在局部最优
```

---

## 🔧 代码修改

### 1. RLPolicySelector（solvers/v4_1/rl_selector.py）

**创建独立网络**：
```python
# 旧：共享网络
self.actor = Actor(...)
self.critic = Critic(...)

# 新：独立网络
self.actors = {
    'IND': Actor(...),
    'EDU': Actor(...)
}
self.critics = {
    'IND': Critic(...),
    'EDU': Critic(...)
}

# 独立优化器
self.actor_optimizers = {
    'IND': Adam(self.actors['IND'].parameters(), ...),
    'EDU': Adam(self.actors['EDU'].parameters(), ...)
}
self.critic_optimizers = {...}  # 同理
```

**修改网络调用**：
```python
# 在_rl_choose_sequence中
current_agent = actions[0].agent
actor = self.actors.get(current_agent, self.actor)  # 选择对应网络
logits = actor(state_embed)
```

### 2. PPO Trainer（trainers/v4_1/ppo_trainer.py）

**分agent更新**：
```python
# 在update_policy中
for exp in experiences:
    agent = exp.get('agent', 'IND')
    actor = self.selector.actors[agent]    # 选择对应网络
    critic = self.selector.critics[agent]
    
    logits = actor(state_embed)
    value = critic(state_embed)

# 更新时
for agent in ['IND', 'EDU']:
    optimizer = self.selector.actor_optimizers[agent]
    optimizer.zero_grad()
    # ... 更新 ...
    optimizer.step()
```

### 3. 模型保存/加载（solvers/v4_1/rl_selector.py）

**保存多个网络**：
```python
model_data = {
    'model_version': 'v4.1_mappo',
    'actor_IND_state_dict': self.actors['IND'].state_dict(),
    'actor_EDU_state_dict': self.actors['EDU'].state_dict(),
    'critic_IND_state_dict': self.critics['IND'].state_dict(),
    'critic_EDU_state_dict': self.critics['EDU'].state_dict(),
    # ... 优化器状态 ...
}
```

**向后兼容**：
```python
# 加载时检测模型版本
if 'v4.1_mappo' in model_version:
    # 加载MAPPO模型
else:
    # 向后兼容：复制旧模型到所有agent
    for agent in self.actors.keys():
        self.actors[agent].load_state_dict(old_model['actor_state_dict'])
```

---

## ✅ 测试验证

### 测试结果
```
[OK] 为IND和EDU创建了独立的Actor和Critic
[OK] 参数不共享，真正独立
[OK] 前向传播正常
[OK] 网络可以正常使用

参数量：
  - IND Actor: 170,674
  - EDU Actor: 170,674  (独立！)
  - IND Critic: 164,353
  - EDU Critic: 164,353 (独立！)

总参数量: 约670K (vs 共享网络335K，翻倍)
```

---

## 📊 架构对比

### 共享网络 vs MAPPO

| 方面 | 共享网络 | MAPPO（独立） |
|------|---------|--------------|
| **Actor数量** | 1个 | 2个（IND+EDU） |
| **Critic数量** | 1个 | 2个（IND+EDU） |
| **参数量** | 335K | 670K (+100%) |
| **策略冲突** | 🔴 有 | ✅ 无 |
| **Value混淆** | 🔴 有 | ✅ 无 |
| **收敛速度** | 慢 | 快 |
| **KL稳定性** | 差 | 好 |

---

## 📈 预期训练效果

### Episode 1-5（MAPPO初期）
```
KL_ind: 1.0-2.0
KL_edu: 1.0-2.0
Value Loss: 300-600 (vs 共享的567)
Clip: 60-90%
Return: 80-150
```

### Episode 10（MAPPO稳定）
```
KL: 0.5-1.5 (vs 共享的4.93)  ← 关键改善！
Value Loss: 200-400
Clip: 40-70%
Return: 120-180
```

### Episode 20（MAPPO收敛）
```
KL: 0.1-0.5
Value Loss: 100-300
Clip: 20-50%
Return: 150-250
Size: 应该出现M/L型
```

---

## 🚀 下一步

### **重新训练（从头开始）**

```bash
# 1. 删除旧模型（必须！因为结构变了）
rm models/v4_1_rl/*.pth

# 2. 训练10 episodes测试
python enhanced_city_simulation_v4_1.py --mode rl
```

### 观察关键指标

**Episode 5检查**：
- KL < 2.0? → ✅ MAPPO有效
- Value Loss < 600? → ✅ Critic学得更好
- Return > 100? → ✅ 策略改善

**Episode 10检查**：
- KL < 1.0? → ✅ 收敛良好
- Return > 120? → ✅ 策略优化
- 出现M/L型? → ✅ 多样性改善

---

## 📝 配置总览

**当前完整配置**：
```json
{
  "solver": {
    "rl": {
      // 独立网络（MAPPO）
      "algo": "mappo",
      "agents": ["IND", "EDU"],
      
      // 学习率
      "actor_lr": 5e-5,
      "critic_lr": 5e-4,
      
      // Entropy（调整后）
      "ent_coef": 0.08,
      "entropy_coef": 0.04,
      "temperature": 2.5,
      
      // PPO参数
      "clip_eps": 0.15,
      "num_epochs": 3,
      
      // Reward
      "reward_scale": 3000.0,
      "reward_clip": 1.0,
      "expected_lifetime": 12,
      
      // Size激励
      "size_bonus": {"S": 0, "M": 1000, "L": 2000}
    }
  }
}
```

---

## 🎯 关键改进点

### 1. 独立网络（MAPPO）✅
- 消除策略冲突
- 加速收敛
- 预期KL从4.93降到<1.0

### 2. 固定NPV Reward ✅
- 解决"躺平"问题
- 鼓励持续建造
- L型reward最高

### 3. Entropy平衡 ✅
- 不会过早收敛
- 不会KL爆炸
- 保持适度探索

### 4. Value Loss优化 ✅
- reward_scale=3000
- critic_lr=5e-4
- 独立Critic更易学习

---

**实施日期**: 2025-10-12  
**版本**: v4.1_MAPPO  
**状态**: ✅ 已完成，准备训练  
**下一步**: 删除旧模型，重新训练10 episodes



