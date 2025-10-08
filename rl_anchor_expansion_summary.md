# PPO Anchor + Fixed Expansion 方案设计摘要

## 一、核心思路
在保持现有 PPO 训练逻辑不变的前提下，通过引入 **Anchor + Fixed Expansion Policy** 模式，让智能体在执行阶段能够生成多槽位（multi-slot）的动作序列。

- **训练阶段**：仍然只训练单个锚点（anchor slot）的选择策略。
- **执行阶段**：根据锚点，通过一个固定的扩展规则（ExpansionPolicy）生成完整序列。

> ✅ 不改 PPO 核心逻辑，✅ 保持训练稳定，✅ 实现多槽布局效果。

---

## 二、架构概览

```
RL Policy Selector (Actor)
│
├── 训练阶段
│     1. 选出 anchor 槽位
│     2. ExpansionPolicy.expand(state, anchor) → 多槽序列
│     3. 环境执行序列动作
│     4. PPO 更新仅基于 anchor 的 log_prob
│
└── 执行阶段
      1. 载入已训练 actor
      2. anchor = argmax(logits)
      3. ExpansionPolicy.expand(state, anchor)
      4. 执行多槽序列动作
```

---

## 三、代码实现要点

### 1️⃣ ExpansionPolicy 模板
```python
class ExpansionPolicy:
    def __init__(self, temperature=1.0, rule="nearest-k"):
        self.temperature = temperature
        self.rule = rule

    def expand(self, state, anchor_slot, k=5):
        # 示例规则：选择距离 anchor 最近的 k 个可用槽位
        candidates = get_valid_slots(state)
        scores = [distance(anchor_slot, c) for c in candidates]
        selected = sorted(candidates, key=lambda x: scores[x])[:k]
        return selected, 0.0  # logprob 常数
```

### 2️⃣ 采样阶段修改
```python
logits, _ = actor(state_embed)
dist = Categorical(logits=logits / temperature)
anchor = dist.sample()
old_logp_anchor = dist.log_prob(anchor)

sequence, _ = expansion.expand(state, anchor, k=5)

experience = {
    "anchor": anchor.item(),
    "old_logp": old_logp_anchor.item(),
    "reward": reward,
}
```

### 3️⃣ PPO 更新阶段
```python
old_logps = torch.tensor([e['old_logp'] for e in experiences]).to(device)
actions   = torch.tensor([e['anchor'] for e in experiences]).to(device)

logits, _ = actor(states)
dist = Categorical(logits=logits)
current_logps = dist.log_prob(actions)

ratio = torch.exp(current_logps - old_logps)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

> 所有 PPO 机制（clip、KL、entropy、value_loss）保持不变。

---

## 四、执行阶段逻辑
```python
actor.eval()
with torch.no_grad():
    logits, _ = actor(state_embed)
    anchor = torch.argmax(logits)

sequence, _ = expansion.expand(state, anchor, k=5)
state, reward, done = env.step(sequence)
```

---

## 五、优点
| 项目 | 说明 |
|------|------|
| ✅ 稳定性 | PPO 更新逻辑完全保持，KL/clip 不受扩展影响 |
| ✅ 简单实现 | 不改 actor/critic 结构，只新增 ExpansionPolicy |
| ✅ 可扩展 | 后续可替换为自回归或分层扩展策略 |
| ✅ 一致性 | 训练和执行使用同一扩展规则，性能稳定 |

---

## 六、扩展与升级方向
- **可变长度序列**：加入 early-stop 或 STOP 动作（见后续方案 B）。
- **自回归扩展**：让策略在子步层面决定是否继续扩展。
- **分层 RL**：高层选模板，低层填槽位。

---

## 七、集成建议
- 将本逻辑单独放入 `expansion_policy.py`；
- 在 `collect_experience()` 调用 ExpansionPolicy；
- `update_policy()` 逻辑不动；
- 执行时添加 `run_trained_policy()` 入口：加载模型 → anchor → expand → step。

