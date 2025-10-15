
# PRD（Cursor对接版）：PPO/MAPPO 掩码一致性 & log_prob / ratio / KL 修复指南

> 目标：用**最小侵入**改动，修复你当前“`clip_fraction = 1.0`、`approx_kl ≫ 0.1`、`entropy ≈ log(|A|)` 恒定、`ratio ≫ 1`”的**结构性问题**：更新阶段与采样阶段的**分布/掩码/动作索引不一致**，以及 `old_log_prob` / `new_log_prob` 的错误使用。本文给出**面向 Cursor 的完整实现步骤**、代码段、断言与自测脚本，落地即可复现“回到健康区间”的训练曲线。

---

## 0. 现象复盘（来自你的日志）

- `ratio.mean ≈ 8.26`，`ratio.std ≈ 2.76`
- `clip_fraction = 1.0000`
- `approx_kl ≈ 5.23`
- `entropy.mean ≈ 1.60942 ≈ log(5)`（动作数≈5）
- `new_log_prob[:5] ≈ -1.61`（= -log(5)，**新分布近似均匀**）
- `valid_mask[:5] = [[True], [False], [False], [False], [True]]`（**更新阶段把已采样动作判为非法**）

**结论**：更新阶段在**另一套分布/掩码**上计算 `new_log_prob`（而且是均匀分布），与采样时不一致 ⇒ `ratio/kl/clip` 失真，调参无效。

---

## 1. 范围（Scope）

- ✅ 修复：
  - 采样阶段 *vs.* 更新阶段的**掩码一致性**（mask on logits）
  - **旧分布**下动作的 `old_log_prob` **缓存与 detach**
  - **回放同一动作索引**，在**同一掩码**下计算 `new_log_prob`
  - **ratio / approx_kl / entropy** 统一来自同一分布对象
  - MAPPO：按 `(episode_id, timestep, agent_id)` **严格对齐**回放批次

- ❌ 不在本 PRD 的改动：
  - 算法更换、网络结构大改
  - 复杂奖励重构（仅提供可选缩放与 sanity-check）

---

## 2. 设计原则

1) **同一语境**：采样 & 更新必须使用**同一掩码**和**同一动作**；
2) **分布优先**：`log_prob` 一律通过 `Distribution.log_prob(action)` 获得，不手搓 `log(action_probs[idx])`；
3) **历史快照**：`old_log_prob` 必须是采样时保存，并在更新时 `.detach()`；
4) **mean 归约**：所有损失项/指标对 batch 使用 `.mean()`，避免数值膨胀；
5) **可断言/可回滚**：每个关键步骤都有断言与仪表输出；一键开关 fallback。

---

## 3. 采样路径（Cursor可直接替换粘贴）

```python
# === 采样阶段（Actor前向 + 掩码 + 采样 + 缓存）===

# logits: [B, A] from actor(obs)
logits = actor(obs)                                   # shape [B, A]

# mask: [B, A]，True=合法动作（来自环境/规则层）
# 把非法动作置为 -inf（数值上用 -1e9 近似）
masked_logits = logits.masked_fill(~mask.bool(), -1e9)

# 构造“旧分布”并采样动作
dist_old = torch.distributions.Categorical(logits=masked_logits)
action   = dist_old.sample()                          # shape [B]
old_logp = dist_old.log_prob(action).detach()         # 关键：detach！

# 断言：采样的动作必须合法
assert mask.gather(1, action.unsqueeze(1)).all(), "sampled illegal action under mask"

# —— 缓存到经验池（MAPPO: 带上三元主键）——
buffer.store(
    obs=obs, action=action, old_logp=old_logp, mask=mask,
    agent_id=agent_id, timestep=timestep, episode_id=episode_id
)
```

> 注意：如果你有“掩码后重映射索引”的实现（把有效动作压紧为 0..k-1），必须同时缓存 `local_idx` 与 `valid_ids`（见 §5）。

---

## 4. 更新路径（Cursor可直接替换粘贴）

```python
# === 更新阶段（重放同一掩码 + 同一动作 → new_logp / ratio / KL）===

# 回放批次：确保按照 (episode_id, timestep, agent_id) 排序对齐
obs, action, old_logp, mask, adv, ret = batch.obs, batch.action, batch.old_logp, batch.mask, batch.adv, batch.ret

# Actor 前向
new_logits = actor(obs)                               # 必须与采样时的 obs 对齐！

# 应用“同一”掩码（不要重新构造另一套）
new_masked_logits = new_logits.masked_fill(~mask.bool(), -1e9)

# 构造“新分布”并求 new_logp/entropy
dist_new = torch.distributions.Categorical(logits=new_masked_logits)
new_logp = dist_new.log_prob(action)                  # 对“那次采样的动作”求 logp
entropy = dist_new.entropy().mean()

# PPO 核心：ratio / approx_kl
ratio     = torch.exp(new_logp - old_logp)            # r_t = exp(Δlogp)
approx_kl = (old_logp - new_logp).mean()              # ≈ KL(old||new)

# PPO-Clip（所有项用 mean，不要 sum）
epsilon = clip_eps  # 如 0.2
pg_obj  = ratio * adv
pg_clip = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
policy_loss = -torch.min(pg_obj, pg_clip).mean()

value_pred  = critic(obs)
value_loss  = 0.5 * (ret - value_pred).pow(2).mean()

loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
optimizer.step()
```

**健康期望**：
- `approx_kl ∈ [0.01, 0.03]`
- `clip_fraction ∈ [0.05, 0.30]`
- `ratio.mean ≈ 1.0 ± 0.1`
- `entropy` 不再固定为 `log(|A|)`，而是缓慢下降

---

## 5. 若动作索引在“掩码后重映射”（必看）

有些实现会将 `mask=True` 的索引压紧为 `0..K-1` 在局部分布上采样：

```python
# 采样时（per-env）
valid_ids = torch.nonzero(mask[env], as_tuple=False).squeeze(-1)  # 原始动作ID
local_logits = logits[env, valid_ids]
local_dist = torch.distributions.Categorical(logits=local_logits)
local_idx  = local_dist.sample()        # [0..K-1]
action_id  = valid_ids[local_idx]       # 回到原始动作空间

old_logp = local_dist.log_prob(local_idx).detach()

buffer.store(valid_ids=valid_ids, local_idx=local_idx, action_id=action_id, old_logp=old_logp, ...)
```

更新时**重建相同的局部分布**并用 `local_idx` 求 `new_logp`：

```python
# 更新时（per-env重放）
valid_ids   = batch.valid_ids[env]
local_logits = new_logits[env, valid_ids]
local_dist   = torch.distributions.Categorical(logits=local_logits)

new_logp = local_dist.log_prob(batch.local_idx[env])
```

> **严禁**：对 `action_id` 直接 `log_prob`，或用 `min(idx, len-1)` 兜底。那会把非法/越界硬映射成“最后一个动作”，制造伪信号。

---

## 6. MAPPO：对齐与聚合要点

- 经验条目键：`(episode_id, timestep, agent_id)`；
- 回放前**按该三元组排序**，确保 `obs/action/old_logp/mask/adv/ret` 同步；
- centralized critic 的 `obs_ctxt` 与 actor 的 `obs_indiv` 必须对齐到**同一 `(t, agent)`**；
- 不同 agent 的 `mask` 可能不同，**严禁广播/复用**。

断言样例：
```python
assert (batch.agent_id[:-1] <= batch.agent_id[1:]).all()
assert (batch.timestep[:-1] <= batch.timestep[1:]).all()
```

---

## 7. 必加断言与自测（一次 mini-batch 判因）

```python
# 1) 新旧 logp 合理范围
print("old_logp[:5]", old_logp[:5].detach().cpu().numpy())
print("new_logp[:5]", new_logp[:5].detach().cpu().numpy())  # 应在 [-20, 0]

# 2) diff 不应普遍 > 2
print("diff[:5]", (new_logp - old_logp)[:5].detach().cpu().numpy())

# 3) ratio 分布是否健康
print("ratio.mean/std", ratio.mean().item(), ratio.std().item())  # ≈1.0 ± 0.1

# 4) 近似 KL
print("approx_kl", approx_kl.item())  # < 0.03 为宜

# 5) 熵来自“同一分布”
print("entropy", entropy.item())  # 不应恒等于 log(|A|)

# 6) 掩码一致性
print("mask_equal", torch.equal(mask, cached_mask))

# 7) 动作合法性（更新阶段）
assert mask.gather(1, action.unsqueeze(1)).all(), "replay action illegal under mask"
```

---

## 8. 额外安全带（防炸，不治本）

```python
# 目标 KL 守卫（建议）
target_kl = 0.015
if approx_kl.item() > 2.0 * target_kl:
    print("[warn] early stop this epoch due to KL")
    break

# 归一化优势（强烈建议）
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
adv = adv.clamp(-10, 10)  # 急救式限幅（可选）

# 奖励缩放到 ~[-1, 1]（若 reward 方差大）
reward = torch.clamp(reward / reward_scale, -1.0, 1.0)
```

---

## 9. 回归测试步骤（两轮内看到“回魂”）

1) 保持现有超参不变（先不再调参）；  
2) 替换采样/更新路径为本 PRD 版本；  
3) 打印第一个 mini-batch 的上面 7 条自测信息；  
4) 期望立刻看到：`ratio.mean≈1.0`、`clip_fraction≈0.1~0.3`、`approx_kl<0.05`、`entropy` 不再固定；  
5) 若仍异常，多半是 **obs 对齐** 或 **mask 不是同一份**（回到 §6 检查三元对齐与 mask 来源）。

---

## 10. 回滚开关（可选）

```python
if cfg.feature_flags.strict_backward_compat:
    # 保持旧路径（仅用于灰度/对比）
    ...
else:
    # 使用本 PRD 的一致性路径
    ...
```

---

## 11. 成功标准（SLO）

- `clip_fraction ∈ [0.05, 0.30]`
- `approx_kl ∈ [0.01, 0.03]`（批内大于 0.05 触发早停）
- `entropy` 非常数，随训练缓慢下降
- `ratio.mean ≈ 1.0 ± 0.1`
- value loss 回落到可控区（取决于 reward 缩放）

---

## 12. 常见“踩坑到死锁”的反例清单（务必避免）

- 在更新阶段：
  - ❌ 用**另一套** mask/均匀分布来算 `new_log_prob`；
  - ❌ 对 `action_probs[idx]` 手算 `log`；
  - ❌ `old_log_prob` 用新策略重算或未 `detach()`；
  - ❌ `.sum()` 替代 `.mean()`；
  - ❌ 对“原始动作ID”直接 `log_prob`，而采样在“局部重映射”索引上进行；
  - ❌ MAPPO 合并批次时未按 `(episode_id, timestep, agent_id)` 对齐；
  - ❌ 把越界索引用 `min(idx, len-1)` 兜底（将非法动作伪造成“最后一个动作”）。

---

**一键总结**：  
> 让**采样时的那套分布**，在更新时**原样重现**，对**同一动作**求 `new_log_prob`。  
> 只要做到这一点，`ratio/kl/clip` 会立刻从“死锁”回到“可调”。
