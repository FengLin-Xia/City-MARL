# Cursor 对接文档：PPO/MAPPO 掩码一致性 & log_prob/ratio/KL 修复（含可粘贴代码）

> 目的：修复 **Valid samples 极低、clip_fraction=1.0、approx_kl≫0.1、entropy≈log(K)** 等“死锁”问题。根因是**采样与更新阶段的分布/掩码/局部索引不一致**（尤其是局部动作子集未持久化，训练时用默认 K=5 重建分布）。本文件提供**最小侵入**改动，直接可丢进 Cursor 执行改造。

---

## 改动概览

- 采样阶段：**持久化“局部分布语境”**  
  - `action_index`（局部索引 0..K-1）  
  - `num_actions`（子集大小 K）  
  - `subset_indices`（局部顺序对应的全局动作 ID 列表）  
  - `state_embed`（用于 actor/critic 的同一前向输入）  
  - `old_log_prob`（来自“局部分布+局部索引”的 logp，并 `detach()`）

- 训练阶段：**按缓存重放“同一局部分布”**  
  - 用 `subset_indices` 重建局部 logits  
  - 对 `action_index` 计算 `new_log_prob`、`ratio`、`approx_kl`、`entropy`  
  - **只用有效样本**参与更新（valid_ratio < 0.9 直接跳过该 mini-batch）

- MAPPO 对齐：batch 必须按 `(episode_id, timestep, agent_id)` 排序对齐。

---

## 1) 采样阶段修改（`solvers/v4_1/rl_selector.py` 约第 296–320 行）

> 关键：在采样处“带走”**局部索引**与**局部子集的顺序**。如果没有全局动作 ID，请使用你在 actor 前向时对应的原始索引/ID。

```python
# 利用：使用该agent的策略网络选择锚点动作
with torch.no_grad():
    # 1) 前向，与训练保持一致
    logits = actor(state_embed)  # shape [1, A]

    # 2) 局部动作子集（当前有效动作列表与顺序）
    num_actions = len(action_subset)                       # 子集大小 K
    subset_indices = torch.tensor(
        [a.global_id for a in action_subset],             # 若无 global_id，换成 a.action_id / a.index
        device=self.device, dtype=torch.long
    )

    # 3) 局部 logits（严格按子集顺序）
    valid_logits = logits[0, :num_actions]                # 你当前做法OK，配合 num_actions 的持久化
    dist = torch.distributions.Categorical(logits=valid_logits)

    # 4) 采样 + 局部索引（0..K-1）
    selected_idx = dist.sample().item()                   # 局部索引
    selected_action = action_subset[selected_idx]

    # 5) old_log_prob（采样时的局部分布 + 局部索引）
    old_log_prob = dist.log_prob(
        torch.tensor(selected_idx, device=self.device)
    ).detach()

    # 6) 缓存用于重放的一致前向输入
    cached_state_embed = state_embed.detach().clone()
```

> 如需掩码（非法动作）支持，请在构造 `valid_logits` 前将非法动作 logits 置为 `-1e9`，并确保训练阶段使用**同一掩码**。

---

## 2) 写入经验（`enhanced_city_simulation_v4_1.py` 约第 479–510 行）

> 将“局部分布语境”写入经验池，训练阶段据此重建分布。

```python
experience = {
    'state': state.copy(),
    'state_embed': cached_state_embed,  # ✅ 与采样一致的嵌入（actor/critic输入）

    'action': selected_sequence,
    'agent': current_agent,
    'month': env.current_month,

    # ✅ 关键：局部分布语境
    'old_log_prob': old_log_prob,       # 来自局部dist + 局部action_index
    'action_index': selected_idx,       # 局部索引 0..K-1
    'num_actions': num_actions,         # 子集大小 K
    'subset_indices': subset_indices,   # 局部顺序对应的全局动作 ID 列表

    # 业务字段（保持原有）
    'selected_slots': [action.footprint_slots for action in selected_sequence.actions] if selected_sequence and selected_sequence.actions else [],
    'action_scores': [action.score for action in selected_sequence.actions] if selected_sequence and selected_sequence.actions else [],
    'action_costs': [action.cost for action in selected_sequence.actions] if selected_sequence and selected_sequence.actions else [],
    'sequence_score': selected_sequence.score if selected_sequence else 0.0,
    'available_actions_count': len(actions),
    'candidate_slots_count': len(set(actions[i].footprint_slots[0] for i in range(len(actions)) if actions[i].footprint_slots)) if actions else 0,
    'detailed_actions': detailed_actions,
    # 如有 episode_id/timestep 也写入，便于 MAPPO 对齐：
    # 'episode_id': episode_id, 'timestep': t
}
```

---

## 3) 训练阶段：重放“同一局部分布”（`trainers/v4_1/ppo_trainer.py`）

### 3.1 取前向输入：优先使用缓存的 `state_embed`（约第 367–408 行）

```python
for i, exp in enumerate(experiences):
    # —— 取缓存（与采样时一致）——
    state_embed = exp.get('state_embed', None)
    if state_embed is None:
        # 兼容旧数据的兜底（不推荐）
        sequence = exp['action']
        first_action = sequence.actions[0] if sequence and sequence.actions else None
        state_embed = self.selector._encode_state_for_rl([first_action]) if first_action else self.selector._encode_state_for_rl([])

    # 选择 agent 对应的网络
    agent = exp.get('agent', 'IND')
    actor = self.selector.actors.get(agent, self.selector.actor)
    critic = self.selector.critics.get(agent, self.selector.critic)

    logits = actor(state_embed)     # [1, A]
    value  = critic(state_embed)
    if value.dim() > 1: value = value.squeeze()
    if value.dim() == 0: value = value.unsqueeze(0)

    # —— 重放“局部分布”与“局部索引” ——
    action_idx   = exp.get('action_index', -1)          # 局部索引
    subset_ids   = exp.get('subset_indices', None)      # 局部顺序（全局ID列表）
    num_actions  = exp.get('num_actions', None)

    if action_idx < 0 or subset_ids is None or num_actions is None:
        log_prob = torch.tensor([float('-inf')], device=self.device)
        entropy  = torch.tensor([0.0], device=self.device)
    else:
        # 若当初 valid_logits=logits[0, :num_actions]，可直接这么切；
        # 更稳健：按 subset_ids 切出“同一局部顺序”的 logits
        local_logits = logits[0, subset_ids] if subset_ids.numel() == num_actions else logits[0, :num_actions]
        dist = torch.distributions.Categorical(logits=local_logits)

        if 0 <= action_idx < local_logits.shape[0]:
            log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device))
            entropy  = dist.entropy()
            if log_prob.dim() == 0: log_prob = log_prob.unsqueeze(0)
            if entropy.dim() == 0:  entropy  = entropy.unsqueeze(0)
        else:
            log_prob = torch.tensor([float('-inf')], device=self.device)
            entropy  = torch.tensor([0.0], device=self.device)

    # 收集 log_prob / entropy / value 到列表，后续拼 batch（此处略）
```

### 3.2 old_log_prob 聚合：只用有效样本（约第 470–500 行）

```python
# 获取旧策略的动作概率（从经验中读取）
old_log_probs_list, valid_flags = [], []

for exp in experiences:
    olp = exp.get('old_log_prob', None)
    aid = exp.get('action_index', -1)
    k   = exp.get('num_actions', None)
    sid = exp.get('subset_indices', None)

    is_valid = (olp is not None) and (aid is not None) and (aid >= 0) and (k is not None) and (sid is not None)
    valid_flags.append(is_valid)

    if is_valid:
        if olp.dim() == 0: olp = olp.unsqueeze(0)
        elif olp.dim() > 1: olp = olp.squeeze()
        old_log_probs_list.append(olp)

valid_mask = torch.tensor(valid_flags, device=self.device, dtype=torch.bool)
if valid_mask.float().mean().item() < 0.9:
    print(f"[skip] valid_ratio={valid_mask.float().mean().item():.2f} (<0.90), skip this mini-batch")
    continue

old_log_probs = torch.stack(old_log_probs_list).to(self.device)
# 其他张量（new_logp/adv/ret/obs 等）也按 valid_mask 筛选后再参与 loss 计算
```

---

## 4) 断言与健康检查（首次 mini-batch 必过）

在第一个 mini-batch 更新里插入：
```python
print("Valid samples:", valid_mask.sum().item(), "/", valid_mask.numel())

print("old_logp[:5]", old_log_probs[:5].detach().cpu().numpy())
print("new_logp[:5]", new_logp[:5].detach().cpu().numpy())
print("diff[:5]", (new_logp[:5] - old_log_probs[:5]).detach().cpu().numpy())

print("ratio.mean/std", ratio.mean().item(), ratio.std().item())
print("approx_kl", approx_kl.item())
print("entropy", entropy.mean().item())

# 断言
assert torch.isfinite(new_logp).all() and torch.isfinite(old_log_probs).all()
assert new_logp.max() <= 0.0 and old_log_probs.max() <= 0.0, "log_prob 应 ≤ 0"
assert approx_kl < 0.2, f"KL too big {approx_kl.item():.3f}"
assert valid_mask.float().mean().item() > 0.9, "有效样本比例过低"
```

**健康区间**：
- `Valid samples: 全部/全部`
- `ratio.mean ≈ 1.0 ± 0.1`
- `clip_fraction ≈ 0.1~0.3`
- `approx_kl ≈ 0.01~0.03`
- `entropy` 不再固定为 `log(K)`

---

## 5) MAPPO 对齐要点（如使用）

- 回放前**按 `(episode_id, timestep, agent_id)` 排序**，确保 `obs/action/old_logp/mask/adv/ret` 同步；  
- centralized critic 的上下文输入与 actor 的个体输入对齐同一 `(t, agent)`；  
- 不同 agent 的 `subset_indices/num_actions` 可能不同，**严禁广播/复用**。

样例断言：
```python
assert (batch.agent_id[:-1] <= batch.agent_id[1:]).all()
assert (batch.timestep[:-1] <= batch.timestep[1:]).all()
```

---

## 6) 常见踩坑（务必避免）

- 在**更新阶段**：
  - ❌ 用**另一套** mask/均匀分布计算 `new_log_prob/entropy`；
  - ❌ 只存 `action_id`，未存 `action_index/num_actions/subset_indices`；
  - ❌ 对 `action_probs[idx]` 手搓 `torch.log`；
  - ❌ `old_log_prob` 未 `detach()` 或用新策略重算；
  - ❌ `.sum()` 替代 `.mean()`（放大梯度）；
  - ❌ 使用默认 `num_actions=5` 兜底（导致 entropy≈log(5)、ratio/kl 失真）。

---

## 7) 回滚开关（可选）

```python
if cfg.feature_flags.strict_backward_compat:
    # 使用旧路径（仅做灰度/对比）
    pass
else:
    # 使用本次修复后的“局部分布一致性”路径
    pass
```

---

## 8) 成功标准（SLO）

- `clip_fraction ∈ [0.05, 0.30]`
- `approx_kl ∈ [0.01, 0.03]`（批内大于 0.05 则提前停止该 epoch）
- `entropy` 非常数，随训练缓慢下降
- `ratio.mean ≈ 1.0 ± 0.1`
- `Valid samples` ≥ 90%（理想 100%）

---

### 附：若仍需紧急“安全带”（不治本，仅防炸）
```python
target_kl = 0.015
if approx_kl.item() > 2.0 * target_kl:
    print("[warn] early stop this epoch due to KL")
    break

# 归一化优势（建议）
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
adv = adv.clamp(-10, 10)  # 急救限幅（可选）
```
