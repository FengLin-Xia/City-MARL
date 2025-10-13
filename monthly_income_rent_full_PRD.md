# PRD：月度收益与租金现金流改造（面向 Cursor 对接）

> 目标：在**不推翻现有规则**前提下，补齐“**每月持续计入收益/租金**”与“**预算软约束**”的实现缺口，修复因一次性成本+一次性收益导致的**收益率失真**与**KL/clip 失控**问题。本文给出**最小修改集**、配置项变更、代码插入点、伪代码/示例实现、验收标准与回滚方案。

---

## 1. 背景与问题陈述

- 现状：建筑**成本（capex）**在建造当步一次性扣除，**月度收益/租金**没有按月持续计入，只在建造步产生一次奖励信号（或被重复计入多处质量项），导致：
  - 单步净现金流出现**巨负尖峰**，优势/价值方差飙升→ **KL≈2、clip≈1**；
  - 智能体“感知到”建造必亏→保守或发散，难以学到“先投入、后回本”的节奏；
  - 协作/租金信号名义存在但**未产生真实预算转移**，失去可学习性。

- 目标：  
  1) 建造当月扣一次**成本**；其后**每月累加**所有在营资产的**月度收益**与**租金**到预算与奖励；  
  2) 引入**预算软约束**（允许合理负债、惩罚软化）；  
  3) 统一奖励缩放，确保送入 PPO 的 reward 落在 **[-1, 1]** 数量级；  
  4) 保持对现有系统**最小侵入**与**可回滚**。

---

## 2. 范围（Scope）

- ✅ 新增/调整：  
  - 资产生命周期与“在营资产”集合管理（Active Assets）  
  - 月度收益累计与租金结算的**预算真实转移**  
  - 预算软约束与奖励缩放  
  - 配置项（JSON）新增/改名/默认值  
  - 训练端统计（reward 分布、KL、clip）观测点

- ❌ 不做：  
  - 复杂金融工具（折现率 r、NPV 图全量实现）  
  - 税收/折旧/动态利率  
  - 现有观测/动作空间的结构性变化（除非必要）

---

## 3. 关键设计原则

1) **现金流对齐**：一次性成本、**持续性收益/租金**。  
2) **单一语义**：reward/cost **只记一次**（去重），其余用于诊断而非再次加总。  
3) **软约束**：负债允许但惩罚**线性温和**，避免训练初期被强惩罚淹没。  
4) **可配置、可回滚**：一键开关/回退旧逻辑。  
5) **数值稳定**：统一 reward 缩放，便于 PPO 收敛。

---

## 4. 配置变更（JSON Schema）

### 4.1 新增配置

```json
{
  "rl_env": {
    "cashflow": {
      "enable_monthly_income": true,
      "income_scale": 500.0,
      "monthly_grant": 300.0,
      "use_amortization": false,
      "amortization_horizon": 20
    },
    "rent": {
      "enable_rent": true,
      "rent_rate": 0.03,
      "distance_scale": 10.0,
      "direction": "IND_TO_EDU"
    },
    "budget_policy": {
      "max_debt": -10000.0,
      "debt_penalty_coef": 0.01,
      "bankruptcy_threshold": -20000.0,
      "bankruptcy_penalty": 0.0
    },
    "safety": {
      "clip_budget_penalty": 200.0,
      "reward_clip": 5.0
    },
    "feature_flags": {
      "strict_backward_compat": false
    }
  }
}
```

---

## 5. 数据结构与状态

```python
class Asset:
    id: str
    owner: AgentId
    kind: str
    cost: float
    monthly_income: float
    active: bool
    pos: Tuple[float, float]

class AgentState:
    budget: float
    active_assets: Set[Asset]
```

---

## 6. 逻辑改造点（最小入侵）

```python
def step(...):
    monthly_income_ind = sum(a.monthly_income for a in IND.active_assets)
    monthly_income_edu = sum(a.monthly_income for a in EDU.active_assets)

    rent_ind_to_edu = 0.0
    if cfg.rent.enable_rent:
        for a in IND.active_assets:
            base = a.cost * cfg.rent.rent_rate
            if cfg.rent.distance_scale:
                w = exp(-distance_to_nearest_edu(a.pos) / cfg.rent.distance_scale)
            else:
                w = 1.0
            rent_ind_to_edu += base * w

    if cfg.rent.direction in ["IND_TO_EDU", "BIDIRECTIONAL"]:
        EDU.budget += rent_ind_to_edu
        IND.budget -= rent_ind_to_edu

    IND.budget += cfg.cashflow.monthly_grant
    EDU.budget += cfg.cashflow.monthly_grant

    IND.budget += monthly_income_ind
    EDU.budget += monthly_income_edu

    if action_ind.build:
        IND.budget -= action_ind.asset.cost
        IND.active_assets.add(action_ind.asset)
    if action_edu.build:
        EDU.budget -= action_edu.asset.cost
        EDU.active_assets.add(action_edu.asset)

    def debt_penalty(budget):
        if budget >= 0: return 0.0
        penalty = min(abs(budget) * cfg.budget_policy.debt_penalty_coef,
                      cfg.safety.clip_budget_penalty)
        return penalty

    penalty_ind = debt_penalty(IND.budget)
    penalty_edu = debt_penalty(EDU.budget)

    if not cfg.cashflow.use_amortization:
        net_ind = (monthly_income_ind
                   - (action_ind.asset.cost if action_ind.build else 0.0)
                   - penalty_ind
                   - (rent_ind_to_edu if cfg.rent.direction in ["IND_TO_EDU","BIDIRECTIONAL"] else 0.0))
        net_edu = (monthly_income_edu
                   - (action_edu.asset.cost if action_edu.build else 0.0)
                   - penalty_edu
                   + (rent_ind_to_edu if cfg.rent.direction in ["IND_TO_EDU","BIDIRECTIONAL"] else 0.0))
    else:
        T = cfg.cashflow.amortization_horizon
        net_ind = sum(a.monthly_income - a.cost / T for a in IND.active_assets) - penalty_ind
        net_edu = sum(a.monthly_income - a.cost / T for a in EDU.active_assets) - penalty_edu

    scale = cfg.cashflow.income_scale
    r_ind = clip(net_ind / scale, -cfg.safety.reward_clip, cfg.safety.reward_clip)
    r_edu = clip(net_edu / scale, -cfg.safety.reward_clip, cfg.safety.reward_clip)

    reward = {"IND": r_ind, "EDU": r_edu}
    info.update({
        "monthly_income_ind": monthly_income_ind,
        "monthly_income_edu": monthly_income_edu,
        "rent_ind_to_edu": rent_ind_to_edu,
        "penalty_ind": penalty_ind,
        "penalty_edu": penalty_edu,
        "net_ind": net_ind,
        "net_edu": net_edu
    })
    return obs, reward, done, info
```

---

## 7. 训练侧最小联动

- 优势标准化  
- 值函数裁剪  
- KL 守卫  
- lr: 5e-5 ~ 1e-4  
- ent_coef: 0.005 ~ 0.01  

---

## 8. 迁移与兼容

- `strict_backward_compat=true` 时禁用本改造  
- 默认启用新逻辑  

---

## 9. 测试与验收

- 单元测试：资产激活、预算转移、软惩罚、缩放范围  
- 集成测试：KL∈[0.01,0.03]、clip∈[0.05,0.30]、value_loss <50、预算曲线回正  

---

## 10. 风险与缓解

- 租金过强 → 降 rent_rate  
- 预算惩罚过弱 → 提高 coef  
- 奖励过大 → 提高 scale  

---

## 11. 开发任务清单

- [ ] 新增配置 schema  
- [ ] 扩展 AgentState / Asset  
- [ ] 修改 step 流程  
- [ ] 移除 reward 重复计入  
- [ ] 加测试与指标  

---

## 12. 成功标准（SLO）

- `KL ≤ 0.03`  
- `clip_fraction ≤ 0.35`  
- `value_loss < 50`  
- 回本期≈理论值 ±20%
