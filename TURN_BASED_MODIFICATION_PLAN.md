# Turn-Based模式修改计划

## 目标
将v4.1从"每月两个agent同时行动"改为"每月一个agent行动，轮流"（类似v4.0）

---

## 当前机制 vs 目标机制

### 当前v4.1机制
```
Month 0: EDU行动 → IND行动
Month 1: EDU行动 → IND行动
...
Month 19: EDU行动 → IND行动

总决策步骤：20个月 × 2个agent = 40步
```

### 目标Turn-Based机制
```
Month 0: IND行动
Month 1: EDU行动  
Month 2: IND行动
Month 3: EDU行动
...
Month 19: EDU行动

总决策步骤：20个月 × 1个agent = 20步
```

---

## 需要修改的代码

### 1. Environment._advance_turn() 逻辑 🔴 核心修改

**位置**：`envs/v4_1/city_env.py` 第494-510行

**当前代码**：
```python
def _advance_turn(self) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
    """推进回合（智能体轮换模式：EDU→IND→下个月）"""
    # 智能体轮换逻辑
    self.agent_turn = (self.agent_turn + 1) % len(self.rl_cfg['agents'])
    self.current_agent = self.rl_cfg['agents'][self.agent_turn]
    
    # 如果轮换回第一个智能体(EDU)，进入下个月
    if self.agent_turn == 0:
        self.current_month += 1
```

**修改为Turn-Based**：
```python
def _advance_turn(self) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
    """推进回合（Turn-Based模式：每月一个agent，轮流行动）"""
    
    # 【新增】检查是否启用turn-based模式
    turn_based = self.v4_cfg.get('enumeration', {}).get('turn_based', False)
    
    if turn_based:
        # Turn-Based模式：先进月，再换agent
        self.current_month += 1
        
        # 轮换到下一个agent
        self.agent_turn = (self.agent_turn + 1) % len(self.rl_cfg['agents'])
        self.current_agent = self.rl_cfg['agents'][self.agent_turn]
    else:
        # 原v4.1模式：先换agent，轮回时进月
        self.agent_turn = (self.agent_turn + 1) % len(self.rl_cfg['agents'])
        self.current_agent = self.rl_cfg['agents'][self.agent_turn]
        
        if self.agent_turn == 0:
            self.current_month += 1
    
    # ... 后续代码不变 ...
```

**关键差异**：
- **v4.1**：先换agent，agent轮回时才进月
- **Turn-Based**：先进月，再换agent（每月月初换人）

---

### 2. Environment.reset() 初始化 🟡 次要修改

**位置**：`envs/v4_1/city_env.py` 第154-164行

**需要考虑**：设置`first_agent`

**当前代码**：
```python
def reset(self) -> Dict[str, Any]:
    self.current_month = 0
    self.current_agent = self.rl_cfg['agents'][0]  # 总是从第一个开始
    self.agent_turn = 0
```

**建议修改**：
```python
def reset(self) -> Dict[str, Any]:
    self.current_month = 0
    
    # 【新增】支持first_agent配置
    first_agent = self.v4_cfg.get('enumeration', {}).get('first_agent', None)
    if first_agent and first_agent in self.rl_cfg['agents']:
        self.agent_turn = self.rl_cfg['agents'].index(first_agent)
        self.current_agent = first_agent
    else:
        self.agent_turn = 0
        self.current_agent = self.rl_cfg['agents'][0]
```

---

### 3. 配置文件添加参数 🟢 配置修改

**位置**：`configs/city_config_v4_1.json`

**需要添加**：
```json
"enumeration": {
  "turn_based": true,      // 启用turn-based模式
  "first_agent": "IND"     // 第一个行动的agent（可选，默认agents[0]）
}
```

**完整配置示例**：
```json
{
  "growth_v4_1": {
    "hubs": {
      "mode": "explicit",
      "candidate_mode": "cumulative",
      "list": [
        {"xy": [122, 80], "R0": 6, "dR": 1.5},
        {"xy": [112, 121], "R0": 6, "dR": 1.5}
      ]
    },
    "enumeration": {
      "turn_based": true,
      "first_agent": "IND",
      "length_max": 5,
      "use_skip": true
    },
    ...
  }
}
```

---

### 4. 训练循环调整 🟡 训练参数

**位置**：`trainers/v4_1/ppo_trainer.py` 或 `enhanced_city_simulation_v4_1.py`

**影响的参数**：

| 参数 | v4.1值 | Turn-Based值 | 说明 |
|------|--------|-------------|------|
| `episode_length` | 40 | 20 | 每episode的步数减半 |
| `rollout_steps` | 10 | 10 | 可保持不变 |
| `total_months` | 20 | 20 | 保持不变 |

**可能需要调整**：
```json
{
  "simulation": {
    "total_months": 20  // 保持20个月
  },
  "solver": {
    "rl": {
      "rollout_steps": 10,  // 可以保持
      // 注意：episode_length是自动计算的（total_months * agents数量）
      // Turn-Based下会自动变成20
    }
  }
}
```

**代码中可能需要确认**：
- Episode结束条件：`done = (self.current_month >= self.total_months)`
- 这个逻辑应该不需要改，因为都是到20个月结束

---

### 5. Cooperation机制调整 ⚠️ 语义修改

**位置**：`envs/v4_1/city_env.py` 第424-460行

**问题**：
当前cooperation reward计算了"同月内另一个agent的建筑数"：
```python
def _calculate_cooperation_reward(self, agent: str, action: Action) -> float:
    if agent == 'EDU':
        ind_buildings = len(self.buildings['industrial'])
        cooperation_bonus += ind_buildings * 0.05  # 基于IND的建筑数
```

**Turn-Based下的含义变化**：
- **v4.1**：同月内IND已经建了，EDU看到IND的新建筑
- **Turn-Based**：IND是上个月建的，EDU看到的是历史建筑

**建议**：
保持代码不变，但理解语义变化：
- Cooperation从"同月协作"变成"历史累积"
- 实际上更合理（EDU看到IND过去的建设成果）

**或者**：如果觉得cooperation不再有意义，可以：
```python
# 在turn_based模式下禁用cooperation
if self.v4_cfg.get('enumeration', {}).get('turn_based', False):
    cooperation_bonus = 0.0
else:
    cooperation_bonus = self._calculate_cooperation_reward(agent, action)
```

---

## 配置文件完整修改示例

### 在`configs/city_config_v4_1.json`中添加

**方案A：最小修改（推荐）**
```json
{
  "growth_v4_1": {
    "enumeration": {
      "turn_based": true,
      "first_agent": "IND"
    },
    // ... 其他配置保持不变 ...
  }
}
```

**方案B：同时调整cooperation**
```json
{
  "growth_v4_1": {
    "enumeration": {
      "turn_based": true,
      "first_agent": "IND"
    },
    // ... 
  },
  "solver": {
    "rl": {
      "cooperation_lambda": 0.0,  // 禁用cooperation（因为turn-based下意义不同）
      // ... 其他配置 ...
    }
  }
}
```

---

## 修改步骤（按顺序）

### Step 1: 添加配置参数
```bash
# 编辑 configs/city_config_v4_1.json
# 在 "growth_v4_1" 下添加 "enumeration" 配置
```

### Step 2: 修改Environment._advance_turn()
```bash
# 编辑 envs/v4_1/city_env.py
# 在_advance_turn()中添加turn_based判断逻辑
```

### Step 3: 修改Environment.reset()（可选）
```bash
# 编辑 envs/v4_1/city_env.py
# 在reset()中添加first_agent支持
```

### Step 4: 测试
```bash
# 运行一个简单的episode测试
python enhanced_city_simulation_v4_1.py --mode rl --eval_only
```

### Step 5: 检查输出
- 查看`chosen_month_XX.txt`文件
- 确认每个月只有一个agent的动作
- 验证agent轮流顺序正确

---

## 预期效果

### 输出文件变化

**v4.1模式**（当前）：
```
chosen_month_00.txt: 3个EDU动作 + 3个IND动作
chosen_month_01.txt: 3个EDU动作 + 3个IND动作
```

**Turn-Based模式**（修改后）：
```
chosen_month_00.txt: 3个IND动作
chosen_month_01.txt: 3个EDU动作
chosen_month_02.txt: 3个IND动作
chosen_month_03.txt: 3个EDU动作
```

### 训练指标变化

| 指标 | v4.1 | Turn-Based | 变化 |
|------|------|-----------|------|
| Episode steps | 40 | 20 | -50% |
| 训练速度 | 基准 | 约2倍快 | +100% |
| 样本数/episode | 40 | 20 | -50% |
| Agent协调 | 同月协作 | 历史观察 | 语义变化 |

---

## 潜在风险与注意事项

### 风险1：Episode长度减半
- **影响**：每个episode的学习信号减少
- **缓解**：可能需要更多episodes来达到相同效果
- **调整**：考虑增加`max_updates`从5到10

### 风险2：Agent看不到对方的即时决策
- **影响**：无法学习"同月协作"策略
- **判断**：当前cooperation本来就很弱（<0.2%），影响有限

### 风险3：训练曲线可能变化
- **影响**：之前的训练结果不可直接对比
- **应对**：需要重新建立baseline

### 风险4：代码中可能有隐藏的假设
- **检查点**：
  - 是否有地方假设`episode_length=40`？
  - 是否有地方假设"一个月内有两个agent"？
  - Budget更新逻辑是否依赖月内顺序？

---

## 测试检查清单

修改完成后，检查以下项目：

- [ ] 配置文件加载正确
- [ ] reset()时current_agent是first_agent
- [ ] 第一个月只有first_agent行动
- [ ] Agent按月轮换（IND→EDU→IND→EDU...）
- [ ] Episode在20个月后正确结束
- [ ] chosen_month_XX.txt文件只包含一个agent
- [ ] Training能正常运行不报错
- [ ] Budget更新逻辑正常
- [ ] 输出的建筑数量合理（约20个月×1个agent×3个动作=60个建筑）

---

## 回滚方案

如果turn-based模式有问题，可以快速回滚：

```json
{
  "enumeration": {
    "turn_based": false,  // 改回false即可
    "first_agent": "IND"
  }
}
```

代码中的条件判断会自动切回v4.1模式。

---

## 与v4.0的对比

| 方面 | v4.0 | v4.1 Turn-Based | v4.1 原模式 |
|------|------|----------------|------------|
| 每月agent数 | 1 | 1 | 2 |
| Episode长度 | 20 | 20 | 40 |
| Agent切换 | 每月 | 每月 | 每步 |
| Cooperation | 无 | 历史累积 | 同月协作 |
| 决策方式 | 参数化 | RL | RL |

**Turn-Based v4.1 ≈ v4.0的结构 + v4.1的RL训练**

---

**文档版本**：v1.0  
**创建时间**：2025-10-11  
**状态**：待实施

