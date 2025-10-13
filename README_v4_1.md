# v4.1 增强城市模拟系统

## 概述

v4.1版本在原有城市模拟系统基础上，新增了强化学习(RL)求解模式，支持PPO和MAPPO算法。系统现在支持两种求解模式：

- **参数化模式 (Parametric Mode)**: 使用穷举/束搜索的传统方法
- **RL模式 (RL Mode)**: 使用强化学习策略进行动作选择

## 核心特性

### 🔧 配置总开关
通过`solver.mode`字段控制求解模式，完全无回退：
- `"param"`: 参数化模式
- `"rl"`: RL模式

### 🤖 RL算法支持
- **PPO**: 单智能体Proximal Policy Optimization
- **MAPPO**: 多智能体PPO，支持EDU和IND双智能体协作

### 📊 统一评测框架
- 严格的黑白对比实验
- 统一的性能指标
- 自动生成对比报告

## 文件结构

```
project/
├── configs/
│   └── city_config_v4_1.json          # v4.1配置文件
├── solvers/v4_1/
│   ├── param_selector.py              # 参数化选择器
│   └── rl_selector.py                 # RL选择器
├── rl/v4_1/
│   ├── models.py                      # 神经网络模型
│   ├── algo_ppo.py                    # PPO/MAPPO算法
│   ├── utils.py                       # 工具函数
│   └── buffers.py                     # 经验回放缓冲区
├── models/v4_1_rl/                    # 模型保存目录
├── logs/v4_1_rl/                      # 训练日志目录
├── reports/v4_1_rl/                   # 评测报告目录
└── enhanced_city_simulation_v4_1.py   # 主程序入口
```

## 使用方法

### 1. 基础配置

编辑`configs/city_config_v4_1.json`：

```json
{
  "solver": {
    "mode": "rl",                    // "param" | "rl"
    "rl": {
      "algo": "mappo",               // "ppo" | "mappo"
      "agents": ["EDU", "IND"],      // 智能体列表
      "clip_eps": 0.2,
      "gamma": 0.99,
      "gae_lambda": 0.95,
      "lr": 3e-4,
      "ent_coef": 0.01,
      "vf_coef": 0.5,
      "K_epochs": 4,
      "rollout_steps": 4096,
      "mini_batch_size": 1024,
      "max_updates": 2000,
      "cooperation_lambda": 0.2,     // 协作奖励权重
      "seed": 42,
      "eval_every": 20,
      "save_every": 50
    }
  }
}
```

### 2. 运行命令

#### 训练RL模型
```bash
python enhanced_city_simulation_v4_1.py --config configs/city_config_v4_1.json --mode rl
```

#### 评估RL模型
```bash
python enhanced_city_simulation_v4_1.py --config configs/city_config_v4_1.json --mode rl --eval_only --model_path models/v4_1_rl/final_model.pth
```

#### 运行参数化模式
```bash
python enhanced_city_simulation_v4_1.py --config configs/city_config_v4_1.json --mode param --eval_only
```

#### 对比两种模式
```bash
python enhanced_city_simulation_v4_1.py --config configs/city_config_v4_1.json --compare --model_path models/v4_1_rl/final_model.pth
```

### 3. 输出结果

- **训练模式**: 生成训练日志和模型文件
- **评估模式**: 生成性能指标和布局快照
- **对比模式**: 生成详细的对比报告

## 技术细节

### 状态编码
- **栅格通道**: 占用状态、功能类型、锁定期、地价、河岸距离
- **全局统计**: 月份、预算、建筑计数等
- **CNN编码**: 200×200栅格 → 512维特征向量

### 动作空间
- **动态大小**: 根据当前状态动态生成合法动作池
- **特征编码**: 槽位坐标、局部地价统计、邻接特征
- **掩码机制**: 确保只选择合法动作

### 奖励设计
- **基础奖励**: 使用现有ActionScorer计算
- **协作奖励**: 双智能体间的协作激励
- **稀疏奖励**: 长期目标导向的最终评估

### 训练策略
1. **Phase 1**: 单智能体预训练
2. **Phase 2**: 双智能体联合训练
3. **Phase 3**: 微调和稳定性提升

## 性能指标

### 收益指标
- `total_return`: 总收益
- `edu_return`: EDU智能体收益
- `ind_return`: IND智能体收益

### 效率指标
- `steps_per_second`: 每秒决策数
- `ms_per_decision`: 单次决策耗时

### 协作指标
- `cooperation_improvement`: 相对单独最优的改进度
- `pareto_efficiency`: Pareto效率

## 开发状态

### ✅ 已完成
- [x] 配置系统设计
- [x] 文件结构搭建
- [x] RL模型架构
- [x] 动作掩码机制
- [x] PPO/MAPPO算法框架
- [x] 经验回放缓冲区
- [x] 主程序入口

### 🚧 进行中
- [ ] 环境接口集成
- [ ] 状态编码实现
- [ ] 动作特征提取
- [ ] 训练循环完善
- [ ] 评估框架实现

### 📋 待实现
- [ ] 与现有城市模拟系统集成
- [ ] 完整的训练和评估流程
- [ ] 性能优化和调试
- [ ] 详细的实验报告

## 注意事项

1. **环境依赖**: 确保CUDA和PyTorch环境正常
2. **内存管理**: 大栅格数据需要足够的GPU内存
3. **训练稳定性**: 建议从简单配置开始，逐步增加复杂度
4. **随机种子**: 使用固定种子确保实验可复现

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目采用MIT许可证。

