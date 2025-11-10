# 🏙️ Enhanced City Simulation v5.0

> **基于多智能体强化学习的城市发展模拟系统**  
> 支持动态Hub激活、预算解锁动作、管道架构和配置驱动的模块化设计

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v5.0-orange.svg)](https://github.com/FengLin-Xia/City-MARL)

## ✨ 核心特性

### 🎯 **v5.0 革命性功能**
- **🔄 动态Hub激活** - 指定月份激活新Hub，影响地价场演化
- **💰 预算解锁动作** - Agent预算达标后解锁新动作类型
- **🏗️ 管道架构** - 模块化设计，支持流水线处理
- **⚙️ 配置驱动** - 所有参数通过JSON配置，无需硬编码

### 🧠 **多智能体系统**
- **政府智能体** - 公共设施规划和政策制定
- **企业智能体** - 商业建筑和工业发展决策
- **居民智能体** - 住宅需求和社区建设

### 🌍 **城市模拟引擎**
- **高斯地价场** - 多Hub驱动的动态地价演化
- **槽位系统** - 严格范围外扩的生长机制
- **建筑类型** - 住宅、商业、工业、公共设施
- **实时可视化** - 训练过程动态展示

## 🚀 快速开始

### 📦 环境安装

```bash
# 使用conda（推荐）
conda env create -f environment.yml
conda activate city-marl

# 或使用pip
pip install -r requirements-core.txt
```

### 🎮 运行v5.0系统

```bash
# 完整模式（训练+导出）
python enhanced_city_simulation_v5_0.py --mode complete --episodes 10

# 仅训练模式
python enhanced_city_simulation_v5_0.py --mode training --episodes 5

# 仅导出模式
python enhanced_city_simulation_v5_0.py --mode export
```

### 🧪 测试功能

```bash
# 烟雾测试
python scripts/smoke_1025.py

# 解锁动作测试
python scripts/test_unlock_actions.py

# 简单测试
python test_v5_simple.py
```

## 📁 项目架构

```
marl/
├── 🎯 v5.0 核心系统
│   ├── enhanced_city_simulation_v5_0.py    # 主程序
│   ├── integration/v5_0/                   # 集成系统
│   │   ├── pipeline.py                     # 管道核心
│   │   ├── training_pipeline.py            # 训练管道
│   │   └── export_pipeline.py              # 导出管道
│   └── envs/v5_0/                         # v5.0环境
│       ├── city_env.py                     # 城市环境
│       └── budget_pool.py                  # 预算池系统
│
├── 🔧 中间件系统
│   ├── action_mw/                          # 动作中间件
│   │   └── unlock_gate.py                 # 预算解锁中间件
│   ├── reward_terms/                       # 奖励模块
│   │   ├── reward_manager.py               # 奖励管理器
│   │   └── action_diversity_reward.py     # 动作多样性奖励
│   └── scheduler/                          # 调度模块
│
├── 🏗️ 环境系统
│   ├── envs/
│   │   ├── v4_1/                          # v4.1环境
│   │   ├── v5_0/                          # v5.0环境
│   │   └── land_price_evo.py              # 地价演化
│   └── logic/                             # 逻辑模块
│       ├── enhanced_sdf_system.py         # SDF系统
│       └── v5_enumeration.py              # v5枚举
│
├── 📊 导出系统
│   ├── exporters/v5_0/                    # v5.0导出器
│   │   ├── txt_exporter.py                # TXT导出
│   │   ├── table_generator.py             # 表格生成
│   │   └── monthly_summary_png.py         # 月度摘要PNG
│   └── outputs/                           # 输出目录
│
├── 🧠 训练系统
│   ├── trainers/v5_0/                     # v5.0训练器
│   │   └── ppo_trainer.py                # PPO训练器
│   └── models/                            # 模型文件
│
├── ⚙️ 配置系统
│   ├── configs/
│   │   ├── city_config_v5_0.json         # v5.0主配置
│   │   ├── agent_config.json             # 智能体配置
│   │   └── building_config.json          # 建筑配置
│   └── contracts/                         # 契约层
│
├── 🛠️ 工具脚本
│   ├── scripts/
│   │   ├── smoke_1025.py                 # 烟雾测试
│   │   └── test_unlock_actions.py        # 解锁测试
│   └── utils/
│       └── event_bus.py                  # 事件总线
│
└── 📚 文档
    ├── docs/                              # 项目文档
    ├── 1025-*.md                          # 任务文档
    └── project_rules.json                 # 项目规则
```

## 🎮 使用指南

### 🔧 配置系统

v5.0采用完全配置驱动的设计，所有功能都可通过JSON配置启用/禁用：

```json
{
  "env": {
    "land_price": {
      "evolution": {
        "enabled": true,
        "hubs_schedule": [
          {
            "hub_id": "hub2",
            "activation_month": 6,
            "fade_in_months": 3
          }
        ]
      }
    }
  },
  "action_middleware": {
    "unlock_gate": {
      "enabled": true,
      "unlock_rules": [
        {
          "action_id": "large_industrial",
          "after_month": 12,
          "budget_threshold": 1000000
        }
      ]
    }
  }
}
```

### 🎯 运行模式

#### 1. 完整模式
```bash
python enhanced_city_simulation_v5_0.py --mode complete --episodes 10 --output-dir outputs/v5_0_results
```

#### 2. 训练模式
```bash
python enhanced_city_simulation_v5_0.py --mode training --episodes 5 --config configs/city_config_v5_0.json
```

#### 3. 导出模式
```bash
python enhanced_city_simulation_v5_0.py --mode export --input-dir outputs/training_results
```

### 🧪 测试与调试

```bash
# 运行烟雾测试
python scripts/smoke_1025.py

# 测试解锁动作功能
python scripts/test_unlock_actions.py

# 检查系统状态
python verify_system.py
```

## 📊 功能特性

### 🌟 **v5.0 核心功能**

| 功能 | 描述 | 状态 |
|------|------|------|
| 动态Hub激活 | 指定月份激活新Hub，影响地价场 | ✅ 完成 |
| 预算解锁动作 | Agent预算达标后解锁新动作 | ✅ 完成 |
| 管道架构 | 模块化流水线处理 | ✅ 完成 |
| 配置驱动 | JSON配置所有参数 | ✅ 完成 |
| 事件总线 | 调试和监控系统 | ✅ 完成 |

### 🏗️ **建筑系统**

- **住宅建筑** - 满足居民需求
- **商业建筑** - 提供商业服务
- **工业建筑** - 传统工业 + 高级工业
- **公共设施** - 教育、医疗、政府

### 🎯 **智能体行为**

- **政府智能体** - 公共设施规划，政策制定
- **企业智能体** - 商业投资，工业发展
- **居民智能体** - 住宅需求，社区建设

## 🔬 技术栈

- **深度学习**: PyTorch 2.5.1
- **强化学习**: Stable-Baselines3, PettingZoo
- **可视化**: Matplotlib, Pygame
- **数据处理**: NumPy, Pandas
- **配置管理**: JSON, YAML
- **环境管理**: Conda

## 📈 性能指标

- **训练速度**: ~100 episodes/hour
- **内存使用**: <4GB RAM
- **GPU支持**: CUDA 11.8+
- **并发处理**: 多智能体并行

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- PyTorch 团队提供的深度学习框架
- Stable-Baselines3 团队提供的强化学习库
- PettingZoo 团队提供的多智能体环境

## 📞 联系方式

- 项目链接: [https://github.com/FengLin-Xia/City-MARL](https://github.com/FengLin-Xia/City-MARL)
- 问题反馈: [Issues](https://github.com/FengLin-Xia/City-MARL/issues)

---

<div align="center">

**🌟 如果这个项目对您有帮助，请给它一个星标！**

Made with ❤️ by [FengLin-Xia](https://github.com/FengLin-Xia)

</div>