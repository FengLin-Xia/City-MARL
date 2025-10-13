# MARL - 多智能体强化学习城市环境

一个基于地形和道路规划的多智能体强化学习系统，支持Blender地形导入和实时可视化。

## 🚀 快速开始

### 环境设置
```bash
# 使用conda环境
conda env create -f environment.yml
conda activate city-marl

# 或使用pip
pip install -r requirements-core.txt
```

### 运行Flask服务器
```bash
python main.py
```

### 训练地形道路规划
```bash
python train_with_uploaded_terrain.py
```

## 📁 项目结构

```
marl/
├── agents/              # 智能体定义
│   ├── base_agent.py
│   ├── landvalue_agent.py
│   ├── traffic_agent.py
│   ├── terrain_policy.py
│   └── zoning_agent.py
├── envs/                # 环境定义
│   ├── city_env.py
│   ├── road_env.py
│   ├── terrain_road_env.py
│   ├── terrain_system.py
│   ├── pathfinding.py
│   ├── land_system.py
│   └── utils.py
├── models/              # 神经网络模型
│   ├── policy_net.py
│   └── value_net.py
├── training/            # 训练脚本
│   ├── config.py
│   ├── train_multi.py
│   ├── train_single.py
│   └── train_terrain_road.py
├── tests/               # 测试和演示
│   ├── check_cuda.py
│   ├── test_terrain_env.py
│   ├── demo_visualization.py
│   ├── demo_replay.py
│   └── test_terrain_upload.py
├── scripts/             # 工具脚本
│   ├── setup_conda_env.py
│   ├── setup_env.py
│   ├── git_upload.bat
│   └── blender_upload_terrain.py
├── data/                # 数据文件
│   ├── episodes/        # 训练回放
│   ├── terrain/         # 地形数据
│   └── results/         # 训练结果
├── main.py              # Flask服务器
├── train_with_uploaded_terrain.py
├── requirements.txt
├── requirements-core.txt
├── environment.yml
└── README.md
```

## 🔧 主要功能

### 1. 地形系统
- 支持OBJ文件导入
- 高程信息提取
- 地形类型分类

### 2. 路径规划
- A*算法
- Dijkstra算法
- 实时路径可视化

### 3. 强化学习
- PPO算法
- DQN算法
- 多智能体环境

### 4. Blender集成
- OBJ文件导出
- Flask API通信
- 实时数据交换

### 5. 可视化
- 实时训练可视化
- 回放系统
- 地形渲染

## 📊 使用流程

1. **启动Flask服务器**: `python main.py`
2. **Blender上传地形**: 运行`scripts/blender_upload_terrain.py`
3. **IDE接收地形**: 自动处理上传的地形数据
4. **开始训练**: `python train_with_uploaded_terrain.py`
5. **查看回放**: `python -m tests.demo_replay`

## 🛠️ 开发工具

- **环境管理**: `scripts/setup_conda_env.py`
- **Git上传**: `scripts/git_upload.bat`
- **CUDA检查**: `python -m tests.check_cuda`

## 📝 许可证

MIT License

