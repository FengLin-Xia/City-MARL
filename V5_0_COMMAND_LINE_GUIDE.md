# v5.0 命令行使用指南

## 🎯 基本命令格式

```bash
python enhanced_city_simulation_v5_0.py [选项]
```

## 📋 命令行参数详解

### **基础参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | `configs/city_config_v5_0.json` | 配置文件路径 |
| `--episodes` | int | `2` | 训练轮数 |
| `--output_dir` | str | `./outputs` | 输出目录 |

### **模式选择**

| 参数 | 选项 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | `complete`, `training`, `export`, `eval` | `complete` | 运行模式 |
| `--eval_only` | flag | False | 仅评估模式 |
| `--model_path` | str | None | 预训练模型路径 |

### **高级功能**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--compare_v4` | flag | False | 对比v4.1和v5.0 |
| `--performance_monitor` | flag | False | 启用性能监控 |
| `--pipeline_config` | str | None | 自定义管道配置文件 |

### **导出选项**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_data` | str | None | 输入数据路径（导出模式必需） |
| `--export_format` | `txt`, `tables`, `all` | `all` | 导出格式 |
| `--export_compatible` | flag | False | 导出v4.1兼容格式 |

### **其他选项**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--verbose` | flag | False | 详细输出 |
| `--save_results` | flag | True | 保存结果到文件 |

## 🚀 常用命令示例

### **1. 基础训练**
```bash
# 使用默认配置进行2轮训练
python enhanced_city_simulation_v5_0.py

# 指定训练轮数
python enhanced_city_simulation_v5_0.py --episodes 10

# 指定输出目录
python enhanced_city_simulation_v5_0.py --episodes 5 --output_dir ./results
```

### **2. 不同模式运行**

#### **完整模式 (默认)**
```bash
# 完整模式：训练 + 导出
python enhanced_city_simulation_v5_0.py --mode complete --episodes 5
```

#### **仅训练模式**
```bash
# 只进行训练，不导出
python enhanced_city_simulation_v5_0.py --mode training --episodes 10
```

#### **仅导出模式**
```bash
# 导出已有数据
python enhanced_city_simulation_v5_0.py --mode export --input_data ./data/episodes.json
```

#### **评估模式**
```bash
# 使用预训练模型进行评估
python enhanced_city_simulation_v5_0.py --mode eval --model_path ./checkpoints/model.pth
```

### **3. 高级功能**

#### **性能监控**
```bash
# 启用性能监控
python enhanced_city_simulation_v5_0.py --performance_monitor --episodes 5
```

#### **v4.1对比**
```bash
# 对比v4.1和v5.0性能
python enhanced_city_simulation_v5_0.py --compare_v4 --episodes 5
```

#### **详细输出**
```bash
# 启用详细输出
python enhanced_city_simulation_v5_0.py --verbose --episodes 3
```

### **4. 导出功能**

#### **导出所有格式**
```bash
# 导出txt和tables格式
python enhanced_city_simulation_v5_0.py --mode export --input_data ./data/episodes.json --export_format all
```

#### **导出特定格式**
```bash
# 只导出txt格式
python enhanced_city_simulation_v5_0.py --mode export --input_data ./data/episodes.json --export_format txt

# 只导出tables格式
python enhanced_city_simulation_v5_0.py --mode export --input_data ./data/episodes.json --export_format tables
```

#### **v4.1兼容导出**
```bash
# 导出v4.1兼容格式
python enhanced_city_simulation_v5_0.py --mode export --input_data ./data/episodes.json --export_compatible
```

### **5. 自定义配置**

#### **使用自定义配置文件**
```bash
# 使用自定义配置
python enhanced_city_simulation_v5_0.py --config ./my_config.json --episodes 5
```

#### **自定义管道配置**
```bash
# 使用自定义管道配置
python enhanced_city_simulation_v5_0.py --pipeline_config ./pipeline_config.json --episodes 5
```

## 🔧 参数组合示例

### **开发测试**
```bash
# 快速测试，详细输出
python enhanced_city_simulation_v5_0.py --episodes 1 --verbose --performance_monitor
```

### **功能验证**
```bash
# 中等规模训练，性能监控
python enhanced_city_simulation_v5_0.py --episodes 5 --performance_monitor --save_results
```

### **正式训练**
```bash
# 大规模训练，完整功能
python enhanced_city_simulation_v5_0.py --episodes 50 --performance_monitor --compare_v4 --verbose
```

### **生产环境**
```bash
# 生产环境，稳定运行
python enhanced_city_simulation_v5_0.py --episodes 100 --output_dir ./production_results
```

## 📊 输出说明

### **控制台输出**
```
v5.0 增强城市模拟系统
配置文件: configs/city_config_v5_0.json
运行模式: complete
训练轮数: 5
输出目录: ./outputs

============================================================
运行v5.0完整模式 (Complete Mode)
============================================================

[训练进度...]
[导出进度...]

总运行时间: 120.50 秒
```

### **文件输出**
```
outputs/
├── episodes/           # 训练数据
├── checkpoints/        # 模型检查点
├── exports/           # 导出文件
├── logs/              # 日志文件
└── results.json       # 结果摘要
```

## ⚠️ 注意事项

### **必需参数**
- 导出模式需要 `--input_data`
- 评估模式需要 `--model_path`

### **参数验证**
```bash
# 错误：导出模式缺少输入数据
python enhanced_city_simulation_v5_0.py --mode export
# 错误: 导出模式需要指定 --input_data

# 错误：评估模式缺少模型路径
python enhanced_city_simulation_v5_0.py --eval_only
# 错误: 评估模式需要指定 --model_path
```

### **性能建议**
- 开发测试：`--episodes 1-3`
- 功能验证：`--episodes 5-10`
- 正式训练：`--episodes 50+`
- 生产环境：`--episodes 100+`

## 🎯 快速开始

### **最简单的命令**
```bash
python enhanced_city_simulation_v5_0.py
```

### **推荐的开发命令**
```bash
python enhanced_city_simulation_v5_0.py --episodes 3 --verbose --performance_monitor
```

### **推荐的训练命令**
```bash
python enhanced_city_simulation_v5_0.py --episodes 20 --performance_monitor --save_results
```

## 📝 总结

v5.0系统提供了丰富的命令行选项：

- ✅ **4种运行模式**: complete, training, export, eval
- ✅ **灵活的参数配置**: 支持自定义配置和管道
- ✅ **高级功能**: 性能监控、v4.1对比
- ✅ **多种导出格式**: txt, tables, v4.1兼容
- ✅ **详细的输出控制**: verbose模式、结果保存

**推荐从基础命令开始，根据需要逐步添加高级功能！** 🚀
