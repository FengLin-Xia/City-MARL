# v5.0训练命令指南

## 🚀 基础训练命令

### 1. **完整模式训练** (推荐)
```bash
python enhanced_city_simulation_v5_0.py --mode complete --episodes 100
```

### 2. **仅训练模式**
```bash
python enhanced_city_simulation_v5_0.py --mode training --episodes 100
```

### 3. **评估模式**
```bash
python enhanced_city_simulation_v5_0.py --mode eval --eval_only --model_path ./models/v5_0_rl/model.pth
```

## 📋 详细参数说明

### **基础参数**
- `--config`: 配置文件路径 (默认: `configs/city_config_v5_0.json`)
- `--episodes`: 训练轮数 (默认: 2)
- `--output_dir`: 输出目录 (默认: `./outputs`)

### **模式选择**
- `--mode complete`: 完整模式 (训练+导出)
- `--mode training`: 仅训练模式
- `--mode export`: 仅导出模式
- `--mode eval`: 评估模式

### **高级功能**
- `--compare_v4`: 对比v4.1和v5.0
- `--performance_monitor`: 启用性能监控
- `--verbose`: 详细输出
- `--save_results`: 保存结果到文件

## 🎯 常用训练命令示例

### **1. 基础训练**
```bash
# 使用默认配置训练100轮
python enhanced_city_simulation_v5_0.py --episodes 100

# 使用自定义配置训练
python enhanced_city_simulation_v5_0.py --config configs/city_config_v5_0.json --episodes 100
```

### **2. 完整训练流程**
```bash
# 完整模式：训练+导出+评估
python enhanced_city_simulation_v5_0.py --mode complete --episodes 100 --output_dir ./outputs/v5_0_training
```

### **3. 性能监控训练**
```bash
# 启用性能监控和详细输出
python enhanced_city_simulation_v5_0.py --mode complete --episodes 100 --performance_monitor --verbose
```

### **4. 对比训练**
```bash
# 对比v4.1和v5.0性能
python enhanced_city_simulation_v5_0.py --mode complete --episodes 100 --compare_v4
```

### **5. 评估已训练模型**
```bash
# 评估预训练模型
python enhanced_city_simulation_v5_0.py --mode eval --eval_only --model_path ./models/v5_0_rl/model.pth
```

### **6. 导出训练结果**
```bash
# 导出训练数据
python enhanced_city_simulation_v5_0.py --mode export --input_data ./outputs/training_data.json --export_format all
```

## 🔧 配置文件说明

### **默认配置文件**: `configs/city_config_v5_0.json`

主要配置项：
- `mappo`: MAPPO算法参数
- `agents`: 智能体配置
- `action_mw`: 动作中间件
- `reward_mechanisms`: 奖励机制
- `hubs`: Hub候选范围配置
- `river_restrictions`: 河流限制配置

## 📊 输出结果

### **训练输出**
- 模型文件: `./outputs/models/v5_0_rl/`
- 训练日志: `./outputs/logs/`
- 性能指标: `./outputs/metrics/`

### **导出文件**
- TXT格式: `./outputs/exports/sequences.txt`
- 表格格式: `./outputs/exports/action_tables/`
- 可视化: `./outputs/visualizations/`

## 🎯 训练流程

### **1. 准备阶段**
```bash
# 检查配置文件
python enhanced_city_simulation_v5_0.py --config configs/city_config_v5_0.json --episodes 1 --verbose
```

### **2. 训练阶段**
```bash
# 开始训练
python enhanced_city_simulation_v5_0.py --mode complete --episodes 100 --performance_monitor
```

### **3. 评估阶段**
```bash
# 评估训练结果
python enhanced_city_simulation_v5_0.py --mode eval --eval_only --model_path ./outputs/models/v5_0_rl/model.pth
```

### **4. 导出阶段**
```bash
# 导出训练数据
python enhanced_city_simulation_v5_0.py --mode export --input_data ./outputs/training_data.json
```

## 🚨 注意事项

### **1. 系统要求**
- Python 3.8+
- 足够的磁盘空间 (建议 > 1GB)
- 内存建议 > 4GB

### **2. 配置文件**
- 确保配置文件路径正确
- 检查所有必需参数是否设置
- 验证路径引用是否正确

### **3. 输出目录**
- 确保输出目录有写入权限
- 建议使用绝对路径
- 定期清理旧文件

## 🔍 故障排除

### **常见问题**
1. **配置文件错误**: 检查JSON格式和路径
2. **内存不足**: 减少episodes数量
3. **权限问题**: 检查输出目录权限
4. **依赖缺失**: 安装所需Python包

### **调试命令**
```bash
# 详细输出调试
python enhanced_city_simulation_v5_0.py --episodes 1 --verbose

# 检查配置
python enhanced_city_simulation_v5_0.py --config configs/city_config_v5_0.json --episodes 1 --verbose
```

## 📈 性能优化

### **训练加速**
- 使用GPU加速 (如果支持)
- 调整batch_size
- 优化网络结构

### **内存优化**
- 减少episodes数量
- 调整缓冲区大小
- 使用流式处理

## 🎯 最佳实践

1. **从小规模开始**: 先用少量episodes测试
2. **监控性能**: 使用`--performance_monitor`
3. **保存结果**: 使用`--save_results`
4. **对比验证**: 使用`--compare_v4`
5. **详细日志**: 使用`--verbose`

## 📝 总结

v5.0系统提供了完整的命令行接口，支持：
- ✅ 多种训练模式
- ✅ 性能监控
- ✅ 结果导出
- ✅ 模型评估
- ✅ 对比分析

推荐使用完整模式进行训练，可以获得最佳的训练效果和完整的输出结果！

