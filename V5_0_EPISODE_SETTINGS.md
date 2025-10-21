# v5.0 Episode设置详解

## 🎯 Episode设置机制

### **1. 命令行参数设置**
```bash
# 基础设置
python enhanced_city_simulation_v5_0.py --episodes 100

# 默认值
python enhanced_city_simulation_v5_0.py  # 默认2轮
```

### **2. 配置文件设置**

#### **环境时间模型**
```json
"env": {
  "time_model": { 
    "step_unit": "month", 
    "total_steps": 30 
  }
}
```
- `step_unit`: 时间单位（月）
- `total_steps`: 每个episode的总步数（30个月）

#### **评估设置**
```json
"eval": {
  "frequency_steps": 5000,    // 每5000步评估一次
  "episodes": 8,              // 评估时使用8个episode
  "seeds": [101,102,103,104], // 评估种子
  "fixed_maps": ["map_A","map_B"]
}
```

#### **检查点设置**
```json
"checkpointing": {
  "save_best_metric": "eval/return_mean",
  "save_every_steps": 10000,  // 每10000步保存一次
  "max_to_keep": 5            // 最多保存5个检查点
}
```

## 📊 Episode处理流程

### **1. 训练管道中的Episode处理**

#### **初始化**
```python
initial_data = {
    "num_episodes": num_episodes,    # 总episode数
    "output_dir": output_dir,         # 输出目录
    "current_episode": 0,            # 当前episode
    "step_logs": [],                 # 步骤日志
    "env_states": []                 # 环境状态
}
```

#### **Episode循环**
```python
def _update_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
    # 更新当前轮次
    current_episode = data.get("current_episode", 0) + 1
    data["current_episode"] = current_episode
    
    # 更新全局状态
    self.pipeline.state_manager.update_global_state("current_episode", current_episode)
    self.pipeline.state_manager.update_global_state("training_phase", "episode_completed")
    
    print(f"  - Episode {current_episode} completed")
    return data
```

### **2. 每个Episode包含的步骤**

#### **Episode内部流程**
1. **环境重置** - 重置城市状态
2. **经验收集** - 收集智能体经验
3. **训练步骤** - 更新模型参数
4. **状态更新** - 更新系统状态
5. **结果导出** - 导出训练数据
6. **清理** - 清理临时数据

#### **Episode时间线**
- **每个Episode**: 30个月（根据`total_steps`配置）
- **每个智能体**: 按调度顺序执行
- **每个步骤**: 包含动作选择、执行、奖励计算

## 🔧 Episode设置方法

### **1. 命令行设置**
```bash
# 少量episode测试
python enhanced_city_simulation_v5_0.py --episodes 10

# 中等规模训练
python enhanced_city_simulation_v5_0.py --episodes 100

# 大规模训练
python enhanced_city_simulation_v5_0.py --episodes 1000
```

### **2. 配置文件设置**

#### **修改环境时间模型**
```json
"env": {
  "time_model": { 
    "step_unit": "month", 
    "total_steps": 50    // 每个episode 50个月
  }
}
```

#### **修改评估频率**
```json
"eval": {
  "frequency_steps": 1000,  // 每1000步评估
  "episodes": 5,            // 评估5个episode
  "seeds": [101,102,103,104,105]
}
```

#### **修改检查点频率**
```json
"checkpointing": {
  "save_every_steps": 5000,  // 每5000步保存
  "max_to_keep": 10          // 保存10个检查点
}
```

## 📈 Episode性能优化

### **1. 内存管理**
- 每个episode结束后清理临时数据
- 使用流式处理减少内存占用
- 定期保存检查点

### **2. 性能监控**
```bash
# 启用性能监控
python enhanced_city_simulation_v5_0.py --episodes 100 --performance_monitor
```

### **3. 并行处理**
- 支持多智能体并行训练
- 异步经验收集
- 批量模型更新

## 🎯 推荐Episode设置

### **1. 开发测试**
```bash
# 快速测试
python enhanced_city_simulation_v5_0.py --episodes 5 --verbose
```

### **2. 功能验证**
```bash
# 中等规模验证
python enhanced_city_simulation_v5_0.py --episodes 50 --performance_monitor
```

### **3. 正式训练**
```bash
# 大规模训练
python enhanced_city_simulation_v5_0.py --episodes 500 --performance_monitor --verbose
```

### **4. 生产环境**
```bash
# 生产级训练
python enhanced_city_simulation_v5_0.py --episodes 1000 --performance_monitor --compare_v4
```

## 📊 Episode监控

### **1. 训练进度**
- 当前episode数
- 完成百分比
- 预计剩余时间

### **2. 性能指标**
- 每episode平均奖励
- 训练损失
- 收敛速度

### **3. 资源使用**
- 内存占用
- CPU使用率
- 磁盘I/O

## 🔍 故障排除

### **1. Episode过少**
- 增加`--episodes`参数
- 检查配置文件设置
- 验证训练数据生成

### **2. Episode过多**
- 减少`--episodes`参数
- 调整检查点频率
- 优化内存使用

### **3. 性能问题**
- 启用性能监控
- 调整批处理大小
- 优化网络结构

## 📝 总结

v5.0系统的Episode设置非常灵活：

- ✅ **命令行控制**: 通过`--episodes`参数设置
- ✅ **配置文件**: 通过JSON配置详细参数
- ✅ **动态调整**: 支持运行时调整
- ✅ **性能监控**: 实时监控训练进度
- ✅ **检查点**: 自动保存训练状态

推荐从少量episode开始测试，然后逐步增加训练规模！🚀
