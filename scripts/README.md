# 脚本分类说明

本目录包含了项目中的各种测试、调试、分析和检查脚本，按功能分类组织。

## 📁 目录结构

### 🧪 testing/ - 测试脚本
包含各种功能测试脚本，用于验证系统各个组件的正确性。

**主要脚本：**
- `test_hub3.py` - Hub3功能测试
- `test_gaussian_land_price.py` - 高斯地价场测试
- `test_isocontour_extraction.py` - 等值线提取测试
- `test_v3_1_visualization.py` - v3.1可视化测试
- `test_long_term.py` - 长期模拟测试

### 🐛 debugging/ - 调试脚本
包含用于调试和排查问题的脚本。

**主要脚本：**
- `debug_hub3_*.py` - Hub3相关调试脚本
- `debug_building_generation_flow.py` - 建筑生成流程调试
- `debug_slot_activation.py` - 槽位激活调试
- `debug_visualization.py` - 可视化调试

### 📊 analysis/ - 分析脚本
包含用于数据分析和结果评估的脚本。

**主要脚本：**
- `analyze_hub_building_distribution.py` - Hub建筑分布分析
- `analyze_sdf_evolution.py` - SDF演化分析
- `analyze_land_price_distribution.py` - 地价分布分析
- `analyze_long_term.py` - 长期趋势分析

### ✅ checking/ - 检查脚本
包含用于验证和检查系统状态的脚本。

**主要脚本：**
- `check_hub3_*.py` - Hub3状态检查
- `check_building_counts.py` - 建筑数量检查
- `check_slot_usage.py` - 槽位使用检查
- `check_data_coverage.py` - 数据覆盖检查

## 🚀 使用说明

1. **运行测试脚本**：验证功能是否正常
2. **使用调试脚本**：排查问题根源
3. **执行分析脚本**：了解系统行为
4. **运行检查脚本**：验证系统状态

## 📝 注意事项

- 大部分脚本都是临时创建的，用于特定问题的调试
- 建议在运行前先查看脚本内容，了解其功能
- 某些脚本可能需要特定的数据文件或配置
- 调试完成后，可以考虑删除不再需要的临时脚本
