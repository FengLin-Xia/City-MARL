# 动作表格生成指南

## 📋 功能说明

为每个月的动作生成可视化表格，显示：
- 每个动作的详细信息（Agent、Size、Slot、Cost、Reward）
- Budget变化（每个动作后的预算）
- 总计行（总Cost、总Reward、最终Budget）

---

## 🎨 表格格式

### **表格列：**
| # | Agent | Size | Slot | Cost | Reward | Budget |
|---|-------|------|------|------|--------|--------|
| 1 | IND | L | s_232 | 1100 | 117 | 5000 → 4017 |
| 2 | IND | M | s_233 | 952 | 110 | 4017 → 3175 |
| 3 | IND | S | s_181 | 574 | 100 | 3175 → 2701 |
| **Total** | | | | **2626** | **327** | **Final: 2701** |

### **样式：**
- ✅ 白色文字
- ✅ 透明背景
- ✅ 白色边框
- ✅ 表头和总计行加粗

---

## 🚀 使用方法

### **1. 生成v4.0的表格**
```bash
python generate_action_tables.py --mode v4.0
```

**输出：**
- `enhanced_simulation_v4_0_output/action_tables/month_XX_IND.png`
- `enhanced_simulation_v4_0_output/action_tables/month_XX_EDU.png`

---

### **2. 生成v4.1的表格**
```bash
python generate_action_tables.py --mode v4.1
```

**输出：**
- `enhanced_simulation_v4_1_output/action_tables/month_XX_IND.png`
- `enhanced_simulation_v4_1_output/action_tables/month_XX_EDU.png`

---

### **3. 同时生成v4.0和v4.1**
```bash
python generate_action_tables.py --mode both
```

---

### **4. 限制处理月份数**
```bash
python generate_action_tables.py --mode v4.0 --max_months 10
```

---

### **5. 指定自定义路径**
```bash
python generate_action_tables.py --mode v4.1 --v4_1_history models/v4_1_rl/slot_selection_history.json
```

---

## 📊 Budget计算逻辑

### **当前实现（模拟）：**
```python
# 初始budget
IND: 5000 kGBP
EDU: 4000 kGBP

# 每个动作后
budget = budget - cost + reward

# 示例
Month 0, Action 1 (IND):
  初始: 5000
  Cost: -1100
  Reward: +117
  最终: 5000 - 1100 + 117 = 4017
```

### **未来实现（真实Budget系统）：**
当Budget系统实现后，可以：
1. 从环境中读取真实的budget值
2. 显示负债情况（红色标记）
3. 显示破产警告

---

## 🎯 表格用途

### **1. 演示文档**
- 清晰展示每月的决策过程
- 适合PPT、报告

### **2. 调试分析**
- 快速查看cost/reward分布
- 发现异常动作

### **3. 策略对比**
- 对比v4.0（参数化）vs v4.1（RL）
- 对比不同训练轮次的策略

### **4. 视频制作**
- 逐月播放表格
- 展示城市发展过程

---

## 📁 输出文件结构

```
enhanced_simulation_v4_0_output/
  action_tables/
    month_00_IND.png  ← Month 0, IND的5个动作
    month_01_EDU.png  ← Month 1, EDU的动作
    month_02_IND.png
    ...

enhanced_simulation_v4_1_output/
  action_tables/
    month_00_IND.png  ← RL模型的决策
    month_00_EDU.png
    ...
```

---

## ⚙️ 自定义选项

### **修改初始Budget：**
```python
# 在 generate_action_tables.py 中
budgets = {'IND': 5000, 'EDU': 4000}  # 修改这里
```

### **修改表格样式：**
```python
# 字体大小
table.set_fontsize(11)  # 改为其他值

# 表格高度
fig_height = len(table_data) * 0.6 + 1.5  # 调整系数

# 列宽
colWidths=[0.08, 0.12, 0.08, 0.15, 0.15, 0.15, 0.27]  # 调整比例
```

### **修改数值格式：**
```python
# Cost/Reward保留小数
str(cost)  # 改为 f"{cost:.1f}"

# Budget格式
f"{before} → {after}"  # 改为其他格式
```

---

## 🔧 集成到主程序（可选）

如果想让v4.0/v4.1自动生成表格，可以：

### **在v4.0主循环末尾添加：**
```python
# 在 enhanced_city_simulation_v4_0.py 的 main() 函数末尾
from generate_action_tables import process_v4_0_output
process_v4_0_output(out_dir, total_months)
```

### **在v4.1评估后添加：**
```python
# 在 enhanced_city_simulation_v4_1.py 的 evaluate_rl_model() 函数末尾
from generate_action_tables import process_v4_1_output
process_v4_1_output('models/v4_1_rl/slot_selection_history.json')
```

---

## ✅ 完成状态

- [x] 创建表格生成函数
- [x] 支持v4.0格式
- [x] 支持v4.1格式
- [x] Budget列（模拟）
- [x] 白色文字、透明背景
- [x] 总计行
- [x] 命令行参数
- [x] 测试通过

---

## 🚀 下一步

**等v4.0运行完成后：**
```bash
# 生成所有月份的表格
python generate_action_tables.py --mode both
```

**查看生成的表格：**
- `enhanced_simulation_v4_0_output/action_tables/`
- `enhanced_simulation_v4_1_output/action_tables/`

---

**文档维护者：** AI Assistant  
**最后更新：** 2025-10-09




