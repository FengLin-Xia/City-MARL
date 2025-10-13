# 城市仿真Demo - 快速入门指南

## 🚀 5分钟快速上手

### 第一步：检查环境
```bash
# 确保在项目目录下
cd marl

# 检查必要文件是否存在
ls configs/colors.json data/poi_example.json logic/ viz/
```

### 第二步：安装依赖
```bash
# 安装核心依赖
pip install matplotlib pillow numpy
```

### 第三步：运行快速测试
```bash
# 运行3天仿真（约1-2分钟）
python quick_test.py
```

### 第四步：查看结果
```bash
# 查看动画播放
python view_results.py
# 选择 2 观看完整动画
```

### 第五步：生成分享文件
```bash
# 生成GIF文件
python create_gif.py
# 选择 2 生成按天动画
```

## 🎯 预期效果

运行成功后，你应该看到：

### 1. 仿真过程
- 控制台显示仿真进度
- 生成 `output_frames/` 目录
- 每天生成多个PNG图片

### 2. 动画播放
- 居民（白点）在城市中移动
- 热力图显示移动轨迹
- POI设施用不同颜色标识
- 新增设施用红色边框高亮

### 3. 城市发展
- 热力走廊逐渐形成
- 政府添加学校和诊所
- 企业扩容住宅和零售设施
- 城市布局不断优化

## 🔍 观察要点

### 第一眼看到什么？
1. **两个蓝色大点**：城市枢纽
2. **灰色线条**：主干线
3. **彩色小点**：各类POI设施
4. **白色移动点**：居民
5. **紫色热力图**：移动轨迹

### 发展过程观察
1. **第1天**：基础布局，居民开始移动
2. **第3-5天**：热力走廊形成
3. **第7-10天**：设施优化，城市成熟

## 🛠️ 常见问题

### Q: 没有找到图片文件？
A: 确保先运行仿真：
```bash
python quick_test.py
```

### Q: 动画播放卡顿？
A: 尝试按天播放模式：
```bash
python view_results.py
# 选择 3
```

### Q: GIF文件太大？
A: 使用快速预览模式：
```bash
python create_gif.py
# 选择 3
```

### Q: 想修改参数？
A: 编辑 `main_demo.py` 中的参数：
```python
self.days = 5  # 改为5天
self.daily_residents = 50  # 改为50人/天
```

## 📊 快速验证

### 检查清单
- [ ] 仿真正常运行
- [ ] 生成PNG图片
- [ ] 动画播放流畅
- [ ] GIF文件生成
- [ ] 热力图显示正常
- [ ] POI颜色正确

### 成功标志
- 控制台显示"仿真完成"
- `output_frames/` 目录有图片
- 动画中看到居民移动
- 热力图有颜色变化

## 🎮 下一步探索

### 1. 运行完整仿真
```bash
python main_demo.py  # 10天完整仿真
```

### 2. 尝试不同模式
```bash
# 修改移动方式
# 在main_demo.py中设置 self.movement_mode = "astar"

# 调整参数
# 修改logic/placement.py中的阈值
```

### 3. 自定义配置
- 修改 `data/poi_example.json` 中的初始布局
- 调整 `configs/colors.json` 中的颜色
- 更改仿真参数和决策规则

## 📚 深入学习

### 阅读文档
- `CITY_SIMULATION_DEMO.md` - 完整技术文档
- `ANIMATION_GUIDE.md` - 动画功能指南
- `README_demo.md` - 项目介绍

### 理解代码
- `main_demo.py` - 主仿真逻辑
- `logic/` - 核心算法实现
- `viz/ide.py` - 可视化引擎

### 扩展功能
- 添加新的智能体类型
- 实现更复杂的决策逻辑
- 优化可视化效果

## 🎉 恭喜！

你已经成功运行了城市仿真Demo！

这个系统展示了：
- **智能体建模**的基本原理
- **城市发展**的动态过程
- **可视化技术**的应用
- **仿真系统**的设计思路

继续探索，发现更多有趣的现象！🏙️✨



