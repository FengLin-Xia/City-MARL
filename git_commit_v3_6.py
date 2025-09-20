#!/usr/bin/env python3
"""
Enhanced City Simulation v3.6 Git Commit Script
提交所有与 v3.6 系统相关的文件到 git
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并打印结果"""
    print(f"\n{'='*50}")
    if description:
        print(f"执行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        if result.stdout:
            print("输出:")
            print(result.stdout)
        if result.stderr:
            print("错误:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"执行失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 Enhanced City Simulation v3.6 Git Commit Script")
    print("=" * 60)
    
    # 检查是否在 git 仓库中
    if not os.path.exists('.git'):
        print("❌ 错误: 当前目录不是 git 仓库")
        sys.exit(1)
    
    # 定义要提交的文件列表
    v3_6_files = [
        # 核心系统文件
        "enhanced_city_simulation_v3_6.py",
        "visualize_building_placement_v3_6.py", 
        "test_finance_system.py",
        
        # PRD 文档
        "enhanced_city_simulation_prd_v3.6.txt",
        
        # 配置文件
        "configs/city_config_v3_5.json",
        "configs/city_config_v3_5_backup.json",
        "restore_original_config.py",
        
        # 核心逻辑模块
        "logic/enhanced_sdf_system.py",
        
        # 输出文件（选择性提交）
        "enhanced_simulation_v3_6_output/simplified/",
        "enhanced_simulation_v3_6_output/building_placement_animation_v3_6.gif",
        "enhanced_simulation_v3_6_output/finance_visualizations/",
    ]
    
    # 检查文件是否存在
    existing_files = []
    missing_files = []
    
    for file_path in v3_6_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} (不存在)")
    
    if missing_files:
        print(f"\n⚠️  警告: {len(missing_files)} 个文件不存在")
        choice = input("是否继续提交存在的文件? (y/N): ").strip().lower()
        if choice != 'y':
            print("取消提交")
            sys.exit(0)
    
    print(f"\n📁 准备提交 {len(existing_files)} 个文件/目录")
    
    # 确认提交
    print("\n" + "=" * 60)
    print("📋 提交清单:")
    for i, file_path in enumerate(existing_files, 1):
        print(f"{i:2d}. {file_path}")
    
    print("\n" + "=" * 60)
    choice = input("确认提交这些文件? (y/N): ").strip().lower()
    if choice != 'y':
        print("取消提交")
        sys.exit(0)
    
    # 执行 git 操作
    print("\n🔧 开始 Git 操作...")
    
    # 1. 检查 git 状态
    if not run_command("git status", "检查 Git 状态"):
        print("❌ Git 状态检查失败")
        sys.exit(1)
    
    # 2. 添加文件到暂存区
    for file_path in existing_files:
        if not run_command(f"git add \"{file_path}\"", f"添加文件: {file_path}"):
            print(f"❌ 添加文件失败: {file_path}")
            # 继续处理其他文件，不退出
    
    # 3. 检查暂存区状态
    run_command("git status", "检查暂存区状态")
    
    # 4. 提交信息
    commit_message = """feat: Enhanced City Simulation v3.6 - Complete System Implementation

🎯 核心特性:
- 单池槽位系统 + Hub外扩R(m)增长模式
- 分位数分类 + 月度锁定重判机制  
- 河流地形约束 + 侧边影响控制
- 工业后处理 + Hub2工业中心转换
- 三智能体财务评估系统完整实现

📊 可视化系统:
- 建筑放置动画 + R(m)环带显示
- 重判可视化 + 黄色三角标记
- 财务热力图 (利润密度/ROI/地价相关性)
- 综合财务仪表板 (4象限分析)

🔧 槽位生成系统:
- 每Hub独立槽位pattern (grid/hex/radial)
- 河流区域排除 + Hub同侧约束
- 可配置密度 + disjoint模式

💰 财务系统:
- 政府/企业/居民三方视角
- 月度财务CSV + 季度汇总JSON
- 工业转换后的收入/成本更新
- 英文标签专业可视化

📁 文件结构:
- enhanced_city_simulation_v3_6.py (核心系统)
- visualize_building_placement_v3_6.py (可视化)  
- enhanced_city_simulation_prd_v3.6.txt (PRD文档)
- simplified/simplified_buildings_XX.txt (简化输出)
- finance_visualizations/ (财务图表)

✨ 技术亮点:
- 延迟重判执行 + 非对称滞后
- 比例模式放置 + 软上限控制
- 河流Point-in-Polygon算法
- 混合地价场 (Hub+道路+河流边界)
- 工业建筑容量/成本/收入差异化

🧪 验证完成:
- 47个月长期模拟稳定运行
- 重判机制正常工作 (8月起生效)
- 财务数据完整输出
- 可视化动画流畅播放"""
    
    # 5. 执行提交
    if not run_command(f'git commit -m "{commit_message}"', "提交更改"):
        print("❌ 提交失败")
        sys.exit(1)
    
    print("\n✅ 提交成功!")
    print("\n🎉 Enhanced City Simulation v3.6 系统已成功提交到 Git!")
    
    # 6. 显示最新提交信息
    run_command("git log -1 --oneline", "显示最新提交")
    
    print("\n📋 后续建议:")
    print("1. 推送到远程仓库: git push origin main")
    print("2. 创建发布标签: git tag -a v3.6 -m 'Enhanced City Simulation v3.6'")
    print("3. 推送标签: git push origin v3.6")

if __name__ == "__main__":
    main()
