#!/usr/bin/env python3
"""
Conda环境设置脚本
自动创建和配置conda环境
"""

import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}成功")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description}失败")
            print(f"错误信息: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description}出错: {e}")
        return False
    return True

def main():
    """主函数"""
    print("🚀 Conda环境设置脚本")
    print("=" * 50)
    
    # 检查conda是否可用
    if not run_command("conda --version", "检查conda版本"):
        print("❌ 未找到conda，请先安装Anaconda或Miniconda")
        return
    
    # 检查环境是否已存在
    env_name = "city-marl"
    result = subprocess.run(f"conda env list | grep {env_name}", shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"⚠️ 环境 {env_name} 已存在")
        choice = input("是否重新创建环境? (y/n): ").strip().lower()
        if choice == 'y':
            if not run_command(f"conda env remove -n {env_name} -y", f"删除现有环境 {env_name}"):
                return
        else:
            print("🔄 使用现有环境")
            if not run_command(f"conda activate {env_name}", f"激活环境 {env_name}"):
                return
            print("✅ 环境设置完成")
            return
    
    # 创建新环境
    print(f"🔄 创建conda环境: {env_name}")
    
    # 方法1：使用environment.yml
    if os.path.exists("environment.yml"):
        print("📦 使用environment.yml创建环境...")
        if not run_command(f"conda env create -f environment.yml", "从environment.yml创建环境"):
            print("❌ 使用environment.yml创建失败，尝试手动创建...")
            # 方法2：手动创建
            if not run_command(f"conda create -n {env_name} python=3.10 -y", f"创建环境 {env_name}"):
                return
            if not run_command(f"conda activate {env_name}", f"激活环境 {env_name}"):
                return
            if not run_command(f"conda install -n {env_name} pip -y", "安装pip"):
                return
            if not run_command(f"pip install -r requirements-core.txt", "安装核心依赖"):
                return
    else:
        print("📦 手动创建环境...")
        if not run_command(f"conda create -n {env_name} python=3.10 -y", f"创建环境 {env_name}"):
            return
        if not run_command(f"conda activate {env_name}", f"激活环境 {env_name}"):
            return
        if not run_command(f"conda install -n {env_name} pip -y", "安装pip"):
            return
        if not run_command(f"pip install -r requirements-core.txt", "安装核心依赖"):
            return
    
    print("\n✅ Conda环境设置完成!")
    print(f"🎯 环境名称: {env_name}")
    print("\n📋 使用方法:")
    print(f"  激活环境: conda activate {env_name}")
    print(f"  退出环境: conda deactivate")
    print(f"  查看环境: conda env list")
    print(f"  删除环境: conda env remove -n {env_name}")

if __name__ == "__main__":
    main()
