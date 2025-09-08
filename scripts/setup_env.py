#!/usr/bin/env python3
"""
MARL 项目环境设置脚本
自动安装依赖并验证环境配置
"""

import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            return True
        else:
            print(f"❌ {description} 失败")
            print(f"错误信息: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} 出错: {e}")
        return False

def check_python_version():
    """检查Python版本"""
    print(f"🐍 Python版本: {sys.version}")
    if sys.version_info >= (3, 8):
        print("✅ Python版本符合要求 (>=3.8)")
        return True
    else:
        print("❌ Python版本过低，需要3.8或更高版本")
        return False

def install_requirements():
    """安装依赖包"""
    print("\n📦 开始安装依赖包...")
    
    # 升级pip
    run_command("python -m pip install --upgrade pip", "升级pip")
    
    # 安装核心依赖
    success = run_command("pip install -r requirements-core.txt", "安装核心依赖")
    
    if success:
        print("✅ 核心依赖安装完成")
        return True
    else:
        print("❌ 核心依赖安装失败")
        return False

def verify_installation():
    """验证安装"""
    print("\n🔍 验证安装...")
    
    # 测试导入
    test_imports = [
        ("torch", "PyTorch"),
        ("torch.cuda", "CUDA支持"),
        ("pettingzoo", "PettingZoo"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("gymnasium", "Gymnasium"),
        ("pygame", "Pygame"),
        ("matplotlib", "Matplotlib"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
    ]
    
    all_success = True
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} 导入成功")
        except ImportError as e:
            print(f"❌ {name} 导入失败: {e}")
            all_success = False
    
    return all_success

def test_cuda():
    """测试CUDA"""
    print("\n🚀 测试CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
            
            # 测试CUDA张量
            x = torch.randn(2, 2).cuda()
            y = torch.randn(2, 2).cuda()
            z = torch.mm(x, y)
            print("✅ CUDA张量运算测试通过")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU")
            return False
    except Exception as e:
        print(f"❌ CUDA测试失败: {e}")
        return False

def test_environments():
    """测试环境"""
    print("\n🎮 测试强化学习环境...")
    
    # 测试PettingZoo
    try:
        from pettingzoo.mpe import simple_v3
        env = simple_v3.parallel_env()
        obs, _ = env.reset(seed=0)
        print("✅ PettingZoo环境测试通过")
    except Exception as e:
        print(f"❌ PettingZoo环境测试失败: {e}")
        return False
    
    # 测试Gymnasium
    try:
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=0)
        print("✅ Gymnasium环境测试通过")
    except Exception as e:
        print(f"❌ Gymnasium环境测试失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 MARL 项目环境设置")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 安装依赖
    if not install_requirements():
        print("\n❌ 依赖安装失败，请检查错误信息")
        sys.exit(1)
    
    # 验证安装
    if not verify_installation():
        print("\n❌ 依赖验证失败，请检查错误信息")
        sys.exit(1)
    
    # 测试CUDA
    test_cuda()
    
    # 测试环境
    if not test_environments():
        print("\n❌ 环境测试失败，请检查错误信息")
        sys.exit(1)
    
    print("\n🎉 环境设置完成！")
    print("=" * 50)
    print("✅ 所有依赖已安装并验证")
    print("✅ 强化学习环境正常工作")
    print("✅ 可以开始MARL项目开发")
    
    print("\n📝 使用说明:")
    print("1. 激活conda环境: conda activate city-marl")
    print("2. 运行测试: python tests/test_env.py")
    print("3. 开始开发: python main.py")

if __name__ == "__main__":
    main()
