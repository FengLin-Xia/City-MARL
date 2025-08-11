#!/usr/bin/env python3
"""
CUDA和PyTorch配置检查脚本
检查虚拟环境中的CUDA和PyTorch是否正确配置
"""

import sys
import platform
import subprocess

def check_python_version():
    """检查Python版本"""
    print("=" * 50)
    print("Python版本信息:")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"平台: {platform.platform()}")
    print()

def check_pytorch():
    """检查PyTorch安装和版本"""
    print("=" * 50)
    print("PyTorch配置信息:")
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"PyTorch安装路径: {torch.__file__}")
        
        # 检查CUDA是否可用
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            
            # 显示每个GPU的信息
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # 检查当前设备
            current_device = torch.cuda.current_device()
            print(f"当前CUDA设备: {current_device}")
            
            # 测试CUDA张量
            try:
                x = torch.randn(3, 3).cuda()
                y = torch.randn(3, 3).cuda()
                z = torch.mm(x, y)
                print("CUDA张量运算测试: 成功")
            except Exception as e:
                print(f"CUDA张量运算测试: 失败 - {e}")
        else:
            print("CUDA不可用，PyTorch将使用CPU")
            
        # 检查MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"MPS (Apple Silicon) 可用: {torch.backends.mps.is_available()}")
        
    except ImportError as e:
        print(f"PyTorch未安装或导入失败: {e}")
    except Exception as e:
        print(f"检查PyTorch时出错: {e}")
    
    print()

def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print("=" * 50)
    print("NVIDIA驱动信息:")
    
    try:
        # 尝试运行nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("nvidia-smi输出:")
            print(result.stdout)
        else:
            print("nvidia-smi命令失败或未找到")
    except FileNotFoundError:
        print("nvidia-smi命令未找到，可能NVIDIA驱动未安装")
    except subprocess.TimeoutExpired:
        print("nvidia-smi命令超时")
    except Exception as e:
        print(f"检查NVIDIA驱动时出错: {e}")
    
    print()

def check_environment():
    """检查环境变量"""
    print("=" * 50)
    print("环境变量:")
    
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH']
    for var in cuda_vars:
        value = os.environ.get(var, '未设置')
        if value != '未设置' and len(value) > 100:
            value = value[:100] + "..."
        print(f"{var}: {value}")
    
    print()

def run_basic_tests():
    """运行基本的PyTorch测试"""
    print("=" * 50)
    print("基本PyTorch测试:")
    
    try:
        import torch
        
        # CPU测试
        print("CPU测试:")
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        z = torch.mm(x, y)
        print(f"CPU矩阵乘法结果:\n{z}")
        
        # CUDA测试
        if torch.cuda.is_available():
            print("\nCUDA测试:")
            x_cuda = torch.randn(2, 2).cuda()
            y_cuda = torch.randn(2, 2).cuda()
            z_cuda = torch.mm(x_cuda, y_cuda)
            print(f"CUDA矩阵乘法结果:\n{z_cuda}")
            
            # 测试设备间数据传输
            x_cpu = x_cuda.cpu()
            print("CUDA到CPU数据传输: 成功")
        
        print("所有基本测试通过!")
        
    except Exception as e:
        print(f"测试失败: {e}")
    
    print()

if __name__ == "__main__":
    import os
    
    print("CUDA和PyTorch配置检查")
    print("=" * 50)
    
    check_python_version()
    check_pytorch()
    check_nvidia_driver()
    check_environment()
    run_basic_tests()
    
    print("检查完成!")
