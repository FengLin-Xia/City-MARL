#!/usr/bin/env python3
"""
测试有序边界上传功能
"""

import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_test_obj_file():
    """创建一个简单的测试OBJ文件"""
    obj_content = """# Test terrain with boundary
v 0.0 0.0 0.0
v 10.0 0.0 0.0
v 10.0 10.0 0.0
v 0.0 10.0 0.0
v 3.0 3.0 0.0
v 7.0 3.0 0.0
v 7.0 7.0 0.0
v 3.0 7.0 0.0
f 1 2 3
f 1 3 4
f 5 6 7
f 5 7 8
"""
    
    test_file = Path("test_terrain.obj")
    with open(test_file, 'w') as f:
        f.write(obj_content)
    
    return test_file

def create_test_ordered_boundary():
    """创建测试用的有序边界数据"""
    return {
        'boundary_loops': [
            # 主边界
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]],
            # 内部空洞
            [[3.0, 3.0, 0.0], [7.0, 3.0, 0.0], [7.0, 7.0, 0.0], [3.0, 7.0, 0.0]]
        ],
        'loop_count': 2,
        'total_points': 8
    }

def test_upload_with_ordered_boundary():
    """测试带有序边界的上传"""
    print("🧪 测试有序边界上传功能...")
    
    # 创建测试文件
    test_file = create_test_obj_file()
    print(f"✅ 创建测试OBJ文件: {test_file}")
    
    # 创建有序边界数据
    ordered_boundary = create_test_ordered_boundary()
    print(f"✅ 创建有序边界数据: {ordered_boundary['loop_count']} 个循环")
    
    # 准备上传数据
    files = {'file': open(test_file, 'rb')}
    data = {
        'ordered_boundary': json.dumps(ordered_boundary)
    }
    
    try:
        # 上传到Flask服务器
        print("🔄 上传到Flask服务器...")
        response = requests.post(
            "http://localhost:5000/upload_terrain",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 上传成功!")
            print(f"📊 地形信息: {result.get('terrain_info', {})}")
            
            # 检查返回的地形数据
            terrain_info = result.get('terrain_info', {})
            if 'mask' in terrain_info:
                mask = np.array(terrain_info['mask'])
                print(f"✅ 掩码创建成功")
                print(f"   掩码形状: {mask.shape}")
                print(f"   有效点数: {np.sum(mask)} / {mask.size}")
                print(f"   覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
                
                # 可视化掩码
                visualize_mask(mask, terrain_info.get('grid_size', [150, 150]))
            
            return result
        else:
            print(f"❌ 上传失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Flask服务器")
        print("请确保服务器已启动: python main.py")
        return None
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        return None
    finally:
        # 清理测试文件
        try:
            test_file.unlink()
            print(f"🧹 清理测试文件: {test_file}")
        except:
            pass

def visualize_mask(mask, grid_size):
    """可视化掩码"""
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 显示掩码
        im = ax.imshow(mask.T, cmap='gray', origin='lower', aspect='equal')
        ax.set_title('有序边界掩码')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig("test_ordered_boundary_mask.png", dpi=300, bbox_inches='tight')
        print("✅ 掩码可视化已保存到: test_ordered_boundary_mask.png")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

def test_get_terrain():
    """测试获取地形数据"""
    try:
        response = requests.get("http://localhost:5000/get_terrain")
        
        if response.status_code == 200:
            result = response.json()
            print("📊 当前地形数据:")
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"❌ 获取地形数据失败: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 获取地形数据出错: {e}")
        return None

if __name__ == "__main__":
    print("🚀 测试有序边界上传功能")
    print("=" * 50)
    
    # 测试上传
    result = test_upload_with_ordered_boundary()
    
    if result:
        print("\n📊 测试结果:")
        print(f"   状态: {result.get('status', 'unknown')}")
        print(f"   消息: {result.get('message', 'no message')}")
        
        # 测试获取地形数据
        print("\n🔄 测试获取地形数据...")
        terrain_data = test_get_terrain()
        
        if terrain_data:
            print("✅ 地形数据获取成功")
        else:
            print("❌ 地形数据获取失败")
    
    print("\n✅ 测试完成")
