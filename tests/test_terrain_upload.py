#!/usr/bin/env python3
"""
测试地形上传和训练流程
"""

import requests
import json
import numpy as np
import os
from pathlib import Path

def test_flask_upload():
    """测试Flask上传功能"""
    print("🧪 测试Flask地形上传功能...")
    
    # 创建一个简单的测试地形数据
    test_terrain = {
        'height_map': np.random.uniform(0, 100, (20, 20)).tolist(),
        'grid_size': [20, 20],
        'vertices_count': 400,
        'faces_count': 722
    }
    
    # 保存为临时JSON文件
    test_file = "test_terrain.json"
    with open(test_file, 'w') as f:
        json.dump(test_terrain, f)
    
    try:
        # 上传文件
        with open(test_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                "http://localhost:5000/upload_terrain",
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 地形上传测试成功!")
            print(f"📊 上传结果: {result}")
            return True
        else:
            print(f"❌ 上传测试失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 上传测试出错: {e}")
        return False
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)

def test_terrain_download():
    """测试地形下载功能"""
    print("\n📥 测试地形下载功能...")
    
    try:
        response = requests.get("http://localhost:5000/get_terrain")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 地形下载测试成功!")
            print(f"📊 地形信息: {result}")
            return True
        else:
            print(f"❌ 下载测试失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 下载测试出错: {e}")
        return False

def test_terrain_file_download():
    """测试地形文件下载功能"""
    print("\n💾 测试地形文件下载功能...")
    
    try:
        response = requests.get("http://localhost:5000/download_terrain")
        
        if response.status_code == 200:
            # 保存下载的文件
            with open("downloaded_terrain.json", "wb") as f:
                f.write(response.content)
            print("✅ 地形文件下载测试成功!")
            print("📁 文件已保存为: downloaded_terrain.json")
            return True
        else:
            print(f"❌ 文件下载测试失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 文件下载测试出错: {e}")
        return False

def create_sample_obj():
    """创建一个简单的OBJ文件用于测试"""
    print("\n📝 创建示例OBJ文件...")
    
    obj_content = """# Simple terrain mesh
v 0.0 0.0 0.0
v 1.0 0.0 0.5
v 0.0 1.0 0.3
v 1.0 1.0 0.8
v 0.5 0.5 1.0

f 1 2 3
f 2 4 3
f 3 4 5
f 1 3 5
f 1 5 2
f 2 5 4
"""
    
    with open("sample_terrain.obj", "w") as f:
        f.write(obj_content)
    
    print("✅ 示例OBJ文件已创建: sample_terrain.obj")
    return "sample_terrain.obj"

def test_obj_upload():
    """测试OBJ文件上传"""
    print("\n📤 测试OBJ文件上传...")
    
    obj_file = create_sample_obj()
    
    try:
        with open(obj_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                "http://localhost:5000/upload_terrain",
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ OBJ文件上传测试成功!")
            print(f"📊 处理结果: {result}")
            return True
        else:
            print(f"❌ OBJ上传测试失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ OBJ上传测试出错: {e}")
        return False
    finally:
        # 清理测试文件
        if os.path.exists(obj_file):
            os.remove(obj_file)

def main():
    """主测试函数"""
    print("🚀 地形上传和训练流程测试")
    print("=" * 50)
    
    # 检查Flask服务器是否运行
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            print("✅ Flask服务器正在运行")
        else:
            print("❌ Flask服务器响应异常")
            return
    except:
        print("❌ 无法连接到Flask服务器")
        print("请先启动服务器: python main.py")
        return
    
    # 运行测试
    tests = [
        ("JSON地形上传", test_flask_upload),
        ("地形数据获取", test_terrain_download),
        ("地形文件下载", test_terrain_file_download),
        ("OBJ文件上传", test_obj_upload)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # 显示测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过! 系统准备就绪")
        print("\n📋 下一步:")
        print("1. 在Blender中运行 blender_upload_terrain.py")
        print("2. 在IDE中运行 train_with_uploaded_terrain.py")
    else:
        print("⚠️ 部分测试失败，请检查系统配置")

if __name__ == "__main__":
    main()
