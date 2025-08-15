#!/usr/bin/env python3
"""
Flask客户端测试脚本
演示如何向Flask服务器发送请求
"""

import requests
import json
import time

# Flask服务器地址
BASE_URL = "http://localhost:5000"

def test_llm_decide():
    """测试主要决策接口"""
    print("🧪 测试 /llm_decide 接口...")
    
    # 模拟Blender发送的状态数据
    test_state = {
        "planes": [
            {
                "height": 2,
                "planesHeight": 4,
                "color": 1
            },
            {
                "height": 0,
                "planesHeight": 4,
                "color": 0
            },
            {
                "height": 1,
                "planesHeight": 4,
                "color": 2
            }
        ],
        "heights": [2, 0, 1],
        "placements": [[1, 2], [0, 0], [2, 1]]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/llm_decide",
            json=test_state,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            action = response.json()
            print(f"✅ 成功接收动作: {action}")
            return action
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Flask服务器")
        print("请确保服务器已启动: python main.py")
        return None
    except Exception as e:
        print(f"❌ 请求出错: {e}")
        return None

def test_status():
    """测试状态查询接口"""
    print("\n📊 测试 /status 接口...")
    
    try:
        response = requests.get(f"{BASE_URL}/status")
        
        if response.status_code == 200:
            status = response.json()
            print(f"✅ 服务器状态: {status}")
            return status
        else:
            print(f"❌ 状态查询失败: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 状态查询出错: {e}")
        return None

def test_set_strategy():
    """测试策略设置接口"""
    print("\n⚙️ 测试 /set_strategy 接口...")
    
    strategies = ["random", "greedy"]
    
    for strategy in strategies:
        try:
            response = requests.post(
                f"{BASE_URL}/set_strategy",
                json={"strategy": strategy},
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 策略设置成功: {strategy}")
                print(f"   结果: {result}")
            else:
                print(f"❌ 策略设置失败: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 策略设置出错: {e}")

def simulate_blender_communication():
    """模拟Blender与Flask的通信"""
    print("\n🎮 模拟Blender通信...")
    
    # 模拟多个回合的通信
    for round_num in range(3):
        print(f"\n--- 第 {round_num + 1} 回合 ---")
        
        # 生成随机状态
        import random
        num_planes = random.randint(2, 5)
        planes = []
        
        for i in range(num_planes):
            plane = {
                "height": random.randint(0, 3),
                "planesHeight": 4,
                "color": random.randint(0, 4)
            }
            planes.append(plane)
        
        state = {
            "planes": planes,
            "heights": [p["height"] for p in planes],
            "placements": [[i, p["height"]] for i, p in enumerate(planes)]
        }
        
        print(f"📤 发送状态: {len(planes)} 个planes")
        
        # 发送请求
        action = test_llm_decide()
        if action:
            print(f"📥 接收动作: {action}")
        
        time.sleep(1)  # 模拟时间间隔

def main():
    """主测试函数"""
    print("🚀 Flask客户端测试开始")
    print("=" * 50)
    
    # 测试各个接口
    test_status()
    test_set_strategy()
    test_llm_decide()
    
    # 模拟完整通信
    simulate_blender_communication()
    
    print("\n" + "=" * 50)
    print("✅ 测试完成!")

if __name__ == "__main__":
    main()
