#!/usr/bin/env python3
"""
测试Flask服务器与Blender接口的兼容性
"""

import requests
import json
import time

def test_flask_server():
    """测试Flask服务器"""
    base_url = "http://localhost:5000"
    
    # 测试1: 检查服务器是否运行
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            print("✅ 服务器运行正常")
            print(f"当前策略: {response.json().get('agent_strategy')}")
        else:
            print(f"❌ 服务器响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保Flask服务器已启动")
        return False
    except Exception as e:
        print(f"❌ 连接错误: {e}")
        return False
    
    # 测试2: 模拟Blender状态数据
    test_state = {
        "planes": [
            {
                "plane": 0,
                "group": 0,
                "color": "",
                "colorId": 1,
                "planesHeight": 4,
                "height": 2,
                "placements": [1, 2, -1, -1]
            },
            {
                "plane": 1,
                "group": 0,
                "color": "",
                "colorId": 2,
                "planesHeight": 3,
                "height": 1,
                "placements": [3, -1, -1]
            },
            {
                "plane": 2,
                "group": 1,
                "color": "",
                "colorId": 0,
                "planesHeight": 5,
                "height": 0,
                "placements": [-1, -1, -1, -1, -1]
            }
        ]
    }
    
    # 测试3: 发送状态并获取动作
    try:
        response = requests.post(
            f"{base_url}/llm_decide",
            json=test_state,
            timeout=10
        )
        
        if response.status_code == 200:
            action = response.json()
            print("✅ 成功获取动作决策")
            print(f"动作: {json.dumps(action, indent=2, ensure_ascii=False)}")
            
            # 验证动作格式
            required_keys = ["plane", "layer", "color"]
            if all(key in action for key in required_keys):
                print("✅ 动作格式正确")
            else:
                print("❌ 动作格式不正确")
                return False
                
        else:
            print(f"❌ 获取动作失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False
    
    # 测试4: 设置策略
    try:
        response = requests.post(
            f"{base_url}/set_strategy",
            json={"strategy": "greedy"},
            timeout=5
        )
        
        if response.status_code == 200:
            print("✅ 策略设置成功")
        else:
            print(f"❌ 策略设置失败: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 策略设置请求失败: {e}")
    
    # 测试5: 多次请求测试
    print("\n🔄 进行多次请求测试...")
    for i in range(3):
        try:
            response = requests.post(
                f"{base_url}/llm_decide",
                json=test_state,
                timeout=5
            )
            
            if response.status_code == 200:
                action = response.json()
                print(f"第{i+1}次请求: {action}")
            else:
                print(f"第{i+1}次请求失败: {response.status_code}")
                
        except Exception as e:
            print(f"第{i+1}次请求异常: {e}")
        
        time.sleep(1)  # 间隔1秒
    
    print("\n✅ 所有测试完成!")
    return True

if __name__ == "__main__":
    print("开始测试Flask服务器...")
    test_flask_server()
