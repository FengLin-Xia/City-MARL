#!/usr/bin/env python3
"""
测试配置文件加载
"""

import json

def test_config_loading():
    """测试配置文件加载"""
    print("=== 测试配置文件加载 ===")
    
    try:
        # 尝试UTF-16编码
        with open('configs/city_config_v5_0.json', 'r', encoding='utf-16') as f:
            config = json.load(f)
        
        print("PASS: 配置文件加载成功 (UTF-16)")
        print(f"  - schema_version: {config.get('schema_version')}")
        print(f"  - multi_action.enabled: {config.get('multi_action', {}).get('enabled', False)}")
        print(f"  - max_actions_per_step: {config.get('multi_action', {}).get('max_actions_per_step', 'N/A')}")
        return config
        
    except Exception as e:
        print(f"FAIL: 配置文件加载失败: {e}")
        return None

if __name__ == "__main__":
    config = test_config_loading()
    if config:
        print("\n配置测试通过！")
    else:
        print("\n配置测试失败！")
"""
测试配置文件加载
"""

import json

def test_config_loading():
    """测试配置文件加载"""
    print("=== 测试配置文件加载 ===")
    
    try:
        # 尝试UTF-16编码
        with open('configs/city_config_v5_0.json', 'r', encoding='utf-16') as f:
            config = json.load(f)
        
        print("PASS: 配置文件加载成功 (UTF-16)")
        print(f"  - schema_version: {config.get('schema_version')}")
        print(f"  - multi_action.enabled: {config.get('multi_action', {}).get('enabled', False)}")
        print(f"  - max_actions_per_step: {config.get('multi_action', {}).get('max_actions_per_step', 'N/A')}")
        return config
        
    except Exception as e:
        print(f"FAIL: 配置文件加载失败: {e}")
        return None

if __name__ == "__main__":
    config = test_config_loading()
    if config:
        print("\n配置测试通过！")
    else:
        print("\n配置测试失败！")
