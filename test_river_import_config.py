#!/usr/bin/env python3
"""
测试v5.0河流导入配置

验证河流文件路径和格式是否正确
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_river_import_config():
    """测试河流导入配置"""
    print("=" * 60)
    print("测试v5.0河流导入配置")
    print("=" * 60)
    
    # 加载v5.0配置
    with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
        config_v5 = json.load(f)
    
    # 加载v4.1配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        config_v4 = json.load(f)
    
    print("\n1. v5.0河流配置检查:")
    paths = config_v5.get("paths", {})
    river_path = paths.get("river_geojson")
    print(f"   河流文件路径: {river_path}")
    
    # 检查文件是否存在
    if river_path and os.path.exists(river_path):
        print("   [PASS] 河流文件存在")
    else:
        print("   [FAIL] 河流文件不存在")
        print(f"   预期路径: {river_path}")
    
    print("\n2. v4.1河流配置检查:")
    terrain_features = config_v4.get("terrain_features", {})
    rivers = terrain_features.get("rivers", [])
    print(f"   v4.1河流数量: {len(rivers)}")
    
    if rivers:
        river_coords = rivers[0].get("coordinates", [])
        print(f"   v4.1河流坐标点数量: {len(river_coords)}")
        if river_coords:
            print(f"   第一个坐标点: {river_coords[0]}")
            print(f"   最后一个坐标点: {river_coords[-1]}")
        print("   [PASS] v4.1河流数据存在")
    else:
        print("   [FAIL] v4.1河流数据不存在")
    
    print("\n3. 河流数据格式对比:")
    
    # 检查v4.1的河流数据格式
    if rivers and rivers[0].get("coordinates"):
        v4_coords = rivers[0]["coordinates"]
        print(f"   v4.1格式: 坐标数组，共{len(v4_coords)}个点")
        print(f"   示例坐标: {v4_coords[:3]}")
    
    # 检查是否存在river.txt文件
    river_txt_path = "river.txt"
    if os.path.exists(river_txt_path):
        print(f"   [INFO] 发现river.txt文件")
        with open(river_txt_path, 'r') as f:
            lines = f.readlines()
        print(f"   river.txt行数: {len(lines)}")
        if lines:
            print(f"   第一行: {lines[0].strip()}")
            print(f"   最后一行: {lines[-1].strip()}")
    
    print("\n4. 配置问题分析:")
    
    # 检查v5.0配置中的河流相关设置
    env_config = config_v5.get("env", {})
    river_restrictions = env_config.get("river_restrictions", {})
    river_premium = env_config.get("river_premium", {})
    
    print("   v5.0河流相关配置:")
    print(f"     river_restrictions.enabled: {river_restrictions.get('enabled', 'NOT_FOUND')}")
    print(f"     river_restrictions.affects_agents: {river_restrictions.get('affects_agents', 'NOT_FOUND')}")
    print(f"     river_premium.RiverPmax_pct: {river_premium.get('RiverPmax_pct', 'NOT_FOUND')}")
    
    print("\n5. 修复建议:")
    
    # 检查是否需要修复配置
    issues = []
    
    if not river_path or not os.path.exists(river_path):
        issues.append("河流文件路径不存在")
    
    if not river_restrictions.get('enabled'):
        issues.append("河流限制功能未启用")
    
    if not river_premium.get('RiverPmax_pct'):
        issues.append("河流溢价参数缺失")
    
    if issues:
        print("   发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"     {i}. {issue}")
        
        print("\n   修复建议:")
        print("   1. 检查河流文件路径是否正确")
        print("   2. 确认河流数据格式是否匹配")
        print("   3. 验证河流相关功能配置")
    else:
        print("   [PASS] 河流配置正常")
    
    print("\n6. 河流数据转换建议:")
    
    # 检查是否需要数据转换
    if os.path.exists("river.txt") and (not river_path or not os.path.exists(river_path)):
        print("   建议将river.txt转换为GeoJSON格式:")
        print("   1. 读取river.txt中的坐标数据")
        print("   2. 转换为GeoJSON格式")
        print("   3. 保存为data/river.geojson")
        print("   4. 更新配置文件路径")
    
    print("\n" + "=" * 60)
    print("河流导入配置测试完成!")
    print("=" * 60)
    
    return len(issues) == 0


def test_river_data_conversion():
    """测试河流数据转换"""
    print("\n" + "=" * 60)
    print("测试河流数据转换")
    print("=" * 60)
    
    # 检查river.txt文件
    if not os.path.exists("river.txt"):
        print("   [FAIL] river.txt文件不存在")
        return False
    
    print("   读取river.txt文件...")
    with open("river.txt", 'r') as f:
        lines = f.readlines()
    
    print(f"   总行数: {len(lines)}")
    
    # 解析坐标数据
    coordinates = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    coordinates.append([x, y])
                except ValueError:
                    continue
    
    print(f"   解析坐标点数量: {len(coordinates)}")
    
    if coordinates:
        print(f"   第一个坐标: {coordinates[0]}")
        print(f"   最后一个坐标: {coordinates[-1]}")
        
        # 创建GeoJSON格式
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": "river",
                        "type": "river"
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    }
                }
            ]
        }
        
        # 保存GeoJSON文件
        os.makedirs("data", exist_ok=True)
        with open("data/river.geojson", 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        
        print("   [PASS] 河流数据转换成功")
        print("   保存为: data/river.geojson")
        return True
    else:
        print("   [FAIL] 无法解析河流坐标数据")
        return False


if __name__ == "__main__":
    try:
        # 测试河流导入配置
        success1 = test_river_import_config()
        
        # 测试河流数据转换
        success2 = test_river_data_conversion()
        
        if success1 and success2:
            print("\n" + "=" * 60)
            print("所有测试通过!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("部分测试失败，请检查配置!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
