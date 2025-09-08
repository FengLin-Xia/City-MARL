#!/usr/bin/env python3
"""
调试Hub3地价场强度
检查Hub3周围的地价场数值是否达到等值线阈值
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_hub3_land_price():
    """分析Hub3周围的地价场强度"""
    
    # Hub3位置
    hub3_x, hub3_y = 67, 94
    
    # 检查几个关键月份
    months_to_check = [2, 10, 20, 30]
    
    print("=== Hub3 地价场强度分析 ===")
    print(f"Hub3 位置: ({hub3_x}, {hub3_y})")
    print()
    
    for month in months_to_check:
        try:
            # 读取地价场数据
            filename = f"enhanced_simulation_v3_1_output/land_price_frame_month_{month:02d}.json"
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            land_price_field = np.array(data['land_price_field'])
            
            # 检查Hub3位置的地价场值
            hub3_value = land_price_field[hub3_y, hub3_x]
            
            # 检查Hub3周围区域的地价场值
            radius = 10  # 检查10像素范围内的值
            max_value = 0
            high_value_points = []
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    x, y = hub3_x + dx, hub3_y + dy
                    if 0 <= x < 110 and 0 <= y < 110:
                        value = land_price_field[y, x]
                        max_value = max(max_value, value)
                        if value > 0.3:  # 高值区域
                            high_value_points.append((x, y, value))
            
            # 计算地价场的统计信息
            total_max = np.max(land_price_field)
            total_mean = np.mean(land_price_field)
            total_std = np.std(land_price_field)
            
            # 计算百分位数
            percentiles = [99, 95, 90, 80, 70, 60, 50]
            percentile_values = {}
            for p in percentiles:
                percentile_values[p] = np.percentile(land_price_field, p)
            
            print(f"--- Month {month} ---")
            print(f"Hub3 位置地价场值: {hub3_value:.4f}")
            print(f"Hub3 周围最大地价场值: {max_value:.4f}")
            print(f"全局最大地价场值: {total_max:.4f}")
            print(f"全局平均地价场值: {total_mean:.4f}")
            print(f"全局标准差: {total_std:.4f}")
            
            print("全局百分位数:")
            for p in percentiles:
                print(f"  {p}%: {percentile_values[p]:.4f}")
            
            # 检查Hub3是否达到等值线阈值
            commercial_threshold = percentile_values[99]  # 商业建筑阈值
            residential_threshold = percentile_values[80]  # 住宅建筑阈值
            
            print(f"商业建筑阈值 (99%): {commercial_threshold:.4f}")
            print(f"住宅建筑阈值 (80%): {residential_threshold:.4f}")
            
            if hub3_value >= commercial_threshold:
                print("✅ Hub3 达到商业建筑阈值")
            elif hub3_value >= residential_threshold:
                print("⚠️  Hub3 只达到住宅建筑阈值")
            else:
                print("❌ Hub3 未达到任何建筑阈值")
            
            print(f"Hub3 周围高值点数量 (>0.3): {len(high_value_points)}")
            if high_value_points:
                print("前5个高值点:")
                for i, (x, y, val) in enumerate(sorted(high_value_points, key=lambda x: x[2], reverse=True)[:5]):
                    print(f"  ({x}, {y}): {val:.4f}")
            
            print()
            
        except FileNotFoundError:
            print(f"❌ 文件不存在: {filename}")
        except Exception as e:
            print(f"❌ 读取文件出错: {e}")
    
    # 检查其他Hub的地价场值作为对比
    print("=== 对比其他Hub的地价场值 ===")
    hub1_x, hub1_y = 20, 55
    hub2_x, hub2_y = 90, 55
    
    try:
        # 使用第20月的数据进行对比
        filename = "enhanced_simulation_v3_1_output/land_price_frame_month_20.json"
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        land_price_field = np.array(data['land_price_field'])
        
        hub1_value = land_price_field[hub1_y, hub1_x]
        hub2_value = land_price_field[hub2_y, hub2_x]
        hub3_value = land_price_field[hub3_y, hub3_x]
        
        print(f"Hub1 ({hub1_x}, {hub1_y}): {hub1_value:.4f}")
        print(f"Hub2 ({hub2_x}, {hub2_y}): {hub2_value:.4f}")
        print(f"Hub3 ({hub3_x}, {hub3_y}): {hub3_value:.4f}")
        
        # 计算相对强度
        max_hub_value = max(hub1_value, hub2_value, hub3_value)
        print(f"Hub3 相对强度: {hub3_value/max_hub_value:.2%}")
        
    except Exception as e:
        print(f"❌ 对比分析出错: {e}")

if __name__ == "__main__":
    analyze_hub3_land_price()
