#!/usr/bin/env python3
"""
诊断可视化中的建筑突变问题
"""

import json
from pathlib import Path

def analyze_building_evolution():
    """分析建筑演化过程"""
    print("🔍 分析建筑演化过程...")
    
    # 加载每日统计数据
    with open('enhanced_simulation_output/daily_stats.json', 'r', encoding='utf-8') as f:
        daily_stats = json.load(f)
    
    print(f"📊 总天数: {len(daily_stats)}")
    
    # 分析建筑数量变化
    print("\n🏗️ 建筑数量变化:")
    print("天数 | 公共建筑 | 住宅建筑 | 商业建筑 | 总建筑数")
    print("-" * 50)
    
    for i, stats in enumerate(daily_stats):
        if i % 30 == 0 or i < 10:  # 显示前10天和每30天
            day = stats['day']
            public = stats['public_buildings']
            residential = stats['residential_buildings']
            commercial = stats['commercial_buildings']
            total = public + residential + commercial
            print(f"{day:3d} | {public:8d} | {residential:8d} | {commercial:8d} | {total:8d}")
    
    # 检查突变点
    print("\n🚨 检查建筑数量突变:")
    for i in range(1, len(daily_stats)):
        prev = daily_stats[i-1]
        curr = daily_stats[i]
        
        prev_total = prev['public_buildings'] + prev['residential_buildings'] + prev['commercial_buildings']
        curr_total = curr['public_buildings'] + curr['residential_buildings'] + curr['commercial_buildings']
        
        if curr_total != prev_total:
            print(f"第 {curr['day']} 天: {prev_total} -> {curr_total} (变化: {curr_total - prev_total})")
            print(f"  公共: {prev['public_buildings']} -> {curr['public_buildings']}")
            print(f"  住宅: {prev['residential_buildings']} -> {curr['residential_buildings']}")
            print(f"  商业: {prev['commercial_buildings']} -> {curr['commercial_buildings']}")

def analyze_final_city_state():
    """分析最终城市状态"""
    print("\n🏙️ 分析最终城市状态...")
    
    # 加载最终城市状态
    with open('enhanced_simulation_output/city_state_output.json', 'r', encoding='utf-8') as f:
        city_state = json.load(f)
    
    buildings = city_state['buildings']
    
    print(f"📊 最终建筑统计:")
    print(f"公共建筑: {len(buildings['public'])} 个")
    print(f"住宅建筑: {len(buildings['residential'])} 个")
    print(f"商业建筑: {len(buildings['commercial'])} 个")
    
    print(f"\n🏠 住宅建筑详情:")
    for building in buildings['residential']:
        print(f"  {building['id']}: 位置 {building['xy']}, 容量 {building['capacity']}, 使用 {building['current_usage']}")
    
    print(f"\n🏪 商业建筑详情:")
    for building in buildings['commercial']:
        print(f"  {building['id']}: 位置 {building['xy']}, 容量 {building['capacity']}, 使用 {building['current_usage']}")
    
    print(f"\n🏛️ 公共建筑详情:")
    for building in buildings['public']:
        print(f"  {building['id']}: 位置 {building['xy']}, 容量 {building['capacity']}, 使用 {building['current_usage']}")

def check_rendering_frequency():
    """检查渲染频率"""
    print("\n🎬 检查渲染频率...")
    
    # 检查图片文件
    image_dir = Path('enhanced_simulation_output/images')
    if image_dir.exists():
        image_files = sorted(image_dir.glob('day_*.png'))
        print(f"📁 找到 {len(image_files)} 个渲染图片")
        
        if len(image_files) > 0:
            print("前10个渲染帧:")
            for i, img_file in enumerate(image_files[:10]):
                day = int(img_file.stem.split('_')[1])
                print(f"  {i+1}. {img_file.name} (第{day}天)")
            
            if len(image_files) > 10:
                print("最后10个渲染帧:")
                for i, img_file in enumerate(image_files[-10:]):
                    day = int(img_file.stem.split('_')[1])
                    print(f"  {len(image_files)-9+i}. {img_file.name} (第{day}天)")

def main():
    """主函数"""
    print("🔍 可视化问题诊断工具")
    print("=" * 50)
    
    analyze_building_evolution()
    analyze_final_city_state()
    check_rendering_frequency()
    
    print("\n💡 问题分析:")
    print("1. 如果建筑数量在某个时间点突然增加，说明企业智能体建设了新建筑")
    print("2. 如果可视化中建筑突然消失，可能是因为:")
    print("   - 渲染帧显示的是中间状态，而不是最终状态")
    print("   - 建筑ID生成逻辑导致重复ID")
    print("   - 可视化脚本读取了错误的数据文件")
    print("3. 建议检查渲染频率和建筑ID的唯一性")

if __name__ == "__main__":
    main()
