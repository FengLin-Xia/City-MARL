#!/usr/bin/env python3
"""
快速查看模拟结果
避免matplotlib崩溃问题
"""

import json
from pathlib import Path

def show_simulation_results():
    """显示模拟结果"""
    print("📊 Enhanced City Simulation Results")
    print("=" * 50)
    
    try:
        # 加载最终城市状态
        with open('enhanced_simulation_output/city_state_output.json', 'r', encoding='utf-8') as f:
            city_state = json.load(f)
        
        # 加载每日统计数据
        with open('enhanced_simulation_output/daily_stats.json', 'r', encoding='utf-8') as f:
            daily_stats = json.load(f)
        
        print("✅ 数据加载成功")
        
    except FileNotFoundError as e:
        print(f"❌ 数据文件未找到: {e}")
        return
    
    # 显示基本信息
    simulation_info = city_state['simulation_info']
    print(f"\n🏙️ 模拟基本信息:")
    print(f"  模拟时长: {simulation_info['day']} 个月")
    print(f"  最终人口: {simulation_info['total_residents']} 人")
    print(f"  建筑总数: {simulation_info['total_buildings']} 个")
    print(f"  平均地价: {simulation_info['average_land_price']:.1f}")
    
    # 显示建筑分布
    buildings = city_state['buildings']
    print(f"\n🏗️ 建筑分布:")
    print(f"  公共建筑: {len(buildings['public'])} 个")
    for i, building in enumerate(buildings['public']):
        print(f"    {i+1}. {building['id']}: 位置 {building['xy']}, 容量 {building['capacity']}")
    
    print(f"  住宅建筑: {len(buildings['residential'])} 个")
    for i, building in enumerate(buildings['residential']):
        print(f"    {i+1}. {building['id']}: 位置 {building['xy']}, 容量 {building['capacity']}, 使用 {building['current_usage']}")
    
    print(f"  商业建筑: {len(buildings['commercial'])} 个")
    for i, building in enumerate(buildings['commercial']):
        print(f"    {i+1}. {building['id']}: 位置 {building['xy']}, 容量 {building['capacity']}, 使用 {building['current_usage']}")
    
    # 显示地价统计
    land_prices = city_state['land_prices']
    print(f"\n💰 地价统计:")
    print(f"  最高地价: {land_prices['max_price']:.1f}")
    print(f"  最低地价: {land_prices['min_price']:.1f}")
    print(f"  平均地价: {land_prices['avg_price']:.1f}")
    
    # 显示地价分布
    price_dist = land_prices['price_distribution']
    print(f"  地价分布:")
    for range_key, count in price_dist.items():
        print(f"    {range_key}: {count} 个位置")
    
    # 显示人口增长
    print(f"\n📈 人口增长:")
    initial_pop = daily_stats[0]['population']
    final_pop = daily_stats[-1]['population']
    growth_rate = ((final_pop - initial_pop) / initial_pop) * 100
    print(f"  初始人口: {initial_pop} 人")
    print(f"  最终人口: {final_pop} 人")
    print(f"  增长率: {growth_rate:.1f}%")
    
    # 显示建筑增长
    print(f"\n🏗️ 建筑增长:")
    initial_buildings = daily_stats[0]['public_buildings'] + daily_stats[0]['residential_buildings'] + daily_stats[0]['commercial_buildings']
    final_buildings = daily_stats[-1]['public_buildings'] + daily_stats[-1]['residential_buildings'] + daily_stats[-1]['commercial_buildings']
    print(f"  初始建筑: {initial_buildings} 个")
    print(f"  最终建筑: {final_buildings} 个")
    print(f"  新增建筑: {final_buildings - initial_buildings} 个")
    
    # 显示关键时间点
    print(f"\n⏰ 关键时间点:")
    for i, stats in enumerate(daily_stats):
        if i % 6 == 0 or i < 5:  # 显示前5个月和每6个月
            month = stats['month']
            pop = stats['population']
            total_buildings = stats['public_buildings'] + stats['residential_buildings'] + stats['commercial_buildings']
            print(f"  第 {month} 个月: 人口 {pop}, 建筑 {total_buildings}")
    
    # 检查图片文件
    print(f"\n📁 可视化文件:")
    image_dir = Path('enhanced_simulation_output/images')
    if image_dir.exists():
        image_files = sorted(image_dir.glob('month_*.png'))
        print(f"  渲染图片: {len(image_files)} 个")
        print(f"  渲染频率: 每月一次")
        print(f"  图片范围: Month 0 - Month {len(image_files)-1}")
        
        # 检查是否有缺失的图片
        expected_months = set(range(0, simulation_info['day']+1, 1))
        actual_months = set()
        for img_file in image_files:
            month = int(img_file.stem.split('_')[1])
            actual_months.add(month)
        
        missing_months = expected_months - actual_months
        if missing_months:
            print(f"  ⚠️ 缺失的渲染帧: {sorted(missing_months)}")
        else:
            print(f"  ✅ 所有渲染帧完整")
    else:
        print(f"  ❌ 图片目录不存在")

def show_file_structure():
    """显示输出文件结构"""
    print(f"\n📂 输出文件结构:")
    output_dir = Path('enhanced_simulation_output')
    if output_dir.exists():
        for item in output_dir.rglob('*'):
            if item.is_file():
                rel_path = item.relative_to(output_dir)
                size = item.stat().st_size
                print(f"  {rel_path} ({size} bytes)")
    else:
        print(f"  ❌ 输出目录不存在")

def main():
    """主函数"""
    show_simulation_results()
    show_file_structure()
    
    print(f"\n💡 提示:")
    print(f"1. 查看 'enhanced_simulation_output/final_city_layout.png' 获取最终城市布局")
    print(f"2. 查看 'enhanced_simulation_output/images/' 目录获取所有渲染帧")
    print(f"3. 使用图片查看器逐帧查看城市演化过程")

if __name__ == "__main__":
    main()
