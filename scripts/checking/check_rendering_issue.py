#!/usr/bin/env python3
"""
检查Day 80-85的渲染问题
"""

import json
from pathlib import Path

def check_rendering_data():
    """检查渲染数据的一致性"""
    print("🔍 检查Day 80-85的渲染数据...")
    
    # 加载最终城市状态
    with open('enhanced_simulation_output/city_state_output.json', 'r', encoding='utf-8') as f:
        final_city_state = json.load(f)
    
    print(f"📊 最终城市状态:")
    print(f"公共建筑: {len(final_city_state['buildings']['public'])} 个")
    print(f"住宅建筑: {len(final_city_state['buildings']['residential'])} 个")
    print(f"商业建筑: {len(final_city_state['buildings']['commercial'])} 个")
    print(f"居民: {len(final_city_state['residents'])} 人")
    
    # 检查渲染脚本中使用的数据结构
    print(f"\n🏗️ 渲染脚本使用的数据结构:")
    print(f"public_pois = city_state['public']")
    print(f"residential_pois = city_state['residential']") 
    print(f"retail_pois = city_state['commercial']")
    
    # 检查数据结构是否匹配
    buildings = final_city_state['buildings']
    print(f"\n🔍 数据结构检查:")
    print(f"city_state['public'] 存在: {'public' in final_city_state}")
    print(f"city_state['residential'] 存在: {'residential' in final_city_state}")
    print(f"city_state['commercial'] 存在: {'commercial' in final_city_state}")
    
    if 'public' in final_city_state:
        print(f"city_state['public'] 长度: {len(final_city_state['public'])}")
    if 'residential' in final_city_state:
        print(f"city_state['residential'] 长度: {len(final_city_state['residential'])}")
    if 'commercial' in final_city_state:
        print(f"city_state['commercial'] 长度: {len(final_city_state['commercial'])}")
    
    print(f"city_state['buildings']['public'] 长度: {len(buildings['public'])}")
    print(f"city_state['buildings']['residential'] 长度: {len(buildings['residential'])}")
    print(f"city_state['buildings']['commercial'] 长度: {len(buildings['commercial'])}")

def check_rendering_script():
    """检查渲染脚本的逻辑"""
    print(f"\n🎬 检查渲染脚本逻辑...")
    
    print(f"渲染脚本中的数据结构访问:")
    print(f"public_pois = self.city_state['public']")
    print(f"residential_pois = self.city_state['residential']")
    print(f"retail_pois = self.city_state['commercial']")
    
    print(f"\n⚠️ 潜在问题:")
    print(f"1. 渲染脚本访问的是 city_state['public'] 等")
    print(f"2. 但实际数据存储在 city_state['buildings']['public'] 等")
    print(f"3. 这可能导致渲染时找不到建筑数据！")

def check_image_files():
    """检查图片文件"""
    print(f"\n📁 检查图片文件...")
    
    image_dir = Path('enhanced_simulation_output/images')
    if image_dir.exists():
        # 检查Day 80-85的图片
        for day in range(80, 86):
            img_file = image_dir / f'day_{day:03d}.png'
            if img_file.exists():
                size = img_file.stat().st_size
                print(f"Day {day}: {img_file.name} ({size} bytes)")
            else:
                print(f"Day {day}: 文件不存在")

def main():
    """主函数"""
    print("🔍 Day 80-85渲染问题诊断")
    print("=" * 50)
    
    check_rendering_data()
    check_rendering_script()
    check_image_files()
    
    print(f"\n💡 问题分析:")
    print(f"如果Day 80-85的建筑突然消失，可能的原因是:")
    print(f"1. 渲染脚本访问了错误的数据路径")
    print(f"2. 数据结构不匹配导致渲染失败")
    print(f"3. 某些帧的渲染过程中出现了异常")

if __name__ == "__main__":
    main()
