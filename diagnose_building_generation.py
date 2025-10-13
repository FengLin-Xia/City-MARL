#!/usr/bin/env python3
"""
诊断商业建筑生成和滞后替代系统问题
"""

import json
import numpy as np
from logic.enhanced_sdf_system import GaussianLandPriceSystem
from logic.isocontour_building_system import IsocontourBuildingSystem
from logic.hysteresis_system import HysteresisSystem

def diagnose_building_generation():
    """诊断建筑生成问题"""
    print("🔍 诊断商业建筑生成和滞后替代系统问题")
    print("=" * 80)
    
    # 加载配置
    config = json.load(open('configs/city_config_v3_1.json', encoding='utf-8'))
    
    # 初始化系统
    land_price_system = GaussianLandPriceSystem(config)
    isocontour_system = IsocontourBuildingSystem(config)
    hysteresis_system = HysteresisSystem(config)
    
    # 初始化系统
    transport_hubs = [[20, 55], [90, 55]]
    map_size = [110, 110]
    land_price_system.initialize_system(transport_hubs, map_size)
    
    print("1. 📊 等值线分析:")
    print("-" * 40)
    
    # 分析初始等值线
    land_price_field = land_price_system.get_land_price_field()
    isocontour_system.initialize_system(land_price_field, transport_hubs, map_size)
    contour_data = isocontour_system.get_contour_data_for_visualization()
    
    commercial_contours = contour_data.get('commercial_contours', [])
    residential_contours = contour_data.get('residential_contours', [])
    
    print(f"商业等值线数量: {len(commercial_contours)}")
    for i, contour in enumerate(commercial_contours):
        print(f"  等值线 {i+1}: 长度 {len(contour)}")
    
    print(f"住宅等值线数量: {len(residential_contours)}")
    for i, contour in enumerate(residential_contours):
        print(f"  等值线 {i+1}: 长度 {len(contour)}")
    
    print("\n2. 🏗️ 槽位生成分析:")
    print("-" * 40)
    
    # 模拟槽位生成
    def create_slots_from_contour(contour, building_type):
        """从等值线创建槽位"""
        if len(contour) < 20:
            return []
        
        # 等弧长采样
        if building_type == 'commercial':
            arc_spacing = 30  # 25-35m的平均值
        else:  # residential
            arc_spacing = 45  # 35-55m的平均值
        
        # 计算总弧长
        total_length = 0.0
        for i in range(len(contour) - 1):
            p1 = contour[i]
            p2 = contour[i + 1]
            distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            total_length += distance
        
        # 计算槽位数量
        num_slots = max(1, int(total_length / arc_spacing))
        return num_slots
    
    commercial_slots = 0
    for contour in commercial_contours:
        commercial_slots += create_slots_from_contour(contour, 'commercial')
    
    residential_slots = 0
    for contour in residential_contours:
        residential_slots += create_slots_from_contour(contour, 'residential')
    
    print(f"商业建筑槽位总数: {commercial_slots}")
    print(f"住宅建筑槽位总数: {residential_slots}")
    
    print("\n3. 🔄 滞后替代系统分析:")
    print("-" * 40)
    
    # 创建模拟城市状态
    city_state = {
        'residential': [
            {
                'id': 'res_1',
                'type': 'residential',
                'xy': [30, 55],
                'capacity': 200,
                'current_usage': 150
            },
            {
                'id': 'res_2',
                'type': 'residential',
                'xy': [40, 55],
                'capacity': 200,
                'current_usage': 180
            }
        ],
        'commercial': [
            {
                'id': 'com_1',
                'type': 'commercial',
                'xy': [50, 55],
                'capacity': 800,
                'current_usage': 600
            }
        ],
        'public': [],
        'residents': [],
        'transport_hubs': transport_hubs
    }
    
    # 测试滞后替代条件
    hysteresis_system.update_quarter(0)
    conversion_result = hysteresis_system.evaluate_conversion_conditions(city_state, land_price_system)
    
    print(f"滞后替代评估结果:")
    print(f"  应该转换: {conversion_result['should_convert']}")
    print(f"  原因: {conversion_result['reason']}")
    
    if 'candidates' in conversion_result:
        print(f"  候选建筑数量: {len(conversion_result['candidates'])}")
        for candidate in conversion_result['candidates']:
            print(f"    候选: {candidate['building_id']}, 评分差异: {candidate['score_difference']:.3f}")
    
    print("\n4. 📈 地价场演化分析:")
    print("-" * 40)
    
    # 分析地价场变化
    months_to_test = [0, 6, 12, 18, 23]
    
    for month in months_to_test:
        land_price_system.update_land_price_field(month)
        field = land_price_system.get_land_price_field()
        
        # 重新初始化等值线
        isocontour_system.initialize_system(field, transport_hubs, map_size)
        contour_data = isocontour_system.get_contour_data_for_visualization()
        
        commercial_contours = contour_data.get('commercial_contours', [])
        residential_contours = contour_data.get('residential_contours', [])
        
        print(f"月份 {month:2d}: 商业等值线 {len(commercial_contours)}, 住宅等值线 {len(residential_contours)}")
    
    print("\n5. 🎯 问题诊断:")
    print("-" * 40)
    
    # 诊断商业建筑生成问题
    if commercial_slots < 5:
        print("❌ 商业建筑槽位不足:")
        print("   - 商业等值线数量少")
        print("   - 等值线长度可能不够")
        print("   - arc_spacing设置可能过大")
    
    # 诊断滞后替代问题
    if not conversion_result['should_convert']:
        print("❌ 滞后替代条件不满足:")
        print(f"   - 原因: {conversion_result['reason']}")
        if 'consecutive_quarters' in conversion_result:
            print(f"   - 连续满足季度: {conversion_result['consecutive_quarters']}")
        if 'cooldown_remaining' in conversion_result:
            print(f"   - 冷却期剩余: {conversion_result['cooldown_remaining']}")
    
    print("\n6. 💡 建议解决方案:")
    print("-" * 40)
    
    print("商业建筑生成问题:")
    print("  1. 检查等值线分位数设置 (95, 90, 85)")
    print("  2. 调整arc_spacing参数 (当前25-35m)")
    print("  3. 降低min_segment_length_factor (当前3.0)")
    print("  4. 增加地价场强度")
    
    print("\n滞后替代问题:")
    print("  1. 检查delta_bid参数 (当前0.15)")
    print("  2. 检查L_quarters参数 (当前2)")
    print("  3. 检查res_min_share参数 (当前0.35)")
    print("  4. 确保有足够的住宅建筑")
    print("  5. 确保地价场变化足够大")

if __name__ == "__main__":
    diagnose_building_generation()


