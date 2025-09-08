#!/usr/bin/env python3
"""
创建进一步优化的哑铃状配置
大幅缩小hub影响范围
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from enhanced_city_simulation_v3_3 import EnhancedCitySimulationV3_3

def create_ultra_compact_dumbbell_config():
    """创建超紧凑的哑铃状配置"""
    print("🔧 创建超紧凑哑铃状配置...")
    
    # 超紧凑配置
    config_ultra_compact = {
        'city': {
            'map_size': [110, 110],
            'meters_per_pixel': 2.0,
            'trunk_road': [[20, 55], [90, 55]],
            'transport_hubs': [[25, 55], [85, 55]]  # 进一步扩大距离
        },
        'government_backbone': {
            'road_corridor': {
                'sigma_perp_m': 40,
                'setback_m': {'commercial': 8, 'residential': 10, 'industrial': 14}
            },
            'hubs': {
                'commercial': {'sigma_perp_m': 15, 'sigma_parallel_m': 45},  # 大幅减小
                'industrial': {'sigma_perp_m': 18, 'sigma_parallel_m': 54}
            },
            'zoning': {
                'hub_com_radius_m': 80,   # 大幅减小分区半径
                'hub_ind_radius_m': 100,
                'mid_corridor_residential': True
            },
            'quotas_per_quarter': {
                'residential': [10, 20, 15, 25],
                'commercial': [5, 12, 8, 15],
                'industrial': [4, 10, 6, 12]
            },
            'strict_layering': True,
            'dead_slots_ratio_max': 0.05
        },
        'gaussian_land_price_system': {
            'w_r': 0.6, 'w_c': 0.5, 'w_i': 0.5, 'w_cor': 0.2, 'bias': 0.0,
            'hub_sigma_base_m': 15,  # 大幅减小基础σ
            'road_sigma_base_m': 20,
            'hub_growth_rate': 0.03, 'road_growth_rate': 0.02,
            'max_hub_multiplier': 1.2,  # 减小最大倍数
            'max_road_multiplier': 2.0,
            'normalize': True, 'smoothstep_tau': 0.0
        },
        'scoring_weights': {
            'commercial': {
                'f_price': 0.35, 'f_hub_com': 0.25, 'f_road': 0.20,
                'f_heat': 0.15, 'f_access': 0.05,
                'crowding': -0.03, 'junction_penalty': -0.02
            },
            'industrial': {
                'f_price': -0.20, 'f_hub_ind': 0.45, 'f_road': 0.25,
                'f_access': 0.05, 'crowding': -0.10, 'junction_penalty': -0.05
            },
            'residential': {
                'f_price': 0.10, 'f_road': 0.45, 'f_access': 0.15,
                'f_hub_com': -0.15, 'f_hub_ind': -0.10, 'crowding': -0.05
            }
        },
        'isocontour_layout': {
            'commercial': {'levels': [0.9, 0.85, 0.8], 'arc_spacing_m': [25, 35]},  # 更高阈值
            'industrial': {'levels': [0.5, 0.6, 0.7], 'arc_spacing_m': [35, 55]},  # 更低阈值
            'residential': {'band': [0.45, 0.65], 'arc_spacing_m': [35, 55]},
            'normal_offset_m': 1.0, 'jitter_m': 0.5
        }
    }
    
    # 保存配置文件
    with open('configs/city_config_v3_3_ultra_compact.json', 'w') as f:
        json.dump(config_ultra_compact, f, indent=2)
    
    print("✅ 已创建超紧凑配置文件: configs/city_config_v3_3_ultra_compact.json")
    
    # 显示关键改进
    print("\n🔧 超紧凑配置改进:")
    print(f"  1. 枢纽距离: 108米 → 120米 (60像素)")
    print(f"  2. 商业枢纽影响范围: 200米 → 80米")
    print(f"  3. 工业枢纽影响范围: 250米 → 100米")
    print(f"  4. 基础σ: 30米 → 15米")
    print(f"  5. 最大倍数: 1.5 → 1.2")
    print(f"  6. 商业等值线阈值: [0.85,0.78,0.71] → [0.9,0.85,0.8]")
    print(f"  7. 工业等值线阈值: [0.6,0.7,0.8] → [0.5,0.6,0.7]")
    
    return config_ultra_compact

def create_extreme_dumbbell_config():
    """创建极端哑铃状配置"""
    print("\n🔧 创建极端哑铃状配置...")
    
    # 极端配置
    config_extreme = {
        'city': {
            'map_size': [110, 110],
            'meters_per_pixel': 2.0,
            'trunk_road': [[20, 55], [90, 55]],
            'transport_hubs': [[22, 55], [88, 55]]  # 最大距离
        },
        'government_backbone': {
            'road_corridor': {
                'sigma_perp_m': 40,
                'setback_m': {'commercial': 8, 'residential': 10, 'industrial': 14}
            },
            'hubs': {
                'commercial': {'sigma_perp_m': 12, 'sigma_parallel_m': 36},  # 极小
                'industrial': {'sigma_perp_m': 15, 'sigma_parallel_m': 45}
            },
            'zoning': {
                'hub_com_radius_m': 60,   # 极小分区半径
                'hub_ind_radius_m': 80,
                'mid_corridor_residential': True
            },
            'quotas_per_quarter': {
                'residential': [10, 20, 15, 25],
                'commercial': [5, 12, 8, 15],
                'industrial': [4, 10, 6, 12]
            },
            'strict_layering': True,
            'dead_slots_ratio_max': 0.05
        },
        'gaussian_land_price_system': {
            'w_r': 0.6, 'w_c': 0.5, 'w_i': 0.5, 'w_cor': 0.2, 'bias': 0.0,
            'hub_sigma_base_m': 12,  # 极小基础σ
            'road_sigma_base_m': 20,
            'hub_growth_rate': 0.03, 'road_growth_rate': 0.02,
            'max_hub_multiplier': 1.1,  # 极小最大倍数
            'max_road_multiplier': 2.0,
            'normalize': True, 'smoothstep_tau': 0.0
        },
        'scoring_weights': {
            'commercial': {
                'f_price': 0.35, 'f_hub_com': 0.25, 'f_road': 0.20,
                'f_heat': 0.15, 'f_access': 0.05,
                'crowding': -0.03, 'junction_penalty': -0.02
            },
            'industrial': {
                'f_price': -0.20, 'f_hub_ind': 0.45, 'f_road': 0.25,
                'f_access': 0.05, 'crowding': -0.10, 'junction_penalty': -0.05
            },
            'residential': {
                'f_price': 0.10, 'f_road': 0.45, 'f_access': 0.15,
                'f_hub_com': -0.15, 'f_hub_ind': -0.10, 'crowding': -0.05
            }
        },
        'isocontour_layout': {
            'commercial': {'levels': [0.95, 0.9, 0.85], 'arc_spacing_m': [25, 35]},  # 极高阈值
            'industrial': {'levels': [0.4, 0.5, 0.6], 'arc_spacing_m': [35, 55]},  # 极低阈值
            'residential': {'band': [0.45, 0.65], 'arc_spacing_m': [35, 55]},
            'normal_offset_m': 1.0, 'jitter_m': 0.5
        }
    }
    
    # 保存配置文件
    with open('configs/city_config_v3_3_extreme.json', 'w') as f:
        json.dump(config_extreme, f, indent=2)
    
    print("✅ 已创建极端配置文件: configs/city_config_v3_3_extreme.json")
    
    # 显示关键改进
    print("\n🔧 极端配置改进:")
    print(f"  1. 枢纽距离: 120米 → 132米 (66像素)")
    print(f"  2. 商业枢纽影响范围: 80米 → 60米")
    print(f"  3. 工业枢纽影响范围: 100米 → 80米")
    print(f"  4. 基础σ: 15米 → 12米")
    print(f"  5. 最大倍数: 1.2 → 1.1")
    print(f"  6. 商业等值线阈值: [0.9,0.85,0.8] → [0.95,0.9,0.85]")
    print(f"  7. 工业等值线阈值: [0.5,0.6,0.7] → [0.4,0.5,0.6]")
    
    return config_extreme

def test_compact_configs():
    """测试紧凑配置"""
    print("\n🧪 测试紧凑配置...")
    
    configs = [
        ('configs/city_config_v3_3_ultra_compact.json', '超紧凑配置'),
        ('configs/city_config_v3_3_extreme.json', '极端配置')
    ]
    
    results = []
    
    for config_file, config_name in configs:
        print(f"\n📊 测试 {config_name}...")
        
        # 加载配置
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # 创建模拟系统
        simulation = EnhancedCitySimulationV3_3(config)
        simulation.initialize_simulation()
        
        # 运行短期模拟（6个月）
        simulation.run_simulation(total_months=6)
        
        # 分析结果
        result = analyze_compact_results(config_file, config_name)
        results.append((config_name, result))
    
    # 比较结果
    compare_compact_results(results)

def analyze_compact_results(config_file, config_name):
    """分析紧凑配置结果"""
    import os
    
    # 重建建筑状态
    output_dir = 'enhanced_simulation_v3_3_output'
    buildings = rebuild_building_state(output_dir, 6)
    
    # 获取枢纽位置
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    hub_com = config['city']['transport_hubs'][0]
    hub_ind = config['city']['transport_hubs'][1]
    
    # 计算距离
    distance = np.sqrt((hub_ind[0] - hub_com[0])**2 + (hub_ind[1] - hub_com[1])**2) * 2.0
    
    # 统计建筑
    total_buildings = 0
    building_counts = {}
    for building_type, building_list in buildings.items():
        if building_type != 'public':
            count = len(building_list)
            building_counts[building_type] = count
            total_buildings += count
    
    # 分析集聚效果
    com_distances = []
    ind_distances = []
    
    for building_type, building_list in buildings.items():
        if building_type == 'public':
            continue
            
        for building in building_list:
            pos = building['xy']
            
            dist_to_com = np.sqrt((pos[0] - hub_com[0])**2 + (pos[1] - hub_com[1])**2) * 2.0
            dist_to_ind = np.sqrt((pos[0] - hub_ind[0])**2 + (pos[1] - hub_ind[1])**2) * 2.0
            
            com_distances.append(dist_to_com)
            ind_distances.append(dist_to_ind)
    
    # 获取分区半径
    hub_com_radius = config['government_backbone']['zoning']['hub_com_radius_m']
    hub_ind_radius = config['government_backbone']['zoning']['hub_ind_radius_m']
    
    # 计算集聚效果
    com_nearby = sum(1 for d in com_distances if d <= hub_com_radius)
    ind_nearby = sum(1 for d in ind_distances if d <= hub_ind_radius)
    overlap_count = sum(1 for d_com, d_ind in zip(com_distances, ind_distances) 
                       if d_com <= hub_com_radius and d_ind <= hub_ind_radius)
    
    result = {
        'distance': distance,
        'total_buildings': total_buildings,
        'building_counts': building_counts,
        'com_nearby': com_nearby,
        'ind_nearby': ind_nearby,
        'overlap_count': overlap_count,
        'overlap_ratio': overlap_count / total_buildings if total_buildings > 0 else 0,
        'hub_com_radius': hub_com_radius,
        'hub_ind_radius': hub_ind_radius
    }
    
    print(f"  {config_name} 结果:")
    print(f"    枢纽距离: {distance:.1f}米")
    print(f"    总建筑数: {total_buildings}")
    print(f"    建筑分布: {building_counts}")
    print(f"    商业枢纽{hub_com_radius}m内: {com_nearby}个 ({com_nearby/total_buildings*100:.1f}%)")
    print(f"    工业枢纽{hub_ind_radius}m内: {ind_nearby}个 ({ind_nearby/total_buildings*100:.1f}%)")
    print(f"    重叠区域: {overlap_count}个 ({overlap_count/total_buildings*100:.1f}%)")
    
    return result

def rebuild_building_state(output_dir, target_month):
    """重建完整的建筑状态"""
    import os
    # 加载基础状态
    with open(os.path.join(output_dir, 'building_positions_month_00.json'), 'r') as f:
        base_data = json.load(f)
    
    buildings = base_data['buildings'].copy()
    
    # 应用增量更新
    for month in range(1, target_month + 1):
        delta_file = os.path.join(output_dir, f'building_delta_month_{month:02d}.json')
        if os.path.exists(delta_file):
            with open(delta_file, 'r') as f:
                delta_data = json.load(f)
            
            for building in delta_data.get('new_buildings', []):
                building_type = building['building_type']
                buildings[building_type].append(building)
    
    return buildings

def compare_compact_results(results):
    """比较紧凑配置结果"""
    print("\n📊 紧凑配置对比分析:")
    print("=" * 80)
    
    for config_name, result in results:
        print(f"\n{config_name}:")
        print(f"  枢纽距离: {result['distance']:.1f}米")
        print(f"  影响范围: 商业{result['hub_com_radius']}m, 工业{result['hub_ind_radius']}m")
        print(f"  总建筑数: {result['total_buildings']}")
        print(f"  重叠比例: {result['overlap_ratio']:.1%}")
        
        if result['overlap_ratio'] < 0.3:
            print("  ✅ 成功形成哑铃状分布！")
        elif result['overlap_ratio'] < 0.5:
            print("  ⚠️ 部分形成哑铃状分布")
        else:
            print("  ❌ 仍有较多重叠")
    
    # 推荐最佳配置
    best_config = min(results, key=lambda x: x[1]['overlap_ratio'])
    print(f"\n🏆 推荐配置: {best_config[0]}")
    print(f"   重叠比例: {best_config[1]['overlap_ratio']:.1%}")

def main():
    """主函数"""
    print("🔧 创建超紧凑哑铃状配置")
    
    # 创建超紧凑配置
    ultra_compact_config = create_ultra_compact_dumbbell_config()
    
    # 创建极端配置
    extreme_config = create_extreme_dumbbell_config()
    
    # 测试配置
    test_compact_configs()
    
    print("\n✅ 配置创建和测试完成！")
    print("  生成的文件:")
    print("  - configs/city_config_v3_3_ultra_compact.json: 超紧凑配置")
    print("  - configs/city_config_v3_3_extreme.json: 极端配置")

if __name__ == "__main__":
    main()
