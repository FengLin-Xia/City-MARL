#!/usr/bin/env python3
"""
修复SDF参数 - 实现渐进式城市发展
1. 增加Hub影响半径，确保两个hub之间有SDF重叠
2. 添加SDF演化配置，实现线SDF随时间扩展
"""

import json
import os

def fix_sdf_parameters():
    """修复SDF系统参数"""
    
    print("🔧 修复SDF参数 - 实现渐进式城市发展")
    print("=" * 50)
    
    # 配置文件路径
    config_file = 'configs/city_config_v2_3.json'
    
    # 检查配置文件是否存在
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    try:
        # 加载配置文件
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✅ 加载配置文件成功: {config_file}")
        
        # 获取米到像素的转换比例
        meters_per_pixel = config.get('sdf_system', {}).get('meters_per_pixel', 2.0)
        print(f"📏 米到像素转换比例: {meters_per_pixel} m/px")
        
        # 创建新的配置结构
        new_config = config.copy()
        
        # 1. 修复SDF系统参数
        if 'sdf_system' not in new_config:
            new_config['sdf_system'] = {}
        
        sdf_system = new_config['sdf_system']
        
        # 修复Hub影响半径 - 确保两个hub之间有SDF重叠
        # 两个hub距离176px，需要影响半径至少88px (176/2)
        # 为了有良好的重叠，设置为120px (240m)
        sdf_system['lambda_point_m'] = 240  # 从100m增加到240m
        sdf_system['lambda_point_px'] = int(240 / meters_per_pixel)  # 120px
        
        # 道路法向衰减 - 初始较小，随时间扩展
        sdf_system['lambda_perp_m'] = 80  # 从120m减少到80m，初始较小
        sdf_system['lambda_perp_px'] = int(80 / meters_per_pixel)  # 40px
        
        # 道路切向衰减 - 控制沿道路方向的扩展
        sdf_system['lambda_tangential_m'] = 150  # 从200m减少到150m
        sdf_system['lambda_tangential_px'] = int(150 / meters_per_pixel)  # 75px
        
        # 启用切向衰减
        sdf_system['use_tangential_decay'] = True
        
        # 添加SDF演化配置
        sdf_system['evolution'] = {
            "enabled": True,
            "road_expansion_rate": 0.15,  # 每季度道路SDF扩展15%
            "max_road_influence": 3.0,    # 最大扩展倍数
            "evolution_stages": {
                "initial": {
                    "months": [0, 6],     # 0-6个月：初始阶段
                    "road_multiplier": 1.0,  # 道路SDF不扩展
                    "description": "初始阶段：只有Hub点SDF，道路SDF最小"
                },
                "early_growth": {
                    "months": [6, 12],    # 6-12个月：早期增长
                    "road_multiplier": 1.5,  # 道路SDF开始扩展
                    "description": "早期增长：道路SDF开始扩展，住宅区形成"
                },
                "mid_growth": {
                    "months": [12, 18],   # 12-18个月：中期增长
                    "road_multiplier": 2.0,  # 道路SDF显著扩展
                    "description": "中期增长：道路SDF显著扩展，商业区扩张"
                },
                "mature": {
                    "months": [18, 24],   # 18-24个月：成熟阶段
                    "road_multiplier": 2.5,  # 道路SDF最大扩展
                    "description": "成熟阶段：道路SDF最大扩展，城市完全发展"
                }
            }
        }
        
        print(f"📐 SDF系统参数修复:")
        print(f"   λ_hub_m = {sdf_system['lambda_point_m']} m → λ_hub_px = {sdf_system['lambda_point_px']} px")
        print(f"   λ⊥_m = {sdf_system['lambda_perp_m']} m → λ⊥_px = {sdf_system['lambda_perp_px']} px")
        print(f"   λ∥_m = {sdf_system['lambda_tangential_m']} m → λ∥_px = {sdf_system['lambda_tangential_px']} px")
        print(f"   启用切向衰减: {sdf_system['use_tangential_decay']}")
        
        # 2. 修复等值线布局参数 - 解决法向偏移和切向抖动为0的问题
        if 'isocontour_layout' not in new_config:
            new_config['isocontour_layout'] = {}
        
        isocontour_layout = new_config['isocontour_layout']
        
        # 增加最小偏移值，避免为0
        isocontour_layout['normal_offset_m'] = 2.0  # 从1.0m增加到2.0m
        isocontour_layout['jitter_m'] = 1.0         # 从0.5m增加到1.0m
        
        # 重新计算像素单位参数
        isocontour_layout['normal_offset_px'] = int(2.0 / meters_per_pixel)  # 1px
        isocontour_layout['jitter_px'] = int(1.0 / meters_per_pixel)         # 0.5px，取整为1px
        
        print(f"🏗️ 等值线布局参数修复:")
        print(f"   法向偏移: {isocontour_layout['normal_offset_m']} m → {isocontour_layout['normal_offset_px']} px")
        print(f"   切向抖动: {isocontour_layout['jitter_m']} m → {isocontour_layout['jitter_px']} px")
        
        # 3. 添加SDF可视化配置
        if 'visualization' not in new_config:
            new_config['visualization'] = {}
        
        visualization = new_config['visualization']
        visualization['sdf_evolution_visualization'] = {
            "enabled": True,
            "save_sdf_frames": True,
            "sdf_frames_interval": 3,  # 每3个月保存一帧SDF
            "color_maps": {
                "hub_sdf": "Reds",      # Hub SDF用红色
                "road_sdf": "Blues",    # 道路SDF用蓝色
                "combined_sdf": "RdYlBu_r"  # 组合SDF用红黄蓝
            },
            "include_contours": True,
            "contour_levels": [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        print(f"🎨 SDF演化可视化配置:")
        print(f"   保存SDF帧: {visualization['sdf_evolution_visualization']['save_sdf_frames']}")
        print(f"   帧间隔: {visualization['sdf_evolution_visualization']['sdf_frames_interval']} 个月")
        
        # 保存修复后的配置
        backup_file = config_file.replace('.json', '_sdf_fixed.json')
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 原配置备份到: {backup_file}")
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 修复后的配置保存到: {config_file}")
        
        # 4. 验证关键参数
        print(f"\n🔍 关键参数验证:")
        print(f"   两个hub距离: 176 px = {176 * meters_per_pixel} m")
        print(f"   Hub影响半径: {sdf_system['lambda_point_px']} px = {sdf_system['lambda_point_px'] * meters_per_pixel} m")
        
        if sdf_system['lambda_point_px'] * 2 >= 176:
            print(f"   ✅ Hub影响半径足够，两个hub之间有SDF重叠")
            overlap_distance = (sdf_system['lambda_point_px'] * 2 - 176) / 2
            print(f"   重叠区域: ±{overlap_distance:.1f} px = ±{overlap_distance * meters_per_pixel:.1f} m")
        else:
            print(f"   ❌ Hub影响半径仍然不足")
        
        print(f"\n💡 SDF演化机制:")
        print(f"   初始阶段 (0-6月): 只有Hub点SDF，道路SDF最小")
        print(f"   早期增长 (6-12月): 道路SDF开始扩展，住宅区形成")
        print(f"   中期增长 (12-18月): 道路SDF显著扩展，商业区扩张")
        print(f"   成熟阶段 (18-24月): 道路SDF最大扩展，城市完全发展")
        
        print(f"\n🚀 下一步:")
        print(f"   1. 修改SDF生成代码，实现演化逻辑")
        print(f"   2. 创建SDF演化可视化脚本")
        print(f"   3. 测试SDF随时间的变化效果")
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")

if __name__ == "__main__":
    fix_sdf_parameters()


