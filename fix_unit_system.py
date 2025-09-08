#!/usr/bin/env python3
"""
修复单位系统 - 统一距离参数
将所有距离参数从米换算到像素，确保单位一致性
"""

import json
import os

def fix_unit_system():
    """修复配置文件中的单位系统"""
    
    print("🔧 修复单位系统 - 统一距离参数")
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
        
        # 创建新的配置结构，添加像素单位参数
        new_config = config.copy()
        
        # 1. 修复SDF系统参数
        if 'sdf_system' not in new_config:
            new_config['sdf_system'] = {}
        
        sdf_system = new_config['sdf_system']
        
        # 添加像素单位参数
        sdf_system['lambda_perp_px'] = int(sdf_system.get('lambda_perp_m', 120) / meters_per_pixel)
        sdf_system['lambda_point_px'] = int(sdf_system.get('lambda_point_m', 100) / meters_per_pixel)
        sdf_system['lambda_tangential_px'] = int(sdf_system.get('lambda_tangential_m', 200) / meters_per_pixel)
        
        print(f"📐 SDF系统像素参数:")
        print(f"   λ⊥_px = {sdf_system['lambda_perp_px']} px (原 {sdf_system.get('lambda_perp_m', 120)} m)")
        print(f"   λ_hub_px = {sdf_system['lambda_point_px']} px (原 {sdf_system.get('lambda_point_m', 100)} m)")
        print(f"   λ∥_px = {sdf_system['lambda_tangential_px']} px (原 {sdf_system.get('lambda_tangential_m', 200)} m)")
        
        # 2. 修复等值线布局参数
        if 'isocontour_layout' not in new_config:
            new_config['isocontour_layout'] = {}
        
        isocontour_layout = new_config['isocontour_layout']
        
        # 商业建筑参数
        if 'commercial' not in isocontour_layout:
            isocontour_layout['commercial'] = {}
        
        commercial = isocontour_layout['commercial']
        commercial['depth_px'] = int(commercial.get('depth_m', 20) / meters_per_pixel)
        commercial['gap_px'] = int(commercial.get('gap_m', 10) / meters_per_pixel)
        commercial['arc_spacing_px'] = [
            int(spacing / meters_per_pixel) for spacing in commercial.get('arc_spacing_m', [25, 35])
        ]
        
        # 住宅建筑参数
        if 'residential' not in isocontour_layout:
            isocontour_layout['residential'] = {}
        
        residential = isocontour_layout['residential']
        residential['depth_px'] = int(residential.get('depth_m', 14) / meters_per_pixel)
        residential['gap_px'] = int(residential.get('gap_m', 26) / meters_per_pixel)
        residential['arc_spacing_px'] = [
            int(spacing / meters_per_pixel) for spacing in residential.get('arc_spacing_m', [35, 55])
        ]
        
        # 通用参数
        isocontour_layout['normal_offset_px'] = int(isocontour_layout.get('normal_offset_m', 1.0) / meters_per_pixel)
        isocontour_layout['jitter_px'] = int(isocontour_layout.get('jitter_m', 0.5) / meters_per_pixel)
        
        print(f"🏗️ 等值线布局像素参数:")
        print(f"   商业建筑:")
        print(f"     深度: {commercial['depth_px']} px (原 {commercial.get('depth_m', 20)} m)")
        print(f"     间隔: {commercial['gap_px']} px (原 {commercial.get('gap_m', 10)} m)")
        print(f"     弧长间距: {commercial['arc_spacing_px']} px (原 {commercial.get('arc_spacing_m', [25, 35])} m)")
        print(f"   住宅建筑:")
        print(f"     深度: {residential['depth_px']} px (原 {residential.get('depth_m', 14)} m)")
        print(f"     间隔: {residential['gap_px']} px (原 {residential.get('gap_m', 26)} m)")
        print(f"     弧长间距: {residential['arc_spacing_px']} px (原 {residential.get('arc_spacing_m', [35, 55])} m)")
        print(f"   通用:")
        print(f"     法向偏移: {isocontour_layout['normal_offset_px']} px (原 {isocontour_layout.get('normal_offset_m', 1.0)} m)")
        print(f"     切向抖动: {isocontour_layout['jitter_px']} px (原 {isocontour_layout.get('jitter_m', 0.5)} m)")
        
        # 3. 修复分带参数
        if 'bands' not in new_config:
            new_config['bands'] = {}
        
        bands = new_config['bands']
        bands['front_no_residential_px'] = [
            int(distance / meters_per_pixel) for distance in bands.get('front_no_residential_m', [60, 120])
        ]
        bands['residential_side_band_px'] = [
            int(distance / meters_per_pixel) for distance in bands.get('residential_side_band_m', [120, 260])
        ]
        
        print(f"🏘️ 分带像素参数:")
        print(f"   前排禁住宅区: {bands['front_no_residential_px']} px (原 {bands.get('front_no_residential_m', [60, 120])} m)")
        print(f"   住宅侧带: {bands['residential_side_band_px']} px (原 {bands.get('residential_side_band_m', [120, 260])} m)")
        
        # 4. 修复公共设施参数
        if 'public_facility_system' in new_config:
            facility_system = new_config['public_facility_system']
            if 'facility_types' in facility_system:
                for facility_type, facility_config in facility_system['facility_types'].items():
                    if 'service_radius' in facility_config:
                        facility_config['service_radius_px'] = int(facility_config['service_radius'] / meters_per_pixel)
                        print(f"🏥 {facility_type}服务半径: {facility_config['service_radius_px']} px (原 {facility_config['service_radius']} m)")
        
        # 5. 添加单位系统说明
        new_config['unit_system'] = {
            "meters_per_pixel": meters_per_pixel,
            "description": "所有距离参数同时提供米和像素单位，确保兼容性",
            "conversion_note": "像素参数 = 米参数 / meters_per_pixel"
        }
        
        # 保存修复后的配置
        backup_file = config_file.replace('.json', '_backup.json')
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 原配置备份到: {backup_file}")
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 修复后的配置保存到: {config_file}")
        
        # 6. 验证关键参数
        print(f"\n🔍 关键参数验证:")
        print(f"   两个hub距离: 176 px = {176 * meters_per_pixel} m")
        print(f"   Hub影响半径: {sdf_system['lambda_point_px']} px = {sdf_system['lambda_point_px'] * meters_per_pixel} m")
        
        if sdf_system['lambda_point_px'] * 2 < 176:
            print(f"   ⚠️  警告: Hub影响半径过小，两个hub之间可能没有SDF重叠")
            print(f"   建议: 增加 lambda_point_m 到 {176 * meters_per_pixel / 2:.0f} m 以上")
        else:
            print(f"   ✅ Hub影响半径足够，两个hub之间有SDF重叠")
        
        print(f"\n💡 下一步建议:")
        print(f"   1. 检查SDF场生成逻辑，确保使用像素单位参数")
        print(f"   2. 修复等值线生成，使用几何等值线而非分位数")
        print(f"   3. 实现基于道路的线SDF")
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")

if __name__ == "__main__":
    fix_unit_system()


