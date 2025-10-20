#!/usr/bin/env python3
"""
调试对岸候选情况
检查对岸是否有A/B/C型建筑的候选槽位
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import load_river_coords, build_river_components, _get_river_buffer_px
from logic.v4_enumeration import ActionEnumerator, ActionScorer
from logic.v4_enumeration import Action

def load_slots_from_points_file(file_path: str, map_size: Tuple[int, int]):
    """从槽位文件加载槽位数据"""
    slots = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    slot_id = parts[0]
                    # 处理可能的逗号分隔
                    x_str = parts[1].replace(',', '')
                    y_str = parts[2].replace(',', '')
                    try:
                        x = float(x_str)
                        y = float(y_str)
                        
                        # 创建槽位对象
                        slot = type('Slot', (), {
                            'id': slot_id,
                            'x': x, 'y': y,
                            'fx': x, 'fy': y,
                            'building_level': 3  # 默认等级
                        })()
                        slots[slot_id] = slot
                    except ValueError as ve:
                        print(f"解析槽位 {slot_id} 坐标失败: {ve}")
                        continue
    except Exception as e:
        print(f"加载槽位文件失败: {e}")
    return slots

def is_other_side_slot(slot, river_center_y, edu_hub_y):
    """检查槽位是否在对岸"""
    y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
    return (y > river_center_y) != (edu_hub_y > river_center_y)

def debug_other_side_candidates(config_path: str):
    """调试对岸候选情况"""
    print("=== 对岸候选调试 ===")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 获取河流信息
    rivers = cfg.get('terrain_features', {}).get('rivers', [])
    coords = rivers[0].get('coordinates', []) if rivers else []
    
    # 计算河流中心线
    try:
        from envs.v4_1.city_env import load_river_coords, river_center_y_from_coords
        center_y = river_center_y_from_coords({'coordinates': coords}) if coords else None
        print(f"河流中心线Y坐标: {center_y}")
    except Exception as e:
        print(f"计算河流中心线失败: {e}")
        # 手动计算河流中心线
        if coords:
            y_coords = [point[1] for point in coords]
            center_y = sum(y_coords) / len(y_coords)
            print(f"手动计算河流中心线Y坐标: {center_y}")
        else:
            print("没有河流坐标数据")
            return
    
    # 获取EDU hub位置
    hubs = cfg.get('city', {}).get('transport_hubs', [[125, 75], [112, 121]])
    edu_hub_y = hubs[1][1] if len(hubs) > 1 else hubs[0][1]
    print(f"EDU hub Y坐标: {edu_hub_y}")
    
    # 加载槽位
    slots_source = cfg.get('growth_v4_1', {}).get('slots', {}).get('path', 'slots_with_angle.txt')
    map_size = (200, 200)  # 默认地图大小
    slots = load_slots_from_points_file(slots_source, map_size)
    print(f"总槽位数: {len(slots)}")
    
    # 分析对岸槽位
    other_side_slots = []
    same_side_slots = []
    
    for slot_id, slot in slots.items():
        if is_other_side_slot(slot, center_y, edu_hub_y):
            other_side_slots.append(slot)
        else:
            same_side_slots.append(slot)
    
    print(f"同侧槽位数: {len(same_side_slots)}")
    print(f"对岸槽位数: {len(other_side_slots)}")
    
    # 分析对岸槽位的分布
    if other_side_slots:
        other_y_coords = [float(getattr(slot, 'fy', getattr(slot, 'y', 0.0))) for slot in other_side_slots]
        print(f"对岸槽位Y坐标范围: [{min(other_y_coords):.1f}, {max(other_y_coords):.1f}]")
        print(f"对岸槽位Y坐标均值: {np.mean(other_y_coords):.1f}")
        
        # 检查对岸槽位是否适合A/B/C型建筑
        print("\n=== 对岸槽位适合性分析 ===")
        suitable_for_abc = 0
        for slot in other_side_slots:
            # 检查槽位等级（假设等级>=3适合A/B/C）
            level = getattr(slot, 'building_level', 3)
            if level >= 3:
                suitable_for_abc += 1
        
        print(f"对岸适合A/B/C的槽位数: {suitable_for_abc}/{len(other_side_slots)}")
        
        # 显示前几个对岸槽位的详细信息
        print("\n=== 前5个对岸槽位详情 ===")
        for i, slot in enumerate(other_side_slots[:5]):
            y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
            level = getattr(slot, 'building_level', 3)
            print(f"  {slot.id}: y={y:.1f}, level={level}")
    else:
        print("X 没有找到对岸槽位！")
    
    # 检查河流过滤是否影响对岸候选
    print("\n=== 河流过滤分析 ===")
    print("EDU A/B/C型建筑是否绕过河流过滤:")
    
    # 检查配置中的河流过滤设置
    river_filter_cfg = cfg.get('growth_v4_1', {}).get('river_filter', {})
    edu_bypass = river_filter_cfg.get('edu_bypass', False)
    edu_abc_bypass = river_filter_cfg.get('edu_abc_bypass', False)
    
    print(f"  EDU完全绕过河流过滤: {edu_bypass}")
    print(f"  EDU A/B/C绕过河流过滤: {edu_abc_bypass}")
    
    if not edu_abc_bypass:
        print("WARNING EDU A/B/C没有绕过河流过滤，可能被过滤掉了！")
    else:
        print("OK EDU A/B/C已绕过河流过滤")

if __name__ == "__main__":
    config_path = "configs/city_config_v4_1.json"
    debug_other_side_candidates(config_path)
