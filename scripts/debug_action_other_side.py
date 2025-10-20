#!/usr/bin/env python3
"""
调试动作池中的对岸A/B/C候选
检查动作池中是否有对岸的A/B/C动作
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

def is_other_side_slot(slot, river_center_y, edu_hub_y):
    """检查槽位是否在对岸"""
    y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
    return (y > river_center_y) != (edu_hub_y > river_center_y)

def debug_action_other_side(config_path: str):
    """调试动作池中的对岸A/B/C候选"""
    print("=== 动作池对岸A/B/C候选调试 ===")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 获取河流信息
    rivers = cfg.get('terrain_features', {}).get('rivers', [])
    coords = rivers[0].get('coordinates', []) if rivers else []
    
    # 计算河流中心线
    if coords:
        y_coords = [point[1] for point in coords]
        center_y = sum(y_coords) / len(y_coords)
        print(f"河流中心线Y坐标: {center_y}")
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
    
    slots = {}
    try:
        with open(slots_source, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    slot_id = parts[0]
                    x_str = parts[1].replace(',', '')
                    y_str = parts[2].replace(',', '')
                    try:
                        x = float(x_str)
                        y = float(y_str)
                        
                        slot = type('Slot', (), {
                            'id': slot_id,
                            'x': x, 'y': y,
                            'fx': x, 'fy': y,
                            'building_level': 3
                        })()
                        slots[slot_id] = slot
                    except ValueError:
                        continue
    except Exception as e:
        print(f"加载槽位文件失败: {e}")
        return
    
    print(f"总槽位数: {len(slots)}")
    
    # 模拟枚举A/B/C动作
    print("\n=== 模拟枚举A/B/C动作 ===")
    
    # 获取所有空闲槽位
    free_slots = set(slots.keys())
    
    # 创建动作枚举器
    enumerator = ActionEnumerator(slots)
    
    # 枚举EDU A/B/C动作
    abc_actions = []
    for size in ['A', 'B', 'C']:
        # A/B/C按单槽位规则枚举
        footprints = enumerator._enumerate_single_slots(free_slots)
        for fp in footprints:
            action = Action(
                agent='EDU',
                size=size,
                footprint_slots=list(fp),
                zone='mid',
                LP_norm=1.0,
                adjacency='4-neighbor'
            )
            abc_actions.append(action)
    
    print(f"枚举到A/B/C动作总数: {len(abc_actions)}")
    
    # 分析对岸A/B/C动作
    other_side_abc = []
    same_side_abc = []
    
    for action in abc_actions:
        if not action.footprint_slots:
            continue
            
        # 检查动作的第一个槽位是否在对岸
        first_slot_id = action.footprint_slots[0]
        slot = slots.get(first_slot_id)
        if slot is None:
            continue
            
        if is_other_side_slot(slot, center_y, edu_hub_y):
            other_side_abc.append(action)
        else:
            same_side_abc.append(action)
    
    print(f"同侧A/B/C动作数: {len(same_side_abc)}")
    print(f"对岸A/B/C动作数: {len(other_side_abc)}")
    
    if other_side_abc:
        print("\n=== 对岸A/B/C动作详情 ===")
        for i, action in enumerate(other_side_abc[:10]):  # 显示前10个
            slot_id = action.footprint_slots[0]
            slot = slots[slot_id]
            y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
            print(f"  {action.size}型: slot={slot_id}, y={y:.1f}")
    else:
        print("❌ 没有找到对岸A/B/C动作！")
        
        # 分析原因
        print("\n=== 原因分析 ===")
        print("检查对岸槽位是否适合A/B/C型建筑...")
        
        other_side_slots = []
        for slot_id, slot in slots.items():
            if is_other_side_slot(slot, center_y, edu_hub_y):
                other_side_slots.append(slot)
        
        print(f"对岸槽位数: {len(other_side_slots)}")
        
        if other_side_slots:
            # 检查对岸槽位是否适合A/B/C
            suitable_slots = []
            for slot in other_side_slots:
                level = getattr(slot, 'building_level', 3)
                if level >= 3:  # A/B/C需要等级>=3
                    suitable_slots.append(slot)
            
            print(f"对岸适合A/B/C的槽位数: {len(suitable_slots)}")
            
            if suitable_slots:
                print("对岸有适合的槽位，但枚举时没有生成动作")
                print("可能原因：")
                print("1. 槽位被其他约束过滤掉了")
                print("2. 枚举逻辑有问题")
                print("3. 槽位被占用或不可用")
            else:
                print("对岸槽位等级不够，不适合A/B/C型建筑")
        else:
            print("对岸没有槽位")

if __name__ == "__main__":
    config_path = "configs/city_config_v4_1.json"
    debug_action_other_side(config_path)
