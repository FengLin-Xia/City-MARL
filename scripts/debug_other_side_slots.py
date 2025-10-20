#!/usr/bin/env python3
"""
调试对岸槽位情况
检查对岸槽位是否存在以及等级是否足够
"""

import sys
import os
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_other_side_slots():
    """调试对岸槽位情况"""
    print("=== 对岸槽位调试 ===")
    
    # 加载配置
    config_path = "configs/city_config_v4_1.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 获取河流信息
    rivers = cfg.get('terrain_features', {}).get('rivers', [])
    coords = rivers[0].get('coordinates', []) if rivers else []
    
    if not coords:
        print("没有河流坐标数据")
        return
    
    # 计算河流中心线
    y_coords = [point[1] for point in coords]
    center_y = sum(y_coords) / len(y_coords)
    print(f"河流中心线Y坐标: {center_y}")
    
    # 获取EDU hub位置
    hubs = cfg.get('city', {}).get('transport_hubs', [[125, 75], [112, 121]])
    edu_hub_y = hubs[1][1] if len(hubs) > 1 else hubs[0][1]
    print(f"EDU hub Y坐标: {edu_hub_y}")
    
    # 加载槽位
    slots_source = cfg.get('growth_v4_1', {}).get('slots', {}).get('path', 'slots_with_angle.txt')
    
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
                        
                        # 解析等级信息
                        level = 3  # 默认等级
                        if len(parts) > 3:
                            try:
                                level = int(parts[3])
                            except ValueError:
                                pass
                        
                        slot = type('Slot', (), {
                            'id': slot_id,
                            'x': x, 'y': y,
                            'fx': x, 'fy': y,
                            'building_level': level
                        })()
                        slots[slot_id] = slot
                    except ValueError:
                        continue
    except Exception as e:
        print(f"加载槽位文件失败: {e}")
        return
    
    print(f"总槽位数: {len(slots)}")
    
    # 分析对岸槽位
    other_side_slots = []
    same_side_slots = []
    
    for slot_id, slot in slots.items():
        y = float(getattr(slot, 'fy', getattr(slot, 'y', 0.0)))
        level = getattr(slot, 'building_level', 3)
        
        # 对岸判断
        if (y > center_y) != (edu_hub_y > center_y):
            other_side_slots.append((slot_id, y, level))
        else:
            same_side_slots.append((slot_id, y, level))
    
    print(f"同侧槽位数: {len(same_side_slots)}")
    print(f"对岸槽位数: {len(other_side_slots)}")
    
    if other_side_slots:
        print("\n=== 对岸槽位详情 ===")
        # 按等级分组
        level_groups = {}
        for slot_id, y, level in other_side_slots:
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append((slot_id, y))
        
        for level in sorted(level_groups.keys()):
            slots_at_level = level_groups[level]
            print(f"等级{level}: {len(slots_at_level)}个槽位")
            if level >= 3:  # A/B/C需要等级>=3
                print(f"  OK 适合A/B/C型建筑")
                for slot_id, y in slots_at_level[:5]:  # 显示前5个
                    print(f"    {slot_id}: y={y:.1f}")
            else:
                print(f"  X 等级不够，不适合A/B/C型建筑")
    else:
        print("X 没有找到对岸槽位！")
        
        # 分析原因
        print("\n=== 原因分析 ===")
        print("检查槽位Y坐标分布...")
        y_values = [float(getattr(slot, 'fy', getattr(slot, 'y', 0.0))) for slot in slots.values()]
        y_values.sort()
        print(f"Y坐标范围: {y_values[0]:.1f} - {y_values[-1]:.1f}")
        print(f"河流中心线: {center_y:.1f}")
        print(f"EDU hub: {edu_hub_y:.1f}")
        
        # 检查是否有槽位在河流上方
        above_river = [y for y in y_values if y < center_y]
        below_river = [y for y in y_values if y > center_y]
        print(f"河流上方槽位: {len(above_river)}个")
        print(f"河流下方槽位: {len(below_river)}个")

if __name__ == "__main__":
    debug_other_side_slots()
