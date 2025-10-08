#!/usr/bin/env python3
"""
v4.1 RL序列导出为简化TXT格式
基于v4.0的export_v4_sequences_txt.py，适配RL模型的输出

- 从RL模型的slot_selection_history.json读取执行结果
- 导出每个月的最终选择序列为TXT格式
- 格式与v4.0完全一致

输出格式：
a(x,y,0)angle
- EDU: S→0, M→1, L→2
- IND: S→3, M→4, L→5
- IND M/L使用多槽位格式: {a(x1,y1,0)angle1, a(x2,y2,0)angle2, ...}

输出目录：enhanced_simulation_v4_1_output/v4_txt/
"""

import os
import re
import json
import argparse
from typing import Dict, Tuple, List, Optional


AGENT_SIZE_CODE: Dict[Tuple[str, str], int] = {
    ('EDU', 'S'): 0, ('EDU', 'M'): 1, ('EDU', 'L'): 2,
    ('IND', 'S'): 3, ('IND', 'M'): 4, ('IND', 'L'): 5,
}


def load_config(path: str) -> Dict:
    """加载配置文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def load_slots_info(slotpoints_path: str, map_size: List[int] = [200, 200]) -> Dict[str, Tuple[float, float, float]]:
    """使用与主程序相同的加载逻辑读取槽位文件。
    - 返回 sid → (x, y, angle_deg)
    - 应用相同的过滤条件：画布范围判断和去重
    """
    if not os.path.exists(slotpoints_path):
        raise FileNotFoundError(f'slotpoints not found: {slotpoints_path}')
    
    W, H = int(map_size[0]), int(map_size[1])
    sid2info: Dict[str, Tuple[float, float, float]] = {}
    seen: set = set()
    
    with open(slotpoints_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) < 2:
                continue
            try:
                xf = float(nums[0]); yf = float(nums[1])
            except Exception:
                continue
            # 画布范围判断（保留浮点）
            if xf < 0.0 or yf < 0.0 or xf >= float(W) or yf >= float(H):
                continue
            key = (xf, yf)
            if key in seen:
                continue
            seen.add(key)
            ang = float(nums[2]) if len(nums) >= 3 else 0.0
            sid = f's_{len(sid2info)}'
            sid2info[sid] = (xf, yf, ang)
    return sid2info


def fmt_entry(agent: str, size: str, x: float, y: float, angle_deg: float) -> str:
    """格式化单个动作条目"""
    code = AGENT_SIZE_CODE.get((agent, size), 0)
    return f"{code}({x:.3f}, {y:.3f}, 0){angle_deg:.2f}"


def export_sequence_txt(actions: List[Dict], sid2info: Dict[str, Tuple[float, float, float]]) -> str:
    """将动作序列导出为TXT格式"""
    parts: List[str] = []
    
    for action in actions:
        agent = str(action.get('agent', 'EDU')).upper()
        size = str(action.get('size', 'S')).upper()
        slot_id = action.get('slot_id', '')
        
        # 获取槽位信息
        if slot_id in sid2info:
            x, y, angle = sid2info[slot_id]
        else:
            # 如果槽位ID不在映射中，尝试从action中直接获取坐标
            x = action.get('x', 0.0)
            y = action.get('y', 0.0)
            angle = action.get('angle', 0.0)
        
        # 检查是否有多个槽位（IND M/L的特殊情况）
        footprint_slots = action.get('footprint_slots', [])
        if len(footprint_slots) > 1 and agent == 'IND' and size in ('M', 'L'):
            # 多槽位格式：{a(x1,y1,0)angle1, a(x2,y2,0)angle2, ...}
            sub_parts = []
            for slot_id in footprint_slots:
                if slot_id in sid2info:
                    sx, sy, sangle = sid2info[slot_id]
                else:
                    sx, sy, sangle = x, y, angle  # 回退到主槽位信息
                sub_parts.append(fmt_entry(agent, size, sx, sy, sangle))
            if sub_parts:
                parts.append('{' + ', '.join(sub_parts) + '}')
        else:
            # 单槽位格式
            parts.append(fmt_entry(agent, size, x, y, angle))
    
    return ', '.join(parts)


def load_rl_history(history_path: str) -> Dict:
    """加载RL选择历史"""
    if not os.path.exists(history_path):
        raise FileNotFoundError(f'RL history not found: {history_path}')
    
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_monthly_sequences(history: Dict) -> Dict[int, List[Dict]]:
    """从RL历史中提取每月的最终选择序列"""
    monthly_sequences = {}
    
    episodes = history.get('episodes', [])
    
    for episode in episodes:
        episode_id = episode.get('episode_id', 0)
        steps = episode.get('steps', [])
        
        # 按月份分组步骤
        monthly_steps = {}
        for step in steps:
            month = step.get('month', 0)
            if month not in monthly_steps:
                monthly_steps[month] = []
            monthly_steps[month].append(step)
        
        # 为每个月创建最终选择序列
        for month, month_steps in monthly_steps.items():
            if month not in monthly_sequences:
                monthly_sequences[month] = []
            
            # 收集该月的所有动作
            month_actions = []
            for step in month_steps:
                selected_slots = step.get('selected_slots', [])
                agent = step.get('agent', 'EDU')
                
                # 从selected_slots中提取动作信息
                for slot_list in selected_slots:
                    for slot_id in slot_list:
                        # 创建动作记录
                        action = {
                            'agent': agent,
                            'size': 'M',  # 默认尺寸，可以从其他信息推断
                            'slot_id': slot_id,
                            'footprint_slots': [slot_id]
                        }
                        month_actions.append(action)
            
            monthly_sequences[month].append({
                'episode_id': episode_id,
                'actions': month_actions
            })
    
    return monthly_sequences


def export_best_episode_sequences(history: Dict, sid2info: Dict[str, Tuple[float, float, float]], output_dir: str):
    """导出最佳Episode的序列"""
    episodes = history.get('episodes', [])
    if not episodes:
        print("No episodes found in history")
        return
    
    # 找到最佳Episode（最高回报）
    best_episode = max(episodes, key=lambda x: x.get('episode_return', 0))
    best_episode_id = best_episode.get('episode_id', 0)
    best_return = best_episode.get('episode_return', 0)
    
    print(f"Exporting best episode: ID {best_episode_id}, return {best_return:.2f}")
    
    steps = best_episode.get('steps', [])
    
    # 按月份分组
    monthly_steps = {}
    for step in steps:
        month = step.get('month', 0)
        if month not in monthly_steps:
            monthly_steps[month] = []
        monthly_steps[month].append(step)
    
    # 导出每月的最终选择
    for month in sorted(monthly_steps.keys()):
        month_steps = monthly_steps[month]
        
        # 收集该月的所有动作（优先使用detailed_actions）
        month_actions = []
        for step in month_steps:
            detailed_actions = step.get('detailed_actions', [])
            
            if detailed_actions:
                # 使用详细动作信息
                for detailed_action in detailed_actions:
                    # 使用slot_positions中的位置信息
                    slot_positions = detailed_action.get('slot_positions', [])
                    if slot_positions:
                        for slot_pos in slot_positions:
                            action = {
                                'agent': detailed_action['agent'],
                                'size': detailed_action['size'],
                                'slot_id': slot_pos['slot_id'],
                                'x': slot_pos['x'],
                                'y': slot_pos['y'],
                                'angle': slot_pos['angle'],
                                'footprint_slots': detailed_action['footprint_slots']
                            }
                            month_actions.append(action)
                    else:
                        # 回退到基本信息
                        action = {
                            'agent': detailed_action['agent'],
                            'size': detailed_action['size'],
                            'slot_id': detailed_action['slot_id'],
                            'footprint_slots': detailed_action['footprint_slots']
                        }
                        month_actions.append(action)
            else:
                # 回退到旧的selected_slots方式
                selected_slots = step.get('selected_slots', [])
                agent = step.get('agent', 'EDU')
                
                for slot_list in selected_slots:
                    for slot_id in slot_list:
                        action = {
                            'agent': agent,
                            'size': 'M',  # 默认尺寸
                            'slot_id': slot_id,
                            'footprint_slots': [slot_id]
                        }
                        month_actions.append(action)
        
        # 导出为TXT
        if month_actions:
            txt_content = export_sequence_txt(month_actions, sid2info)
            output_file = os.path.join(output_dir, f'chosen_month_{month:02d}.txt')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            
            print(f"  Exported month {month}: {len(month_actions)} actions -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Export v4.1 RL sequences to simplified TXT format')
    parser.add_argument('--config', default='configs/city_config_v4_1.json', help='config path')
    parser.add_argument('--history_path', default='models/v4_1_rl/slot_selection_history.json', 
                       help='RL slot selection history path')
    parser.add_argument('--output_dir', default='enhanced_simulation_v4_1_output', help='base output dir')
    parser.add_argument('--slots_file', default='slots_with_angle.txt', help='slots file path')
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    map_size = cfg.get('city', {}).get('map_size', [200, 200])
    
    # 加载槽位信息
    print(f"Loading slots from: {args.slots_file}")
    sid2info = load_slots_info(args.slots_file, map_size)
    print(f"Loaded {len(sid2info)} slots")
    
    # 加载RL历史
    print(f"Loading RL history from: {args.history_path}")
    history = load_rl_history(args.history_path)
    
    # 创建输出目录
    txt_dir = os.path.join(args.output_dir, 'v4_txt')
    os.makedirs(txt_dir, exist_ok=True)
    
    # 导出最佳Episode的序列
    export_best_episode_sequences(history, sid2info, txt_dir)
    
    print(f'Exported RL sequences to: {txt_dir}')


if __name__ == '__main__':
    main()
