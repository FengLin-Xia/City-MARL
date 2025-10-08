#!/usr/bin/env python3
"""
可视化最佳Episode中槽位的实际地理位置
在地图上显示选择的槽位分布
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Any, Tuple
import argparse

def load_slot_positions(slots_file: str) -> Dict[str, Tuple[float, float]]:
    """加载槽位位置数据"""
    slot_positions = {}
    
    try:
        with open(slots_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # 尝试解析格式: x, y, angle
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    slot_id = f"s_{i}"  # 生成槽位ID
                    slot_positions[slot_id] = (x, y)
                except ValueError:
                    continue
            else:
                # 尝试解析空格分隔格式: slot_id x y
                parts = line.split()
                if len(parts) >= 3:
                    slot_id = parts[0]
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        slot_positions[slot_id] = (x, y)
                    except ValueError:
                        continue
                    
    except FileNotFoundError:
        print(f"警告: 槽位文件 {slots_file} 不存在")
        return {}
    
    return slot_positions

def load_slot_selection_history(history_path: str) -> Dict:
    """加载槽位选择历史数据"""
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_best_episode(history: Dict) -> Dict:
    """找到episode_return最高的episode"""
    episodes = history.get('episodes', [])
    if not episodes:
        raise ValueError("No episodes found in history")
    
    best_episode = max(episodes, key=lambda ep: ep['episode_return'])
    return best_episode

def extract_selected_slots(best_episode: Dict) -> Dict[str, Dict]:
    """提取选择的槽位信息"""
    selected_slots = {}
    
    for step in best_episode['steps']:
        agent = step['agent']
        month = step['month']
        reward = step['reward']
        action_score = step['action_scores'][0] if step['action_scores'] else 0.0
        
        for slot_list in step['selected_slots']:
            for slot_id in slot_list:
                if slot_id not in selected_slots:
                    selected_slots[slot_id] = {
                        'agent': agent,
                        'month': month,
                        'reward': reward,
                        'action_score': action_score,
                        'selection_order': len(selected_slots)
                    }
    
    return selected_slots

def visualize_slot_positions(history_path: str, slots_file: str, output_dir: str = None):
    """可视化槽位位置分布"""
    
    # 加载数据
    history = load_slot_selection_history(history_path)
    best_episode = find_best_episode(history)
    slot_positions = load_slot_positions(slots_file)
    selected_slots = extract_selected_slots(best_episode)
    
    if output_dir is None:
        output_dir = os.path.dirname(history_path)
    
    print(f"最佳Episode: ID {best_episode['episode_id']} (Update {best_episode['update']})")
    print(f"Episode Return: {best_episode['episode_return']:.2f}")
    print(f"总选择槽位数: {len(selected_slots)}")
    print(f"槽位位置数据: {len(slot_positions)} 个槽位")
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 所有槽位 + 选择的槽位
    ax1 = axes[0, 0]
    
    # 绘制所有槽位 (浅灰色)
    all_x = [pos[0] for pos in slot_positions.values()]
    all_y = [pos[1] for pos in slot_positions.values()]
    ax1.scatter(all_x, all_y, c='lightgray', alpha=0.3, s=20, label='所有槽位')
    
    # 绘制选择的槽位
    edu_x = []
    edu_y = []
    ind_x = []
    ind_y = []
    edu_rewards = []
    ind_rewards = []
    
    for slot_id, info in selected_slots.items():
        if slot_id in slot_positions:
            x, y = slot_positions[slot_id]
            if info['agent'] == 'EDU':
                edu_x.append(x)
                edu_y.append(y)
                edu_rewards.append(info['reward'])
            else:
                ind_x.append(x)
                ind_y.append(y)
                ind_rewards.append(info['reward'])
    
    # EDU槽位 (蓝色，大小表示奖励)
    if edu_x:
        scatter1 = ax1.scatter(edu_x, edu_y, c=edu_rewards, cmap='Blues', 
                             s=[r*5 for r in edu_rewards], alpha=0.8, 
                             label=f'EDU槽位 ({len(edu_x)})', edgecolors='darkblue', linewidth=1)
        plt.colorbar(scatter1, ax=ax1, label='EDU奖励')
    
    # IND槽位 (红色，大小表示奖励)
    if ind_x:
        scatter2 = ax1.scatter(ind_x, ind_y, c=ind_rewards, cmap='Reds', 
                             s=[r*5 for r in ind_rewards], alpha=0.8, 
                             label=f'IND槽位 ({len(ind_x)})', edgecolors='darkred', linewidth=1)
        plt.colorbar(scatter2, ax=ax1, label='IND奖励')
    
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.set_title('槽位位置分布 (大小表示奖励)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 按月份显示选择顺序
    ax2 = axes[0, 1]
    
    # 绘制所有槽位
    ax2.scatter(all_x, all_y, c='lightgray', alpha=0.2, s=10)
    
    # 按月份和智能体绘制
    month_colors = plt.cm.viridis(np.linspace(0, 1, 10))  # 10个月
    
    for slot_id, info in selected_slots.items():
        if slot_id in slot_positions:
            x, y = slot_positions[slot_id]
            month = info['month']
            agent = info['agent']
            
            color = month_colors[month]
            marker = 'o' if agent == 'EDU' else 's'
            size = 100 if agent == 'EDU' else 80
            
            ax2.scatter(x, y, c=[color], s=size, marker=marker, alpha=0.8,
                       edgecolors='black', linewidth=1)
            
            # 添加月份标签
            ax2.annotate(f"{month}", (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('X坐标')
    ax2.set_ylabel('Y坐标')
    ax2.set_title('槽位选择时间顺序 (数字表示月份)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 动作分数热力图
    ax3 = axes[1, 0]
    
    # 绘制所有槽位
    ax3.scatter(all_x, all_y, c='lightgray', alpha=0.2, s=10)
    
    # 绘制选择的槽位，颜色表示动作分数
    selected_x = []
    selected_y = []
    action_scores = []
    agents = []
    
    for slot_id, info in selected_slots.items():
        if slot_id in slot_positions:
            x, y = slot_positions[slot_id]
            selected_x.append(x)
            selected_y.append(y)
            action_scores.append(info['action_score'])
            agents.append(info['agent'])
    
    if selected_x:
        scatter3 = ax3.scatter(selected_x, selected_y, c=action_scores, 
                             cmap='RdYlBu_r', s=100, alpha=0.8,
                             edgecolors='black', linewidth=1)
        plt.colorbar(scatter3, ax=ax3, label='动作分数')
    
    ax3.set_xlabel('X坐标')
    ax3.set_ylabel('Y坐标')
    ax3.set_title('槽位动作分数分布')
    ax3.grid(True, alpha=0.3)
    
    # 4. 智能体选择模式
    ax4 = axes[1, 1]
    
    # 绘制所有槽位
    ax4.scatter(all_x, all_y, c='lightgray', alpha=0.2, s=10)
    
    # 绘制EDU和IND的选择区域
    if edu_x:
        ax4.scatter(edu_x, edu_y, c='blue', s=120, alpha=0.7, 
                   marker='o', label=f'EDU ({len(edu_x)})', edgecolors='darkblue')
    
    if ind_x:
        ax4.scatter(ind_x, ind_y, c='red', s=120, alpha=0.7, 
                   marker='s', label=f'IND ({len(ind_x)})', edgecolors='darkred')
    
    # 计算并显示选择区域的重心
    if edu_x:
        edu_center_x = np.mean(edu_x)
        edu_center_y = np.mean(edu_y)
        ax4.scatter(edu_center_x, edu_center_y, c='darkblue', s=200, 
                   marker='*', label='EDU重心')
    
    if ind_x:
        ind_center_x = np.mean(ind_x)
        ind_center_y = np.mean(ind_y)
        ax4.scatter(ind_center_x, ind_center_y, c='darkred', s=200, 
                   marker='*', label='IND重心')
    
    ax4.set_xlabel('X坐标')
    ax4.set_ylabel('Y坐标')
    ax4.set_title('智能体选择区域分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'最佳Episode槽位位置分析 (Episode {best_episode["episode_id"]})\n'
                f'Episode Return: {best_episode["episode_return"]:.2f}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'best_episode_{best_episode["episode_id"]}_slot_positions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n槽位位置分析图已保存到: {output_path}")
    
    # 输出槽位选择统计
    print("\n=== 槽位选择统计 ===")
    print(f"总选择槽位数: {len(selected_slots)}")
    print(f"EDU选择槽位数: {len(edu_x)}")
    print(f"IND选择槽位数: {len(ind_x)}")
    
    if edu_x:
        print(f"EDU选择区域重心: ({np.mean(edu_x):.1f}, {np.mean(edu_y):.1f})")
        print(f"EDU平均奖励: {np.mean(edu_rewards):.2f}")
    
    if ind_x:
        print(f"IND选择区域重心: ({np.mean(ind_x):.1f}, {np.mean(ind_y):.1f})")
        print(f"IND平均奖励: {np.mean(ind_rewards):.2f}")
    
    # 输出详细槽位信息
    print("\n=== 详细槽位选择信息 ===")
    for slot_id, info in sorted(selected_slots.items(), key=lambda x: (x[1]['month'], x[1]['agent'])):
        if slot_id in slot_positions:
            x, y = slot_positions[slot_id]
            print(f"Month {info['month']} {info['agent']}: {slot_id} "
                  f"位置({x:.1f}, {y:.1f}) 奖励={info['reward']:.2f} "
                  f"动作分数={info['action_score']:.4f}")
    
    return selected_slots

def main():
    parser = argparse.ArgumentParser(description='可视化最佳Episode的槽位位置分布')
    parser.add_argument('--history_path', type=str, 
                       default='models/v4_1_rl/slot_selection_history.json',
                       help='槽位选择历史文件路径')
    parser.add_argument('--slots_file', type=str, 
                       default='slots_with_angle.txt',
                       help='槽位位置文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认与历史文件同目录)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.history_path):
        print(f"错误: 历史文件不存在: {args.history_path}")
        return
    
    if not os.path.exists(args.slots_file):
        print(f"错误: 槽位文件不存在: {args.slots_file}")
        return
    
    try:
        selected_slots = visualize_slot_positions(args.history_path, args.slots_file, args.output_dir)
        print(f"\n可视化完成! 分析了 {len(selected_slots)} 个选择的槽位")
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
