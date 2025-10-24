#!/usr/bin/env python3
"""
可视化v4.1 RL系统的城市布局
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_config():
    """加载配置"""
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def simulate_city_layout():
    """模拟城市布局"""
    print("=== 模拟城市布局 ===")
    
    from solvers.v4_1.rl_selector import RLPolicySelector
    from envs.v4_1.city_env import CityEnvironment
    
    cfg = load_config()
    
    # 初始化
    selector = RLPolicySelector(cfg)
    env = CityEnvironment(cfg)
    state = env.reset(seed=42)  # 使用表现最好的种子
    
    print(f"开始模拟，初始智能体: {state['current_agent']}")
    
    step = 0
    building_log = []
    
    while step < 30:  # 限制步数
        current_agent = state['current_agent']
        
        # 获取动作池
        actions, action_feats, mask = env.get_action_pool(current_agent)
        
        if not actions:
            print(f"第{step}步: {current_agent}没有可用动作，结束")
            break
        
        # 选择动作
        _, selected_action = selector.choose_action_sequence(
            slots=env.slots,
            candidates=set(actions[i].footprint_slots[0] for i in range(len(actions)) if actions[i].footprint_slots),
            occupied=env._get_occupied_slots(),
            lp_provider=env._create_lp_provider(),
            agent_types=[current_agent],
            sizes={'EDU': ['S', 'M', 'L'], 'IND': ['S', 'M', 'L'], 'Council': ['A', 'B', 'C']}
        )
        
        if selected_action is None:
            print(f"第{step}步: {current_agent}没有选择动作，结束")
            break
        
        # 执行动作
        next_state, reward, done, info = env.step(current_agent, selected_action)
        
        # 记录建筑信息
        if selected_action.footprint_slots:
            slot_id = selected_action.footprint_slots[0]
            slot = env.slots.get(slot_id)
            if slot:
                building_log.append({
                    'step': step,
                    'agent': current_agent,
                    'slot_id': slot_id,
                    'x': slot.x,
                    'y': slot.y,
                    'size': selected_action.size,
                    'reward': reward,
                    'score': selected_action.score
                })
                print(f"第{step}步: {current_agent} 在 ({slot.x}, {slot.y}) 建造 {selected_action.size} 建筑，奖励={reward:.3f}")
        
        state = next_state
        step += 1
        
        if done:
            print(f"Episode在第{step}步结束")
            break
    
    return env, building_log

def create_city_layout_visualization():
    """创建城市布局可视化"""
    print("=== 创建城市布局可视化 ===")
    
    # 模拟城市布局
    env, building_log = simulate_city_layout()
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左图：最终城市布局
    print("绘制最终城市布局...")
    
    # 绘制所有槽位
    for slot_id, slot in env.slots.items():
        if slot_id in env._get_occupied_slots():
            # 检查是哪种建筑类型
            is_edu = any(b['slot_id'] == slot_id for b in building_log if b['agent'] == 'EDU')
            is_ind = any(b['slot_id'] == slot_id for b in building_log if b['agent'] == 'IND')
            
            if is_edu:
                color = 'lightblue'
                alpha = 0.8
            elif is_ind:
                color = 'lightcoral'
                alpha = 0.8
            else:
                color = 'lightgray'
                alpha = 0.3
        else:
            color = 'lightgray'
            alpha = 0.1
        
        rect = Rectangle((slot.x-0.5, slot.y-0.5), 1, 1, 
                        facecolor=color, edgecolor='black', linewidth=0.3, alpha=alpha)
        ax1.add_patch(rect)
    
    # 绘制交通枢纽
    for i, hub in enumerate(env.hubs):
        circle = Circle((hub[0], hub[1]), 4, color='gold', alpha=0.9, edgecolor='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(hub[0], hub[1], f'HUB{i+1}', ha='center', va='center', 
                fontweight='bold', fontsize=12, color='black')
    
    # 绘制河流（如果有）
    if hasattr(env, 'river_coords') and env.river_coords:
        river_y = np.mean([coord[1] for coord in env.river_coords])
        ax1.axhline(y=river_y, color='blue', linewidth=3, alpha=0.6, label='River')
    
    ax1.set_xlim(0, env.map_size[0])
    ax1.set_ylim(0, env.map_size[1])
    ax1.set_title('Final City Layout - v4.1 RL System', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [
        patches.Patch(color='lightblue', label='EDU Buildings'),
        patches.Patch(color='lightcoral', label='IND Buildings'),
        patches.Patch(color='lightgray', label='Available Slots'),
        patches.Patch(color='gold', label='Transport Hubs'),
        patches.Patch(color='blue', label='River (if exists)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 右图：建筑建造时序
    print("绘制建筑建造时序...")
    
    # 按建造顺序绘制
    colors = {'EDU': 'blue', 'IND': 'red'}
    sizes_map = {'S': 20, 'M': 40, 'L': 60}
    
    for i, building in enumerate(building_log):
        color = colors[building['agent']]
        size = sizes_map.get(building['size'], 30)
        
        # 绘制建筑
        circle = Circle((building['x'], building['y']), size/40, 
                       color=color, alpha=0.7, edgecolor='black', linewidth=1)
        ax2.add_patch(circle)
        
        # 添加序号标签
        ax2.text(building['x'], building['y'], str(i+1), 
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # 绘制交通枢纽
    for i, hub in enumerate(env.hubs):
        circle = Circle((hub[0], hub[1]), 4, color='gold', alpha=0.9, edgecolor='black', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(hub[0], hub[1], f'HUB{i+1}', ha='center', va='center', 
                fontweight='bold', fontsize=12, color='black')
    
    ax2.set_xlim(0, env.map_size[0])
    ax2.set_ylim(0, env.map_size[1])
    ax2.set_title('Building Construction Timeline', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('X Coordinate', fontsize=12)
    ax2.set_ylabel('Y Coordinate', fontsize=12)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 添加时序图例
    timeline_legend = [
        patches.Patch(color='blue', label='EDU Buildings (numbered by order)'),
        patches.Patch(color='red', label='IND Buildings (numbered by order)'),
        patches.Patch(color='gold', label='Transport Hubs'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Small (S)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=15, label='Medium (M)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=20, label='Large (L)')
    ]
    ax2.legend(handles=timeline_legend, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('v4_1_city_layout.png', dpi=300, bbox_inches='tight')
    print("[OK] 城市布局可视化已保存: v4_1_city_layout.png")
    
    # 打印建筑统计
    print("\n=== 建筑统计 ===")
    edu_buildings = [b for b in building_log if b['agent'] == 'EDU']
    ind_buildings = [b for b in building_log if b['agent'] == 'IND']
    
    print(f"总建筑数: {len(building_log)}")
    print(f"EDU建筑数: {len(edu_buildings)}")
    print(f"IND建筑数: {len(ind_buildings)}")
    
    # 按大小统计
    size_stats = {}
    for building in building_log:
        size = building['size']
        agent = building['agent']
        key = f"{agent}_{size}"
        size_stats[key] = size_stats.get(key, 0) + 1
    
    print("\n建筑大小分布:")
    for key, count in sorted(size_stats.items()):
        print(f"  {key}: {count}")
    
    # 奖励统计
    total_reward = sum(b['reward'] for b in building_log)
    edu_reward = sum(b['reward'] for b in edu_buildings)
    ind_reward = sum(b['reward'] for b in ind_buildings)
    
    print(f"\n奖励统计:")
    print(f"  总奖励: {total_reward:.3f}")
    print(f"  EDU奖励: {edu_reward:.3f}")
    print(f"  IND奖励: {ind_reward:.3f}")
    
    return fig

def main():
    """主函数"""
    print("=== v4.1 RL系统城市布局可视化 ===")
    
    try:
        fig = create_city_layout_visualization()
        plt.show()
        print("[SUCCESS] 城市布局可视化完成！")
        
    except Exception as e:
        print(f"[ERROR] 可视化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

