#!/usr/bin/env python3
"""
可视化v4.1 RL系统的最优结果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from collections import defaultdict
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_config():
    """加载配置"""
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def simulate_best_episode():
    """模拟最优episode"""
    print("=== 模拟最优episode ===")
    
    from solvers.v4_1.rl_selector import RLPolicySelector
    from envs.v4_1.city_env import CityEnvironment
    
    cfg = load_config()
    
    # 初始化
    selector = RLPolicySelector(cfg)
    env = CityEnvironment(cfg)
    state = env.reset(seed=42)  # 使用表现最好的种子
    
    print(f"开始模拟，初始智能体: {state['current_agent']}")
    
    # 记录轨迹
    trajectory = []
    building_history = []
    
    step = 0
    while step < 30:  # 限制步数避免无限循环
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
        
        # 记录
        trajectory.append({
            'step': step,
            'agent': current_agent,
            'action': selected_action,
            'reward': reward,
            'position': (selected_action.footprint_slots[0], selected_action.size) if selected_action.footprint_slots else None
        })
        
        building_history.append({
            'step': step,
            'edu_buildings': len(env.buildings['public']),
            'ind_buildings': len(env.buildings['industrial']),
            'total_reward': sum(sum(rewards) for rewards in env.monthly_rewards.values())
        })
        
        print(f"第{step}步: {current_agent} -> 奖励={reward:.3f}, 建筑位置={selected_action.footprint_slots[0] if selected_action.footprint_slots else 'None'}")
        
        state = next_state
        step += 1
        
        if done:
            print(f"Episode在第{step}步结束")
            break
    
    return trajectory, building_history, env

def create_visualization():
    """创建可视化图表"""
    print("=== 创建可视化图表 ===")
    
    # 模拟最优episode
    trajectory, building_history, env = simulate_best_episode()
    
    # 创建图表
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 奖励趋势图
    ax1 = plt.subplot(3, 3, 1)
    steps = [t['step'] for t in trajectory]
    rewards = [t['reward'] for t in trajectory]
    agents = [t['agent'] for t in trajectory]
    
    colors = {'EDU': 'blue', 'IND': 'red'}
    for i, (step, reward, agent) in enumerate(zip(steps, rewards, agents)):
        ax1.bar(step, reward, color=colors[agent], alpha=0.7, label=agent if i == 0 or agents[i-1] != agent else "")
    
    ax1.set_title('Step-wise Rewards by Agent', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 累积奖励图
    ax2 = plt.subplot(3, 3, 2)
    cumulative_rewards = np.cumsum(rewards)
    ax2.plot(steps, cumulative_rewards, 'g-', linewidth=2, marker='o')
    ax2.set_title('Cumulative Rewards', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Reward')
    ax2.grid(True, alpha=0.3)
    
    # 3. 建筑数量增长
    ax3 = plt.subplot(3, 3, 3)
    bh_steps = [b['step'] for b in building_history]
    edu_counts = [b['edu_buildings'] for b in building_history]
    ind_counts = [b['ind_buildings'] for b in building_history]
    
    ax3.plot(bh_steps, edu_counts, 'b-', linewidth=2, marker='s', label='EDU Buildings')
    ax3.plot(bh_steps, ind_counts, 'r-', linewidth=2, marker='^', label='IND Buildings')
    ax3.set_title('Building Count Growth', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Building Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 智能体奖励分布
    ax4 = plt.subplot(3, 3, 4)
    edu_rewards = [t['reward'] for t in trajectory if t['agent'] == 'EDU']
    ind_rewards = [t['reward'] for t in trajectory if t['agent'] == 'IND']
    
    ax4.hist([edu_rewards, ind_rewards], bins=10, alpha=0.7, label=['EDU', 'IND'], color=['blue', 'red'])
    ax4.set_title('Reward Distribution by Agent', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Reward Value')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 建筑地图可视化
    ax5 = plt.subplot(3, 3, (5, 6))
    
    # 绘制槽位
    for slot_id, slot in env.slots.items():
        if slot_id in env._get_occupied_slots():
            # 已占用的槽位
            if slot_id in [b['footprint_slots'][0] for b in env.buildings['public'] if 'footprint_slots' in b]:
                color = 'lightblue'  # EDU建筑
            else:
                color = 'lightcoral'  # IND建筑
        else:
            color = 'lightgray'  # 空闲槽位
        
        rect = Rectangle((slot.x-0.5, slot.y-0.5), 1, 1, 
                        facecolor=color, edgecolor='black', linewidth=0.5)
        ax5.add_patch(rect)
    
    # 绘制交通枢纽
    for hub in env.hubs:
        circle = patches.Circle((hub[0], hub[1]), 3, color='gold', alpha=0.8)
        ax5.add_patch(circle)
        ax5.text(hub[0], hub[1], 'HUB', ha='center', va='center', fontweight='bold')
    
    ax5.set_xlim(0, env.map_size[0])
    ax5.set_ylim(0, env.map_size[1])
    ax5.set_title('City Layout - Final State', fontsize=14, fontweight='bold')
    ax5.set_xlabel('X Coordinate')
    ax5.set_ylabel('Y Coordinate')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [
        patches.Patch(color='lightblue', label='EDU Buildings'),
        patches.Patch(color='lightcoral', label='IND Buildings'),
        patches.Patch(color='lightgray', label='Available Slots'),
        patches.Patch(color='gold', label='Transport Hubs')
    ]
    ax5.legend(handles=legend_elements, loc='upper right')
    
    # 6. 性能统计
    ax6 = plt.subplot(3, 3, 7)
    stats_data = {
        'Total Reward': sum(rewards),
        'EDU Reward': sum([t['reward'] for t in trajectory if t['agent'] == 'EDU']),
        'IND Reward': sum([t['reward'] for t in trajectory if t['agent'] == 'IND']),
        'Steps': len(trajectory),
        'EDU Buildings': len(env.buildings['public']),
        'IND Buildings': len(env.buildings['industrial'])
    }
    
    bars = ax6.bar(range(len(stats_data)), list(stats_data.values()), 
                   color=['green', 'blue', 'red', 'orange', 'lightblue', 'lightcoral'])
    ax6.set_title('Performance Statistics', fontsize=14, fontweight='bold')
    ax6.set_xticks(range(len(stats_data)))
    ax6.set_xticklabels(list(stats_data.keys()), rotation=45, ha='right')
    ax6.set_ylabel('Value')
    
    # 在柱子上添加数值
    for bar, value in zip(bars, stats_data.values()):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 7. 动作类型分布
    ax7 = plt.subplot(3, 3, 8)
    action_sizes = [t['action'].size for t in trajectory]
    size_counts = {'S': action_sizes.count('S'), 'M': action_sizes.count('M'), 'L': action_sizes.count('L')}
    
    colors_size = {'S': 'lightgreen', 'M': 'orange', 'L': 'darkred'}
    wedges, texts, autotexts = ax7.pie(list(size_counts.values()), 
                                       labels=list(size_counts.keys()),
                                       colors=[colors_size[k] for k in size_counts.keys()],
                                       autopct='%1.1f%%', startangle=90)
    ax7.set_title('Action Size Distribution', fontsize=14, fontweight='bold')
    
    # 8. 奖励效率分析
    ax8 = plt.subplot(3, 3, 9)
    efficiency_data = []
    for i, t in enumerate(trajectory):
        if t['position']:
            efficiency = t['reward'] / (i + 1)  # 奖励除以步数
            efficiency_data.append(efficiency)
        else:
            efficiency_data.append(0)
    
    ax8.plot(steps, efficiency_data, 'purple', linewidth=2, marker='o')
    ax8.set_title('Reward Efficiency Over Time', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Step')
    ax8.set_ylabel('Reward per Step')
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v4_1_best_results_visualization.png', dpi=300, bbox_inches='tight')
    print("[OK] 可视化图表已保存: v4_1_best_results_visualization.png")
    
    # 打印详细统计
    print("\n=== 详细统计信息 ===")
    print(f"总奖励: {sum(rewards):.3f}")
    print(f"EDU奖励: {sum([t['reward'] for t in trajectory if t['agent'] == 'EDU']):.3f}")
    print(f"IND奖励: {sum([t['reward'] for t in trajectory if t['agent'] == 'IND']):.3f}")
    print(f"总步数: {len(trajectory)}")
    print(f"EDU建筑数: {len(env.buildings['public'])}")
    print(f"IND建筑数: {len(env.buildings['industrial'])}")
    print(f"平均步奖励: {np.mean(rewards):.3f}")
    print(f"奖励标准差: {np.std(rewards):.3f}")
    
    return fig

def main():
    """主函数"""
    print("=== v4.1 RL系统最优结果可视化 ===")
    
    try:
        fig = create_visualization()
        plt.show()
        print("[SUCCESS] 可视化完成！")
        
    except Exception as e:
        print(f"[ERROR] 可视化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
