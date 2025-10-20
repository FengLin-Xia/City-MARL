#!/usr/bin/env python3
"""
可视化Budget历史曲线
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_results(model_dir='models/v4_1_rl'):
    """加载训练结果"""
    model_path = Path(model_dir)
    
    # 查找最新的训练结果
    result_files = list(model_path.glob('training_results_*.json'))
    if not result_files:
        print(f"未找到训练结果文件")
        return None
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"加载: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def visualize_budget_history(results):
    """可视化Budget历史"""
    if not results:
        return
    
    # 提取budget历史
    budget_history = results.get('budget_history', {})
    
    if not budget_history:
        print("训练结果中没有budget历史数据")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Budget System Analysis', fontsize=16, fontweight='bold')
    
    # 1. Budget随时间变化
    ax1 = axes[0, 0]
    for agent, history in budget_history.items():
        if history:
            ax1.plot(history, label=agent, linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Budget (kGBP)')
    ax1.set_title('Budget Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Line')
    
    # 2. Budget分布
    ax2 = axes[0, 1]
    for agent, history in budget_history.items():
        if history:
            ax2.hist(history, bins=20, alpha=0.6, label=agent)
    ax2.set_xlabel('Budget (kGBP)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Budget Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Budget统计
    ax3 = axes[1, 0]
    stats_data = []
    labels = []
    
    for agent, history in budget_history.items():
        if history:
            stats_data.append([
                np.mean(history),
                np.min(history),
                np.max(history),
                history[-1]
            ])
            labels.append(agent)
    
    if stats_data:
        x = np.arange(len(labels))
        width = 0.2
        
        metrics = ['Mean', 'Min', 'Max', 'Final']
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, metric in enumerate(metrics):
            values = [stats[i] for stats in stats_data]
            ax3.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.7)
        
        ax3.set_xlabel('Agent')
        ax3.set_ylabel('Budget (kGBP)')
        ax3.set_title('Budget Statistics')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. 负债分析
    ax4 = axes[1, 1]
    debt_stats = {}
    
    for agent, history in budget_history.items():
        if history:
            debt_count = sum(1 for b in history if b < 0)
            debt_ratio = debt_count / len(history) * 100
            max_debt = min(history)
            debt_stats[agent] = {
                'debt_ratio': debt_ratio,
                'max_debt': max_debt,
                'debt_count': debt_count
            }
    
    if debt_stats:
        agents = list(debt_stats.keys())
        debt_ratios = [debt_stats[a]['debt_ratio'] for a in agents]
        max_debts = [debt_stats[a]['max_debt'] for a in agents]
        
        x = np.arange(len(agents))
        width = 0.35
        
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(x - width/2, debt_ratios, width, label='Debt Ratio (%)', color='red', alpha=0.7)
        bars2 = ax4_twin.bar(x + width/2, max_debts, width, label='Max Debt', color='darkred', alpha=0.7)
        
        ax4.set_xlabel('Agent')
        ax4.set_ylabel('Debt Ratio (%)', color='red')
        ax4_twin.set_ylabel('Max Debt (kGBP)', color='darkred')
        ax4.set_title('Debt Analysis')
        ax4.set_xticks(x)
        ax4.set_xticklabels(agents)
        ax4.tick_params(axis='y', labelcolor='red')
        ax4_twin.tick_params(axis='y', labelcolor='darkred')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加图例
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # 保存
    output_path = 'budget_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存: {output_path}")
    
    # 显示
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*80)
    print("Budget统计信息:")
    print("="*80)
    
    for agent, history in budget_history.items():
        if history:
            print(f"\n{agent}:")
            print(f"  初始: {history[0]:.2f}")
            print(f"  最终: {history[-1]:.2f}")
            print(f"  平均: {np.mean(history):.2f}")
            print(f"  最小: {np.min(history):.2f}")
            print(f"  最大: {np.max(history):.2f}")
            print(f"  标准差: {np.std(history):.2f}")
            
            if agent in debt_stats:
                print(f"  负债次数: {debt_stats[agent]['debt_count']}")
                print(f"  负债比例: {debt_stats[agent]['debt_ratio']:.1f}%")
                print(f"  最大负债: {debt_stats[agent]['max_debt']:.2f}")

if __name__ == '__main__':
    results = load_training_results()
    
    if results:
        visualize_budget_history(results)
    else:
        print("\n请先运行训练:")
        print("  python enhanced_city_simulation_v4_1.py --mode rl")




