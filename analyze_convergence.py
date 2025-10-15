#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_convergence():
    """分析训练收敛情况"""
    
    # 读取训练结果
    with open('models/v4_1_rl/training_results_20251014_021231.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== 训练收敛分析 ===")
    print(f"训练轮数: {data['training_updates']}")
    print(f"最终平均回报: {data['final_avg_return']:.4f}")
    
    # 分析总回报趋势
    total_returns = data['training_metrics']['episode_returns']
    edu_returns = data['training_metrics']['edu_returns']
    ind_returns = data['training_metrics']['ind_returns']
    
    print(f"\n总回报趋势:")
    print(f"Episode 1: {total_returns[0]:.4f}")
    print(f"Episode 5: {total_returns[4]:.4f}")
    print(f"Episode 10: {total_returns[9]:.4f}")
    
    # 计算回报变化
    early_avg = np.mean(total_returns[:3])  # 前3个episode
    late_avg = np.mean(total_returns[-3:])  # 后3个episode
    improvement = late_avg - early_avg
    improvement_pct = (improvement / early_avg) * 100
    
    print(f"\n收敛分析:")
    print(f"前3个episode平均回报: {early_avg:.4f}")
    print(f"后3个episode平均回报: {late_avg:.4f}")
    print(f"改进幅度: {improvement:.4f} ({improvement_pct:.2f}%)")
    
    # 计算回报方差（稳定性）
    variance = np.var(total_returns)
    std = np.std(total_returns)
    
    print(f"\n稳定性分析:")
    print(f"回报方差: {variance:.6f}")
    print(f"回报标准差: {std:.4f}")
    print(f"变异系数: {(std/np.mean(total_returns)*100):.2f}%")
    
    # 检查是否收敛
    if abs(improvement_pct) < 2.0:
        print(f"\n结论: 模型已经收敛 (改进幅度 < 2%)")
    elif improvement_pct > 5.0:
        print(f"\n结论: 模型还在显著改进中 (改进幅度 > 5%)")
    else:
        print(f"\n结论: 模型可能还在缓慢改进中")
    
    # 分析建筑选择分布
    print(f"\n建筑选择分析:")
    print(f"总选择数: {data['slot_selection_stats']['total_selections']}")
    print(f"EDU选择数: {data['slot_selection_stats']['edu_selections']}")
    print(f"IND选择数: {data['slot_selection_stats']['ind_selections']}")
    print(f"平均动作得分: {data['slot_selection_stats']['avg_action_score']:.6f}")
    
    # 检查动作得分的稳定性
    if data['slot_selection_stats']['avg_action_score'] < 0.1:
        print("注意: 平均动作得分较低，可能存在探索不足的问题")
    
    return {
        'converged': abs(improvement_pct) < 2.0,
        'improvement_pct': improvement_pct,
        'variance': variance,
        'final_return': data['final_avg_return']
    }

def plot_training_curves():
    """绘制训练曲线"""
    with open('models/v4_1_rl/training_results_20251014_021231.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_returns = data['training_metrics']['episode_returns']
    edu_returns = data['training_metrics']['edu_returns']
    ind_returns = data['training_metrics']['ind_returns']
    
    episodes = range(1, len(total_returns) + 1)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(episodes, total_returns, 'b-o', label='总回报')
    plt.title('总回报趋势')
    plt.xlabel('Episode')
    plt.ylabel('回报')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(episodes, edu_returns, 'g-o', label='EDU回报')
    plt.plot(episodes, ind_returns, 'r-o', label='IND回报')
    plt.title('分Agent回报趋势')
    plt.xlabel('Episode')
    plt.ylabel('回报')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    # 计算移动平均
    window = 3
    if len(total_returns) >= window:
        moving_avg = np.convolve(total_returns, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg, 'purple', linewidth=2, label=f'{window}期移动平均')
        plt.plot(episodes, total_returns, 'b-o', alpha=0.3, label='原始数据')
    plt.title('回报平滑趋势')
    plt.xlabel('Episode')
    plt.ylabel('回报')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # 计算回报变化率
    returns_diff = np.diff(total_returns)
    plt.plot(episodes[1:], returns_diff, 'orange', marker='o')
    plt.title('回报变化率')
    plt.xlabel('Episode')
    plt.ylabel('回报变化')
    plt.grid(True)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('training_convergence_analysis.png', dpi=150, bbox_inches='tight')
    print("\n训练曲线图已保存为: training_convergence_analysis.png")

if __name__ == "__main__":
    convergence_info = analyze_convergence()
    plot_training_curves()
    
    print(f"\n=== 最终建议 ===")
    if convergence_info['converged']:
        print("✅ 模型已经收敛，增加训练轮数意义不大")
        print("建议：")
        print("1. 检查模型是否真的学会了选择M/L型建筑")
        print("2. 如果仍然只选择S型建筑，问题可能在于：")
        print("   - 动作池本身不平衡（M/L型建筑可用槽位太少）")
        print("   - 探索策略需要进一步调整")
        print("   - 奖励信号需要重新设计")
    else:
        print("❌ 模型还未收敛，可以继续训练")
        print(f"当前改进幅度: {convergence_info['improvement_pct']:.2f}%")
        print("建议继续训练直到收敛")


