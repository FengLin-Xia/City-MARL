"""
v5.0 月度汇总PNG导出器

从StepLog数据中提取cost、reward、budget信息，按月聚合后生成透明底白字的PNG图片。
输出格式：三个数字用5个空格分隔，如 "358     16     20"
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from contracts import StepLog, EnvironmentState


class V5MonthlySummaryPNGExporter:
    """v5.0 月度汇总PNG导出器"""
    
    def __init__(self, config_path: str):
        """
        初始化导出器
        
        Args:
            config_path: v5.0配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # 设置matplotlib样式
        plt.style.use('default')
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found, using default config")
            return {}
    
    def export_monthly_summary_png(self, step_logs: List[StepLog], 
                                  env_states: List[EnvironmentState], 
                                  output_dir: str) -> List[str]:
        """
        导出月度汇总PNG
        
        Args:
            step_logs: 步骤日志列表
            env_states: 对应的环境状态列表
            output_dir: 输出目录
            
        Returns:
            生成的文件路径列表
        """
        print(f"[MonthlyPNG] Starting export for {len(step_logs)} step logs")
        
        # 如果数量不匹配，使用step_logs为准
        if len(step_logs) != len(env_states):
            print(f"Warning: StepLogs({len(step_logs)})和EnvironmentStates({len(env_states)})数量不匹配")
            env_states = env_states[-len(step_logs):] if len(env_states) > len(step_logs) else env_states
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 按月份分组数据
        monthly_data = self._group_by_month(step_logs, env_states)
        
        print(f"[MonthlyPNG] Found {len(monthly_data)} months: {sorted(monthly_data.keys())}")
        
        generated_files = []
        
        # 为每个月生成PNG
        for month, (month_step_logs, month_env_states) in monthly_data.items():
            print(f"[MonthlyPNG] Processing month {month} with {len(month_step_logs)} logs")
            
            # 计算月度汇总
            summary = self._calculate_monthly_summary(month_step_logs, month_env_states)
            print(f"[MonthlyPNG] Month {month} summary: {summary}")
            
            # 生成PNG
            filepath = self._generate_png(summary, month, output_dir)
            generated_files.append(filepath)
            print(f"[MonthlyPNG] Generated month summary PNG: {filepath}")
        
        print(f"[MonthlyPNG] Successfully generated {len(generated_files)} monthly summary PNG files")
        return generated_files
    
    def _group_by_month(self, step_logs: List[StepLog], 
                       env_states: List[EnvironmentState]) -> Dict[int, Tuple[List[StepLog], List[EnvironmentState]]]:
        """按月份分组数据，一个月对应一个phase"""
        monthly_data = {}
        
        # 按月份分组，一个月就是一个phase
        for log, state in zip(step_logs, env_states):
            month = state.month
            if month not in monthly_data:
                monthly_data[month] = ([], [])
            monthly_data[month][0].append(log)
            monthly_data[month][1].append(state)
        
        return monthly_data
    
    def _calculate_monthly_summary(self, step_logs: List[StepLog], 
                                 env_states: List[EnvironmentState]) -> Dict[str, float]:
        """计算月度汇总数据"""
        total_cost = 0.0
        total_reward = 0.0
        total_budget = 0.0
        
        # 聚合所有agent的数据（cost和reward累加）
        for log, state in zip(step_logs, env_states):
            # 从reward_terms获取cost和reward
            if log.reward_terms:
                # cost是负值，取绝对值
                cost = abs(log.reward_terms.get("cost", 0))
                # reward可能包含多个字段，使用revenue作为主要奖励
                reward = log.reward_terms.get("revenue", 0)
                
                total_cost += cost
                total_reward += reward
                print(f"  - {log.agent}: cost={cost:.1f}, reward={reward:.1f}")
            else:
                print(f"  - {log.agent}: no reward_terms")
            
        # 修复：预算只取最后一个StepLog的budget_snapshot总和（避免重复累加）
        if step_logs:
            last_log = step_logs[-1]
            if last_log.budget_snapshot:
                if isinstance(last_log.budget_snapshot, dict):
                    # 如果是字典，累加所有agent的预算
                    total_budget = sum(
                        sum(agent_budget.values()) if isinstance(agent_budget, dict)
                        else float(agent_budget)
                        for agent_budget in last_log.budget_snapshot.values()
                    )
                else:
                    total_budget = float(last_log.budget_snapshot)
                print(f"  - Budget from last step_log: {total_budget:.1f}")
            else:
                print(f"  - Warning: last step_log has no budget_snapshot")
        
        print(f"  - Monthly totals: cost={total_cost:.1f}, reward={total_reward:.1f}, budget={total_budget:.1f}")
        
        return {
            "cost": total_cost,
            "reward": total_reward,
            "budget": total_budget
        }
    
    def _generate_png(self, summary: Dict[str, float], month: int, output_dir: str) -> str:
        """生成PNG图片"""
        # 取整数值
        cost = int(round(summary["cost"]))
        reward = int(round(summary["reward"]))
        budget = int(round(summary["budget"]))
        
        # 格式化文本：budget在最前面，三个数字用16个空格分隔
        text = f"{budget}                {cost}                {reward}"
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis('off')
        
        # 设置文本
        ax.text(0.5, 0.5, text, 
                fontsize=24, 
                fontweight='bold',
                color='white',
                ha='center', 
                va='center',
                transform=ax.transAxes)
        
        # 设置透明背景
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        
        # 保存文件
        filename = f"month_{month:02d}_summary.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, 
                    dpi=150, 
                    bbox_inches='tight',
                    facecolor='none', 
                    edgecolor='none',
                    transparent=True)
        plt.close()
        
        return filepath
