"""
v5.0 动作表格生成器

基于契约对象和配置的表格生成系统。
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from contracts import StepLog, EnvironmentState
from config_loader import ConfigLoader


class V5TableGenerator:
    """v5.0 动作表格生成器"""
    
    def __init__(self, config_path: str):
        """
        初始化表格生成器
        
        Args:
            config_path: v5.0配置文件路径
        """
        self.loader = ConfigLoader()
        self.config = self.loader.load_v5_config(config_path)
        self.action_params = self.config.get("action_params", {})
        
        # 设置matplotlib样式
        plt.style.use('dark_background')
    
    def generate_monthly_tables(self, step_logs: List[StepLog], 
                               env_states: List[EnvironmentState], 
                               output_dir: str) -> List[str]:
        """
        生成月度动作表格
        
        Args:
            step_logs: 步骤日志列表
            env_states: 对应的环境状态列表
            output_dir: 输出目录
            
        Returns:
            生成的文件路径列表
        """
        if len(step_logs) != len(env_states):
            raise ValueError("StepLogs和EnvironmentStates数量不匹配")
        
        # 按月份和智能体分组
        monthly_data = self._group_by_month_and_agent(step_logs, env_states)
        
        # 生成输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = []
        for (month, agent), (month_logs, month_states) in monthly_data.items():
            if not month_logs:
                continue
            
            # 生成表格
            table_file = self._generate_agent_table(
                month, agent, month_logs, month_states, output_dir
            )
            if table_file:
                generated_files.append(table_file)
        
        return generated_files
    
    def _group_by_month_and_agent(self, step_logs: List[StepLog], 
                                 env_states: List[EnvironmentState]) -> Dict[Tuple[int, str], Tuple[List[StepLog], List[EnvironmentState]]]:
        """按月份和智能体分组数据"""
        grouped_data = {}
        
        for log, state in zip(step_logs, env_states):
            month = state.month
            agent = log.agent
            key = (month, agent)
            
            if key not in grouped_data:
                grouped_data[key] = ([], [])
            grouped_data[key][0].append(log)
            grouped_data[key][1].append(state)
        
        return grouped_data
    
    def _generate_agent_table(self, month: int, agent: str, 
                             step_logs: List[StepLog], 
                             env_states: List[EnvironmentState], 
                             output_dir: str) -> Optional[str]:
        """生成单个智能体的动作表格"""
        if not step_logs:
            return None
        
        # 准备表格数据
        table_data = []
        total_cost = 0
        total_reward = 0
        total_prestige = 0
        
        # 计算预算信息
        budget_info = self._calculate_budget_info(step_logs, env_states)
        
        for i, (log, state) in enumerate(zip(step_logs, env_states)):
            # 获取动作参数
            action_params = self._get_action_params(log.chosen[0])
            
            cost = action_params.get("cost", 0)
            reward = action_params.get("reward", 0)
            prestige = action_params.get("prestige", 0)
            
            total_cost += cost
            total_reward += reward
            total_prestige += prestige
            
            # 计算预算变化
            budget_before = budget_info.get(f"budget_before_{i}", 0)
            budget_after = budget_info.get(f"budget_after_{i}", 0)
            
            # 表格行数据
            row = [
                f"Action {i+1}",
                action_params.get("desc", f"ACTION_{log.chosen[0]}"),
                f"{cost}",
                f"{reward}",
                f"{prestige:.2f}",
                f"{budget_before} → {budget_after}"
            ]
            table_data.append(row)
        
        # 添加总计行
        total_row = [
            "TOTAL",
            f"{len(step_logs)} actions",
            f"{total_cost}",
            f"{total_reward}",
            f"{total_prestige:.2f}",
            f"{budget_info.get('final_budget', 0)}"
        ]
        table_data.append(total_row)
        
        # 生成表格图片
        output_path = os.path.join(output_dir, f"month_{month:02d}_{agent}.png")
        self._create_table_image(month, agent, table_data, output_path)
        
        print(f"Generated: {output_path}")
        return output_path
    
    def _get_action_params(self, action_id: int) -> Dict[str, Any]:
        """获取动作参数"""
        return self.action_params.get(str(action_id), {
            "desc": f"ACTION_{action_id}",
            "cost": 0,
            "reward": 0,
            "prestige": 0
        })
    
    def _calculate_budget_info(self, step_logs: List[StepLog], 
                              env_states: List[EnvironmentState]) -> Dict[str, Any]:
        """计算预算信息"""
        budget_info = {}
        
        if not step_logs or not env_states:
            return budget_info
        
        # 获取初始预算
        initial_budget = env_states[0].budgets.get(step_logs[0].agent, 0)
        current_budget = initial_budget
        
        for i, (log, state) in enumerate(zip(step_logs, env_states)):
            budget_before = current_budget
            
            # 计算动作成本
            action_params = self._get_action_params(log.chosen[0])
            cost = action_params.get("cost", 0)
            reward = action_params.get("reward", 0)
            
            # 更新预算
            current_budget = current_budget - cost + reward
            
            budget_info[f"budget_before_{i}"] = budget_before
            budget_info[f"budget_after_{i}"] = current_budget
        
        budget_info["final_budget"] = current_budget
        return budget_info
    
    def _create_table_image(self, month: int, agent: str, 
                           table_data: List[List[str]], output_path: str):
        """创建表格图片"""
        if not table_data:
            return
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        headers = ["#", "Action", "Cost", "Reward", "Prestige", "Budget"]
        
        # 创建表格
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置单元格样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 表头
                cell.set_text_props(color='white', weight='bold', fontsize=12)
                cell.set_facecolor('#2E2E2E')
            elif i == len(table_data):  # 总计行
                cell.set_text_props(color='white', weight='bold', fontsize=11)
                cell.set_facecolor('#1E1E1E')
            else:  # 普通行
                cell.set_text_props(color='white', fontsize=10)
                cell.set_facecolor('#3E3E3E')
            
            cell.set_edgecolor('white')
            cell.set_linewidth(1.0)
        
        # 设置背景
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # 设置标题
        title = f"Month {month} - {agent} Actions"
        ax.set_title(title, color='white', fontsize=16, weight='bold', pad=20)
        
        # 保存图片
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        plt.close()
    
    def generate_summary_table(self, step_logs: List[StepLog], 
                              env_states: List[EnvironmentState], 
                              output_path: str) -> str:
        """生成汇总表格"""
        # 按智能体统计
        agent_stats = {}
        
        for log, state in zip(step_logs, env_states):
            agent = log.agent
            if agent not in agent_stats:
                agent_stats[agent] = {
                    'total_actions': 0,
                    'total_cost': 0,
                    'total_reward': 0,
                    'total_prestige': 0
                }
            
            # 统计信息
            agent_stats[agent]['total_actions'] += len(log.chosen)
            
            for action_id in log.chosen:
                action_params = self._get_action_params(action_id)
                agent_stats[agent]['total_cost'] += action_params.get("cost", 0)
                agent_stats[agent]['total_reward'] += action_params.get("reward", 0)
                agent_stats[agent]['total_prestige'] += action_params.get("prestige", 0)
        
        # 生成汇总表格
        summary_data = []
        for agent, stats in agent_stats.items():
            row = [
                agent,
                str(stats['total_actions']),
                str(stats['total_cost']),
                str(stats['total_reward']),
                f"{stats['total_prestige']:.2f}"
            ]
            summary_data.append(row)
        
        # 创建汇总表格图片
        self._create_summary_table_image(summary_data, output_path)
        
        print(f"Generated summary table: {output_path}")
        return output_path
    
    def _create_summary_table_image(self, summary_data: List[List[str]], output_path: str):
        """创建汇总表格图片"""
        if not summary_data:
            return
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        headers = ["Agent", "Actions", "Total Cost", "Total Reward", "Total Prestige"]
        
        # 创建表格
        table = ax.table(
            cellText=summary_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # 设置单元格样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 表头
                cell.set_text_props(color='white', weight='bold', fontsize=14)
                cell.set_facecolor('#2E2E2E')
            else:  # 普通行
                cell.set_text_props(color='white', fontsize=12)
                cell.set_facecolor('#3E3E3E')
            
            cell.set_edgecolor('white')
            cell.set_linewidth(1.0)
        
        # 设置背景
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # 设置标题
        title = "Training Summary - All Agents"
        ax.set_title(title, color='white', fontsize=16, weight='bold', pad=20)
        
        # 保存图片
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        plt.close()
