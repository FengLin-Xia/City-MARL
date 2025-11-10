#!/usr/bin/env python3
"""
月度汇总PNG导出器

从StepLog数据中提取cost、reward、budget信息，按月聚合后生成透明底白字的PNG图片。
输出格式：三个数字用5个空格分隔，如 "358     16     20"
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from contracts import StepLog, EnvironmentState


class MonthlySummaryPNGExporter:
    """月度汇总PNG导出器"""
    
    def __init__(self):
        """初始化导出器"""
        # 设置matplotlib样式
        plt.style.use('default')
        
    def export_from_step_logs(self, step_logs: List[StepLog], 
                             env_states: List[EnvironmentState], 
                             output_dir: str) -> List[str]:
        """
        从StepLog数据导出月度汇总PNG
        
        Args:
            step_logs: 步骤日志列表
            env_states: 对应的环境状态列表
            output_dir: 输出目录
            
        Returns:
            生成的文件路径列表
        """
        print(f"Debug: step_logs count: {len(step_logs)}, env_states count: {len(env_states)}")
        
        # 如果数量不匹配，使用step_logs为准，忽略多余的env_states
        if len(step_logs) != len(env_states):
            print(f"Warning: StepLogs({len(step_logs)})和EnvironmentStates({len(env_states)})数量不匹配，使用最后{len(step_logs)}个EnvironmentStates")
            env_states = env_states[-len(step_logs):] if len(env_states) > len(step_logs) else env_states
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 按月份分组数据
        monthly_data = self._group_by_month(step_logs, env_states)
        
        print(f"Debug: Found {len(monthly_data)} months: {list(monthly_data.keys())}")
        
        # 生成每月汇总PNG
        generated_files = []
        for month, (month_logs, month_states) in monthly_data.items():
            if not month_logs:
                continue
                
            print(f"Debug: Processing month {month} with {len(month_logs)} logs")
            
            # 计算月度汇总数据
            monthly_summary = self._calculate_monthly_summary(month_logs, month_states)
            
            print(f"Debug: Month {month} summary: {monthly_summary}")
            
            # 生成PNG文件
            png_file = self._generate_monthly_png(month, monthly_summary, output_dir)
            if png_file:
                generated_files.append(png_file)
                print(f"Generated monthly summary PNG: {png_file}")
        
        return generated_files
    
    def export_from_json_results(self, json_file_path: str, output_dir: str) -> List[str]:
        """
        从v5_0_results JSON文件导出月度汇总PNG
        
        Args:
            json_file_path: JSON结果文件路径
            output_dir: 输出目录
            
        Returns:
            生成的文件路径列表
        """
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取step_logs和env_states
        step_logs_data = data.get('training_result', {}).get('step_logs', [])
        env_states_data = data.get('training_result', {}).get('env_states', [])
        
        if not step_logs_data or not env_states_data:
            print(f"No step_logs or env_states found in {json_file_path}")
            return []
        
        # 转换为StepLog和EnvironmentState对象
        step_logs = self._convert_to_step_logs(step_logs_data)
        env_states = self._convert_to_env_states(env_states_data)
        
        # 调用主导出方法
        return self.export_from_step_logs(step_logs, env_states, output_dir)
    
    def _group_by_month(self, step_logs: List[StepLog], 
                       env_states: List[EnvironmentState]) -> Dict[int, Tuple[List[StepLog], List[EnvironmentState]]]:
        """按月份分组数据"""
        monthly_data = {}
        
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
        
        # 聚合所有agent的数据
        for log, state in zip(step_logs, env_states):
            # 从reward_terms获取cost和reward
            if log.reward_terms:
                # cost是负值，取绝对值
                cost = abs(log.reward_terms.get("cost", 0))
                # reward可能包含多个字段，使用revenue作为主要奖励
                reward = log.reward_terms.get("revenue", 0)
                
                total_cost += cost
                total_reward += reward
            
            # 从budget_snapshot获取预算信息
            if log.budget_snapshot:
                if isinstance(log.budget_snapshot, dict):
                    # 如果是字典，累加所有agent的预算
                    for agent_budget in log.budget_snapshot.values():
                        total_budget += float(agent_budget)
                else:
                    # 如果是单个值
                    total_budget += float(log.budget_snapshot)
            else:
                # 回退到环境状态中的budgets
                if hasattr(state, 'budgets') and state.budgets:
                    for agent_budget in state.budgets.values():
                        total_budget += float(agent_budget)
        
        return {
            'cost': total_cost,
            'reward': total_reward,
            'budget': total_budget
        }
    
    def _generate_monthly_png(self, month: int, summary: Dict[str, float], 
                             output_dir: str) -> str:
        """生成月度汇总PNG"""
        # 取整数值
        cost = int(round(summary['cost']))
        reward = int(round(summary['reward']))
        budget = int(round(summary['budget']))
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis('off')
        
        # 设置透明背景
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        
        # 生成文本内容（三个数字用5个空格分隔）
        text_content = f"{cost}     {reward}     {budget}"
        
        # 添加文本
        ax.text(0.5, 0.5, text_content, 
                fontsize=24, 
                color='white', 
                ha='center', 
                va='center',
                weight='bold',
                fontfamily='monospace')
        
        # 保存PNG文件
        filename = f'month_{month:02d}_summary.png'
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight',
                    facecolor='none', edgecolor='none',
                    transparent=True)
        plt.close()
        
        return filepath
    
    def _convert_to_step_logs(self, step_logs_data: List) -> List[StepLog]:
        """将JSON数据转换为StepLog对象"""
        step_logs = []
        for data in step_logs_data:
            try:
                # 如果是字符串格式，需要解析
                if isinstance(data, str):
                    # 解析字符串格式的StepLog
                    parsed_data = self._parse_step_log_string(data)
                    if parsed_data:
                        step_log = StepLog(**parsed_data)
                        step_logs.append(step_log)
                elif isinstance(data, dict):
                    # 字典格式直接使用
                    step_log = StepLog(
                        t=data.get('t', 0),
                        agent=data.get('agent', 'UNKNOWN'),
                        chosen=data.get('chosen', []),
                        reward_terms=data.get('reward_terms', {}),
                        budget_snapshot=data.get('budget_snapshot'),
                        slot_positions=data.get('slot_positions')
                    )
                    step_logs.append(step_log)
            except Exception as e:
                print(f"Error converting StepLog: {e}")
                continue
        
        return step_logs
    
    def _parse_step_log_string(self, step_log_str: str) -> Optional[Dict]:
        """解析字符串格式的StepLog"""
        try:
            import re
            import ast
            
            # 提取StepLog(...)中的内容
            match = re.match(r"StepLog\((.*)\)", step_log_str.strip())
            if not match:
                return None
            
            content = match.group(1)
            
            # 解析参数
            parsed_data = {}
            
            # 解析t=...
            t_match = re.search(r"t=(\d+)", content)
            if t_match:
                parsed_data['t'] = int(t_match.group(1))
            
            # 解析agent=...
            agent_match = re.search(r"agent='([^']+)'", content)
            if agent_match:
                parsed_data['agent'] = agent_match.group(1)
            
            # 解析chosen=...
            chosen_match = re.search(r"chosen=\[([^\]]*)\]", content)
            if chosen_match:
                chosen_str = chosen_match.group(1)
                if chosen_str.strip():
                    parsed_data['chosen'] = [int(x.strip()) for x in chosen_str.split(',')]
                else:
                    parsed_data['chosen'] = []
            
            # 解析reward_terms=...
            reward_match = re.search(r"reward_terms=(\{[^}]*\})", content)
            if reward_match:
                try:
                    parsed_data['reward_terms'] = ast.literal_eval(reward_match.group(1))
                except:
                    parsed_data['reward_terms'] = {}
            
            # 解析budget_snapshot=...
            budget_match = re.search(r"budget_snapshot=(\{[^}]*\})", content)
            if budget_match:
                try:
                    parsed_data['budget_snapshot'] = ast.literal_eval(budget_match.group(1))
                except:
                    parsed_data['budget_snapshot'] = None
            
            # 解析slot_positions=...
            slot_match = re.search(r"slot_positions=(\[.*?\])", content)
            if slot_match:
                try:
                    parsed_data['slot_positions'] = ast.literal_eval(slot_match.group(1))
                except:
                    parsed_data['slot_positions'] = None
            
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing StepLog string: {e}")
            return None
    
    def _convert_to_env_states(self, env_states_data: List) -> List[EnvironmentState]:
        """将JSON数据转换为EnvironmentState对象"""
        env_states = []
        print(f"Debug: Processing {len(env_states_data)} env_states")
        
        for i, data in enumerate(env_states_data):
            try:
                # 如果是字符串格式，需要解析
                if isinstance(data, str):
                    print(f"Debug: Processing string env_state {i}: {data[:100]}...")
                    # 解析字符串格式的EnvironmentState
                    parsed_data = self._parse_env_state_string(data)
                    if parsed_data:
                        env_state = EnvironmentState(**parsed_data)
                        env_states.append(env_state)
                        print(f"Debug: Successfully parsed env_state {i}, month={parsed_data.get('month')}")
                    else:
                        print(f"Debug: Failed to parse env_state {i}")
                elif isinstance(data, dict):
                    # 字典格式直接使用
                    env_state = EnvironmentState(
                        month=data.get('month', 0),
                        land_prices=np.array(data.get('land_prices', [])),
                        buildings=data.get('buildings', []),
                        budgets=data.get('budgets', {}),
                        slots=data.get('slots', [])
                    )
                    env_states.append(env_state)
                    print(f"Debug: Successfully processed dict env_state {i}, month={data.get('month')}")
            except Exception as e:
                print(f"Error converting EnvironmentState {i}: {e}")
                continue
        
        print(f"Debug: Successfully converted {len(env_states)} env_states")
        return env_states
    
    def _parse_env_state_string(self, env_state_str: str) -> Optional[Dict]:
        """解析字符串格式的EnvironmentState"""
        try:
            import re
            import ast
            
            # 使用多行模式匹配
            match = re.match(r"EnvironmentState\((.*)\)", env_state_str.strip(), re.DOTALL)
            if not match:
                print(f"Debug: No match for EnvironmentState pattern")
                return None
            
            content = match.group(1)
            print(f"Debug: Content: {content[:200]}...")
            
            # 解析参数
            parsed_data = {}
            
            # 解析month=...
            month_match = re.search(r"month=(\d+)", content)
            if month_match:
                parsed_data['month'] = int(month_match.group(1))
                print(f"Debug: Found month: {parsed_data['month']}")
            else:
                print("Debug: No month found")
                return None
            
            # 解析land_prices=... (简化处理)
            parsed_data['land_prices'] = np.array([[1.0]])  # 占位符
            
            # 解析buildings=...
            buildings_match = re.search(r"buildings=(\[.*?\])", content, re.DOTALL)
            if buildings_match:
                try:
                    parsed_data['buildings'] = ast.literal_eval(buildings_match.group(1))
                except:
                    parsed_data['buildings'] = []
            else:
                parsed_data['buildings'] = []
            
            # 解析budgets=...
            budgets_match = re.search(r"budgets=(\{[^}]*\})", content, re.DOTALL)
            if budgets_match:
                try:
                    parsed_data['budgets'] = ast.literal_eval(budgets_match.group(1))
                except:
                    parsed_data['budgets'] = {}
            else:
                parsed_data['budgets'] = {}
            
            # 解析slots=... (简化处理)
            parsed_data['slots'] = []
            
            print(f"Debug: Parsed data: {parsed_data}")
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing EnvironmentState string: {e}")
            return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Export monthly summary PNG from StepLog data')
    parser.add_argument('--json_file', type=str, help='Path to v5_0_results JSON file')
    parser.add_argument('--output_dir', type=str, default='outputs/table/monthly_summary', 
                       help='Output directory for PNG files')
    parser.add_argument('--month', type=int, help='Specific month to export (optional)')
    
    args = parser.parse_args()
    
    if not args.json_file:
        print("Please provide --json_file argument")
        return
    
    if not os.path.exists(args.json_file):
        print(f"JSON file not found: {args.json_file}")
        return
    
    # 创建导出器
    exporter = MonthlySummaryPNGExporter()
    
    # 导出PNG文件
    try:
        generated_files = exporter.export_from_json_results(args.json_file, args.output_dir)
        
        if generated_files:
            print(f"\nSuccessfully generated {len(generated_files)} monthly summary PNG files:")
            for file_path in generated_files:
                print(f"  - {file_path}")
        else:
            print("No PNG files were generated")
            
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
