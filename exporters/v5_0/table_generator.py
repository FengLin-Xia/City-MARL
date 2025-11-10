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
    """v5.0 动作表格生成器 - 基于TXT数据+StepLog匹配"""
    
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
        
        # v4.1兼容性映射（与TXT导出器保持一致）
        self.agent_size_mapping = {
            "EDU": {"S": 0, "M": 1, "L": 2},
            "IND": {"S": 3, "M": 4, "L": 5, "A": 9, "B": 10, "C": 11},
            "COUNCIL": {"A": 6, "B": 7, "C": 8}
        }
    
    def generate_monthly_tables(self, step_logs: List[StepLog], 
                               env_states: List[EnvironmentState], 
                               output_dir: str) -> List[str]:
        """
        生成月度动作表格 - 基于TXT数据+StepLog匹配
        
        Args:
            step_logs: 步骤日志列表
            env_states: 对应的环境状态列表
            output_dir: 输出目录
            
        Returns:
            生成的文件路径列表
        """
        if len(step_logs) != len(env_states):
            raise ValueError("StepLogs和EnvironmentStates数量不匹配")
        
        # 生成输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 生成TXT数据（与TXT导出器保持一致）
        txt_data = self._generate_txt_data(step_logs, env_states)
        print(f"[TABLE_DEBUG] Generated TXT data for {len(txt_data)} months")
        
        # 2. 计算跨月预算变化，处理没有动作但有收入的月份
        budget_changes = self._calculate_cross_month_budget_changes(step_logs, env_states)
        
        # 3. 基于TXT数据生成表格
        generated_files = []
        for month, txt_content in txt_data.items():
            if not txt_content.strip():
                continue
                
            print(f"[TABLE_DEBUG] Processing month {month}, txt_content: {txt_content}")
            # 解析TXT数据获取动作信息
            txt_actions = self._parse_txt_actions(txt_content)
            print(f"[TABLE_DEBUG] Parsed {len(txt_actions)} actions for month {month}")
            
            # 按智能体分组TXT动作
            agent_actions = self._group_txt_actions_by_agent(txt_actions)
            
            # 为每个智能体生成表格
            for agent, actions in agent_actions.items():
                if not actions:
                    continue
                    
                # 找到对应的StepLog
                matching_step_log = self._find_matching_step_log(
                    actions, step_logs, month, agent
                )
                
                if matching_step_log:
                    # 生成表格
                    table_file = self._generate_agent_table_from_txt_and_step_log(
                        month, agent, actions, matching_step_log, output_dir, budget_changes
                    )
                    if table_file:
                        generated_files.append(table_file)
        
        return generated_files
    
    def _generate_txt_data(self, step_logs: List[StepLog], 
                          env_states: List[EnvironmentState]) -> Dict[int, str]:
        """生成TXT数据（与TXT导出器保持一致）"""
        # 如果数据超过30个月，说明有多个episode，需要找到最后一个episode的起始位置
        if len(step_logs) > 30:
            print(f"[TABLE_GENERATOR] 检测到多个episode数据({len(step_logs)}条)，寻找最后一个episode")
            
            # 找到最后一个episode的起始位置（从month=1开始的数据）
            last_episode_start = 0
            for i, state in enumerate(env_states):
                if state.month == 1 and i > 0:  # 找到新的episode开始
                    last_episode_start = i
            
            print(f"[TABLE_GENERATOR] 最后一个episode从索引{last_episode_start}开始")
            step_logs = step_logs[last_episode_start:]
            env_states = env_states[last_episode_start:]
        
        # 按月份分组
        monthly_data = {}
        for log, state in zip(step_logs, env_states):
            month = state.month
            if month not in monthly_data:
                monthly_data[month] = ([], [])
            monthly_data[month][0].append(log)
            monthly_data[month][1].append(state)
        
        # 生成TXT内容
        txt_data = {}
        for month, (month_logs, month_states) in monthly_data.items():
            if not month_logs:
                continue
                
            # 生成月度TXT输出
            txt_content = self._generate_monthly_txt_output(month_logs, month_states)
            txt_data[month] = txt_content
        
        return txt_data
    
    def _calculate_cross_month_budget_changes(self, step_logs: List[StepLog], 
                                            env_states: List[EnvironmentState]) -> Dict[Tuple[int, str], float]:
        """计算跨月预算变化，识别没有动作但有收入的月份"""
        budget_changes = {}
        
        # 按智能体和月份分组
        agent_month_budgets = {}
        for log, state in zip(step_logs, env_states):
            agent = log.agent
            month = state.month
            
            if agent not in agent_month_budgets:
                agent_month_budgets[agent] = {}
            
            # 获取预算值
            if log.budget_snapshot and agent in log.budget_snapshot:
                budget = log.budget_snapshot[agent]
            else:
                budget = state.budgets.get(agent, 0)
            
            agent_month_budgets[agent][month] = budget
        
        # 计算每个智能体的跨月预算变化
        for agent, month_budgets in agent_month_budgets.items():
            months = sorted(month_budgets.keys())
            for i in range(len(months) - 1):
                current_month = months[i]
                next_month = months[i + 1]
                
                current_budget = month_budgets[current_month]
                next_budget = month_budgets[next_month]
                
                # 计算预算变化
                budget_change = next_budget - current_budget
                budget_changes[(next_month, agent)] = budget_change
                
                print(f"[BUDGET_DEBUG] {agent} Month {current_month}→{next_month}: {current_budget:.1f}→{next_budget:.1f} (change: {budget_change:+.1f})")
        
        return budget_changes
    
    def _generate_monthly_txt_output(self, step_logs: List[StepLog], 
                                   env_states: List[EnvironmentState]) -> str:
        """生成月度TXT输出（与TXT导出器保持一致）"""
        output_lines = []
        
        for log, state in zip(step_logs, env_states):
            # 获取坐标信息
            coordinates = self._get_coordinates_from_env(log, state)
            
            # 生成v4.1格式行
            line = self._format_v4_line(log, coordinates)
            if line:
                output_lines.append(line)
        
        return '\n'.join(output_lines)
    
    def _get_coordinates_from_env(self, step_log: StepLog, 
                                 env_state: EnvironmentState) -> List[Tuple[float, float, float]]:
        """从环境状态获取坐标（与TXT导出器保持一致）"""
        coordinates = []
        
        # 优先使用StepLog中的槽位位置信息
        if step_log.slot_positions:
            for slot_pos in step_log.slot_positions:
                x = slot_pos.get('x', 0.0)
                y = slot_pos.get('y', 0.0)
                angle = slot_pos.get('angle', 0.0)
                coordinates.append((x, y, angle))
        else:
            # 回退到旧方法
            for action_id in step_log.chosen:
                # 根据动作ID找到对应的槽位
                slot_info = self._find_slot_by_action(action_id, env_state)
                if slot_info:
                    x, y, angle = slot_info
                    coordinates.append((x, y, angle))
                else:
                    # 如果找不到槽位，使用默认坐标
                    coordinates.append((0.0, 0.0, 0.0))
        
        return coordinates
    
    def _find_slot_by_action(self, action_id: int, 
                            env_state: EnvironmentState) -> Optional[Tuple[float, float, float]]:
        """根据动作ID查找槽位坐标"""
        # 从环境状态中查找对应的槽位
        if action_id < len(env_state.slots):
            slot = env_state.slots[action_id]
            if isinstance(slot, dict):
                x = slot.get('x', 0.0)
                y = slot.get('y', 0.0)
                angle = slot.get('angle', 0.0)
                return (x, y, angle)
        
        return (0.0, 0.0, 0.0)
    
    def _format_v4_line(self, step_log: StepLog, 
                       coordinates: List[Tuple[float, float, float]]) -> str:
        """格式化为v4.1格式（与TXT导出器保持一致）"""
        if not step_log.chosen or not coordinates:
            return ""
        
        # 生成v4.1格式输出
        parts = []
        for i, (action_id, (x, y, angle)) in enumerate(zip(step_log.chosen, coordinates)):
            # 获取动作参数
            action_params = self.action_params.get(str(action_id), {})
            desc = action_params.get("desc", f"ACTION_{action_id}")
            
            # 解析动作描述获取智能体和尺寸
            agent, size = self._parse_action_desc(desc)
            
            # 获取v4.1格式的动作编号
            v4_action_id = self._get_v4_action_id(agent, size)
            
            # v4.1格式：a(x,y,z)angle
            part = f"{v4_action_id}({x:.1f},{y:.1f},0){angle:.1f}"
            parts.append(part)
        
        return ', '.join(parts)
    
    def _parse_action_desc(self, desc: str) -> Tuple[str, str]:
        """解析动作描述获取智能体和尺寸"""
        if '_' in desc:
            parts = desc.split('_')
            if len(parts) >= 2:
                agent = parts[0]
                size = parts[1]
                return agent, size
        
        return "EDU", "S"
    
    def _get_v4_action_id(self, agent: str, size: str) -> int:
        """获取v4.1格式的动作编号"""
        return self.agent_size_mapping.get(agent, {}).get(size, 0)
    
    def _parse_txt_actions(self, txt_content: str) -> List[Dict]:
        """解析TXT内容获取动作信息"""
        actions = []
        
        for line in txt_content.strip().split('\n'):
            if not line.strip():
                continue
                
            # 解析格式：action_id(x,y,z)angle, action_id(x,y,z)angle, ...
            parts = line.split(', ')
            for part in parts:
                action_info = self._parse_txt_action_part(part)
                if action_info:
                    actions.append(action_info)
        
        return actions
    
    def _parse_txt_action_part(self, part: str) -> Optional[Dict]:
        """解析TXT动作部分"""
        try:
            # 格式：action_id(x,y,z)angle
            import re
            match = re.match(r'(\d+)\(([^,]+),([^,]+),([^)]+)\)([^,]+)', part.strip())
            if match:
                action_id = int(match.group(1))
                x = float(match.group(2))
                y = float(match.group(3))
                z = float(match.group(4))
                angle = float(match.group(5))
                
                return {
                    'action_id': action_id,
                    'x': x,
                    'y': y,
                    'z': z,
                    'angle': angle
                }
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def _group_txt_actions_by_agent(self, txt_actions: List[Dict]) -> Dict[str, List[Dict]]:
        """按智能体分组TXT动作"""
        agent_actions = {}
        
        for action in txt_actions:
            # 从动作ID推断智能体
            agent = self._get_agent_from_action_id(action['action_id'])
            if agent not in agent_actions:
                agent_actions[agent] = []
            agent_actions[agent].append(action)
        
        return agent_actions
    
    def _get_agent_from_action_id(self, action_id: int) -> str:
        """从动作ID推断智能体"""
        # 根据v4.1映射推断智能体
        if action_id in [0, 1, 2]:
            return "EDU"
        elif action_id in [3, 4, 5]:
            return "IND"
        elif action_id in [6, 7, 8]:
            return "COUNCIL"
        elif action_id in [9, 10, 11]:
            return "IND"
        else:
            return "UNKNOWN"
    
    def _find_matching_step_log(self, txt_actions: List[Dict], 
                               step_logs: List[StepLog], 
                               month: int, agent: str) -> Optional[StepLog]:
        """找到与TXT动作匹配的StepLog"""
        # 找到对应月份和智能体的StepLog
        matching_logs = []
        for log in step_logs:
            if (hasattr(log, 'month') and log.month == month and 
                hasattr(log, 'agent') and log.agent == agent):
                matching_logs.append(log)
        
        print(f"[TABLE_DEBUG] Looking for month={month}, agent={agent}")
        print(f"[TABLE_DEBUG] Found {len(matching_logs)} matching logs")
        if not matching_logs:
            print(f"[TABLE_DEBUG] No matching StepLog found for month={month}, agent={agent}")
            return None
        
        # 如果有多个StepLog，选择最后一个（与TXT导出器保持一致）
        return matching_logs[-1]
    
    def _generate_agent_table_from_txt_and_step_log(self, month: int, agent: str,
                                                   txt_actions: List[Dict],
                                                   step_log: StepLog,
                                                   output_dir: str,
                                                   budget_changes: Dict[Tuple[int, str], float] = None) -> Optional[str]:
        """基于TXT动作和StepLog生成智能体表格"""
        print(f"[TABLE_DEBUG] _generate_agent_table_from_txt_and_step_log called: month={month}, agent={agent}, actions={len(txt_actions)}")
        if not txt_actions or not step_log:
            print(f"[TABLE_DEBUG] Missing data: txt_actions={bool(txt_actions)}, step_log={bool(step_log)}")
            return None
        
        # 准备表格数据
        table_data = []
        total_cost = 0
        total_reward = 0
        total_prestige = 0
        
        # 使用StepLog中的实际数据
        if step_log.reward_terms:
            total_cost = abs(step_log.reward_terms.get("cost", 0))
            # reward_terms中的revenue就是总奖励
            total_reward = step_log.reward_terms.get("revenue", 0)
            total_prestige = step_log.reward_terms.get("prestige", 0)
        
        # 计算预算信息
        if hasattr(step_log, 'budget_snapshot') and step_log.budget_snapshot:
            if isinstance(step_log.budget_snapshot, dict):
                budget_after = step_log.budget_snapshot.get(agent, 0.0)
            else:
                budget_after = float(step_log.budget_snapshot)
        else:
            budget_after = 0.0

        # 使用跨月预算变化计算更准确的预算
        if budget_changes and (month, agent) in budget_changes:
            # 使用跨月预算变化
            total_budget_change = budget_changes[(month, agent)]
            budget_before = budget_after - total_budget_change
            print(f"[BUDGET_DEBUG] Using cross-month budget change for {agent} month {month}: {total_budget_change:+.1f}")
        else:
            # 回退到原始计算
            budget_before = budget_after + total_cost - total_reward
            total_budget_change = budget_after - budget_before
            print(f"[BUDGET_DEBUG] Using step-log budget calculation for {agent} month {month}: {total_budget_change:+.1f}")
        
        # 计算每个动作的预算变化
        action_budget_change = total_budget_change / len(txt_actions) if len(txt_actions) > 0 else 0
        
        # 为每个TXT动作生成一行
        for i, txt_action in enumerate(txt_actions):
            # 平均分配成本和奖励
            action_cost = total_cost / len(txt_actions) if len(txt_actions) > 0 else 0
            action_reward = total_reward / len(txt_actions) if len(txt_actions) > 0 else 0
            action_prestige = total_prestige / len(txt_actions) if len(txt_actions) > 0 else 0
            
            # 计算每行的预算变化（累积式）
            action_budget_before = budget_before + (i * action_budget_change)
            action_budget_after = action_budget_before + action_budget_change
            
            # 获取动作描述
            action_desc = self._get_action_desc_from_id(txt_action['action_id'])
            
            table_data.append({
                'action_num': i + 1,
                'action_desc': action_desc,
                'cost': action_cost,
                'reward': action_reward,
                'prestige': action_prestige,
                'budget_before': action_budget_before,
                'budget_after': action_budget_after
            })
        
        # 生成表格
        print(f"[TABLE_DEBUG] Creating table image for month={month}, agent={agent}, data_count={len(table_data)}")
        try:
            result = self._create_table_image(month, agent, table_data, output_dir)
            print(f"[TABLE_DEBUG] Table image result: {result}")
            return result
        except Exception as e:
            print(f"[TABLE_DEBUG] Error in _create_table_image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_action_desc_from_id(self, action_id: int) -> str:
        """从动作ID获取动作描述"""
        # 根据v4.1映射获取描述
        if action_id in [0, 1, 2]:
            sizes = ["S", "M", "L"]
            return f"EDU_{sizes[action_id]}"
        elif action_id in [3, 4, 5]:
            sizes = ["S", "M", "L"]
            return f"IND_{sizes[action_id-3]}"
        elif action_id in [6, 7, 8]:
            sizes = ["A", "B", "C"]
            return f"COUNCIL_{sizes[action_id-6]}"
        elif action_id in [9, 10, 11]:
            sizes = ["A", "B", "C"]
            return f"IND_{sizes[action_id-9]}"
        else:
            return f"ACTION_{action_id}"
    
    def _group_by_month_and_agent(self, step_logs: List[StepLog], 
                                 env_states: List[EnvironmentState]) -> Dict[Tuple[int, str], Tuple[List[StepLog], List[EnvironmentState]]]:
        """按月份和智能体分组数据"""
        # 如果数据超过30个月，说明有多个episode，需要找到最后一个episode的起始位置
        if len(step_logs) > 30:
            print(f"[TABLE_GENERATOR] 检测到多个episode数据({len(step_logs)}条)，寻找最后一个episode")
            
            # 找到最后一个episode的起始位置（从month=1开始的数据）
            last_episode_start = 0
            for i, state in enumerate(env_states):
                if state.month == 1 and i > 0:  # 找到新的episode开始
                    last_episode_start = i
            
            print(f"[TABLE_GENERATOR] 最后一个episode从索引{last_episode_start}开始")
            step_logs = step_logs[last_episode_start:]
            env_states = env_states[last_episode_start:]
        
        grouped_data = {}
        
        for log, state in zip(step_logs, env_states):
            month = state.month
            agent = log.agent
            key = (month, agent)
            
            if key not in grouped_data:
                grouped_data[key] = ([], [])
            grouped_data[key][0].append(log)
            grouped_data[key][1].append(state)
        
        # 只保留每个智能体每月的最后一个StepLog
        print(f"[TABLE_GENERATOR] 开始合并StepLog，共有{len(grouped_data)}个分组")
        for key in grouped_data:
            logs, states = grouped_data[key]
            print(f"[TABLE_GENERATOR] 检查分组 {key}: 有{len(logs)}个StepLog")
            if len(logs) > 1:
                # 只保留最后一个StepLog
                grouped_data[key] = ([logs[-1]], [states[-1]])
                print(f"[TABLE_GENERATOR] 智能体{key[1]}在月份{key[0]}有{len(logs)}个StepLog，只保留最后一个")
        
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
        
        # 添加可配置的日志输出
        if self.config.get("logging", {}).get("topics", {}).get("export_coords", False):
            print(f"[TABLE_DEBUG] Generating table for month={month}, agent={agent}, logs={len(step_logs)}")
        
        action_count = 0
        for i, (log, state) in enumerate(zip(step_logs, env_states)):
            # 为每个动作生成一行
            for j, action_id in enumerate(log.chosen):
                action_count += 1
                
                # 使用StepLog中的实际奖励数据，而不是配置中的静态参数
                if log.reward_terms:
                    # 获取动作参数用于基础值
                    action_params = self._get_action_params(action_id)
                    base_cost = action_params.get("base_cost", 0)
                    base_reward = action_params.get("base_reward", 0)
                    base_prestige = action_params.get("prestige_base", 0)
                    
                    # 计算动态调整因子（基于总奖励与基础奖励的比例）
                    total_actions = len(log.chosen)
                    total_revenue = log.reward_terms.get("revenue", 0)
                    total_cost = abs(log.reward_terms.get("cost", 0))
                    
                    if total_revenue > 0 and base_reward > 0:
                        # 使用动态比例来估算每个动作的值
                        revenue_ratio = total_revenue / (base_reward * total_actions)
                        cost_ratio = total_cost / (abs(base_cost) * total_actions)
                        
                        # 为每个动作添加随机变化，模拟不同位置和环境的影响
                        import random
                        random.seed(action_id + j + log.t)  # 使用动作ID、位置、时间作为种子
                        cost_variation = 1.0 + (random.random() - 0.5) * 0.1  # ±5% 变化
                        reward_variation = 1.0 + (random.random() - 0.5) * 0.15  # ±7.5% 变化
                        
                        cost = abs(base_cost) * cost_ratio * cost_variation
                        reward = base_reward * revenue_ratio * reward_variation
                        prestige = base_prestige
                    else:
                        # 降级到平均分配
                        cost = total_cost / total_actions
                        reward = total_revenue / total_actions
                        prestige = log.reward_terms.get("prestige", 0) / total_actions
                    
                    # 添加可配置的调试日志
                    if self.config.get("logging", {}).get("topics", {}).get("export_coords", False):
                        print(f"[TABLE_DEBUG] Action {action_id}: cost={cost:.1f}, reward={reward:.1f}, prestige={prestige:.2f}")
                else:
                    # 降级：使用配置参数
                    action_params = self._get_action_params(action_id)
                    cost = action_params.get("cost", 0)
                    reward = action_params.get("reward", 0)
                    prestige = action_params.get("prestige", 0)
                    
                    # 添加可配置的调试日志
                    if self.config.get("logging", {}).get("topics", {}).get("export_coords", False):
                        print(f"[TABLE_DEBUG] Using static config for action {action_id}: cost={cost}, reward={reward}, prestige={prestige}")
                
                total_cost += cost
                total_reward += reward
                total_prestige += prestige
                
                # 计算预算变化
                budget_before = budget_info.get(f"budget_before_{i}", 0)
                budget_after = budget_info.get(f"budget_after_{i}", 0)
                
                # 表格行数据
                row = [
                    f"Action {action_count}",
                    action_params.get("desc", f"ACTION_{action_id}") if 'action_params' in locals() else f"ACTION_{action_id}",
                    f"{cost:.1f}",
                    f"{reward:.1f}",
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
        """计算预算信息 - 使用StepLog中的准确budget数据"""
        budget_info = {}
        
        if not step_logs or not env_states:
            return budget_info
        
        # 直接使用StepLog中的budget_snapshot，这是最准确的数据源
        for i, (log, state) in enumerate(zip(step_logs, env_states)):
            agent = log.agent
            
            # 使用StepLog中的budget_snapshot作为准确数据源
            if log.budget_snapshot and agent in log.budget_snapshot:
                # 当前动作后的预算
                budget_after = log.budget_snapshot[agent]
                
                # 计算动作前的预算（当前预算 + 成本 - 奖励）
                if log.reward_terms:
                    cost = abs(log.reward_terms.get("cost", 0))
                    reward = log.reward_terms.get("revenue", 0)  # 使用revenue作为实际奖励
                    budget_before = budget_after + cost - reward
                else:
                    budget_before = budget_after
                
                # 添加可配置的调试日志
                if self.config.get("logging", {}).get("topics", {}).get("export_coords", False):
                    print(f"[BUDGET_DEBUG] Agent {agent}: before={budget_before:.1f}, after={budget_after:.1f}")
            else:
                # 如果没有budget_snapshot，使用环境状态中的budgets
                budget_after = state.budgets.get(agent, 0)
                budget_before = budget_after
                
                # 添加可配置的调试日志
                if self.config.get("logging", {}).get("topics", {}).get("export_coords", False):
                    print(f"[BUDGET_DEBUG] Using env state budget for agent {agent}: {budget_after:.1f}")
            
            budget_info[f"budget_before_{i}"] = budget_before
            budget_info[f"budget_after_{i}"] = budget_after
        
        # 使用最后一个StepLog的预算作为最终预算
        if step_logs and hasattr(step_logs[-1], 'budget_snapshot') and step_logs[-1].budget_snapshot:
            final_agent = step_logs[-1].agent
            budget_info["final_budget"] = step_logs[-1].budget_snapshot.get(final_agent, 0)
        else:
            # 如果没有budget_snapshot，使用环境状态
            if env_states:
                final_agent = step_logs[-1].agent
                budget_info["final_budget"] = env_states[-1].budgets.get(final_agent, 0)
            else:
                budget_info["final_budget"] = 0
                
        return budget_info
    
    def _create_table_image(self, month: int, agent: str, 
                           table_data: List[Dict], output_dir: str) -> str:
        """创建表格图片"""
        if not table_data:
            return None
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        headers = ["#", "Action", "Cost", "Reward", "Prestige", "Budget"]
        
        # 转换table_data为二维列表格式
        rows = []
        for data in table_data:
            rows.append([
                data['action_num'],
                data['action_desc'],
                f"{int(round(data['cost']))}",
                f"{int(round(data['reward']))}",
                f"{data['prestige']:.2f}",
                f"{int(round(data['budget_before']))} → {int(round(data['budget_after']))}"
            ])
        
        # 添加总计行 - 使用已取整的单个动作值计算总和
        total_cost = sum(int(round(data['cost'])) for data in table_data)
        total_reward = sum(int(round(data['reward'])) for data in table_data)
        total_prestige = sum(data['prestige'] for data in table_data)
        final_budget = int(round(table_data[-1]['budget_after'])) if table_data else 0
        
        rows.append([
            'TOTAL',
            f"{len(table_data)} actions",
            f"{total_cost}",
            f"{total_reward}",
            f"{total_prestige:.2f}",
            f"{final_budget}"
        ])
        
        # 创建表格
        table = ax.table(
            cellText=rows,
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
        filename = f'month_{month:02d}_{agent}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        plt.close()
        
        return filepath
    
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
            
            # 使用StepLog中的实际奖励数据
            if log.reward_terms:
                cost = abs(log.reward_terms.get("cost", 0))  # cost是负值，取绝对值
                reward = log.reward_terms.get("base_reward", 0) + log.reward_terms.get("land_price_reward", 0) + log.reward_terms.get("proximity_reward", 0)
                prestige = log.reward_terms.get("prestige", 0)
                
                agent_stats[agent]['total_cost'] += cost
                agent_stats[agent]['total_reward'] += reward
                agent_stats[agent]['total_prestige'] += prestige
            else:
                # 降级：使用配置参数
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

