"""
v5.0 TXT导出器

基于契约对象和配置的TXT导出系统，完全兼容v4.1格式。
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from contracts import StepLog, EnvironmentState
from config_loader import ConfigLoader


class V5TXTExporter:
    """v5.0 TXT导出器 - 完全兼容v4.1格式"""
    
    def __init__(self, config_path: str):
        """
        初始化TXT导出器
        
        Args:
            config_path: v5.0配置文件路径
        """
        self.loader = ConfigLoader()
        self.config = self.loader.load_v5_config(config_path)
        self.action_params = self.config.get("action_params", {})
        
        # v4.1兼容性映射（保持原有格式）
        self.agent_size_mapping = {
            "EDU": {"S": 0, "M": 1, "L": 2},
            "IND": {"S": 3, "M": 4, "L": 5},
            "COUNCIL": {"A": 6, "B": 7, "C": 8}
        }
    
    def export_step_logs(self, step_logs: List[StepLog], 
                         env_states: List[EnvironmentState], 
                         output_path: str) -> str:
        """
        导出StepLog为v4.1兼容的TXT格式
        
        Args:
            step_logs: 步骤日志列表
            env_states: 对应的环境状态列表
            output_path: 输出文件路径
            
        Returns:
            导出的文件路径
        """
        if len(step_logs) != len(env_states):
            raise ValueError("StepLogs和EnvironmentStates数量不匹配")
        
        # 按月份分组
        monthly_data = self._group_by_month(step_logs, env_states)
        
        # 生成输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 导出每月数据
        exported_files = []
        for month, (month_logs, month_states) in monthly_data.items():
            if not month_logs:
                continue
            
            # 生成月度输出
            month_output = self._generate_monthly_output(month_logs, month_states)
            
            # 保存到文件
            month_file = output_path.replace(".txt", f"_month_{month:02d}.txt")
            with open(month_file, 'w', encoding='utf-8') as f:
                f.write(month_output)
            
            exported_files.append(month_file)
            print(f"Exported month {month}: {len(month_logs)} actions -> {month_file}")
        
        return exported_files
    
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
    
    def _generate_monthly_output(self, step_logs: List[StepLog], 
                                env_states: List[EnvironmentState]) -> str:
        """生成月度输出"""
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
        """从环境状态获取坐标"""
        coordinates = []
        
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
        # 简化实现：从环境状态的槽位中查找
        # 这里需要根据实际的槽位数据结构来实现
        
        # 假设槽位数据格式：{"id": "slot_1", "x": 100, "y": 50, "angle": 45}
        for slot in env_state.slots:
            if hasattr(slot, 'id') and slot.id:
                # 这里需要建立动作ID到槽位的映射关系
                # 简化实现：使用槽位索引作为动作ID
                if action_id < len(env_state.slots):
                    slot = env_state.slots[action_id]
                    return (slot.get('x', 0.0), slot.get('y', 0.0), slot.get('angle', 0.0))
        
        return None
    
    def _format_v4_line(self, step_log: StepLog, 
                        coordinates: List[Tuple[float, float, float]]) -> str:
        """格式化为v4.1格式"""
        if not step_log.chosen or not coordinates:
            return ""
        
        # 获取动作参数
        action_params = self.action_params.get(str(step_log.chosen[0]), {})
        desc = action_params.get("desc", f"ACTION_{step_log.chosen[0]}")
        
        # 解析动作描述获取智能体和尺寸
        agent, size = self._parse_action_desc(desc)
        
        # 获取v4.1格式的动作编号
        v4_action_id = self._get_v4_action_id(agent, size)
        
        # 生成v4.1格式输出
        parts = []
        for i, (action_id, (x, y, angle)) in enumerate(zip(step_log.chosen, coordinates)):
            # v4.1格式：a(x,y,z)angle
            part = f"{v4_action_id}({x:.1f},{y:.1f},0){angle:.1f}"
            parts.append(part)
        
        return ', '.join(parts)
    
    def _parse_action_desc(self, desc: str) -> Tuple[str, str]:
        """解析动作描述获取智能体和尺寸"""
        # 例如：EDU_S -> (EDU, S)
        if '_' in desc:
            parts = desc.split('_')
            if len(parts) >= 2:
                agent = parts[0]
                size = parts[1]
                return agent, size
        
        # 默认值
        return "EDU", "S"
    
    def _get_v4_action_id(self, agent: str, size: str) -> int:
        """获取v4.1格式的动作编号"""
        agent_mapping = self.agent_size_mapping.get(agent, {})
        return agent_mapping.get(size, 0)
    
    def export_simple_format(self, step_logs: List[StepLog], output_path: str) -> str:
        """
        导出简化格式（v5.0原生格式）
        
        Args:
            step_logs: 步骤日志列表
            output_path: 输出文件路径
            
        Returns:
            导出的文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for log in step_logs:
                # 简化格式：t,agent,action_ids
                line = f"{log.t},{log.agent},{','.join(map(str, log.chosen))}"
                f.write(line + '\n')
        
        print(f"Exported simple format: {len(step_logs)} logs -> {output_path}")
        return output_path
    
    def validate_export(self, step_logs: List[StepLog], 
                       exported_data: str) -> bool:
        """验证导出数据的一致性"""
        try:
            # 重新解析导出数据
            parsed_logs = self._parse_exported_data(exported_data)
            
            # 对比关键字段
            if len(parsed_logs) != len(step_logs):
                return False
            
            for original, parsed in zip(step_logs, parsed_logs):
                if original.t != parsed.t:
                    return False
                if original.agent != parsed.agent:
                    return False
                if original.chosen != parsed.chosen:
                    return False
            
            return True
        except Exception:
            return False
    
    def _parse_exported_data(self, exported_data: str) -> List[StepLog]:
        """解析导出的数据（简化实现）"""
        # 这里需要根据实际的导出格式来实现解析
        # 简化实现：返回空列表
        return []
