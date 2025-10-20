"""
v5.0 统一导出系统

集成TXT导出和表格生成的统一接口。
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from contracts import StepLog, EnvironmentState
from .txt_exporter import V5TXTExporter
from .table_generator import V5TableGenerator


@dataclass
class ExportConfig:
    """导出配置"""
    txt_format: str = "v4"  # v4, v5, json
    include_tables: bool = True
    include_summary: bool = True
    output_encoding: str = "utf-8"
    table_style: str = "modern"  # classic, modern, minimal


class V5ExportSystem:
    """v5.0 统一导出系统"""
    
    def __init__(self, config_path: str, export_config: Optional[ExportConfig] = None):
        """
        初始化导出系统
        
        Args:
            config_path: v5.0配置文件路径
            export_config: 导出配置
        """
        self.config_path = config_path
        self.export_config = export_config or ExportConfig()
        
        # 初始化导出器
        self.txt_exporter = V5TXTExporter(config_path)
        self.table_generator = V5TableGenerator(config_path)
    
    def export_all(self, step_logs: List[StepLog], 
                   env_states: List[EnvironmentState], 
                   output_dir: str) -> Dict[str, List[str]]:
        """
        导出所有格式
        
        Args:
            step_logs: 步骤日志列表
            env_states: 对应的环境状态列表
            output_dir: 输出目录
            
        Returns:
            导出结果字典
        """
        if len(step_logs) != len(env_states):
            raise ValueError("StepLogs和EnvironmentStates数量不匹配")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            "txt_files": [],
            "table_files": [],
            "summary_files": []
        }
        
        # 导出TXT格式
        if self.export_config.txt_format == "v4":
            # v4.1兼容格式
            txt_files = self.txt_exporter.export_step_logs(
                step_logs, env_states, 
                os.path.join(output_dir, "v4_compatible.txt")
            )
            results["txt_files"].extend(txt_files)
        
        elif self.export_config.txt_format == "v5":
            # v5.0原生格式
            txt_file = self.txt_exporter.export_simple_format(
                step_logs, 
                os.path.join(output_dir, "v5_native.txt")
            )
            results["txt_files"].append(txt_file)
        
        # 生成动作表格
        if self.export_config.include_tables:
            table_files = self.table_generator.generate_monthly_tables(
                step_logs, env_states, 
                os.path.join(output_dir, "tables")
            )
            results["table_files"].extend(table_files)
        
        # 生成汇总表格
        if self.export_config.include_summary:
            summary_file = self.table_generator.generate_summary_table(
                step_logs, env_states,
                os.path.join(output_dir, "summary.png")
            )
            results["summary_files"].append(summary_file)
        
        return results
    
    def export_txt_only(self, step_logs: List[StepLog], 
                        env_states: List[EnvironmentState], 
                        output_path: str) -> str:
        """仅导出TXT格式"""
        if self.export_config.txt_format == "v4":
            files = self.txt_exporter.export_step_logs(step_logs, env_states, output_path)
            return files[0] if files else output_path
        else:
            return self.txt_exporter.export_simple_format(step_logs, output_path)
    
    def export_tables_only(self, step_logs: List[StepLog], 
                           env_states: List[EnvironmentState], 
                           output_dir: str) -> List[str]:
        """仅导出表格"""
        return self.table_generator.generate_monthly_tables(
            step_logs, env_states, output_dir
        )
    
    def validate_export(self, step_logs: List[StepLog], 
                       exported_data: str) -> bool:
        """验证导出数据"""
        return self.txt_exporter.validate_export(step_logs, exported_data)
    
    def get_export_summary(self, step_logs: List[StepLog]) -> Dict[str, Any]:
        """获取导出摘要"""
        if not step_logs:
            return {}
        
        # 统计信息
        total_steps = len(step_logs)
        agents = list(set(log.agent for log in step_logs))
        months = list(set(log.t for log in step_logs))
        
        # 按智能体统计
        agent_stats = {}
        for log in step_logs:
            agent = log.agent
            if agent not in agent_stats:
                agent_stats[agent] = 0
            agent_stats[agent] += len(log.chosen)
        
        return {
            "total_steps": total_steps,
            "agents": agents,
            "months": months,
            "agent_stats": agent_stats,
            "export_config": self.export_config
        }


def create_export_system(config_path: str, 
                        txt_format: str = "v4",
                        include_tables: bool = True,
                        include_summary: bool = True) -> V5ExportSystem:
    """
    创建导出系统的便捷函数
    
    Args:
        config_path: 配置文件路径
        txt_format: TXT格式 (v4, v5, json)
        include_tables: 是否包含表格
        include_summary: 是否包含汇总
        
    Returns:
        导出系统实例
    """
    export_config = ExportConfig(
        txt_format=txt_format,
        include_tables=include_tables,
        include_summary=include_summary
    )
    
    return V5ExportSystem(config_path, export_config)


def export_v5_training_results(step_logs: List[StepLog], 
                              env_states: List[EnvironmentState],
                              config_path: str,
                              output_dir: str = "./outputs") -> Dict[str, List[str]]:
    """
    导出v5.0训练结果的便捷函数
    
    Args:
        step_logs: 步骤日志列表
        env_states: 环境状态列表
        config_path: 配置文件路径
        output_dir: 输出目录
        
    Returns:
        导出结果
    """
    export_system = create_export_system(
        config_path=config_path,
        txt_format="v4",  # 默认使用v4.1兼容格式
        include_tables=True,
        include_summary=True
    )
    
    return export_system.export_all(step_logs, env_states, output_dir)
