"""
v5.0 导出管道实现

基于管道模式的导出流程。
"""

import os
import sys
import json
import re
import ast
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from contracts import StepLog, EnvironmentState
from exporters.v5_0.export_system import V5ExportSystem
from .pipeline import V5Pipeline


class V5ExportPipeline:
    """v5.0导出管道"""
    
    def __init__(self, config_path: str):
        """
        初始化导出管道
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.pipeline = V5Pipeline(config_path)
        
        # 初始化组件
        self.export_system = None
        
        # 导出数据
        self.step_logs = []
        self.env_states = []
        
        # 设置管道步骤
        self._setup_pipeline_steps()
    
    def _setup_pipeline_steps(self):
        """设置管道步骤"""
        # 初始化步骤
        self.pipeline.add_step("initialize_export_system", self._initialize_export_system, "strict")
        
        # 数据加载步骤
        self.pipeline.add_step("load_training_data", self._load_training_data, "strict")
        self.pipeline.add_step("validate_data", self._validate_data, "strict")
        
        # 导出步骤
        self.pipeline.add_step("export_txt", self._export_txt, "retry", max_retries=2)
        self.pipeline.add_step("export_tables", self._export_tables, "retry", max_retries=2)
        self.pipeline.add_step("export_summary", self._export_summary, "skip")
        self.pipeline.add_step("export_monthly_png", self._export_monthly_png, "skip")
        
        # 验证步骤
        self.pipeline.add_step("validate_export", self._validate_export, "skip")
        
        # 清理步骤
        self.pipeline.add_step("cleanup", self._cleanup, "skip")
    
    def run_export(self, step_logs: List[StepLog], env_states: List[EnvironmentState], 
                   output_dir: str = "./outputs", input_data_path: str = None) -> Dict[str, Any]:
        """
        运行导出管道
        
        Args:
            step_logs: 步骤日志列表
            env_states: 环境状态列表
            output_dir: 输出目录
            
        Returns:
            导出结果
        """
        initial_data = {
            "step_logs": step_logs,
            "env_states": env_states,
            "output_dir": output_dir,
            "input_data_path": input_data_path,
            "export_results": {}
        }
        
        print(f"[EXPORT] Starting export pipeline for {len(step_logs)} step logs")
        
        # 运行管道
        result = self.pipeline.run(initial_data)
        
        if result.success:
            print(f"[EXPORT] Export completed successfully")
        else:
            print(f"[EXPORT] Export completed with errors")
        
        return {
            "success": result.success,
            "data": result.data,
            "export_results": result.data.get("export_results", {}),
            "pipeline_summary": self.pipeline.get_pipeline_summary()
        }
    
    def _initialize_export_system(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """初始化导出系统"""
        print("[EXPORT] Initializing export system...")
        
        # 初始化导出系统
        self.export_system = V5ExportSystem(self.config_path)
        
        # 更新状态
        self.pipeline.state_manager.update_global_state("export_phase", "initialization")
        self.pipeline.state_manager.update_component_state("export_system", {
            "initialized": True
        })
        
        print("  - Export system initialized")
        
        return data
    
    def _load_training_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """加载训练数据"""
        print("[EXPORT] Loading training data...")
        
        step_logs = data.get("step_logs", [])
        env_states = data.get("env_states", [])
        
        # 如果没有提供数据，尝试从JSON文件加载
        if not step_logs or not env_states:
            input_data_path = data.get("input_data_path")
            if input_data_path and os.path.exists(input_data_path):
                print(f"  - Loading data from JSON file: {input_data_path}")
                loaded_data = self._load_from_json_file(input_data_path)
                step_logs = loaded_data.get("step_logs", [])
                env_states = loaded_data.get("env_states", [])
                data["step_logs"] = step_logs
                data["env_states"] = env_states
        
        # 验证数据
        if not step_logs:
            raise ValueError("No step logs provided")
        
        if not env_states:
            raise ValueError("No environment states provided")
        
        if len(step_logs) != len(env_states):
            raise ValueError(f"Step logs ({len(step_logs)}) and env states ({len(env_states)}) count mismatch")
        
        # 更新状态
        self.pipeline.state_manager.update_global_state("export_phase", "data_loaded")
        self.pipeline.state_manager.update_component_state("export_system", {
            "data_loaded": True,
            "step_logs_count": len(step_logs),
            "env_states_count": len(env_states)
        })
        
        print(f"  - Loaded {len(step_logs)} step logs and {len(env_states)} env states")
        
        return data
    
    def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证数据"""
        print("[EXPORT] Validating data...")
        
        step_logs = data.get("step_logs", [])
        env_states = data.get("env_states", [])
        
        # 验证步骤日志
        for i, log in enumerate(step_logs):
            if not isinstance(log, StepLog):
                raise ValueError(f"Invalid StepLog at index {i}")
            
            if not hasattr(log, 'agent') or not hasattr(log, 'chosen'):
                raise ValueError(f"StepLog at index {i} missing required fields")
        
        # 验证环境状态
        for i, state in enumerate(env_states):
            if not isinstance(state, EnvironmentState):
                raise ValueError(f"Invalid EnvironmentState at index {i}")
        
        # 更新状态
        self.pipeline.state_manager.update_global_state("export_phase", "data_validated")
        
        print("  - Data validation completed")
        
        return data
    
    def _export_txt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """导出TXT格式"""
        print("[EXPORT] Exporting TXT format...")
        
        step_logs = data.get("step_logs", [])
        env_states = data.get("env_states", [])
        output_dir = data.get("output_dir", "./outputs")
        
        try:
            # 导出TXT
            txt_file = self.export_system.export_txt_only(step_logs, env_states, 
                                                         os.path.join(output_dir, "export.txt"))
            
            # 更新结果
            export_results = data.get("export_results", {})
            export_results["txt_file"] = txt_file
            data["export_results"] = export_results
            
            # 更新状态
            self.pipeline.state_manager.update_component_state("export_system", {
                "txt_exported": True,
                "txt_file": txt_file
            })
            
            print(f"  - TXT exported to: {txt_file}")
            
        except Exception as e:
            print(f"  - TXT export failed: {e}")
            raise
        
        return data
    
    def _export_tables(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """导出表格"""
        print("[EXPORT] Exporting tables...")
        
        step_logs = data.get("step_logs", [])
        env_states = data.get("env_states", [])
        output_dir = data.get("output_dir", "./outputs")
        
        try:
            # 导出表格
            table_files = self.export_system.export_tables_only(step_logs, env_states, 
                                                               os.path.join(output_dir, "tables"))
            
            # 更新结果
            export_results = data.get("export_results", {})
            export_results["table_files"] = table_files
            data["export_results"] = export_results
            
            # 更新状态
            self.pipeline.state_manager.update_component_state("export_system", {
                "tables_exported": True,
                "table_files": table_files
            })
            
            print(f"  - Tables exported: {len(table_files)} files")
            
        except Exception as e:
            print(f"  - Table export failed: {e}")
            raise
        
        return data
    
    def _export_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """导出汇总"""
        print("[EXPORT] Exporting summary...")
        
        step_logs = data.get("step_logs", [])
        env_states = data.get("env_states", [])
        output_dir = data.get("output_dir", "./outputs")
        
        try:
            # 导出汇总
            summary_file = self.export_system.export_system.generate_summary_table(
                step_logs, env_states, os.path.join(output_dir, "summary.png"))
            
            # 更新结果
            export_results = data.get("export_results", {})
            export_results["summary_file"] = summary_file
            data["export_results"] = export_results
            
            # 更新状态
            self.pipeline.state_manager.update_component_state("export_system", {
                "summary_exported": True,
                "summary_file": summary_file
            })
            
            print(f"  - Summary exported to: {summary_file}")
            
        except Exception as e:
            print(f"  - Summary export failed: {e}")
            # 汇总导出失败不影响整体流程
        
        return data
    
    def _validate_export(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证导出结果"""
        print("[EXPORT] Validating export results...")
        
        export_results = data.get("export_results", {})
        
        # 检查TXT文件
        txt_file = export_results.get("txt_file")
        if txt_file and os.path.exists(txt_file):
            file_size = os.path.getsize(txt_file)
            print(f"  - TXT file size: {file_size} bytes")
        
        # 检查表格文件
        table_files = export_results.get("table_files", [])
        if table_files:
            existing_files = [f for f in table_files if os.path.exists(f)]
            print(f"  - Table files: {len(existing_files)}/{len(table_files)} exist")
        
        # 更新状态
        self.pipeline.state_manager.update_global_state("export_phase", "validation_completed")
        
        print("  - Export validation completed")
        
        return data
    
    def _cleanup(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """清理资源"""
        print("[EXPORT] Cleaning up...")
        
        # 清理组件状态
        self.pipeline.state_manager.update_component_state("export_system", {})
        
        # 更新状态
        self.pipeline.state_manager.update_global_state("export_phase", "completed")
        
        print("  - Cleanup completed")
        
        return data
    
    def _export_monthly_png(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """导出月度汇总PNG"""
        print("[EXPORT] Exporting monthly summary PNG...")
        
        step_logs = data.get("step_logs", [])
        env_states = data.get("env_states", [])
        output_dir = data.get("output_dir", "./outputs")
        
        try:
            # 导出月度汇总PNG
            monthly_png_files = self.export_system.monthly_png_exporter.export_monthly_summary_png(
                step_logs, env_states, 
                os.path.join(output_dir, "monthly_summary")
            )
            
            # 更新结果
            export_results = data.get("export_results", {})
            export_results["monthly_png_files"] = monthly_png_files
            data["export_results"] = export_results
            
            # 更新状态
            self.pipeline.state_manager.update_component_state("export_system", {
                "monthly_png_exported": True,
                "monthly_png_files": monthly_png_files
            })
            
            print(f"  - Monthly PNG exported: {len(monthly_png_files)} files")
            
        except Exception as e:
            print(f"  - Monthly PNG export failed: {e}")
            raise
        
        return data
    
    def _load_from_json_file(self, json_file_path: str) -> Dict[str, Any]:
        """从JSON文件加载训练数据"""
        import json
        import re
        import ast
        import numpy as np
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        raw_step_logs = results.get('training_result', {}).get('step_logs', [])
        raw_env_states = results.get('training_result', {}).get('env_states', [])
        
        # 转换StepLog数据
        step_logs = []
        for data in raw_step_logs:
            try:
                if isinstance(data, str):
                    parsed_data = self._parse_step_log_string(data)
                    if parsed_data:
                        step_log = StepLog(**parsed_data)
                        step_logs.append(step_log)
                elif isinstance(data, dict):
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
        
        # 转换EnvironmentState数据
        env_states = []
        for data in raw_env_states:
            try:
                if isinstance(data, str):
                    parsed_data = self._parse_env_state_string(data)
                    if parsed_data:
                        env_state = EnvironmentState(**parsed_data)
                        env_states.append(env_state)
                elif isinstance(data, dict):
                    env_state = EnvironmentState(
                        month=data.get('month', 0),
                        land_prices=np.array(data.get('land_prices', [])),
                        buildings=data.get('buildings', []),
                        budgets=data.get('budgets', {}),
                        slots=data.get('slots', [])
                    )
                    env_states.append(env_state)
            except Exception as e:
                print(f"Error converting EnvironmentState: {e}")
                continue
        
        return {
            "step_logs": step_logs,
            "env_states": env_states
        }
    
    def _parse_step_log_string(self, step_log_str: str) -> Optional[Dict]:
        """解析字符串格式的StepLog"""
        try:
            match = re.match(r"StepLog\((.*)\)", step_log_str.strip())
            if not match:
                return None
            
            content = match.group(1)
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
                parsed_data['chosen'] = [int(x.strip()) for x in chosen_str.split(',')] if chosen_str.strip() else []
            
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
    
    def _parse_env_state_string(self, env_state_str: str) -> Optional[Dict]:
        """解析字符串格式的EnvironmentState"""
        try:
            match = re.match(r"EnvironmentState\((.*)\)", env_state_str.strip(), re.DOTALL)
            if not match:
                return None
            
            content = match.group(1)
            parsed_data = {}
            
            # 解析month=...
            month_match = re.search(r"month=(\d+)", content)
            if month_match:
                parsed_data['month'] = int(month_match.group(1))
            else:
                return None
            
            # 解析land_prices=... (简化处理)
            parsed_data['land_prices'] = np.array([[1.0]])
            
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
            
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing EnvironmentState string: {e}")
            return None


def create_export_pipeline(config_path: str) -> V5ExportPipeline:
    """
    创建导出管道的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        导出管道实例
    """
    return V5ExportPipeline(config_path)


def run_export_session(config_path: str, step_logs: List[StepLog], 
                      env_states: List[EnvironmentState], 
                      output_dir: str = "./outputs") -> Dict[str, Any]:
    """
    运行导出会话的便捷函数
    
    Args:
        config_path: 配置文件路径
        step_logs: 步骤日志列表
        env_states: 环境状态列表
        output_dir: 输出目录
        
    Returns:
        导出结果
    """
    pipeline = create_export_pipeline(config_path)
    return pipeline.run_export(step_logs, env_states, output_dir)

