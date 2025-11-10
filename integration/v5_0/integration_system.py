"""
v5.0 统一集成系统

集成训练管道和导出管道的统一接口。
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from contracts import StepLog, EnvironmentState
from .training_pipeline import V5TrainingPipeline, run_training_session
from .export_pipeline import V5ExportPipeline, run_export_session


class V5IntegrationSystem:
    """v5.0统一集成系统"""
    
    def __init__(self, config_path: str):
        """
        初始化集成系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        
        # 初始化管道
        self.training_pipeline = V5TrainingPipeline(config_path)
        self.export_pipeline = V5ExportPipeline(config_path)
        
        # 系统状态
        self.system_state = {
            "initialized": False,
            "training_completed": False,
            "export_completed": False,
            "current_phase": "initialization"
        }
    
    def run_complete_session(self, num_episodes: int, output_dir: str = "./outputs") -> Dict[str, Any]:
        """
        运行完整会话（训练 + 导出）
        
        Args:
            num_episodes: 训练轮数
            output_dir: 输出目录
            
        Returns:
            完整会话结果
        """
        print(f"[INTEGRATION] Starting complete session for {num_episodes} episodes")
        
        # 阶段1：训练
        print("[INTEGRATION] Phase 1: Training")
        training_result = self.training_pipeline.run_training(num_episodes, output_dir)
        
        if not training_result["success"]:
            print("[INTEGRATION] Training failed, stopping session")
            return {
                "success": False,
                "phase": "training",
                "error": "Training failed",
                "training_result": training_result
            }
        
        # 阶段2：导出
        print("[INTEGRATION] Phase 2: Export")
        step_logs = training_result.get("step_logs", [])
        env_states = training_result.get("env_states", [])
        
        if not step_logs or not env_states:
            print("[INTEGRATION] No data to export, skipping export phase")
            return {
                "success": True,
                "phase": "training_only",
                "training_result": training_result,
                "export_result": None
            }
        
        export_result = self.export_pipeline.run_export(step_logs, env_states, output_dir)
        
        # 更新系统状态
        self.system_state["training_completed"] = True
        self.system_state["export_completed"] = export_result["success"]
        self.system_state["current_phase"] = "completed"
        
        # 生成综合结果
        success = training_result["success"] and export_result["success"]
        
        result = {
            "success": success,
            "phase": "completed",
            "training_result": training_result,
            "export_result": export_result,
            "system_state": self.system_state.copy(),
            "summary": self._generate_session_summary(training_result, export_result)
        }
        
        if success:
            print("[INTEGRATION] Complete session finished successfully")
        else:
            print("[INTEGRATION] Complete session finished with errors")
        
        return result
    
    def run_training_only(self, num_episodes: int, output_dir: str = "./outputs") -> Dict[str, Any]:
        """
        仅运行训练
        
        Args:
            num_episodes: 训练轮数
            output_dir: 输出目录
            
        Returns:
            训练结果
        """
        print(f"[INTEGRATION] Running training only for {num_episodes} episodes")
        
        result = self.training_pipeline.run_training(num_episodes, output_dir)
        
        # 更新系统状态
        self.system_state["training_completed"] = result["success"]
        self.system_state["current_phase"] = "training_completed"
        
        return result
    
    def run_export_only(self, step_logs: List[StepLog], env_states: List[EnvironmentState], 
                       output_dir: str = "./outputs") -> Dict[str, Any]:
        """
        仅运行导出
        
        Args:
            step_logs: 步骤日志列表
            env_states: 环境状态列表
            output_dir: 输出目录
            
        Returns:
            导出结果
        """
        print(f"[INTEGRATION] Running export only for {len(step_logs)} step logs")
        
        result = self.export_pipeline.run_export(step_logs, env_states, output_dir)
        
        # 更新系统状态
        self.system_state["export_completed"] = result["success"]
        self.system_state["current_phase"] = "export_completed"
        
        return result
    
    def _generate_session_summary(self, training_result: Dict[str, Any], 
                                 export_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成会话摘要"""
        summary = {
            "training": {
                "success": training_result.get("success", False),
                "episodes": training_result.get("data", {}).get("current_episode", 0),
                "step_logs": len(training_result.get("step_logs", [])),
                "env_states": len(training_result.get("env_states", []))
            },
            "export": {
                "success": export_result.get("success", False),
                "files_created": len(export_result.get("export_results", {}).get("table_files", []))
            },
            "performance": {
                "training_time": training_result.get("pipeline_summary", {}).get("performance_summary", {}).get("total_time", 0),
                "export_time": export_result.get("pipeline_summary", {}).get("performance_summary", {}).get("total_time", 0)
            }
        }
        
        return summary
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "system_state": self.system_state.copy(),
            "training_pipeline_summary": self.training_pipeline.pipeline.get_pipeline_summary(),
            "export_pipeline_summary": self.export_pipeline.pipeline.get_pipeline_summary()
        }
    
    def reset_system(self):
        """重置系统"""
        self.system_state = {
            "initialized": False,
            "training_completed": False,
            "export_completed": False,
            "current_phase": "initialization"
        }
        
        # 重置管道
        self.training_pipeline = V5TrainingPipeline(self.config_path)
        self.export_pipeline = V5ExportPipeline(self.config_path)
        
        print("[INTEGRATION] System reset completed")


def create_integration_system(config_path: str) -> V5IntegrationSystem:
    """
    创建集成系统的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        集成系统实例
    """
    return V5IntegrationSystem(config_path)


def run_complete_session(config_path: str, num_episodes: int, 
                        output_dir: str = "./outputs") -> Dict[str, Any]:
    """
    运行完整会话的便捷函数
    
    Args:
        config_path: 配置文件路径
        num_episodes: 训练轮数
        output_dir: 输出目录
        
    Returns:
        完整会话结果
    """
    system = create_integration_system(config_path)
    return system.run_complete_session(num_episodes, output_dir)

