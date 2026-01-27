"""
v5.0 训练管道实现

基于管道模式的训练流程。
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from contracts import StepLog, EnvironmentState
from envs.v5_0.city_env import V5CityEnvironment
from trainers.v5_0.ppo_trainer import V5PPOTrainer
from exporters.v5_0.export_system import V5ExportSystem
from .pipeline import V5Pipeline


class V5TrainingPipeline:
    """v5.0训练管道"""
    
    def __init__(self, config_path: str):
        """
        初始化训练管道
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.pipeline = V5Pipeline(config_path)
        
        # 初始化组件
        self.env = None
        self.trainer = None
        self.export_system = None
        
        # 训练数据（临时，用于每个episode）
        self.step_logs = []
        self.env_states = []
        
        # 最终保存的数据（累积所有episode）
        self.final_step_logs = []
        self.final_env_states = []
        
        # 限制保存的episode数量（避免内存溢出）
        # 如果训练episode数超过max_saved_episodes，只保存最后N个episode的数据
        self.max_saved_episodes = 50  # 默认保存最后50个episode
        
        # 设置管道步骤
        self._setup_pipeline_steps()
    
    def _setup_pipeline_steps(self):
        """设置管道步骤"""
        # 初始化步骤
        self.pipeline.add_step("initialize_components", self._initialize_components, "strict")
        self.pipeline.add_step("reset_environment", self._reset_environment, "strict")
        
        # 训练步骤
        self.pipeline.add_step("collect_experience", self._collect_experience, "retry", max_retries=3)
        self.pipeline.add_step("train_step", self._train_step, "retry", max_retries=2)
        self.pipeline.add_step("update_state", self._update_state, "strict")
        
        # 导出步骤
        self.pipeline.add_step("export_results", self._export_results, "skip")
        
        # 清理步骤
        self.pipeline.add_step("cleanup", self._cleanup, "skip")
    
    def run_training(self, num_episodes: int, output_dir: str = "./outputs") -> Dict[str, Any]:
        """
        运行训练管道（按 episode 循环）
        """
        data = {
            "num_episodes": num_episodes,
            "output_dir": output_dir,
            "current_episode": 0,
            "step_logs": [],
            "env_states": []
        }
        print(f"[TRAINING] Starting training pipeline for {num_episodes} episodes")
        
        # 根据episode数量决定是否保存所有数据
        # 如果episode数 <= max_saved_episodes，保存所有数据
        # 如果episode数 > max_saved_episodes，只保存最后N个episode的数据
        save_all_episodes = num_episodes <= self.max_saved_episodes
        keep_last_n = self.max_saved_episodes
        
        if not save_all_episodes:
            print(f"[TRAINING] Memory optimization: Only keeping last {keep_last_n} episodes' data to prevent OOM")
        
        last_result = None
        for ep in range(1, num_episodes + 1):
            data["current_episode"] = ep - 1  # 当前轮次从0开始计
            last_result = self.pipeline.run(data)
            # 下一轮继续沿用返回的数据
            data = last_result.data if last_result and last_result.data is not None else data
            
            # 决定是否保存当前episode的数据
            should_save_data = save_all_episodes or ep > num_episodes - keep_last_n
            
            # 保存数据到final列表（如果应该保存）
            if should_save_data:
                if ep < num_episodes:
                    current_count = len(self.step_logs)
                    print(f"[TRAINING] Episode {ep} completed, saving data ({current_count} step_logs)")
                else:
                    print(f"[TRAINING] Episode {ep} completed, saving final data")
                self.final_step_logs.extend(self.step_logs)
                self.final_env_states.extend(self.env_states)
            else:
                print(f"[TRAINING] Episode {ep} completed, discarding data to save memory (only keeping last {keep_last_n} episodes)")
            
            # 清空临时数据（避免内存泄漏）
            data["step_logs"] = []
            data["env_states"] = []
            self.step_logs = []
            self.env_states = []
        
        if last_result and last_result.success:
            print(f"[TRAINING] Training completed successfully")
        else:
            print(f"[TRAINING] Training completed with errors")
        
        return {
            "success": bool(last_result and last_result.success),
            "data": data,
            "step_logs": self.final_step_logs,  # 修复：返回所有episode的数据
            "env_states": self.final_env_states,  # 修复：返回所有episode的数据
            "pipeline_summary": self.pipeline.get_pipeline_summary()
        }
    
    def _initialize_components(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """初始化组件"""
        # 已初始化则跳过
        if self.env is not None and self.trainer is not None and self.export_system is not None:
            return data
        print("[TRAINING] Initializing components...")
        # 初始化环境
        self.env = V5CityEnvironment(self.config_path)
        print("  - Environment initialized")
        # 初始化训练器
        self.trainer = V5PPOTrainer(self.config_path)
        print("  - Trainer initialized")
        # 初始化导出系统
        self.export_system = V5ExportSystem(self.config_path)
        print("  - Export system initialized")
        
        # 更新状态
        self.pipeline.state_manager.update_global_state("training_phase", "initialization")
        
        return data
    
    def _reset_environment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """重置环境"""
        print("[TRAINING] Resetting environment...")
        
        # 重置环境
        initial_state = self.env.reset()
        
        # 更新状态
        self.pipeline.state_manager.update_global_state("training_phase", "environment_ready")
        self.pipeline.state_manager.update_component_state("environment", {
            "initialized": True,
            "current_state": initial_state
        })
        
        return data
    
    def _collect_experience(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """收集经验"""
        print("[TRAINING] Collecting experience...")
        try:
            # 收集经验
            # 从配置中获取total_steps
            config = self.pipeline.config
            total_steps = config.get('env', {}).get('time_model', {}).get('total_steps', 30)
            experiences = self.trainer.collect_experience(total_steps)  # 使用配置的total_steps
        except Exception as e:
            import traceback
            print("[TRAINING][collect_experience] Exception raised:\n" + traceback.format_exc())
            # 失败时返回原数据，不中断后续步骤（由管道错误策略处理）
            return data
        
        if experiences:
            # 提取步骤日志和环境状态
            step_logs = [exp.get('step_log') for exp in experiences if exp.get('step_log')]
            env_states = [exp.get('next_state') for exp in experiences if exp.get('next_state')]
            
            # 初始化数据字典
            if "step_logs" not in data:
                data["step_logs"] = []
            if "env_states" not in data:
                data["env_states"] = []
            if "experiences" not in data:
                data["experiences"] = []
            
            # 更新数据
            data["step_logs"].extend(step_logs)
            data["env_states"].extend(env_states)
            data["experiences"].extend(experiences)
            
            # 同时更新实例属性
            self.step_logs.extend(step_logs)
            self.env_states.extend(env_states)
            
            # 更新状态
            self.pipeline.state_manager.update_global_state("total_steps", 
                self.pipeline.state_manager.get_global_state().get("total_steps", 0) + len(experiences))
            
            print(f"  - Collected {len(experiences)} experiences")
        else:
            print("  - No experiences collected")
        
        return data
    
    def _train_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """训练步骤"""
        print("[TRAINING] Training step...")
        
        # 获取经验（直接使用采样阶段返回的结构化数据）
        experiences = data.get("experiences", [])
        
        if experiences:
            # 执行训练
            train_stats = self.trainer.train_step(experiences)
            
            # 更新状态
            self.pipeline.state_manager.update_component_state("trainer", {
                "training_step": train_stats.get("training_step", 0),
                "total_loss": train_stats.get("total_loss", 0),
                "actor_loss": train_stats.get("actor_loss", 0),
                "critic_loss": train_stats.get("critic_loss", 0)
            })
            
            print(f"  - Training completed: loss={train_stats.get('total_loss', 0):.4f}")
            # 训练后清空已消费的经验（避免重复训练）
            data["experiences"] = []
            # 修复：强制内存清理
            import gc
            gc.collect()
        else:
            print("  - No experiences to train on")
        
        return data
    
    def _update_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """更新状态"""
        print("[TRAINING] Updating state...")
        
        # 更新当前轮次
        current_episode = data.get("current_episode", 0) + 1
        data["current_episode"] = current_episode
        
        # 更新全局状态
        self.pipeline.state_manager.update_global_state("current_episode", current_episode)
        self.pipeline.state_manager.update_global_state("training_phase", "episode_completed")
        
        print(f"  - Episode {current_episode} completed")
        
        return data
    
    def _export_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """导出结果"""
        # 按配置控制是否导出
        export_cfg = self.pipeline.config.get("export", {"enabled": True, "every_n_episodes": 0})
        enabled = bool(export_cfg.get("enabled", True))
        every_n = int(export_cfg.get("every_n_episodes", 0))
        current_ep = int(data.get("current_episode", 0)) + 1
        should_export = enabled and (every_n == 0 or (current_ep % every_n == 0) or (current_ep == int(data.get("num_episodes", current_ep))))
        # 强制只在最后一个episode导出
        if every_n == 1:
            should_export = should_export and (current_ep == int(data.get("num_episodes", current_ep)))
        if not should_export:
            print("[TRAINING] Export skipped by config")
            return data
        print("[TRAINING] Exporting results...")
        
        step_logs = data.get("step_logs", [])
        env_states = data.get("env_states", [])
        output_dir = data.get("output_dir", "./outputs")
        
        if step_logs and env_states:
            try:
                # 直接传递对象，不进行字符串转换
                # 导出结果
                results = self.export_system.export_all(step_logs, env_states, output_dir)
                
                # 更新状态
                self.pipeline.state_manager.update_component_state("export_system", {
                    "exported": True,
                    "output_dir": output_dir,
                    "files_created": len(results.get("txt_files", [])) + len(results.get("table_files", []))
                })
                
                print(f"  - Exported {len(results.get('txt_files', []))} TXT files")
                print(f"  - Exported {len(results.get('table_files', []))} table files")
                
            except Exception as e:
                print(f"  - Export failed: {e}")
        else:
            print("  - No data to export")
        # 保留数据供集成系统使用，不清空
        # data["step_logs"] = []
        # data["env_states"] = []
        
        return data
    
    def _cleanup(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """清理资源"""
        print("[TRAINING] Cleaning up...")
        
        # 清理组件状态
        self.pipeline.state_manager.update_component_state("environment", {})
        self.pipeline.state_manager.update_component_state("trainer", {})
        self.pipeline.state_manager.update_component_state("export_system", {})
        
        # 更新状态
        self.pipeline.state_manager.update_global_state("training_phase", "completed")
        
        print("  - Cleanup completed")
        
        return data


def create_training_pipeline(config_path: str) -> V5TrainingPipeline:
    """
    创建训练管道的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        训练管道实例
    """
    return V5TrainingPipeline(config_path)


def run_training_session(config_path: str, num_episodes: int, output_dir: str = "./outputs") -> Dict[str, Any]:
    """
    运行训练会话的便捷函数
    
    Args:
        config_path: 配置文件路径
        num_episodes: 训练轮数
        output_dir: 输出目录
        
    Returns:
        训练结果
    """
    pipeline = create_training_pipeline(config_path)
    return pipeline.run_training(num_episodes, output_dir)












