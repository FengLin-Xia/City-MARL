"""
v5.0 管道模式核心实现

支持管道模式、混合式状态管理、可配置错误处理、可扩展性能监控。
"""

import os
import sys
import time
import psutil
import traceback
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from contracts import StepLog, EnvironmentState
from config_loader import ConfigLoader


class ErrorPolicy(Enum):
    """错误处理策略"""
    STRICT = "strict"      # 严格模式：任何错误都停止
    SKIP = "skip"          # 跳过模式：错误后跳过当前步骤
    RETRY = "retry"        # 重试模式：错误后重试
    FALLBACK = "fallback"  # 降级模式：错误后使用备用方案


@dataclass
class StepInfo:
    """管道步骤信息"""
    name: str
    func: Callable
    error_policy: ErrorPolicy
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class PipelineResult:
    """管道执行结果"""
    success: bool
    data: Any
    error: Optional[Exception] = None
    step_results: List[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class V5StateManager:
    """v5.0状态管理器 - 混合式状态管理"""
    
    def __init__(self):
        # 集中式状态（关键状态）
        self.global_state = {
            "current_episode": 0,
            "total_steps": 0,
            "training_phase": "initialization",
            "last_update": time.time()
        }
        
        # 分布式状态（组件状态）
        self.component_states = {
            "environment": {},
            "trainer": {},
            "export_system": {},
            "scheduler": {}
        }
        
        # 状态历史
        self.state_history = []
    
    def get_global_state(self) -> Dict[str, Any]:
        """获取全局状态"""
        return self.global_state.copy()
    
    def update_global_state(self, key: str, value: Any):
        """更新全局状态"""
        self.global_state[key] = value
        self.global_state["last_update"] = time.time()
        self._save_state_snapshot()
    
    def get_component_state(self, component: str) -> Dict[str, Any]:
        """获取组件状态"""
        return self.component_states.get(component, {}).copy()
    
    def update_component_state(self, component: str, state: Dict[str, Any]):
        """更新组件状态"""
        self.component_states[component] = state
        self._save_state_snapshot()
    
    def _save_state_snapshot(self):
        """保存状态快照"""
        snapshot = {
            "timestamp": time.time(),
            "global_state": self.global_state.copy(),
            "component_states": {k: v.copy() for k, v in self.component_states.items()}
        }
        self.state_history.append(snapshot)
        
        # 保持历史记录在合理范围内
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-500:]
    
    def get_state_history(self, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取状态历史"""
        if component:
            return [snapshot for snapshot in self.state_history 
                   if component in snapshot["component_states"]]
        return self.state_history.copy()


class V5ErrorHandler:
    """v5.0错误处理器 - 可配置错误处理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_policies = config.get("error_handling", {})
        self.retry_counts = {}
        self.error_history = []
    
    def handle_error(self, error: Exception, context: Dict[str, Any], step_name: str) -> str:
        """处理错误"""
        error_type = type(error).__name__
        policy = self.error_policies.get(error_type, "strict")
        
        # 记录错误
        self.error_history.append({
            "timestamp": time.time(),
            "step_name": step_name,
            "error_type": error_type,
            "error_message": str(error),
            "context": context
        })
        
        if policy == "strict":
            return "stop"
        elif policy == "skip":
            print(f"[ERROR] Skipping step {step_name} due to {error_type}: {error}")
            return "skip"
        elif policy == "retry":
            return self._handle_retry(error, context, step_name)
        elif policy == "fallback":
            return self._handle_fallback(error, context, step_name)
        else:
            return "stop"
    
    def _handle_retry(self, error: Exception, context: Dict[str, Any], step_name: str) -> str:
        """处理重试"""
        max_retries = self.error_policies.get("max_retries", 3)
        current_retries = self.retry_counts.get(step_name, 0)
        
        if current_retries < max_retries:
            self.retry_counts[step_name] = current_retries + 1
            retry_delay = self.error_policies.get("retry_delay", 1.0)
            print(f"[RETRY] Retrying step {step_name} (attempt {current_retries + 1}/{max_retries}) after {retry_delay}s")
            time.sleep(retry_delay)
            return "retry"
        else:
            print(f"[ERROR] Max retries exceeded for step {step_name}")
            return "skip"
    
    def _handle_fallback(self, error: Exception, context: Dict[str, Any], step_name: str) -> str:
        """处理降级"""
        print(f"[FALLBACK] Using fallback for step {step_name} due to {type(error).__name__}")
        # 这里可以实现具体的降级逻辑
        return "fallback"
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_types = {}
        for error in self.error_history:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recent_errors": self.error_history[-10:]  # 最近10个错误
        }


class V5PerformanceMonitor:
    """v5.0性能监控器 - 可扩展性能监控"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = {
            "step_times": {},
            "memory_usage": {},
            "throughput": {},
            "error_counts": {}
        }
        
        # 性能阈值
        perf_config = config.get("performance", {})
        self.thresholds = perf_config.get("thresholds", {
            "max_step_time": 10.0,
            "max_memory_usage": 1024 * 1024 * 1024,  # 1GB
            "min_throughput": 10
        })
        
        # 优化配置
        self.optimization = perf_config.get("optimization", {
            "batch_size": 1000,
            "parallel_processing": True,
            "memory_cleanup": True
        })
        
        self.start_time = time.time()
    
    def record_step(self, step_name: str, duration: float, memory_usage: int = None):
        """记录步骤性能"""
        if step_name not in self.metrics["step_times"]:
            self.metrics["step_times"][step_name] = []
        
        self.metrics["step_times"][step_name].append(duration)
        
        if memory_usage is None:
            memory_usage = psutil.Process().memory_info().rss
        
        self.metrics["memory_usage"][step_name] = memory_usage
        
        # 性能警告
        if duration > self.thresholds["max_step_time"]:
            print(f"[PERF] Warning: Step {step_name} took {duration:.2f}s (threshold: {self.thresholds['max_step_time']}s)")
        
        if memory_usage > self.thresholds["max_memory_usage"]:
            print(f"[PERF] Warning: Step {step_name} used {memory_usage / 1024 / 1024:.2f}MB (threshold: {self.thresholds['max_memory_usage'] / 1024 / 1024:.2f}MB)")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        total_time = time.time() - self.start_time
        
        # 计算平均步骤时间
        avg_step_times = {}
        for step_name, times in self.metrics["step_times"].items():
            avg_step_times[step_name] = sum(times) / len(times) if times else 0
        
        # 计算吞吐量
        total_steps = sum(len(times) for times in self.metrics["step_times"].values())
        throughput = total_steps / total_time if total_time > 0 else 0
        
        return {
            "total_time": total_time,
            "total_steps": total_steps,
            "throughput": throughput,
            "average_step_times": avg_step_times,
            "memory_usage": self.metrics["memory_usage"],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        # 检查慢步骤
        for step_name, times in self.metrics["step_times"].items():
            if times:
                avg_time = sum(times) / len(times)
                if avg_time > self.thresholds["max_step_time"]:
                    recommendations.append(f"Consider optimizing step '{step_name}' (avg: {avg_time:.2f}s)")
        
        # 检查内存使用
        for step_name, memory in self.metrics["memory_usage"].items():
            if memory > self.thresholds["max_memory_usage"]:
                recommendations.append(f"Consider memory optimization for step '{step_name}' ({memory / 1024 / 1024:.2f}MB)")
        
        # 检查吞吐量
        total_time = time.time() - self.start_time
        total_steps = sum(len(times) for times in self.metrics["step_times"].values())
        if total_time > 0:
            throughput = total_steps / total_time
            if throughput < self.thresholds["min_throughput"]:
                recommendations.append(f"Low throughput detected: {throughput:.2f} steps/s")
        
        return recommendations


class V5Pipeline:
    """v5.0管道 - 核心管道实现"""
    
    def __init__(self, config_path: str):
        """
        初始化管道
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = ConfigLoader().load_v5_config(config_path)
        
        # 初始化组件
        self.state_manager = V5StateManager()
        self.error_handler = V5ErrorHandler(self.config)
        self.performance_monitor = V5PerformanceMonitor(self.config)
        
        # 管道步骤
        self.steps: List[StepInfo] = []
        
        # 执行历史
        self.execution_history = []
    
    def add_step(self, name: str, func: Callable, error_policy: str = "strict", 
                 max_retries: int = 3, retry_delay: float = 1.0):
        """
        添加管道步骤
        
        Args:
            name: 步骤名称
            func: 步骤函数
            error_policy: 错误处理策略
            max_retries: 最大重试次数
            retry_delay: 重试延迟
        """
        step_info = StepInfo(
            name=name,
            func=func,
            error_policy=ErrorPolicy(error_policy),
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        self.steps.append(step_info)
    
    def run(self, initial_data: Any) -> PipelineResult:
        """
        运行管道
        
        Args:
            initial_data: 初始数据
            
        Returns:
            管道执行结果
        """
        data = initial_data
        step_results = []
        start_time = time.time()
        
        print(f"[PIPELINE] Starting pipeline with {len(self.steps)} steps")
        
        for i, step_info in enumerate(self.steps):
            step_start_time = time.time()
            step_result = {
                "step_name": step_info.name,
                "step_index": i,
                "success": False,
                "duration": 0,
                "error": None,
                "retry_count": 0
            }
            
            try:
                # 执行步骤
                data = self._execute_step(step_info, data, step_result)
                step_result["success"] = True
                
                # 记录性能
                step_duration = time.time() - step_start_time
                memory_usage = psutil.Process().memory_info().rss
                self.performance_monitor.record_step(step_info.name, step_duration, memory_usage)
                
                step_result["duration"] = step_duration
                print(f"[PIPELINE] Step {i+1}/{len(self.steps)} '{step_info.name}' completed in {step_duration:.2f}s")
                
            except Exception as e:
                step_result["error"] = str(e)
                step_result["duration"] = time.time() - step_start_time
                
                # 错误处理
                error_context = {
                    "step_index": i,
                    "step_name": step_info.name,
                    "data_type": type(data).__name__
                }
                
                action = self.error_handler.handle_error(e, error_context, step_info.name)
                
                if action == "stop":
                    print(f"[PIPELINE] Stopping pipeline due to error in step '{step_info.name}'")
                    break
                elif action == "skip":
                    print(f"[PIPELINE] Skipping step '{step_info.name}' due to error")
                    continue
                elif action == "retry":
                    # 重试逻辑
                    retry_success = self._retry_step(step_info, data, step_result)
                    if not retry_success:
                        print(f"[PIPELINE] Retry failed for step '{step_info.name}', skipping")
                        continue
                elif action == "fallback":
                    # 降级逻辑
                    data = self._fallback_step(step_info, data, step_result)
                    step_result["success"] = True
            
            step_results.append(step_result)
        
        # 记录执行历史
        total_duration = time.time() - start_time
        self.execution_history.append({
            "timestamp": start_time,
            "duration": total_duration,
            "steps_executed": len(step_results),
            "successful_steps": sum(1 for r in step_results if r["success"]),
            "step_results": step_results
        })
        
        # 生成结果
        success = all(r["success"] for r in step_results)
        performance_metrics = self.performance_monitor.get_performance_report()
        
        result = PipelineResult(
            success=success,
            data=data,
            step_results=step_results,
            performance_metrics=performance_metrics
        )
        
        print(f"[PIPELINE] Pipeline completed in {total_duration:.2f}s (success: {success})")
        return result
    
    def _execute_step(self, step_info: StepInfo, data: Any, step_result: Dict[str, Any]) -> Any:
        """执行单个步骤"""
        return step_info.func(data)
    
    def _retry_step(self, step_info: StepInfo, data: Any, step_result: Dict[str, Any]) -> bool:
        """重试步骤"""
        for attempt in range(step_info.max_retries):
            try:
                step_result["retry_count"] = attempt + 1
                data = step_info.func(data)
                step_result["success"] = True
                return True
            except Exception as e:
                if attempt < step_info.max_retries - 1:
                    time.sleep(step_info.retry_delay)
                    continue
                else:
                    step_result["error"] = str(e)
                    return False
        return False
    
    def _fallback_step(self, step_info: StepInfo, data: Any, step_result: Dict[str, Any]) -> Any:
        """降级步骤"""
        # 这里可以实现具体的降级逻辑
        # 简化实现：返回原始数据
        print(f"[FALLBACK] Using fallback for step '{step_info.name}'")
        return data
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """获取管道摘要"""
        return {
            "total_steps": len(self.steps),
            "execution_history": self.execution_history,
            "state_summary": self.state_manager.get_global_state(),
            "error_summary": self.error_handler.get_error_summary(),
            "performance_summary": self.performance_monitor.get_performance_report()
        }
