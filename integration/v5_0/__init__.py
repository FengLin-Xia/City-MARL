"""
v5.0 集成系统模块

基于管道模式的统一集成系统。
"""

from .pipeline import V5Pipeline, V5StateManager, V5ErrorHandler, V5PerformanceMonitor
from .training_pipeline import V5TrainingPipeline, create_training_pipeline, run_training_session
from .export_pipeline import V5ExportPipeline, create_export_pipeline, run_export_session
from .integration_system import V5IntegrationSystem, create_integration_system, run_complete_session

__all__ = [
    'V5Pipeline',
    'V5StateManager', 
    'V5ErrorHandler',
    'V5PerformanceMonitor',
    'V5TrainingPipeline',
    'create_training_pipeline',
    'run_training_session',
    'V5ExportPipeline',
    'create_export_pipeline', 
    'run_export_session',
    'V5IntegrationSystem',
    'create_integration_system',
    'run_complete_session'
]
