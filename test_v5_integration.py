"""
测试v5.0集成系统
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from contracts import StepLog, EnvironmentState
from integration.v5_0 import (
    V5IntegrationSystem, 
    create_integration_system, 
    run_complete_session,
    V5TrainingPipeline,
    V5ExportPipeline
)


def create_test_data():
    """创建测试数据"""
    # 创建测试StepLog
    step_logs = [
        StepLog(
            t=0, agent='EDU', chosen=[0, 1],
            reward_terms={'revenue': 160, 'cost': -650, 'prestige': 0.2},
            budget_snapshot={'IND': 15000, 'EDU': 9350, 'COUNCIL': 0}
        ),
        StepLog(
            t=0, agent='IND', chosen=[3, 4],
            reward_terms={'revenue': 150, 'cost': -900, 'prestige': 0.2},
            budget_snapshot={'IND': 14100, 'EDU': 9350, 'COUNCIL': 0}
        )
    ]
    
    # 创建测试环境状态
    env_states = [
        EnvironmentState(
            month=0,
            land_prices=None,
            buildings=[],
            budgets={'IND': 15000, 'EDU': 10000, 'COUNCIL': 0},
            slots=[
                {"id": "slot_1", "x": 100, "y": 50, "angle": 45},
                {"id": "slot_2", "x": 120, "y": 60, "angle": 90}
            ]
        ),
        EnvironmentState(
            month=0,
            land_prices=None,
            buildings=[],
            budgets={'IND': 14100, 'EDU': 9350, 'COUNCIL': 0},
            slots=[
                {"id": "slot_1", "x": 100, "y": 50, "angle": 45},
                {"id": "slot_2", "x": 120, "y": 60, "angle": 90}
            ]
        )
    ]
    
    return step_logs, env_states


def test_integration_system():
    """测试集成系统"""
    print("Testing v5.0 integration system...")
    
    try:
        # 创建集成系统
        system = create_integration_system("configs/city_config_v5_0.json")
        print("Integration system created successfully")
        
        # 测试系统状态
        status = system.get_system_status()
        print(f"System status: {status['system_state']}")
        
        # 测试完整会话
        print("\\nTesting complete session...")
        result = system.run_complete_session(2, "test_outputs/integration")
        print(f"Complete session result: success={result['success']}, phase={result['phase']}")
        
        if result['success']:
            summary = result.get('summary', {})
            print(f"Training: {summary.get('training', {})}")
            print(f"Export: {summary.get('export', {})}")
            print(f"Performance: {summary.get('performance', {})}")
        
        # 测试训练管道
        print("\\nTesting training pipeline...")
        training_result = system.run_training_only(1, "test_outputs/training_only")
        print(f"Training pipeline result: success={training_result['success']}")
        
        # 测试导出管道
        print("\\nTesting export pipeline...")
        step_logs, env_states = create_test_data()
        export_result = system.run_export_only(step_logs, env_states, "test_outputs/export_only")
        print(f"Export pipeline result: success={export_result['success']}")
        
        # 测试系统重置
        print("\\nTesting system reset...")
        system.reset_system()
        status_after_reset = system.get_system_status()
        print(f"System status after reset: {status_after_reset['system_state']}")
        
        print("\\nIntegration system test passed!")
        return True
        
    except Exception as e:
        print(f"Integration system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_pipelines():
    """测试独立管道"""
    print("\\nTesting individual pipelines...")
    
    try:
        # 测试训练管道
        print("Testing training pipeline...")
        training_pipeline = V5TrainingPipeline("configs/city_config_v5_0.json")
        training_result = training_pipeline.run_training(1, "test_outputs/training_pipeline")
        print(f"Training pipeline result: success={training_result['success']}")
        
        # 测试导出管道
        print("Testing export pipeline...")
        step_logs, env_states = create_test_data()
        export_pipeline = V5ExportPipeline("configs/city_config_v5_0.json")
        export_result = export_pipeline.run_export(step_logs, env_states, "test_outputs/export_pipeline")
        print(f"Export pipeline result: success={export_result['success']}")
        
        print("Individual pipelines test passed!")
        return True
        
    except Exception as e:
        print(f"Individual pipelines test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_functions():
    """测试便捷函数"""
    print("\\nTesting convenience functions...")
    
    try:
        # 测试完整会话便捷函数
        print("Testing run_complete_session...")
        result = run_complete_session("configs/city_config_v5_0.json", 1, "test_outputs/convenience")
        print(f"Complete session result: success={result['success']}")
        
        print("Convenience functions test passed!")
        return True
        
    except Exception as e:
        print(f"Convenience functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing v5.0 integration system...")
    
    # 创建测试输出目录
    os.makedirs("test_outputs", exist_ok=True)
    
    # 运行测试
    success1 = test_integration_system()
    success2 = test_individual_pipelines()
    success3 = test_convenience_functions()
    
    if success1 and success2 and success3:
        print("\\nAll v5.0 integration system tests passed!")
    else:
        print("\\nSome v5.0 integration system tests failed!")
