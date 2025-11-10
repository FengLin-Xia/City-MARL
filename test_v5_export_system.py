"""
测试v5.0导出系统
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from contracts import StepLog, EnvironmentState
from exporters.v5_0.export_system import V5ExportSystem, create_export_system, export_v5_training_results


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
        ),
        StepLog(
            t=1, agent='EDU', chosen=[2],
            reward_terms={'revenue': 360, 'cost': -2700, 'prestige': 1.0},
            budget_snapshot={'IND': 14100, 'EDU': 6650, 'COUNCIL': 0}
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
                {"id": "slot_2", "x": 120, "y": 60, "angle": 90},
                {"id": "slot_3", "x": 140, "y": 70, "angle": 135}
            ]
        ),
        EnvironmentState(
            month=0,
            land_prices=None,
            buildings=[],
            budgets={'IND': 14100, 'EDU': 9350, 'COUNCIL': 0},
            slots=[
                {"id": "slot_1", "x": 100, "y": 50, "angle": 45},
                {"id": "slot_2", "x": 120, "y": 60, "angle": 90},
                {"id": "slot_3", "x": 140, "y": 70, "angle": 135}
            ]
        ),
        EnvironmentState(
            month=1,
            land_prices=None,
            buildings=[],
            budgets={'IND': 14100, 'EDU': 6650, 'COUNCIL': 0},
            slots=[
                {"id": "slot_1", "x": 100, "y": 50, "angle": 45},
                {"id": "slot_2", "x": 120, "y": 60, "angle": 90},
                {"id": "slot_3", "x": 140, "y": 70, "angle": 135}
            ]
        )
    ]
    
    return step_logs, env_states


def test_v5_export_system():
    """测试v5.0导出系统"""
    print("Testing v5.0 export system...")
    
    try:
        # 创建测试数据
        step_logs, env_states = create_test_data()
        print(f"Created test data: {len(step_logs)} step logs, {len(env_states)} env states")
        
        # 测试导出系统
        export_system = V5ExportSystem("configs/city_config_v5_0.json")
        print("Export system created successfully")
        
        # 测试TXT导出
        print("\\nTesting TXT export...")
        txt_file = export_system.export_txt_only(
            step_logs, env_states, "test_outputs/v5_test.txt"
        )
        print(f"TXT export: {txt_file}")
        
        # 测试表格生成
        print("\\nTesting table generation...")
        table_files = export_system.export_tables_only(
            step_logs, env_states, "test_outputs/tables"
        )
        print(f"Table files: {table_files}")
        
        # 测试完整导出
        print("\\nTesting complete export...")
        results = export_system.export_all(
            step_logs, env_states, "test_outputs/complete"
        )
        print(f"Export results: {results}")
        
        # 测试便捷函数
        print("\\nTesting convenience function...")
        convenience_results = export_v5_training_results(
            step_logs, env_states, 
            "configs/city_config_v5_0.json",
            "test_outputs/convenience"
        )
        print(f"Convenience results: {convenience_results}")
        
        # 测试导出摘要
        print("\\nTesting export summary...")
        summary = export_system.get_export_summary(step_logs)
        print(f"Export summary: {summary}")
        
        print("\\nExport system test passed!")
        return True
        
    except Exception as e:
        print(f"Export system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_export_formats():
    """测试不同导出格式"""
    print("\\nTesting different export formats...")
    
    try:
        step_logs, env_states = create_test_data()
        
        # 测试v4.1兼容格式
        print("Testing v4.1 compatible format...")
        v4_system = create_export_system(
            "configs/city_config_v5_0.json",
            txt_format="v4",
            include_tables=True,
            include_summary=True
        )
        v4_results = v4_system.export_all(step_logs, env_states, "test_outputs/v4_format")
        print(f"V4 format results: {v4_results}")
        
        # 测试v5.0原生格式
        print("Testing v5.0 native format...")
        v5_system = create_export_system(
            "configs/city_config_v5_0.json",
            txt_format="v5",
            include_tables=True,
            include_summary=True
        )
        v5_results = v5_system.export_all(step_logs, env_states, "test_outputs/v5_format")
        print(f"V5 format results: {v5_results}")
        
        print("Export formats test passed!")
        return True
        
    except Exception as e:
        print(f"Export formats test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing v5.0 export system...")
    
    # 创建测试输出目录
    os.makedirs("test_outputs", exist_ok=True)
    
    # 运行测试
    success1 = test_v5_export_system()
    success2 = test_export_formats()
    
    if success1 and success2:
        print("\\nAll v5.0 export system tests passed!")
    else:
        print("\\nSome v5.0 export system tests failed!")

