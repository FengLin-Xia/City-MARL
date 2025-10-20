"""
v5.0 回归测试

验证v5.0系统与v4.1的兼容性。
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import unittest
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from contracts import StepLog, EnvironmentState, ActionCandidate, Sequence
from config_loader import ConfigLoader
from envs.v5_0.city_env import V5CityEnvironment
from trainers.v5_0.ppo_trainer import V5PPOTrainer
from exporters.v5_0.export_system import V5ExportSystem


@dataclass
class RegressionTestResult:
    """回归测试结果"""
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None


class V5RegressionTester:
    """v5.0回归测试器"""
    
    def __init__(self, v4_config_path: str, v5_config_path: str):
        """
        初始化回归测试器
        
        Args:
            v4_config_path: v4.1配置文件路径
            v5_config_path: v5.0配置文件路径
        """
        self.v4_config_path = v4_config_path
        self.v5_config_path = v5_config_path
        
        # 加载配置
        self.v4_config = self._load_config(v4_config_path)
        self.v5_config = ConfigLoader().load_v5_config(v5_config_path)
        
        # 测试结果
        self.test_results = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load config {config_path}: {e}")
            return {}
    
    def run_all_tests(self) -> List[RegressionTestResult]:
        """运行所有回归测试"""
        print("Running v5.0 regression tests...")
        
        # 配置兼容性测试
        self.test_config_compatibility()
        
        # 动作映射测试
        self.test_action_mapping()
        
        # 环境行为测试
        self.test_environment_behavior()
        
        # 训练器兼容性测试
        self.test_trainer_compatibility()
        
        # 导出系统兼容性测试
        self.test_export_compatibility()
        
        # 性能测试
        self.test_performance()
        
        return self.test_results
    
    def test_config_compatibility(self):
        """测试配置兼容性"""
        test_name = "Config Compatibility"
        try:
            # 检查关键配置项
            v4_rl_config = self.v4_config.get('solver', {}).get('rl', {})
            v5_rl_config = self.v5_config.get('mappo', {})
            
            # 检查学习率
            v4_lr = v4_rl_config.get('lr', 3e-4)
            v5_lr = v5_rl_config.get('lr', 3e-4)
            assert abs(v4_lr - v5_lr) < 1e-6, f"Learning rate mismatch: {v4_lr} vs {v5_lr}"
            
            # 检查折扣因子
            v4_gamma = v4_rl_config.get('gamma', 0.99)
            v5_gamma = v5_rl_config.get('gamma', 0.99)
            assert abs(v4_gamma - v5_gamma) < 1e-6, f"Gamma mismatch: {v4_gamma} vs {v5_gamma}"
            
            # 检查智能体配置（处理名称差异）
            v4_agents = self.v4_config.get('solver', {}).get('rl', {}).get('agents', [])
            v5_agents = self.v5_config.get('agents', {}).get('order', [])
            
            # 标准化智能体名称（处理Council vs COUNCIL的差异）
            v4_agents_normalized = [agent.upper() if agent.lower() == 'council' else agent for agent in v4_agents]
            v5_agents_normalized = [agent.upper() if agent.lower() == 'council' else agent for agent in v5_agents]
            
            assert set(v4_agents_normalized) == set(v5_agents_normalized), f"Agent mismatch: {v4_agents} vs {v5_agents}"
            
            self.test_results.append(RegressionTestResult(test_name, True))
            print(f"[PASS] {test_name}")
            
        except Exception as e:
            self.test_results.append(RegressionTestResult(test_name, False, str(e)))
            print(f"[FAIL] {test_name}: {e}")
    
    def test_action_mapping(self):
        """测试动作映射兼容性"""
        test_name = "Action Mapping"
        try:
            # 检查v5.0动作参数
            v5_action_params = self.v5_config.get('action_params', {})
            
            # 验证动作ID范围
            action_ids = [int(k) for k in v5_action_params.keys()]
            assert min(action_ids) >= 0, "Action IDs must be non-negative"
            assert max(action_ids) <= 8, "Action IDs must be <= 8"
            
            # 验证动作描述格式
            for action_id, params in v5_action_params.items():
                desc = params.get('desc', '')
                assert '_' in desc, f"Action {action_id} desc format invalid: {desc}"
                
                # 验证成本、奖励、声望
                cost = params.get('cost', 0)
                reward = params.get('reward', 0)
                prestige = params.get('prestige', 0)
                
                assert cost >= 0, f"Action {action_id} cost must be non-negative"
                assert reward >= 0, f"Action {action_id} reward must be non-negative"
                assert isinstance(prestige, (int, float)), f"Action {action_id} prestige must be numeric"
            
            # 验证智能体动作ID分配
            agents_config = self.v5_config.get('agents', {}).get('defs', {})
            all_assigned_ids = []
            
            for agent, config in agents_config.items():
                action_ids = config.get('action_ids', [])
                all_assigned_ids.extend(action_ids)
                
                # 验证动作ID在有效范围内
                for action_id in action_ids:
                    assert 0 <= action_id <= 8, f"Agent {agent} has invalid action ID: {action_id}"
            
            # 验证没有重复的动作ID
            assert len(all_assigned_ids) == len(set(all_assigned_ids)), "Duplicate action IDs found"
            
            self.test_results.append(RegressionTestResult(test_name, True))
            print(f"[PASS] {test_name}")
            
        except Exception as e:
            self.test_results.append(RegressionTestResult(test_name, False, str(e)))
            print(f"[FAIL] {test_name}: {e}")
    
    def test_environment_behavior(self):
        """测试环境行为兼容性"""
        test_name = "Environment Behavior"
        try:
            # 创建v5.0环境
            env = V5CityEnvironment(self.v5_config_path)
            
            # 测试环境初始化
            state = env.reset()
            assert isinstance(state, EnvironmentState), "Environment should return EnvironmentState"
            assert state.month == 0, "Initial month should be 0"
            
            # 测试智能体调度
            active_agents = env.scheduler.get_active_agents(0)
            assert len(active_agents) > 0, "Should have active agents"
            
            # 测试动作候选生成
            for agent in active_agents:
                candidates = env.get_action_candidates(agent)
                assert isinstance(candidates, list), f"Should return list of candidates for {agent}"
                
                if candidates:
                    candidate = candidates[0]
                    assert isinstance(candidate, ActionCandidate), "Candidates should be ActionCandidate objects"
                    assert hasattr(candidate, 'id'), "ActionCandidate should have id"
                    assert hasattr(candidate, 'features'), "ActionCandidate should have features"
            
            # 测试动作执行
            if candidates:
                test_sequence = Sequence(agent=active_agents[0], actions=[candidates[0].id])
                next_state, reward, done, info = env.step(active_agents[0], test_sequence)
                
                assert isinstance(next_state, EnvironmentState), "Should return EnvironmentState"
                assert isinstance(reward, (int, float)), "Reward should be numeric"
                assert isinstance(done, bool), "Done should be boolean"
                assert isinstance(info, dict), "Info should be dictionary"
            
            self.test_results.append(RegressionTestResult(test_name, True))
            print(f"[PASS] {test_name}")
            
        except Exception as e:
            self.test_results.append(RegressionTestResult(test_name, False, str(e)))
            print(f"[FAIL] {test_name}: {e}")
    
    def test_trainer_compatibility(self):
        """测试训练器兼容性"""
        test_name = "Trainer Compatibility"
        try:
            # 创建v5.0训练器
            trainer = V5PPOTrainer(self.v5_config_path)
            
            # 测试训练器初始化
            assert hasattr(trainer, 'env'), "Trainer should have environment"
            assert hasattr(trainer, 'selector'), "Trainer should have selector"
            assert hasattr(trainer, 'optimizers'), "Trainer should have optimizers"
            
            # 测试经验收集
            experiences = trainer.collect_experience(5)  # 收集5步经验
            assert isinstance(experiences, list), "Should return list of experiences"
            
            if experiences:
                experience = experiences[0]
                required_fields = ['agent', 'state', 'candidates', 'sequence', 'reward', 'next_state', 'done', 'info']
                for field in required_fields:
                    assert field in experience, f"Experience should have {field} field"
            
            # 测试训练步骤
            if experiences:
                train_stats = trainer.train_step(experiences)
                assert isinstance(train_stats, dict), "Should return training statistics"
                
                required_stats = ['total_loss', 'actor_loss', 'critic_loss', 'entropy_loss', 'training_step']
                for stat in required_stats:
                    assert stat in train_stats, f"Training stats should have {stat}"
            
            self.test_results.append(RegressionTestResult(test_name, True))
            print(f"[PASS] {test_name}")
            
        except Exception as e:
            self.test_results.append(RegressionTestResult(test_name, False, str(e)))
            print(f"[FAIL] {test_name}: {e}")
    
    def test_export_compatibility(self):
        """测试导出系统兼容性"""
        test_name = "Export Compatibility"
        try:
            # 创建测试数据
            step_logs, env_states = self._create_test_data()
            
            # 测试导出系统
            export_system = V5ExportSystem(self.v5_config_path)
            
            # 测试TXT导出
            txt_file = export_system.export_txt_only(
                step_logs, env_states, "test_outputs/regression_test.txt"
            )
            assert os.path.exists(txt_file), "TXT file should be created"
            
            # 测试表格生成
            table_files = export_system.export_tables_only(
                step_logs, env_states, "test_outputs/regression_tables"
            )
            assert isinstance(table_files, list), "Should return list of table files"
            
            # 测试完整导出
            results = export_system.export_all(
                step_logs, env_states, "test_outputs/regression_complete"
            )
            assert 'txt_files' in results, "Results should have txt_files"
            assert 'table_files' in results, "Results should have table_files"
            assert 'summary_files' in results, "Results should have summary_files"
            
            # 测试导出摘要
            summary = export_system.get_export_summary(step_logs)
            assert 'total_steps' in summary, "Summary should have total_steps"
            assert 'agents' in summary, "Summary should have agents"
            assert 'months' in summary, "Summary should have months"
            
            self.test_results.append(RegressionTestResult(test_name, True))
            print(f"[PASS] {test_name}")
            
        except Exception as e:
            self.test_results.append(RegressionTestResult(test_name, False, str(e)))
            print(f"[FAIL] {test_name}: {e}")
    
    def test_performance(self):
        """测试性能"""
        test_name = "Performance"
        try:
            import time
            
            # 测试环境创建性能
            start_time = time.time()
            env = V5CityEnvironment(self.v5_config_path)
            env_creation_time = time.time() - start_time
            
            # 测试训练器创建性能
            start_time = time.time()
            trainer = V5PPOTrainer(self.v5_config_path)
            trainer_creation_time = time.time() - start_time
            
            # 测试经验收集性能
            start_time = time.time()
            experiences = trainer.collect_experience(20)
            experience_collection_time = time.time() - start_time
            
            # 测试导出性能
            step_logs, env_states = self._create_test_data()
            export_system = V5ExportSystem(self.v5_config_path)
            
            start_time = time.time()
            results = export_system.export_all(step_logs, env_states, "test_outputs/performance_test")
            export_time = time.time() - start_time
            
            # 性能指标
            performance_metrics = {
                'env_creation_time': env_creation_time,
                'trainer_creation_time': trainer_creation_time,
                'experience_collection_time': experience_collection_time,
                'export_time': export_time,
                'total_time': env_creation_time + trainer_creation_time + experience_collection_time + export_time
            }
            
            # 性能阈值检查
            assert env_creation_time < 5.0, f"Environment creation too slow: {env_creation_time:.2f}s"
            assert trainer_creation_time < 5.0, f"Trainer creation too slow: {trainer_creation_time:.2f}s"
            assert experience_collection_time < 10.0, f"Experience collection too slow: {experience_collection_time:.2f}s"
            assert export_time < 5.0, f"Export too slow: {export_time:.2f}s"
            
            self.test_results.append(RegressionTestResult(
                test_name, True, 
                performance_metrics=performance_metrics
            ))
            print(f"[PASS] {test_name}")
            print(f"  Performance metrics: {performance_metrics}")
            
        except Exception as e:
            self.test_results.append(RegressionTestResult(test_name, False, str(e)))
            print(f"[FAIL] {test_name}: {e}")
    
    def _create_test_data(self) -> Tuple[List[StepLog], List[EnvironmentState]]:
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
                land_prices=np.ones((200, 200), dtype=np.float32),
                buildings=[],
                budgets={'IND': 15000, 'EDU': 10000, 'COUNCIL': 0},
                slots=[
                    {"id": "slot_1", "x": 100, "y": 50, "angle": 45},
                    {"id": "slot_2", "x": 120, "y": 60, "angle": 90}
                ]
            ),
            EnvironmentState(
                month=0,
                land_prices=np.ones((200, 200), dtype=np.float32),
                buildings=[],
                budgets={'IND': 14100, 'EDU': 9350, 'COUNCIL': 0},
                slots=[
                    {"id": "slot_1", "x": 100, "y": 50, "angle": 45},
                    {"id": "slot_2", "x": 120, "y": 60, "angle": 90}
                ]
            )
        ]
        
        return step_logs, env_states
    
    def generate_report(self) -> str:
        """生成测试报告"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        report = f"""
# v5.0 回归测试报告

## 测试概览
- 总测试数: {total_tests}
- 通过测试: {passed_tests}
- 失败测试: {failed_tests}
- 通过率: {passed_tests/total_tests*100:.1f}%

## 详细结果
"""
        
        for result in self.test_results:
            status = "[PASS]" if result.passed else "[FAIL]"
            report += f"- {result.test_name}: {status}\n"
            
            if result.error_message:
                report += f"  Error: {result.error_message}\n"
            
            if result.performance_metrics:
                report += f"  Performance: {result.performance_metrics}\n"
        
        return report


def run_regression_tests():
    """运行回归测试"""
    print("Starting v5.0 regression tests...")
    
    # 创建测试器
    tester = V5RegressionTester(
        v4_config_path="configs/city_config_v4_1.json",
        v5_config_path="configs/city_config_v5_0.json"
    )
    
    # 运行测试
    results = tester.run_all_tests()
    
    # 生成报告
    report = tester.generate_report()
    print(report)
    
    # 保存报告
    os.makedirs("test_outputs", exist_ok=True)
    with open("test_outputs/regression_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    # 返回结果
    passed_tests = sum(1 for result in results if result.passed)
    total_tests = len(results)
    
    print(f"\\nRegression tests completed: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_regression_tests()
    if success:
        print("\\nAll regression tests passed! v5.0 is compatible with v4.1.")
    else:
        print("\\nSome regression tests failed. Please check the report.")
