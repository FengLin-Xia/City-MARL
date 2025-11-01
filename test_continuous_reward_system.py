"""
持续奖励系统单元测试
"""

import unittest
import numpy as np
from contracts import EnvironmentState, BuildingRegistry, BuildingInfo


class TestContinuousRewardSystem(unittest.TestCase):
    """测试ContinuousRewardSystem类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {
            "continuous_rewards": {
                "enabled": True,
                "gamma": 0.99,
                "action_reward_rates": {
                    "3": 100.0,
                    "4": 150.0,
                    "5": 200.0,
                    "9": 120.0,
                    "10": 180.0,
                    "11": 250.0
                }
            },
            "synergy_effects": {
                "industrial_cluster": {
                    "enabled": True,
                    "bonus_rate": 0.15,
                    "affects_actions": [3, 4, 5],
                    "trigger_actions": [9, 10, 11],
                    "global_effect": True
                }
            },
            "action_params": {
                "3": {"base_reward": 100.0},
                "4": {"base_reward": 150.0},
                "5": {"base_reward": 200.0},
                "9": {"base_reward": 120.0},
                "10": {"base_reward": 180.0},
                "11": {"base_reward": 250.0}
            }
        }
        
        # 创建测试环境状态
        self.state = EnvironmentState(
            month=1,
            land_prices=np.array([[1.0, 2.0], [3.0, 4.0]]),
            buildings=[],
            budgets={"IND": 1000.0, "EDU": 1000.0, "COUNCIL": 1000.0},
            slots=[],
            building_registry=BuildingRegistry()
        )
        
        # 导入ContinuousRewardSystem
        from logic.continuous_reward_system import ContinuousRewardSystem
        self.system = ContinuousRewardSystem(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(self.system.enabled)
        self.assertEqual(self.system.gamma, 0.99)
        self.assertEqual(self.system.action_reward_rates[3], 100.0)
        self.assertEqual(self.system.action_reward_rates[9], 120.0)
    
    def test_calculate_monthly_total_rewards_empty(self):
        """测试空建筑列表的月度总收益计算"""
        rewards = self.system.calculate_monthly_total_rewards(self.state)
        
        self.assertEqual(rewards["IND"], 0.0)
        self.assertEqual(rewards["EDU"], 0.0)
        self.assertEqual(rewards["COUNCIL"], 0.0)
    
    def test_calculate_monthly_total_rewards_with_buildings(self):
        """测试有建筑时的月度总收益计算"""
        # 添加建筑到注册表
        building1 = BuildingInfo(
            building_id="IND_3_1_0",
            action_id=3,
            agent="IND",
            month=1,
            position=(0.0, 0.0),
            cost=100.0,
            reward=100.0
        )
        
        building2 = BuildingInfo(
            building_id="IND_9_1_1",
            action_id=9,
            agent="IND",
            month=1,
            position=(1.0, 1.0),
            cost=200.0,
            reward=120.0
        )
        
        self.state.building_registry.buildings["IND_3_1_0"] = building1
        self.state.building_registry.buildings["IND_9_1_1"] = building2
        self.state.building_registry.buildings_by_agent["IND"] = ["IND_3_1_0", "IND_9_1_1"]
        self.state.building_registry.action_counts[3] = 1
        self.state.building_registry.action_counts[9] = 1
        
        rewards = self.system.calculate_monthly_total_rewards(self.state)
        
        # 动作3: 100.0 * 1.15 = 115.0 (有协同效应)
        # 动作9: 120.0 * 1.0 = 120.0 (无协同效应)
        # 总计: 115.0 + 120.0 = 235.0
        self.assertEqual(rewards["IND"], 235.0)
    
    def test_calculate_delta_reward(self):
        """测试增量奖励计算"""
        # 第一次计算
        delta1 = self.system.calculate_delta_reward("IND", 100.0)
        self.assertEqual(delta1, 100.0)  # 0 -> 100
        
        # 第二次计算
        delta2 = self.system.calculate_delta_reward("IND", 150.0)
        self.assertEqual(delta2, 50.0)  # 100 -> 150
        
        # 第三次计算（无变化）
        delta3 = self.system.calculate_delta_reward("IND", 150.0)
        self.assertEqual(delta3, 0.0)  # 150 -> 150
    
    def test_calculate_potential_shaping(self):
        """测试潜势塑形计算"""
        # 设置历史数据
        self.system.monthly_totals["IND"] = 100.0
        
        # 计算塑形奖励
        shaping = self.system.calculate_potential_shaping("IND", 150.0)
        
        # phi_s = 100 / (1 - 0.99) = 10000
        # phi_sp = 150 / (1 - 0.99) = 15000
        # shaping = 0.99 * 15000 - 10000 = 14850 - 10000 = 4850
        expected_shaping = 0.99 * 15000 - 10000
        self.assertAlmostEqual(shaping, expected_shaping, places=5)
    
    def test_calculate_synergy_multiplier(self):
        """测试协同效应倍数计算"""
        # 无协同效应
        multiplier1 = self.system.calculate_synergy_multiplier(3, self.state)
        self.assertEqual(multiplier1, 1.0)
        
        # 添加触发动作
        self.state.building_registry.action_counts[9] = 1
        
        # 有协同效应
        multiplier2 = self.system.calculate_synergy_multiplier(3, self.state)
        self.assertEqual(multiplier2, 1.15)  # 1.0 + 0.15
    
    def test_activate_synergy(self):
        """测试协同效应激活"""
        # 激活协同效应
        self.system.activate_synergy(9, self.state)
        
        # 检查状态
        self.assertTrue(self.state.synergy_activations.get("industrial_cluster", False))
    
    def test_disabled_system(self):
        """测试禁用系统"""
        disabled_config = self.config.copy()
        disabled_config["continuous_rewards"]["enabled"] = False
        
        from logic.continuous_reward_system import ContinuousRewardSystem
        disabled_system = ContinuousRewardSystem(disabled_config)
        
        # 所有计算都应该返回默认值
        rewards = disabled_system.calculate_monthly_total_rewards(self.state)
        self.assertEqual(rewards, {})
        
        delta = disabled_system.calculate_delta_reward("IND", 100.0)
        self.assertEqual(delta, 0.0)
        
        shaping = disabled_system.calculate_potential_shaping("IND", 100.0)
        self.assertEqual(shaping, 0.0)


if __name__ == "__main__":
    unittest.main()
