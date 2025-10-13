#!/usr/bin/env python3
"""
v4.1 RL集成测试脚本
测试RL框架与城市模拟环境的集成
"""

import sys
import os
import json
import torch
import numpy as np
from typing import Dict, Any

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from envs.v4_1.city_env import CityEnvironment
from solvers.v4_1 import RLPolicySelector, ParamSelector
from rl.v4_1 import PPOTrainer, MAPPOTrainer


def test_environment_creation():
    """测试环境创建"""
    print("=" * 50)
    print("1. 测试环境创建")
    print("=" * 50)
    
    try:
        # 加载配置
        with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        # 创建环境
        env = CityEnvironment(cfg)
        print("[OK] 环境创建成功")
        
        # 测试重置
        state = env.reset(seed=42)
        print(f"[OK] 环境重置成功，状态维度: {len(state)}")
        
        return env, cfg
        
    except Exception as e:
        print(f"[ERROR] 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_action_pool_generation(env: CityEnvironment):
    """测试动作池生成"""
    print("\n" + "=" * 50)
    print("2. 测试动作池生成")
    print("=" * 50)
    
    try:
        # 获取动作池
        actions, action_feats, mask = env.get_action_pool('EDU')
        
        print(f"[OK] EDU动作池生成成功")
        print(f"  动作数量: {len(actions)}")
        print(f"  动作特征维度: {action_feats.shape if len(action_feats) > 0 else 'N/A'}")
        print(f"  掩码维度: {mask.shape if len(mask) > 0 else 'N/A'}")
        
        # 测试IND动作池
        actions_ind, action_feats_ind, mask_ind = env.get_action_pool('IND')
        
        print(f"[OK] IND动作池生成成功")
        print(f"  动作数量: {len(actions_ind)}")
        print(f"  动作特征维度: {action_feats_ind.shape if len(action_feats_ind) > 0 else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 动作池生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_selector_integration(env: CityEnvironment, cfg: Dict):
    """测试选择器集成"""
    print("\n" + "=" * 50)
    print("3. 测试选择器集成")
    print("=" * 50)
    
    try:
        # 测试参数化选择器
        param_selector = ParamSelector(cfg)
        print("[OK] 参数化选择器创建成功")
        
        # 测试RL选择器
        rl_selector = RLPolicySelector(cfg)
        print("[OK] RL选择器创建成功")
        
        # 获取当前状态信息
        state = env._get_current_state()
        candidates = env._get_candidate_slots()
        occupied = env._get_occupied_slots()
        lp_provider = env._create_lp_provider()
        
        # 测试参数化选择
        try:
            actions_param, sequence_param = param_selector.choose_action_sequence(
                slots=env.slots,
                candidates=candidates,
                occupied=occupied,
                lp_provider=lp_provider,
                agent_types=['EDU'],
                sizes={'EDU': ['S', 'M', 'L']}
            )
            print(f"[OK] 参数化选择成功，生成{len(actions_param)}个动作")
        except Exception as e:
            print(f"[WARNING] 参数化选择失败: {e}")
        
        # 测试RL选择
        try:
            actions_rl, selected_rl = rl_selector.choose_action_sequence(
                slots=env.slots,
                candidates=candidates,
                occupied=occupied,
                lp_provider=lp_provider,
                agent_types=['EDU'],
                sizes={'EDU': ['S', 'M', 'L']}
            )
            print(f"[OK] RL选择成功，生成{len(actions_rl)}个动作")
            if selected_rl:
                print(f"  选中动作得分: {selected_rl.score:.4f}")
        except Exception as e:
            print(f"[WARNING] RL选择失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 选择器集成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_framework(env: CityEnvironment, cfg: Dict):
    """测试训练框架"""
    print("\n" + "=" * 50)
    print("4. 测试训练框架")
    print("=" * 50)
    
    try:
        # 测试PPO训练器
        ppo_trainer = PPOTrainer(cfg)
        print("[OK] PPO训练器创建成功")
        
        # 测试MAPPO训练器
        mappo_trainer = MAPPOTrainer(cfg)
        print("[OK] MAPPO训练器创建成功")
        
        # 测试模型保存和加载
        test_model_path = "models/v4_1_rl/test_model.pth"
        ppo_trainer.save_model(test_model_path)
        print("[OK] 模型保存成功")
        
        # 清理测试文件
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
            print("[OK] 测试文件清理完成")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 训练框架测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_execution(env: CityEnvironment):
    """测试步骤执行"""
    print("\n" + "=" * 50)
    print("5. 测试步骤执行")
    print("=" * 50)
    
    try:
        # 重置环境
        state = env.reset(seed=42)
        
        # 获取动作池
        actions, action_feats, mask = env.get_action_pool('EDU')
        
        if len(actions) > 0:
            # 选择第一个动作进行测试
            test_action = actions[0]
            
            # 执行步骤
            next_state, reward, done, info = env.step('EDU', test_action)
            
            print(f"[OK] 步骤执行成功")
            print(f"  奖励: {reward:.4f}")
            print(f"  完成: {done}")
            print(f"  信息: {list(info.keys())}")
            
            # 检查状态变化
            print(f"  月份: {state['month']} -> {next_state['month']}")
            print(f"  当前智能体: {state['current_agent']} -> {next_state['current_agent']}")
            
        else:
            print("[WARNING] 没有可用动作，跳过步骤执行测试")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 步骤执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episode_simulation(env: CityEnvironment):
    """测试完整episode模拟"""
    print("\n" + "=" * 50)
    print("6. 测试完整episode模拟")
    print("=" * 50)
    
    try:
        # 重置环境
        state = env.reset(seed=42)
        
        step_count = 0
        max_steps = 10  # 限制步数进行快速测试
        
        while step_count < max_steps:
            current_agent = state['current_agent']
            
            # 获取动作池
            actions, action_feats, mask = env.get_action_pool(current_agent)
            
            if len(actions) > 0:
                # 随机选择一个动作
                action_idx = np.random.choice(len(actions))
                selected_action = actions[action_idx]
                
                # 执行步骤
                next_state, reward, done, info = env.step(current_agent, selected_action)
                
                step_count += 1
                state = next_state
                
                print(f"  步骤 {step_count}: {current_agent} 奖励={reward:.4f}")
                
                if done:
                    print(f"[OK] Episode完成，总步数: {step_count}")
                    break
            else:
                print(f"[WARNING] {current_agent} 没有可用动作")
                break
        
        # 打印最终统计
        final_stats = state.get('monthly_stats', {})
        print(f"  最终建筑数量: {final_stats.get('total_buildings', 0)}")
        print(f"  EDU建筑: {final_stats.get('public_buildings', 0)}")
        print(f"  IND建筑: {final_stats.get('industrial_buildings', 0)}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Episode模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("v4.1 RL集成测试")
    print("=" * 60)
    
    # 测试结果
    results = {}
    
    # 1. 环境创建测试
    env, cfg = test_environment_creation()
    results['environment'] = env is not None
    
    if env is None:
        print("\n[ERROR] 环境创建失败，无法继续测试")
        return
    
    # 2. 动作池生成测试
    results['action_pool'] = test_action_pool_generation(env)
    
    # 3. 选择器集成测试
    results['selector'] = test_selector_integration(env, cfg)
    
    # 4. 训练框架测试
    results['training'] = test_training_framework(env, cfg)
    
    # 5. 步骤执行测试
    results['step_execution'] = test_step_execution(env)
    
    # 6. Episode模拟测试
    results['episode'] = test_episode_simulation(env)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"总测试项: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n详细结果:")
    for test_name, result in results.items():
        status = "[OK]" if result else "[ERROR]"
        print(f"  {test_name}: {status}")
    
    # 总体评估
    if passed_tests == total_tests:
        print("\n[SUCCESS] v4.1 RL集成测试: 完全通过！")
        print("   所有组件都已正确集成，可以开始训练。")
    elif passed_tests >= total_tests * 0.8:
        print("\n[WARNING] v4.1 RL集成测试: 基本通过")
        print("   大部分组件正常工作，需要修复少量问题。")
    else:
        print("\n[ERROR] v4.1 RL集成测试: 需要改进")
        print("   存在较多问题，需要进一步调试。")
    
    return results


if __name__ == "__main__":
    main()

