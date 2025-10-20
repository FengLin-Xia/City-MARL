"""
测试v5.0城市环境
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from envs.v5_0.city_env import V5CityEnvironment


def test_v5_city_env():
    """测试v5.0城市环境"""
    print("Testing v5.0 city environment...")
    
    try:
        # 创建环境
        env = V5CityEnvironment("configs/city_config_v5_0.json")
        print("Environment created successfully")
        
        # 重置环境
        state = env.reset()
        print(f"Initial state: month={state.month}, budgets={state.budgets}")
        
        # 测试智能体调度
        print("\\nTesting agent scheduling...")
        for step in range(10):
            active_agents = env.scheduler.get_active_agents(step)
            mode = env.scheduler.get_execution_mode(step)
            print(f"Step {step}: agents={active_agents}, mode={mode}")
        
        # 测试动作枚举
        print("\\nTesting action enumeration...")
        for agent in ["EDU", "IND", "COUNCIL"]:
            candidates = env.get_action_candidates(agent)
            print(f"{agent} candidates: {len(candidates)}")
            
            if candidates:
                # 显示前3个候选
                for i, candidate in enumerate(candidates[:3]):
                    print(f"  Candidate {i}: id={candidate.id}, features_shape={candidate.features.shape}")
        
        # 测试动作执行
        print("\\nTesting action execution...")
        from contracts import Sequence
        
        # 创建测试序列
        test_sequence = Sequence(agent="EDU", actions=[0])  # EDU_S
        
        # 执行一步
        next_state, reward, done, info = env.step("EDU", test_sequence)
        print(f"Step result: reward={reward}, done={done}")
        print(f"Next state: month={next_state.month}, budgets={next_state.budgets}")
        
        # 测试观察
        print("\\nTesting observations...")
        for agent in ["EDU", "IND", "COUNCIL"]:
            obs = env.get_observation(agent)
            print(f"{agent} observation shape: {obs.shape}")
        
        # 测试统计信息
        print("\\nTesting statistics...")
        stats = env.get_statistics()
        print(f"Statistics: {stats}")
        
        # 测试步骤日志
        print("\\nTesting step logs...")
        step_logs = env.get_step_logs()
        print(f"Step logs: {len(step_logs)}")
        if step_logs:
            print(f"Last step log: {step_logs[-1]}")
        
        print("\\nCity environment test passed!")
        return True
        
    except Exception as e:
        print(f"City environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing v5.0 city environment...")
    
    success = test_v5_city_env()
    
    if success:
        print("\\nAll v5.0 city environment tests passed!")
    else:
        print("\\nSome v5.0 city environment tests failed!")
