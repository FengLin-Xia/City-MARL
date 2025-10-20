"""
测试v5.0配置加载和验证
"""

import json
from config_loader import ConfigLoader


def test_config_loading():
    """测试配置加载"""
    loader = ConfigLoader()
    
    try:
        # 加载v5.0配置
        config = loader.load_v5_config("configs/city_config_v5_0.json")
        print("Config loading successful")
        
        # 验证关键配置
        print(f"Schema版本: {config.get('schema_version')}")
        print(f"场景名称: {config.get('scenario_name')}")
        
        # 验证智能体配置
        agents = config.get("agents", {})
        print(f"智能体顺序: {agents.get('order')}")
        
        # 验证动作参数
        action_params = config.get("action_params", {})
        print(f"动作参数数量: {len(action_params)}")
        
        # 验证调度器配置
        scheduler = config.get("scheduler", {})
        print(f"调度器类型: {scheduler.get('name')}")
        
        # 验证路径解析
        paths = config.get("paths", {})
        print(f"路径配置: {paths}")
        
        # 测试动作参数获取
        for action_id in range(9):  # 0-8
            params = loader.get_action_params(action_id)
            if params:
                print(f"动作{action_id}: {params.get('desc')}")
        
        # 测试智能体配置获取
        for agent in ["EDU", "IND", "COUNCIL"]:
            agent_config = loader.get_agent_config(agent)
            if agent_config:
                print(f"{agent}动作ID: {agent_config.get('action_ids')}")
        
        print("All tests passed")
        return True
        
    except Exception as e:
        print(f"Config loading failed: {e}")
        return False


def test_scheduler():
    """测试调度器"""
    from scheduler import PhaseCycleScheduler
    
    try:
        loader = ConfigLoader()
        config = loader.load_v5_config("configs/city_config_v5_0.json")
        
        scheduler_config = loader.get_scheduler_config()
        scheduler = PhaseCycleScheduler(scheduler_config["params"])
        
        # 测试不同步骤的调度
        for step in range(10):
            active_agents = scheduler.get_active_agents(step)
            mode = scheduler.get_execution_mode(step)
            phase_info = scheduler.get_phase_info(step)
            
            print(f"步骤{step}: 活跃智能体={active_agents}, 模式={mode}")
            print(f"  阶段信息: {phase_info}")
        
        print("Scheduler test passed")
        return True
        
    except Exception as e:
        print(f"Scheduler test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing v5.0 config system...")
    
    # 测试配置加载
    config_ok = test_config_loading()
    
    # 测试调度器
    scheduler_ok = test_scheduler()
    
    if config_ok and scheduler_ok:
        print("All tests passed! v5.0 config system working correctly")
    else:
        print("Some tests failed, need to fix")
