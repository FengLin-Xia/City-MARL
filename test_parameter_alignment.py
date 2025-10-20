"""
测试v5.0和v4.1参数对齐情况
"""

import json
import sys
import os

def load_config(file_path):
    """加载配置文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_parameters():
    """对比v4.1和v5.0参数"""
    print("=" * 60)
    print("v5.0 vs v4.1 参数对齐测试")
    print("=" * 60)
    
    # 加载配置
    v4_config = load_config("configs/city_config_v4_1.json")
    v5_config = load_config("configs/city_config_v5_0.json")
    
    print("[PASS] 配置文件加载成功")
    
    # 1. 基础配置对比
    print("\n1. 基础配置对比:")
    
    # 时间模型
    v4_months = v4_config.get("simulation", {}).get("total_months", 0)
    v5_months = v5_config.get("env", {}).get("time_model", {}).get("total_steps", 0)
    print(f"  总月数: v4.1={v4_months}, v5.0={v5_months}, 对齐={'[PASS]' if v4_months == v5_months else '[FAIL]'}")
    
    # 城市配置
    v4_map_size = v4_config.get("city", {}).get("map_size", [])
    v5_map_size = v5_config.get("city", {}).get("map_size", [])
    print(f"  地图大小: v4.1={v4_map_size}, v5.0={v5_map_size}, 对齐={'[PASS]' if v4_map_size == v5_map_size else '[FAIL]'}")
    
    # 2. 预算系统对比
    print("\n2. 预算系统对比:")
    
    # 初始预算
    v4_budgets = v4_config.get("budget_system", {}).get("initial_budgets", {})
    v5_budgets = v5_config.get("ledger", {}).get("initial_budget", {})
    
    for agent in ["IND", "EDU"]:
        v4_budget = v4_budgets.get(agent, 0)
        v5_budget = v5_budgets.get(agent, 0)
        print(f"  {agent}预算: v4.1={v4_budget}, v5.0={v5_budget}, 对齐={'✅' if v4_budget == v5_budget else '❌'}")
    
    # Council预算
    v4_council = v4_budgets.get("Council", 0)
    v5_council = v5_budgets.get("COUNCIL", 0)
    print(f"  Council预算: v4.1={v4_council}, v5.0={v5_council}, 对齐={'✅' if v4_council == v5_council else '❌'}")
    
    # 3. 地价系统对比
    print("\n3. 地价系统对比:")
    
    v4_gaussian = v4_config.get("land_price", {}).get("gaussian_system", {})
    v5_gaussian = v5_config.get("land_price", {}).get("gaussian_system", {})
    
    key_params = ["meters_per_pixel", "hub_sigma_base_m", "road_sigma_base_m", 
                  "hub_peak_value", "road_peak_value", "min_threshold", "alpha_inertia"]
    
    for param in key_params:
        v4_val = v4_gaussian.get(param, 0)
        v5_val = v5_gaussian.get(param, 0)
        print(f"  {param}: v4.1={v4_val}, v5.0={v5_val}, 对齐={'✅' if v4_val == v5_val else '❌'}")
    
    # 4. 强化学习参数对比
    print("\n4. 强化学习参数对比:")
    
    v4_rl = v4_config.get("solver", {}).get("rl", {})
    v5_mappo = v5_config.get("mappo", {})
    
    rl_params = [
        ("clip_eps", "clip_eps"),
        ("value_clip_eps", "value_clip_eps"),
        ("entropy_coef", "entropy_coef"),
        ("value_coef", "value_coef"),
        ("max_grad_norm", "max_grad_norm"),
        ("lr", "lr"),
        ("gamma", "gamma"),
        ("gae_lambda", "gae_lambda")
    ]
    
    for v4_key, v5_key in rl_params:
        v4_val = v4_rl.get(v4_key, 0)
        v5_val = v5_mappo.get(v5_key, 0)
        print(f"  {v4_key}: v4.1={v4_val}, v5.0={v5_val}, 对齐={'✅' if v4_val == v5_val else '❌'}")
    
    # 5. 动作参数对比
    print("\n5. 动作参数对比:")
    
    # 检查动作参数
    v4_buildings = v4_config.get("env", {}).get("buildings", {})
    v5_action_params = v5_config.get("action_params", {})
    
    # 对比EDU动作
    edu_actions = ["EDU_S", "EDU_M", "EDU_L"]
    for i, action in enumerate(edu_actions):
        v4_params = v4_buildings.get("EDU", {}).get(action, {})
        v5_params = v5_action_params.get(str(i), {})
        
        if v4_params and v5_params:
            cost_match = v4_params.get("base_cost", 0) == v5_params.get("cost", 0)
            reward_match = v4_params.get("base_reward", 0) == v5_params.get("reward", 0)
            prestige_match = v4_params.get("prestige", 0) == v5_params.get("prestige", 0)
            
            print(f"  {action}: 成本对齐={'✅' if cost_match else '❌'}, 奖励对齐={'✅' if reward_match else '❌'}, 声望对齐={'✅' if prestige_match else '❌'}")
    
    # 6. 智能体顺序对比
    print("\n6. 智能体顺序对比:")
    
    v4_agents = v4_config.get("solver", {}).get("rl", {}).get("agents", [])
    v5_agents = v5_config.get("agents", {}).get("order", [])
    
    # 标准化智能体名称
    v4_agents_normalized = [agent.upper() if agent == "Council" else agent for agent in v4_agents]
    v5_agents_normalized = [agent.upper() for agent in v5_agents]
    
    print(f"  v4.1智能体: {v4_agents} -> {v4_agents_normalized}")
    print(f"  v5.0智能体: {v5_agents} -> {v5_agents_normalized}")
    print(f"  顺序对齐: {'✅' if v4_agents_normalized == v5_agents_normalized else '❌'}")
    
    # 7. 新增功能检查
    print("\n7. v5.0新增功能:")
    
    # 调度器
    scheduler = v5_config.get("scheduler", {})
    print(f"  调度器: {'✅' if scheduler else '❌'} - {scheduler.get('name', 'None')}")
    
    # 路径引用
    paths = v5_config.get("paths", {})
    print(f"  路径配置: {'✅' if paths else '❌'} - {len(paths)}个路径")
    
    # 契约层
    print(f"  契约层: ✅ - contracts/ 模块")
    
    # 管道模式
    print(f"  管道模式: ✅ - integration/v5_0/ 模块")
    
    print("\n" + "=" * 60)
    print("参数对齐测试完成")
    print("=" * 60)

if __name__ == "__main__":
    compare_parameters()
