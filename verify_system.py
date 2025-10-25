#!/usr/bin/env python3
"""验证所有多动作功能"""
import json

print("="*60)
print("验证v5.1多动作功能完整性")
print("="*60)

# 1. 验证数据结构
print("\n1. 数据结构:")
from contracts import AtomicAction, CandidateIndex, Sequence
print("   - AtomicAction OK")
print("   - CandidateIndex OK")
print("   - Sequence OK")

# 2. 验证枚举器
print("\n2. 枚举器:")
from logic.v5_enumeration import V5ActionEnumerator
enum = V5ActionEnumerator.__dict__
if 'enumerate_with_index' in dir(V5ActionEnumerator):
    print("   - enumerate_with_index() OK")
else:
    print("   - enumerate_with_index() MISSING!")

# 3. 验证策略网络
print("\n3. 策略网络:")
from solvers.v5_0.rl_selector import V5ActorNetworkMulti, V5RLSelector
print("   - V5ActorNetworkMulti OK")
if 'select_action_multi' in dir(V5RLSelector):
    print("   - select_action_multi() OK")
else:
    print("   - select_action_multi() MISSING!")

# 4. 验证环境
print("\n4. 环境:")
from envs.v5_0.city_env import V5CityEnvironment
env_methods = dir(V5CityEnvironment)
if '_execute_action_atomic' in env_methods:
    print("   - _execute_action_atomic() OK")
else:
    print("   - _execute_action_atomic() MISSING!")
    
if 'get_action_candidates_with_index' in env_methods:
    print("   - get_action_candidates_with_index() OK")
else:
    print("   - get_action_candidates_with_index() MISSING!")

# 5. 验证配置
print("\n5. 配置:")
with open('configs/city_config_v5_0.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)
    
if 'multi_action' in cfg:
    ma = cfg['multi_action']
    print(f"   - multi_action.enabled = {ma['enabled']}")
    print(f"   - max_actions_per_step = {ma['max_actions_per_step']}")
    print(f"   - mode = {ma['mode']}")
    print(f"   - dup_policy = {ma['dup_policy']}")
else:
    print("   - multi_action MISSING!")

print("\n" + "="*60)
print("验证完成！")
print("="*60)








