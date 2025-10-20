#!/usr/bin/env python3
"""
测试EDU邻近性时序修复方案
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

def test_edu_proximity_timing_fix():
    """测试EDU邻近性时序修复方案"""
    print("=== 测试EDU邻近性时序修复方案 ===")
    
    # 读取当前配置
    with open('configs/city_config_v4_1.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    print("当前配置:")
    print(f"邻近性约束: {cfg['growth_v4_1']['proximity_constraint']}")
    
    # 修复方案1：延迟邻近性约束启动
    print("\n=== 修复方案1：延迟邻近性约束启动 ===")
    
    # 创建修复后的配置
    fixed_cfg = cfg.copy()
    fixed_cfg['growth_v4_1']['proximity_constraint']['apply_after_month'] = 2
    
    print("修复后的配置:")
    print(f"邻近性约束: {fixed_cfg['growth_v4_1']['proximity_constraint']}")
    
    # 保存修复后的配置
    with open('configs/city_config_v4_1_timing_fix.json', 'w', encoding='utf-8') as f:
        json.dump(fixed_cfg, f, indent=2, ensure_ascii=False)
    
    print("修复完成！配置已保存到: configs/city_config_v4_1_timing_fix.json")
    
    # 分析修复效果
    print("\n=== 修复效果分析 ===")
    print("1. Month 0: 邻近性约束未启用，EDU自由选择初始位置")
    print("2. Month 1: 邻近性约束未启用，EDU继续自由扩展")
    print("3. Month 2: 邻近性约束启用，EDU开始形成集群")
    print("4. Month 3+: 邻近性约束持续生效，EDU建筑紧密分布")
    
    print("\n=== 预期效果 ===")
    print("1. EDU在Month 0-1有足够时间建立初始建筑")
    print("2. 从Month 2开始，邻近性约束引导EDU形成集群")
    print("3. 避免早期过度限制导致EDU无法扩展")
    print("4. 保持15px的邻近性范围，确保集群紧凑")
    
    # 创建测试脚本
    test_script = '''#!/usr/bin/env python3
"""
测试EDU邻近性时序修复效果
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.v4_1.city_env import CityEnvironment
import json

def test_timing_fix():
    """测试时序修复效果"""
    print("=== 测试EDU邻近性时序修复效果 ===")
    
    # 加载修复后的配置
    with open('configs/city_config_v4_1_timing_fix.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # 创建环境
    env = CityEnvironment(cfg)
    
    print("修复后的配置:")
    print(f"邻近性约束: {cfg['growth_v4_1']['proximity_constraint']}")
    
    # 测试不同月份的行为
    test_months = [0, 1, 2, 3, 5, 8, 10]
    
    for test_month in test_months:
        print(f"\\n--- 测试月份 {test_month} ---")
        env.current_month = test_month
        env.current_agent = 'EDU'
        
        try:
            # 获取候选槽位
            candidates = env._get_candidate_slots()
            print(f"EDU候选槽位数量: {len(candidates)}")
            
            # 检查邻近性约束状态
            proximity_cfg = env.v4_cfg.get('proximity_constraint', {})
            if proximity_cfg.get('enabled', False) and test_month >= proximity_cfg.get('apply_after_month', 1):
                print(f"  邻近性约束: 已启用 (max_distance={proximity_cfg.get('max_distance', 10.0)})")
                
                # 检查是否有EDU建筑作为参考
                edu_buildings = env.buildings.get('public', [])
                print(f"  参考EDU建筑数量: {len(edu_buildings)}")
                
                if len(edu_buildings) == 0:
                    print("  WARNING: 没有EDU建筑作为邻近性参考")
                else:
                    print("  OK: 有EDU建筑作为邻近性参考")
            else:
                print(f"  邻近性约束: 未启用 (月份{test_month} < apply_after_month)")
                
        except Exception as e:
            print(f"  测试月份{test_month}时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\\n=== 测试完成 ===")

if __name__ == "__main__":
    test_timing_fix()
'''
    
    with open('scripts/test_timing_fix_effect.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("\n测试脚本已创建: scripts/test_timing_fix_effect.py")
    print("运行测试: python scripts/test_timing_fix_effect.py")
    
    print(f"\n=== 修复完成 ===")

if __name__ == "__main__":
    test_edu_proximity_timing_fix()
