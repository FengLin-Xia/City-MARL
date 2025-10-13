#!/usr/bin/env python3
"""
测试数据结构一致性
"""

import json
from pathlib import Path

def test_simulation_data_structure():
    """测试模拟数据结构"""
    print("🔍 测试模拟数据结构...")
    
    # 检查模拟脚本中的数据结构
    print("📊 模拟脚本中的数据结构:")
    print("city_state = {")
    print("    'public': [],")
    print("    'residential': [],")
    print("    'commercial': [],")
    print("    'residents': [],")
    print("    'trunk_road': [[40, 128], [216, 128]],")
    print("    'core_point': [128, 128]")
    print("}")
    
    # 检查输出文件中的数据结构
    print("\n📊 输出文件中的数据结构:")
    with open('enhanced_simulation_output/city_state_output.json', 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    
    print("city_state_output.json = {")
    print("    'buildings': {")
    print("        'public': [...],")
    print("        'residential': [...],")
    print("        'commercial': [...]")
    print("    },")
    print("    'residents': [...]")
    print("}")
    
    # 检查数据是否一致
    print(f"\n🔍 数据一致性检查:")
    print(f"输出文件中的建筑数量:")
    print(f"  公共建筑: {len(output_data['buildings']['public'])}")
    print(f"  住宅建筑: {len(output_data['buildings']['residential'])}")
    print(f"  商业建筑: {len(output_data['buildings']['commercial'])}")
    print(f"  居民: {len(output_data['residents'])}")
    
    # 检查渲染脚本应该使用哪个数据结构
    print(f"\n💡 渲染脚本应该使用的数据结构:")
    print(f"如果渲染是在模拟运行时进行的，应该使用:")
    print(f"  self.city_state['public']")
    print(f"  self.city_state['residential']")
    print(f"  self.city_state['commercial']")
    print(f"")
    print(f"如果渲染是读取输出文件进行的，应该使用:")
    print(f"  city_state['buildings']['public']")
    print(f"  city_state['buildings']['residential']")
    print(f"  city_state['buildings']['commercial']")

def main():
    """主函数"""
    print("🔍 数据结构一致性测试")
    print("=" * 50)
    
    test_simulation_data_structure()
    
    print(f"\n💡 结论:")
    print(f"1. 模拟运行时使用 city_state['public'] 等")
    print(f"2. 输出文件使用 city_state['buildings']['public'] 等")
    print(f"3. 渲染脚本需要根据调用时机选择正确的数据结构")

if __name__ == "__main__":
    main()
