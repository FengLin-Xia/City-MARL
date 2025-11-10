#!/usr/bin/env python3
"""
修复被重复的关键文件
"""

import json

print("="*60)
print("修复重复文件")
print("="*60)

# 1. 修复 rl_selector.py (保留前642行)
print("\n1. 修复 solvers/v5_0/rl_selector.py...")
with open('solvers/v5_0/rl_selector.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
print(f"   原始: {len(lines)} 行")

# 保留到第642行（V5ActorNetworkMulti第一次定义后的空行）
fixed_lines = lines[:642]
with open('solvers/v5_0/rl_selector.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)
print(f"   修复后: {len(fixed_lines)} 行")

# 2. 修复 city_config_v5_0.json (保留到第一个multi_action结束)
print("\n2. 修复 configs/city_config_v5_0.json...")
with open('configs/city_config_v5_0.json', 'r', encoding='utf-16') as f:
    content = f.read()
print(f"   原始大小: {len(content)} 字符")

# 找到第一个闭合大括号（在multi_action之后）
first_close = content.find('\n}\n', content.find('"multi_action"'))
if first_close > 0:
    fixed_content = content[:first_close + 3]  # 包含 \n}\n
    with open('configs/city_config_v5_0.json', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    print(f"   修复后大小: {len(fixed_content)} 字符")
else:
    print("   未找到重复点，跳过")

print("\n" + "="*60)
print("修复完成！")
print("="*60)




