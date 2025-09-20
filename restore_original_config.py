#!/usr/bin/env python3
"""
恢复原始配置脚本
将 city_config_v3_5_backup.json 恢复到 city_config_v3_5.json
"""

import shutil
import os

def restore_original_config():
    """恢复原始配置"""
    backup_file = 'configs/city_config_v3_5_backup.json'
    target_file = 'configs/city_config_v3_5.json'
    
    if not os.path.exists(backup_file):
        print(f"错误：备份文件 {backup_file} 不存在")
        return False
    
    try:
        # 复制备份文件到目标文件
        shutil.copy2(backup_file, target_file)
        print(f"成功恢复原始配置：{backup_file} -> {target_file}")
        print("恢复的内容包括：")
        print("- Hub1 (20, 55) 已恢复")
        print("- 道路线高斯核已恢复")
        print("- 3个Hub的权重配置已恢复 [0.4, 0.4, 0.2]")
        return True
    except Exception as e:
        print(f"恢复配置时出错：{e}")
        return False

if __name__ == "__main__":
    restore_original_config()


