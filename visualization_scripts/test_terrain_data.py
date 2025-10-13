#!/usr/bin/env python3
"""
测试地形数据脚本
检查地形数据是否有NaN值或其他问题
"""

import numpy as np
import json
import os

def test_terrain_data():
    """测试地形数据"""
    terrain_file = "data/terrain/terrain_direct_mesh.json"
    
    if not os.path.exists(terrain_file):
        print(f"❌ 地形文件不存在: {terrain_file}")
        return
    
    with open(terrain_file, 'r') as f:
        terrain_data = json.load(f)
    
    height_map = np.array(terrain_data['height_map'])
    mask = np.array(terrain_data['mask'])
    
    print("=== 地形数据测试 ===")
    print(f"高程图形状: {height_map.shape}")
    print(f"掩码形状: {mask.shape}")
    
    # 检查NaN值
    height_nan_count = np.sum(np.isnan(height_map))
    mask_nan_count = np.sum(np.isnan(mask))
    
    print(f"高程图NaN数量: {height_nan_count}")
    print(f"掩码NaN数量: {mask_nan_count}")
    
    # 检查无穷值
    height_inf_count = np.sum(np.isinf(height_map))
    mask_inf_count = np.sum(np.isinf(mask))
    
    print(f"高程图无穷值数量: {height_inf_count}")
    print(f"掩码无穷值数量: {mask_inf_count}")
    
    # 检查数值范围
    valid_height = height_map[~np.isnan(height_map)]
    if len(valid_height) > 0:
        print(f"高程范围: [{np.min(valid_height):.3f}, {np.max(valid_height):.3f}]")
        print(f"高程均值: {np.mean(valid_height):.3f}")
        print(f"高程标准差: {np.std(valid_height):.3f}")
    
    # 检查掩码
    print(f"掩码True数量: {np.sum(mask)}")
    print(f"掩码False数量: {np.sum(~mask)}")
    
    # 检查有效区域
    valid_region = height_map[mask]
    valid_nan_count = np.sum(np.isnan(valid_region))
    print(f"有效区域NaN数量: {valid_nan_count}")
    
    # 总是生成修复后的数据，确保没有NaN值
    print("🔄 生成修复后的地形数据...")
    height_map_fixed = height_map.copy()
    
    # 将所有NaN值替换为0
    height_map_fixed = np.where(np.isnan(height_map_fixed), 0.0, height_map_fixed)
    
    # 保存修复后的数据
    terrain_data['height_map'] = height_map_fixed.tolist()
    
    output_file = "data/terrain/terrain_direct_mesh_fixed.json"
    with open(output_file, 'w') as f:
        json.dump(terrain_data, f, indent=2)
    
    print(f"✅ 修复后的地形数据已保存到: {output_file}")
    
    # 验证修复结果
    valid_region_fixed = height_map_fixed[mask]
    valid_nan_count_fixed = np.sum(np.isnan(valid_region_fixed))
    print(f"修复后有效区域NaN数量: {valid_nan_count_fixed}")
    
    if valid_nan_count > 0:
        print("⚠️  有效区域存在NaN值，这可能导致训练问题")
        
        # 尝试修复NaN值
        print("🔄 尝试修复NaN值...")
        height_map_fixed = height_map.copy()
        
        # 在有效区域内，用最近邻填充NaN值
        for i in range(height_map.shape[0]):
            for j in range(height_map.shape[1]):
                if mask[i, j] and np.isnan(height_map[i, j]):
                    # 找到最近的非NaN值
                    min_dist = float('inf')
                    nearest_val = 0.0
                    
                    for di in range(-5, 6):
                        for dj in range(-5, 6):
                            ni, nj = i + di, j + dj
                            if (0 <= ni < height_map.shape[0] and 
                                0 <= nj < height_map.shape[1] and 
                                mask[ni, nj] and 
                                not np.isnan(height_map[ni, nj])):
                                dist = di*di + dj*dj
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_val = height_map[ni, nj]
                    
                    height_map_fixed[i, j] = nearest_val
        
        # 保存修复后的数据
        terrain_data['height_map'] = height_map_fixed.tolist()
        
        output_file = "data/terrain/terrain_direct_mesh_fixed.json"
        with open(output_file, 'w') as f:
            json.dump(terrain_data, f, indent=2)
        
        print(f"✅ 修复后的地形数据已保存到: {output_file}")
        
        # 验证修复结果
        valid_region_fixed = height_map_fixed[mask]
        valid_nan_count_fixed = np.sum(np.isnan(valid_region_fixed))
        print(f"修复后有效区域NaN数量: {valid_nan_count_fixed}")
    
    else:
        print("✅ 地形数据正常，没有NaN值")

if __name__ == "__main__":
    test_terrain_data()
