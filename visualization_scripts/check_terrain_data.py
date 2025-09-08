#!/usr/bin/env python3
"""
检查地形数据的真实情况
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt

def check_terrain_data():
    """检查地形数据"""
    terrain_file = "data/terrain/terrain_direct_mesh_fixed.json"
    
    if not os.path.exists(terrain_file):
        print(f"❌ 地形文件不存在: {terrain_file}")
        return
    
    print("🔍 检查地形数据...")
    print("=" * 50)
    
    # 加载地形数据
    with open(terrain_file, 'r') as f:
        terrain_data = json.load(f)
    
    height_map = np.array(terrain_data['height_map'])
    mask = np.array(terrain_data['mask'])
    
    print("📊 基本信息:")
    print(f"   高程图形状: {height_map.shape}")
    print(f"   掩码形状: {mask.shape}")
    print(f"   高程范围: [{np.min(height_map):.3f}, {np.max(height_map):.3f}]")
    print(f"   平均高程: {np.mean(height_map):.3f}")
    
    # 检查掩码
    print(f"\n🔍 掩码分析:")
    print(f"   掩码True数量: {np.sum(mask)}")
    print(f"   掩码False数量: {np.sum(~mask)}")
    print(f"   掩码覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
    
    # 检查有效区域
    valid_heights = height_map[mask]
    invalid_heights = height_map[~mask]
    
    print(f"\n📈 有效区域分析:")
    print(f"   有效区域高程范围: [{np.min(valid_heights):.3f}, {np.max(valid_heights):.3f}]")
    print(f"   有效区域平均高程: {np.mean(valid_heights):.3f}")
    print(f"   无效区域高程范围: [{np.min(invalid_heights):.3f}, {np.max(invalid_heights):.3f}]")
    print(f"   无效区域平均高程: {np.mean(invalid_heights):.3f}")
    
    # 检查NaN值
    print(f"\n⚠️  NaN值检查:")
    print(f"   高程图NaN数量: {np.sum(np.isnan(height_map))}")
    print(f"   掩码NaN数量: {np.sum(np.isnan(mask))}")
    print(f"   有效区域NaN数量: {np.sum(np.isnan(valid_heights))}")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 原始高程图（无掩码）
    im1 = axes[0].imshow(height_map.T, cmap='terrain', aspect='auto', origin='lower')
    axes[0].set_title('原始高程图（无掩码）')
    axes[0].set_xlabel('X坐标')
    axes[0].set_ylabel('Y坐标')
    plt.colorbar(im1, ax=axes[0], label='高程')
    
    # 2. 掩码
    im2 = axes[1].imshow(mask.T, cmap='gray', aspect='auto', origin='lower')
    axes[1].set_title('掩码')
    axes[1].set_xlabel('X坐标')
    axes[1].set_ylabel('Y坐标')
    plt.colorbar(im2, ax=axes[1])
    
    # 3. 应用掩码后的高程图
    masked_height = np.where(mask, height_map, np.nan)
    im3 = axes[2].imshow(masked_height.T, cmap='terrain', aspect='auto', origin='lower')
    axes[2].set_title('应用掩码后的高程图')
    axes[2].set_xlabel('X坐标')
    axes[2].set_ylabel('Y坐标')
    plt.colorbar(im3, ax=axes[2], label='高程')
    
    plt.tight_layout()
    plt.savefig("visualization_output/terrain_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ 对比图已保存到: visualization_output/terrain_comparison.png")
    
    plt.show()
    
    # 建议
    print(f"\n💡 建议:")
    if np.sum(np.isnan(height_map)) > 0:
        print("   - 高程图中存在NaN值，需要处理")
    
    if np.mean(invalid_heights) == 0.0:
        print("   - 无效区域高程都是0，这是正常的")
    
    print("   - 如果要使用无掩码版本，直接使用height_map即可")
    print("   - 如果要使用掩码版本，使用np.where(mask, height_map, np.nan)")

if __name__ == "__main__":
    check_terrain_data()
