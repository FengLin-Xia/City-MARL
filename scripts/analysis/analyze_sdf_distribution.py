#!/usr/bin/env python3
"""
分析SDF场分布，诊断等值线建筑生成问题
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_sdf_distribution():
    """分析SDF场分布"""
    sdf_file = "enhanced_simulation_v2_3_output/sdf_field_month_00.json"
    
    if not os.path.exists(sdf_file):
        print(f"SDF文件不存在: {sdf_file}")
        return
    
    print("正在分析SDF场分布...")
    
    with open(sdf_file, 'r', encoding='utf-8') as f:
        sdf_data = json.load(f)
    
    sdf_field = np.array(sdf_data['sdf_field'])
    
    print(f"SDF场形状: {sdf_field.shape}")
    print(f"SDF值范围: [{np.min(sdf_field):.4f}, {np.max(sdf_field):.4f}]")
    print(f"SDF平均值: {np.mean(sdf_field):.4f}")
    print(f"SDF标准差: {np.std(sdf_field):.4f}")
    
    # 分析不同阈值下的等值线
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    
    print("\n等值线分析:")
    for threshold in thresholds:
        binary = (sdf_field >= threshold).astype(np.uint8)
        area_ratio = np.sum(binary) / binary.size
        print(f"阈值 {threshold:.2f}: 面积占比 {area_ratio:.4f} ({np.sum(binary)} 像素)")
    
    # 可视化SDF场
    plt.figure(figsize=(15, 5))
    
    # SDF场热力图
    plt.subplot(1, 3, 1)
    plt.imshow(sdf_field, cmap='viridis')
    plt.colorbar(label='SDF值')
    plt.title('SDF场分布')
    
    # SDF值直方图
    plt.subplot(1, 3, 2)
    plt.hist(sdf_field.flatten(), bins=50, alpha=0.7)
    plt.axvline(x=0.85, color='red', linestyle='--', label='商业建筑阈值')
    plt.axvline(x=0.55, color='orange', linestyle='--', label='住宅建筑阈值')
    plt.xlabel('SDF值')
    plt.ylabel('频次')
    plt.title('SDF值分布')
    plt.legend()
    
    # 等值线可视化
    plt.subplot(1, 3, 3)
    # 显示几个关键等值线
    for threshold in [0.3, 0.5, 0.7, 0.85]:
        binary = (sdf_field >= threshold).astype(np.uint8)
        if np.sum(binary) > 0:
            plt.contour(sdf_field, levels=[threshold], colors=['red'], alpha=0.7, linewidths=2)
    
    plt.imshow(sdf_field, cmap='viridis', alpha=0.3)
    plt.title('等值线可视化')
    
    plt.tight_layout()
    plt.savefig('sdf_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 检查等值线提取问题
    print("\n等值线提取诊断:")
    commercial_threshold = 0.85
    binary = (sdf_field >= commercial_threshold).astype(np.uint8)
    
    if np.sum(binary) == 0:
        print(f"❌ 商业建筑阈值 {commercial_threshold} 太高，没有符合条件的区域")
        print("建议降低商业建筑阈值或调整SDF生成逻辑")
    else:
        print(f"✅ 商业建筑阈值 {commercial_threshold} 有 {np.sum(binary)} 个像素符合条件")
        
        # 检查连通性
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)
        print(f"连通区域数量: {num_features}")
        
        if num_features > 0:
            # 找到最大的连通区域
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            max_size = np.max(sizes)
            print(f"最大连通区域大小: {max_size} 像素")
            
            if max_size < 100:
                print("⚠️ 连通区域太小，可能无法生成有效的等值线")

if __name__ == "__main__":
    analyze_sdf_distribution()
