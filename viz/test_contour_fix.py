#!/usr/bin/env python3
"""
测试等值线修正效果
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def test_contour_fix():
    """测试等值线修正效果"""
    
    print("🧪 测试等值线修正效果")
    print("=" * 40)
    
    # 加载SDF场数据
    try:
        with open('enhanced_simulation_v2_3_output/sdf_field_month_21.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        sdf_field = np.array(data['sdf_field'])
        print(f"✅ SDF场加载成功，形状: {sdf_field.shape}")
    except Exception as e:
        print(f"❌ 无法加载SDF场数据: {e}")
        return
    
    # 获取SDF场范围
    sdf_min, sdf_max = np.min(sdf_field), np.max(sdf_field)
    print(f"📊 SDF场范围: [{sdf_min:.3f}, {sdf_max:.3f}]")
    
    # 测试修正后的等值线生成
    print("\n🔧 测试修正后的等值线生成:")
    
    # 商业建筑等值线 - 确保是递增的
    commercial_start = 0.85
    commercial_levels = np.linspace(commercial_start, sdf_min + 0.1, 8)
    # matplotlib contour需要递增的值，所以从小到大排列
    commercial_levels = np.sort(commercial_levels)  # 升序排列
    print(f"  商业建筑等值线 (8条):")
    for i, level in enumerate(commercial_levels):
        print(f"    {i+1}: {level:.3f}")
    
    # 住宅建筑等值线 - 确保是递增的
    residential_start = 0.55
    residential_levels = np.linspace(residential_start, sdf_min + 0.1, 10)
    # matplotlib contour需要递增的值，所以从小到大排列
    residential_levels = np.sort(residential_levels)  # 升序排列
    print(f"  住宅建筑等值线 (10条):")
    for i, level in enumerate(residential_levels):
        print(f"    {i+1}: {level:.3f}")
    
    # 验证等值线值是否在合理范围内
    print(f"\n✅ 验证结果:")
    print(f"  所有商业等值线值都在 [{sdf_min + 0.1:.3f}, {commercial_start:.3f}] 范围内")
    print(f"  所有住宅等值线值都在 [{sdf_min + 0.1:.3f}, {residential_start:.3f}] 范围内")
    
    # 检查是否有等值线值超出SDF场范围
    commercial_valid = np.all((commercial_levels >= sdf_min) & (commercial_levels <= sdf_max))
    residential_valid = np.all((residential_levels >= sdf_min) & (residential_levels <= sdf_max))
    
    print(f"  商业等值线有效性: {'✅' if commercial_valid else '❌'}")
    print(f"  住宅等值线有效性: {'✅' if residential_valid else '❌'}")
    
    # 创建简单的可视化
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：SDF场 + 商业等值线
        im1 = ax1.imshow(sdf_field, cmap='viridis', origin='lower')
        commercial_contours = ax1.contour(sdf_field, levels=commercial_levels, 
                                        colors='orange', linewidths=2, alpha=0.8)
        ax1.clabel(commercial_contours, inline=True, fontsize=8, fmt='%.2f')
        ax1.set_title('Commercial Isocontours (Fixed)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # 右图：SDF场 + 住宅等值线
        im2 = ax2.imshow(sdf_field, cmap='viridis', origin='lower')
        residential_contours = ax2.contour(sdf_field, levels=residential_levels, 
                                         colors='blue', linewidths=2, alpha=0.8)
        ax2.clabel(residential_contours, inline=True, fontsize=8, fmt='%.2f')
        ax2.set_title('Residential Isocontours (Fixed)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        plt.show()
        
        print("\n🎨 可视化完成！")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

if __name__ == "__main__":
    test_contour_fix()


