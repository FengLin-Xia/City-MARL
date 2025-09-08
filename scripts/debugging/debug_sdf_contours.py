#!/usr/bin/env python3
"""
SDF等值线诊断脚本
分析SDF分布和等值线生成问题
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import cv2
from pathlib import Path

def analyze_sdf_distribution():
    """分析SDF分布"""
    # 使用存在的SDF文件
    sdf_file = 'enhanced_simulation_v2_3_output/sdf_field_month_21.json'
    
    if not os.path.exists(sdf_file):
        print(f"❌ SDF文件不存在: {sdf_file}")
        return
    
    try:
        print(f"📁 正在加载SDF文件: {sdf_file}")
        with open(sdf_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            sdf_field = np.array(data['sdf_field'])
        print("✅ SDF文件加载成功")
    except Exception as e:
        print(f"❌ 加载SDF文件失败: {e}")
        return
    
    print("📊 SDF场分析结果:")
    print(f"   - 尺寸: {sdf_field.shape}")
    print(f"   - 最小值: {sdf_field.min():.6f}")
    print(f"   - 最大值: {sdf_field.max():.6f}")
    print(f"   - 平均值: {sdf_field.mean():.6f}")
    print(f"   - 标准差: {sdf_field.std():.6f}")
    
    # 分析分位数
    sdf_flat = sdf_field.flatten()
    percentiles = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
    sdf_percentiles = np.percentile(sdf_flat, percentiles)
    
    print("\n📊 分位数分析:")
    for p, val in zip(percentiles, sdf_percentiles):
        print(f"   - {p}%: {val:.6f}")
    
    # 分析各分位数对应的区域面积
    print("\n📊 区域面积分析:")
    for p, val in zip(percentiles, sdf_percentiles):
        area_ratio = np.sum(sdf_field >= val) / sdf_field.size * 100
        print(f"   - {p}% (阈值 {val:.6f}): 覆盖 {area_ratio:.2f}% 的区域")
    
    # 可视化SDF场
    visualize_sdf_field(sdf_field, sdf_percentiles)
    
    # 分析等值线问题
    analyze_contour_issues(sdf_field, sdf_percentiles)

def visualize_sdf_field(sdf_field, sdf_percentiles):
    """可视化SDF场"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SDF场分析 - 等值线问题诊断', fontsize=16, fontweight='bold')
    
    # 1. 原始SDF场
    im1 = axes[0, 0].imshow(sdf_field, cmap='viridis')
    axes[0, 0].set_title('原始SDF场')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. SDF场直方图
    axes[0, 1].hist(sdf_field.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('SDF值分布')
    axes[0, 1].set_xlabel('SDF值')
    axes[0, 1].set_ylabel('像素数量')
    
    # 在直方图上标记分位数
    for i, (p, val) in enumerate(zip([80, 70, 60, 50, 40, 30, 20], sdf_percentiles[2:9])):
        color = 'red' if i < 3 else 'blue'
        axes[0, 1].axvline(val, color=color, linestyle='--', alpha=0.8, 
                           label=f'{p}%: {val:.3f}')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 商业建筑等值线 (80%, 70%, 60%)
    commercial_thresholds = sdf_percentiles[:3]  # 80%, 70%, 60%
    axes[0, 2].imshow(sdf_field, cmap='viridis')
    axes[0, 2].set_title('商业建筑等值线 (80%, 70%, 60%)')
    
    for i, threshold in enumerate(commercial_thresholds):
        binary = (sdf_field >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour = largest_contour.reshape(-1, 2)
            axes[0, 2].plot(largest_contour[:, 0], largest_contour[:, 1], 
                           'r--', linewidth=2, alpha=0.8, 
                           label=f'{80-i*10}%: {threshold:.3f}')
    
    axes[0, 2].legend()
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    
    # 4. 住宅建筑等值线 (50%, 40%, 30%, 20%)
    residential_thresholds = sdf_percentiles[4:8]  # 50%, 40%, 30%, 20%
    axes[1, 0].imshow(sdf_field, cmap='viridis')
    axes[1, 0].set_title('住宅建筑等值线 (50%, 40%, 30%, 20%)')
    
    for i, threshold in enumerate(residential_thresholds):
        binary = (sdf_field >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour = largest_contour.reshape(-1, 2)
            axes[1, 0].plot(largest_contour[:, 0], largest_contour[:, 1], 
                           'b--', linewidth=2, alpha=0.8, 
                           label=f'{50-i*10}%: {threshold:.3f}')
    
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    
    # 5. 问题分析：80%分位数覆盖区域
    problem_threshold = sdf_percentiles[2]  # 80%
    problem_binary = (sdf_field >= problem_threshold).astype(np.uint8) * 255
    axes[1, 1].imshow(problem_binary, cmap='gray')
    axes[1, 1].set_title(f'问题区域: 80%分位数覆盖\n(阈值: {problem_threshold:.3f})')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    
    # 计算问题区域面积
    problem_area = np.sum(problem_binary > 0)
    total_area = problem_binary.size
    problem_ratio = problem_area / total_area * 100
    axes[1, 1].text(0.5, 0.95, f'覆盖面积: {problem_ratio:.1f}%', 
                    transform=axes[1, 1].transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 6. 建议的修正分位数
    suggested_commercial = [95, 90, 85]  # 更严格的商业分位数
    suggested_residential = [80, 70, 60, 50]  # 更合理的住宅分位数
    
    axes[1, 2].imshow(sdf_field, cmap='viridis')
    axes[1, 2].set_title('建议的修正分位数')
    
    # 绘制建议的商业分位数
    for i, p in enumerate(suggested_commercial):
        threshold = np.percentile(sdf_field.flatten(), p)
        binary = (sdf_field >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour = largest_contour.reshape(-1, 2)
            axes[1, 2].plot(largest_contour[:, 0], largest_contour[:, 1], 
                           'r-', linewidth=2, alpha=0.8, 
                           label=f'商业 {p}%: {threshold:.3f}')
    
    # 绘制建议的住宅分位数
    for i, p in enumerate(suggested_residential):
        threshold = np.percentile(sdf_field.flatten(), p)
        binary = (sdf_field >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour = largest_contour.reshape(-1, 2)
            axes[1, 2].plot(largest_contour[:, 0], largest_contour[:, 1], 
                           'b-', linewidth=2, alpha=0.8, 
                           label=f'住宅 {p}%: {threshold:.3f}')
    
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('sdf_contour_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_contour_issues(sdf_field, sdf_percentiles):
    """分析等值线问题"""
    print("\n🔍 等值线问题分析:")
    
    # 分析商业建筑等值线
    commercial_thresholds = sdf_percentiles[:3]  # 80%, 70%, 60%
    print("\n🏢 商业建筑等值线问题:")
    
    for i, threshold in enumerate(commercial_thresholds):
        binary = (sdf_field >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            print(f"  - {80-i*10}%分位数 (阈值 {threshold:.3f}):")
            print(f"    面积: {area:.0f} 像素²")
            print(f"    周长: {perimeter:.0f} 像素")
            print(f"    覆盖区域比例: {area/sdf_field.size*100:.2f}%")
            
            # 检查是否过大
            if area > sdf_field.size * 0.3:  # 超过30%的区域
                print(f"    ⚠️ 问题: 覆盖区域过大!")
    
    # 分析住宅建筑等值线
    residential_thresholds = sdf_percentiles[4:8]  # 50%, 40%, 30%, 20%
    print("\n🏠 住宅建筑等值线问题:")
    
    for i, threshold in enumerate(residential_thresholds):
        binary = (sdf_field >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            print(f"  - {50-i*10}%分位数 (阈值 {threshold:.3f}):")
            print(f"    面积: {area:.0f} 像素²")
            print(f"    周长: {perimeter:.0f} 像素")
            print(f"    覆盖区域比例: {area/sdf_field.size*100:.2f}%")
            
            # 检查是否过大
            if area > sdf_field.size * 0.4:  # 超过40%的区域
                print(f"    ⚠️ 问题: 覆盖区域过大!")

def suggest_fixes():
    """建议修复方案"""
    print("\n💡 修复建议:")
    print("1. 调整分位数设置:")
    print("   - 商业建筑: [95, 90, 85] 替代 [80, 70, 60]")
    print("   - 住宅建筑: [80, 70, 60, 50] 替代 [50, 40, 30, 20]")
    
    print("\n2. 使用更严格的阈值:")
    print("   - 商业建筑: 使用95%分位数作为最内圈")
    print("   - 住宅建筑: 使用80%分位数作为最内圈")
    
    print("\n3. 考虑使用固定阈值:")
    print("   - 基于SDF值的绝对范围设置阈值")
    print("   - 避免分位数带来的跳跃性变化")
    
    print("\n4. 等值线验证:")
    print("   - 检查等值线覆盖面积是否合理")
    print("   - 确保最内圈不会覆盖过大区域")

if __name__ == "__main__":
    print("🔍 SDF等值线问题诊断开始...")
    analyze_sdf_distribution()
    suggest_fixes()
    print("\n✅ 诊断完成！请查看生成的图表和修复建议。")
