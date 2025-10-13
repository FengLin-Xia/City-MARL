#!/usr/bin/env python3
"""
简单的槽位形状测试
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os

def test_slot_generation():
    """测试槽位生成"""
    print("🔍 测试槽位生成...")
    
    # 创建简单的地价场
    map_size = [110, 110]
    land_price_field = np.zeros((map_size[1], map_size[0]))
    
    # 创建两个高斯核
    hub1_pos = [37, 55]  # 商业枢纽
    hub2_pos = [73, 55]  # 工业枢纽
    
    # 商业枢纽高斯核
    for y in range(map_size[1]):
        for x in range(map_size[0]):
            dx = x - hub1_pos[0]
            dy = y - hub1_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            land_price_field[y, x] += np.exp(-(distance**2) / (2 * 20**2))
    
    # 工业枢纽高斯核
    for y in range(map_size[1]):
        for x in range(map_size[0]):
            dx = x - hub2_pos[0]
            dy = y - hub2_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            land_price_field[y, x] += np.exp(-(distance**2) / (2 * 25**2))
    
    # 归一化
    land_price_field = (land_price_field - land_price_field.min()) / (land_price_field.max() - land_price_field.min())
    
    print(f"地价场范围: {land_price_field.min():.3f} - {land_price_field.max():.3f}")
    
    # 测试等值线提取
    test_contour_extraction(land_price_field, map_size)

def test_contour_extraction(land_price_field, map_size):
    """测试等值线提取"""
    print("🔍 测试等值线提取...")
    
    # 测试不同的等值线级别
    levels = [0.85, 0.78, 0.71, 0.60, 0.70, 0.80, 0.55]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, level in enumerate(levels):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # 创建二值图像
        binary_image = (land_price_field >= level).astype(np.uint8) * 255
        
        # 查找轮廓
        contours_found, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制地价场
        im = ax.imshow(land_price_field, cmap='YlOrRd', alpha=0.7)
        
        # 绘制轮廓
        contour_count = 0
        for contour in contours_found:
            # 简化轮廓
            epsilon = 0.02 * cv2.arcLength(contour, True)
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # 绘制原始轮廓
            contour_array = np.array(contour).reshape(-1, 2)
            ax.plot(contour_array[:, 0], contour_array[:, 1], 'b-', linewidth=1, alpha=0.6)
            
            # 绘制简化轮廓
            simplified_array = np.array(simplified_contour).reshape(-1, 2)
            ax.plot(simplified_array[:, 0], simplified_array[:, 1], 'r-', linewidth=2, alpha=0.8)
            
            contour_count += 1
        
        ax.set_title(f'等值线级别 {level}\n({contour_count} 条轮廓)')
        ax.set_aspect('equal')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for i in range(len(levels), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_contour_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 等值线提取测试完成")

def test_slot_sampling():
    """测试槽位采样"""
    print("🔍 测试槽位采样...")
    
    # 创建简单的地价场
    map_size = [110, 110]
    land_price_field = np.zeros((map_size[1], map_size[0]))
    
    # 创建商业枢纽高斯核
    hub_pos = [55, 55]
    for y in range(map_size[1]):
        for x in range(map_size[0]):
            dx = x - hub_pos[0]
            dy = y - hub_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            land_price_field[y, x] = np.exp(-(distance**2) / (2 * 30**2))
    
    # 提取等值线
    level = 0.7
    binary_image = (land_price_field >= level).astype(np.uint8) * 255
    contours_found, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_found:
        print("  未找到等值线")
        return
    
    # 选择最大的轮廓
    largest_contour = max(contours_found, key=cv2.contourArea)
    
    # 简化轮廓
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 转换为坐标列表
    contour_points = []
    for point in simplified_contour:
        x, y = point[0]
        contour_points.append([int(x), int(y)])
    
    print(f"  轮廓点数: {len(contour_points)}")
    
    # 测试等弧长采样
    test_arc_length_sampling(contour_points, map_size)

def test_arc_length_sampling(contour, map_size):
    """测试等弧长采样"""
    print("🔍 测试等弧长采样...")
    
    # 采样参数
    spacing_px = 15  # 15像素间距
    
    # 计算轮廓总长度
    total_length = 0
    segment_lengths = []
    
    for i in range(len(contour)):
        next_i = (i + 1) % len(contour)
        dx = contour[next_i][0] - contour[i][0]
        dy = contour[next_i][1] - contour[i][1]
        segment_length = np.sqrt(dx**2 + dy**2)
        segment_lengths.append(segment_length)
        total_length += segment_length
    
    print(f"  轮廓总长度: {total_length:.2f} 像素")
    
    # 计算采样点数量
    num_samples = max(1, int(total_length / spacing_px))
    actual_spacing = total_length / num_samples
    
    print(f"  采样点数: {num_samples}")
    print(f"  实际间距: {actual_spacing:.2f} 像素")
    
    # 沿轮廓采样
    slots = []
    current_length = 0
    sample_index = 0
    
    for i in range(len(contour)):
        if sample_index >= num_samples:
            break
        
        next_i = (i + 1) % len(contour)
        segment_length = segment_lengths[i]
        
        # 检查是否需要在当前段内采样
        while (sample_index < num_samples and 
               current_length + segment_length >= sample_index * actual_spacing):
            
            # 计算采样点位置
            t = (sample_index * actual_spacing - current_length) / segment_length
            t = max(0, min(1, t))
            
            # 线性插值
            x = int(contour[i][0] + t * (contour[next_i][0] - contour[i][0]))
            y = int(contour[i][1] + t * (contour[next_i][1] - contour[i][1]))
            
            # 检查位置有效性
            if 0 <= x < map_size[0] and 0 <= y < map_size[1]:
                slots.append([x, y])
            
            sample_index += 1
        
        current_length += segment_length
    
    print(f"  生成槽位数: {len(slots)}")
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：轮廓和采样点
    ax1.plot([p[0] for p in contour], [p[1] for p in contour], 'b-', linewidth=2, label='轮廓')
    ax1.scatter([p[0] for p in slots], [p[1] for p in slots], c='red', s=30, alpha=0.8, label='槽位')
    ax1.set_title('等弧长采样结果')
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：采样间距分析
    if len(slots) > 1:
        distances = []
        for i in range(len(slots)):
            next_i = (i + 1) % len(slots)
            dx = slots[next_i][0] - slots[i][0]
            dy = slots[next_i][1] - slots[i][1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
        
        ax2.hist(distances, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(actual_spacing, color='red', linestyle='--', label=f'目标间距: {actual_spacing:.2f}')
        ax2.set_title('槽位间距分布')
        ax2.set_xlabel('间距 (像素)')
        ax2.set_ylabel('频次')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('simple_slot_sampling_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 等弧长采样测试完成")

def main():
    """主函数"""
    print("🔍 简单槽位形状测试")
    
    # 测试槽位生成
    test_slot_generation()
    
    # 测试槽位采样
    test_slot_sampling()
    
    print("\n✅ 测试完成！")
    print("  生成的文件:")
    print("  - simple_contour_test.png: 等值线提取测试")
    print("  - simple_slot_sampling_test.png: 槽位采样测试")

if __name__ == "__main__":
    main()
