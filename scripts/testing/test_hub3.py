#!/usr/bin/env python3
"""
测试 Hub3 功能
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from logic.enhanced_sdf_system import GaussianLandPriceSystem

def test_hub3():
    """测试 Hub3 功能"""
    print("🧪 测试 Hub3 功能...")
    
    # 加载配置
    with open('configs/city_config_v3_1.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取交通枢纽位置
    transport_hubs = config['city']['transport_hubs']
    map_size = config['city']['map_size']
    
    print(f"📍 交通枢纽位置: {transport_hubs}")
    print(f"🗺️ 地图尺寸: {map_size}")
    
    # 初始化高斯核地价场系统
    land_price_system = GaussianLandPriceSystem(config)
    land_price_system.initialize_system(transport_hubs, map_size)
    
    # 获取地价场
    land_price_field = land_price_system.get_land_price_field()
    
    print(f"📊 地价场形状: {land_price_field.shape}")
    print(f"📊 地价场值范围: [{np.min(land_price_field):.3f}, {np.max(land_price_field):.3f}]")
    
    # 检查每个 Hub 附近的地价值
    for i, hub in enumerate(transport_hubs):
        x, y = hub[0], hub[1]
        hub_value = land_price_field[y, x]
        print(f"🎯 Hub {i+1} ({x}, {y}) 地价值: {hub_value:.3f}")
        
        # 检查周围区域的地价值
        radius = 5
        y_min, y_max = max(0, y-radius), min(map_size[1]-1, y+radius)
        x_min, x_max = max(0, x-radius), min(map_size[0]-1, x+radius)
        local_values = land_price_field[y_min:y_max+1, x_min:x_max+1]
        print(f"  周围区域地价值范围: [{np.min(local_values):.3f}, {np.max(local_values):.3f}]")
    
    # 可视化地价场
    plt.figure(figsize=(12, 10))
    
    # 地价场热力图
    plt.subplot(2, 2, 1)
    im = plt.imshow(land_price_field, cmap='viridis', aspect='equal')
    plt.colorbar(im, label='地价值')
    plt.title('🏔️ 高斯核地价场（含 Hub3）')
    plt.xlabel('X (像素)')
    plt.ylabel('Y (像素)')
    
    # 添加交通枢纽标记
    for i, hub in enumerate(transport_hubs):
        plt.plot(hub[0], hub[1], 'ro', markersize=12, label=f'Hub {i+1}')
    plt.legend()
    
    # 地价场等高线图
    plt.subplot(2, 2, 2)
    X, Y = np.meshgrid(np.arange(map_size[0]), np.arange(map_size[1]))
    contours = plt.contour(X, Y, land_price_field, levels=10, colors='black', alpha=0.6)
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(land_price_field, cmap='viridis', aspect='equal', alpha=0.7)
    plt.title('📈 地价场等高线')
    plt.xlabel('X (像素)')
    plt.ylabel('Y (像素)')
    
    # 添加交通枢纽标记
    for i, hub in enumerate(transport_hubs):
        plt.plot(hub[0], hub[1], 'ro', markersize=12, label=f'Hub {i+1}')
    plt.legend()
    
    # 地价场3D视图
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(2, 2, 3, projection='3d')
    ax.plot_surface(X, Y, land_price_field, cmap='viridis', alpha=0.8)
    ax.set_title('🏔️ 地价场3D视图')
    ax.set_xlabel('X (像素)')
    ax.set_ylabel('Y (像素)')
    ax.set_zlabel('地价值')
    
    # 地价场统计
    plt.subplot(2, 2, 4)
    plt.hist(land_price_field.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('📊 地价值分布直方图')
    plt.xlabel('地价值')
    plt.ylabel('频次')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('hub3_test_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Hub3 测试完成！结果已保存到 hub3_test_result.png")

if __name__ == "__main__":
    test_hub3()
