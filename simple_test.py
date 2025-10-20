#!/usr/bin/env python3
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

def test_parcel_loading():
    """测试地块加载"""
    print("开始测试地块加载...")
    
    parcels = {}
    current_parcel = []
    parcel_id = 0
    
    with open('parcel.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[':
                if current_parcel and len(current_parcel) >= 3:
                    try:
                        poly = Polygon(current_parcel)
                        if poly.is_valid:
                            parcels[f'parcel_{parcel_id}'] = poly
                            parcel_id += 1
                    except Exception as e:
                        print(f"警告：地块 {parcel_id} 无效: {e}")
                current_parcel = []
            elif line == ']':
                if current_parcel and len(current_parcel) >= 3:
                    try:
                        poly = Polygon(current_parcel)
                        if poly.is_valid:
                            parcels[f'parcel_{parcel_id}'] = poly
                            parcel_id += 1
                    except Exception as e:
                        print(f"警告：地块 {parcel_id} 无效: {e}")
                current_parcel = []
            elif line and not line.startswith('#'):
                try:
                    coords = [float(x.strip()) for x in line.split(',')]
                    if len(coords) >= 2:
                        current_parcel.append((coords[0], coords[1]))
                except ValueError:
                    continue
    
    print(f"成功加载 {len(parcels)} 个地块")
    
    # 测试第一个地块
    if parcels:
        first_parcel_id = list(parcels.keys())[0]
        first_poly = parcels[first_parcel_id]
        print(f"第一个地块 {first_parcel_id}:")
        print(f"  面积: {first_poly.area:.2f} 平方米")
        print(f"  周长: {first_poly.length:.2f} 米")
        print(f"  质心: ({first_poly.centroid.x:.2f}, {first_poly.centroid.y:.2f})")
        
        # 简单可视化
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制多边形
        if hasattr(first_poly, 'exterior'):
            x, y = first_poly.exterior.xy
            ax.plot(x, y, 'k-', linewidth=2, label='边界')
            ax.fill(x, y, alpha=0.3, color='lightblue')
            
            for interior in first_poly.interiors:
                ix, iy = interior.xy
                ax.plot(ix, iy, 'k-', linewidth=1, alpha=0.7)
                ax.fill(ix, iy, color='white', alpha=0.8)
        
        # 在质心附近生成一些测试点
        centroid = first_poly.centroid
        test_points = []
        for _ in range(20):
            x = np.random.normal(centroid.x, 10)
            y = np.random.normal(centroid.y, 10)
            point = Point(x, y)
            if first_poly.contains(point):
                test_points.append([x, y])
        
        if test_points:
            test_points = np.array(test_points)
            ax.scatter(test_points[:, 0], test_points[:, 1], 
                      c='red', s=30, alpha=0.7, label=f'测试点 ({len(test_points)})')
        
        ax.set_xlabel('X (米)')
        ax.set_ylabel('Y (米)')
        ax.set_title(f'{first_parcel_id} - 地块测试')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig('simple_test_result.png', dpi=150, bbox_inches='tight')
        print("已保存测试结果到 simple_test_result.png")
        plt.close()
    
    return parcels

if __name__ == '__main__':
    parcels = test_parcel_loading()
    print("测试完成！")









