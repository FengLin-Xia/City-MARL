#!/usr/bin/env python3
"""
测试地块解析功能
"""

import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

def load_parcels_from_txt(file_path: str):
    """解析parcel.txt文件"""
    parcels = {}
    current_parcel = []
    parcel_id = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[':
                # 开始新地块
                if current_parcel:
                    # 保存前一个地块
                    if len(current_parcel) >= 3:
                        try:
                            poly = Polygon(current_parcel)
                            if poly.is_valid:
                                parcels[f'parcel_{parcel_id}'] = poly
                                parcel_id += 1
                        except Exception as e:
                            print(f"警告：地块 {parcel_id} 无效: {e}")
                    current_parcel = []
            elif line == ']':
                # 地块结束
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
                # 解析坐标行
                try:
                    coords = [float(x.strip()) for x in line.split(',')]
                    if len(coords) >= 2:
                        current_parcel.append((coords[0], coords[1]))
                except ValueError:
                    continue
    
    print(f"成功加载 {len(parcels)} 个地块")
    return parcels

def visualize_parcels(parcels):
    """可视化地块"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(parcels)))
    
    for i, (parcel_id, poly) in enumerate(parcels.items()):
        # 绘制外边界
        if hasattr(poly, 'exterior'):
            x, y = poly.exterior.xy
            ax.plot(x, y, color=colors[i], linewidth=2, label=parcel_id)
            
            # 填充
            ax.fill(x, y, color=colors[i], alpha=0.3)
            
            # 绘制洞
            for interior in poly.interiors:
                ix, iy = interior.xy
                ax.plot(ix, iy, color=colors[i], linewidth=1, alpha=0.7)
                ax.fill(ix, iy, color='white', alpha=0.8)
    
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_title('地块边界可视化')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('parcels_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 加载地块
    parcels = load_parcels_from_txt('parcel.txt')
    
    # 打印地块信息
    for parcel_id, poly in parcels.items():
        print(f"{parcel_id}: 面积={poly.area:.2f} 平方米, 周长={poly.length:.2f} 米")
        print(f"  边界框: {poly.bounds}")
        print(f"  质心: ({poly.centroid.x:.2f}, {poly.centroid.y:.2f})")
        print()
    
    # 可视化
    visualize_parcels(parcels)





