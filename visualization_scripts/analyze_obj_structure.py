#!/usr/bin/env python3
"""
OBJ文件结构分析脚本
深入分析顶点分布、边界特征和可能的问题
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_obj_file(obj_filepath: str):
    """加载OBJ文件并返回顶点和面数据"""
    vertices = []
    faces = []
    
    with open(obj_filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):  # 顶点
                parts = line.strip().split()[1:]
                if len(parts) >= 3:
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif line.startswith('f '):  # 面
                parts = line.strip().split()[1:]
                if len(parts) >= 3:
                    face = [int(part.split('/')[0]) - 1 for part in parts[:3]]
                    faces.append(face)
    
    return np.array(vertices), np.array(faces)


def analyze_vertex_distribution(vertices):
    """分析顶点分布特征"""
    print("=== 顶点分布分析 ===")
    
    # 基本统计
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    z_coords = vertices[:, 2]
    
    print(f"顶点总数: {len(vertices)}")
    print(f"X坐标范围: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
    print(f"Y坐标范围: [{y_coords.min():.3f}, {y_coords.max():.3f}]")
    print(f"Z坐标范围: [{z_coords.min():.3f}, {z_coords.max():.3f}]")
    
    # 检查是否有重复顶点
    unique_vertices = np.unique(vertices, axis=0)
    print(f"唯一顶点数: {len(unique_vertices)}")
    print(f"重复顶点数: {len(vertices) - len(unique_vertices)}")
    
    # 检查边界顶点
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # 找到边界上的顶点
    boundary_vertices = []
    tolerance = 0.001  # 容差
    
    for i, vertex in enumerate(vertices):
        x, y = vertex[0], vertex[1]
        if (abs(x - x_min) < tolerance or abs(x - x_max) < tolerance or
            abs(y - y_min) < tolerance or abs(y - y_max) < tolerance):
            boundary_vertices.append(i)
    
    print(f"边界顶点数: {len(boundary_vertices)}")
    
    return {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'z_coords': z_coords,
        'unique_vertices': unique_vertices,
        'boundary_vertices': boundary_vertices
    }


def analyze_face_structure(faces, vertices):
    """分析面结构"""
    print("\n=== 面结构分析 ===")
    
    print(f"面总数: {len(faces)}")
    
    # 检查每个顶点被多少个面使用
    vertex_usage = np.zeros(len(vertices), dtype=int)
    for face in faces:
        for vertex_idx in face:
            vertex_usage[vertex_idx] += 1
    
    print(f"平均每个顶点被使用次数: {vertex_usage.mean():.2f}")
    print(f"最少使用次数: {vertex_usage.min()}")
    print(f"最多使用次数: {vertex_usage.max()}")
    
    # 找到边界边（只属于一个面的边）
    edge_count = {}
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1) % 3]
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    print(f"边界边数: {len(boundary_edges)}")
    
    return {
        'vertex_usage': vertex_usage,
        'boundary_edges': boundary_edges
    }


def visualize_analysis(vertices, faces, analysis_data):
    """可视化分析结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('OBJ文件结构分析', fontsize=16)
    
    x_coords = analysis_data['x_coords']
    y_coords = analysis_data['y_coords']
    z_coords = analysis_data['z_coords']
    
    # 1. 顶点分布散点图
    scatter = axes[0, 0].scatter(x_coords, y_coords, c=z_coords, s=1, cmap='terrain', alpha=0.6)
    axes[0, 0].set_title('顶点分布（颜色表示高程）')
    axes[0, 0].set_xlabel('X坐标')
    axes[0, 0].set_ylabel('Y坐标')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='高程')
    
    # 2. 边界顶点高亮
    boundary_vertices = analysis_data['boundary_vertices']
    axes[0, 1].scatter(x_coords, y_coords, s=1, alpha=0.3, c='gray')
    if boundary_vertices:
        boundary_x = x_coords[boundary_vertices]
        boundary_y = y_coords[boundary_vertices]
        axes[0, 1].scatter(boundary_x, boundary_y, s=10, c='red', alpha=0.8, label='边界顶点')
        axes[0, 1].legend()
    axes[0, 1].set_title('边界顶点识别')
    axes[0, 1].set_xlabel('X坐标')
    axes[0, 1].set_ylabel('Y坐标')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 凸包边界
    points_2d = np.column_stack((x_coords, y_coords))
    hull = ConvexHull(points_2d)
    axes[0, 2].scatter(x_coords, y_coords, s=1, alpha=0.3, c='gray')
    for simplex in hull.simplices:
        p1 = points_2d[simplex[0]]
        p2 = points_2d[simplex[1]]
        axes[0, 2].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
    axes[0, 2].set_title('凸包边界')
    axes[0, 2].set_xlabel('X坐标')
    axes[0, 2].set_ylabel('Y坐标')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 顶点使用频率
    vertex_usage = analysis_data['vertex_usage']
    axes[1, 0].hist(vertex_usage, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('顶点使用频率分布')
    axes[1, 0].set_xlabel('使用次数')
    axes[1, 0].set_ylabel('顶点数量')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 高程分布
    axes[1, 1].hist(z_coords, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title('高程分布')
    axes[1, 1].set_xlabel('高程')
    axes[1, 1].set_ylabel('顶点数量')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 边界边可视化
    boundary_edges = analysis_data['boundary_edges']
    axes[1, 2].scatter(x_coords, y_coords, s=1, alpha=0.3, c='gray')
    if boundary_edges:
        for edge in boundary_edges[:100]:  # 只显示前100条边界边
            v1, v2 = edge
            p1 = vertices[v1]
            p2 = vertices[v2]
            axes[1, 2].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=1, alpha=0.7)
    axes[1, 2].set_title('边界边（前100条）')
    axes[1, 2].set_xlabel('X坐标')
    axes[1, 2].set_ylabel('Y坐标')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def check_for_issues(vertices, faces, analysis_data):
    """检查可能的问题"""
    print("\n=== 问题诊断 ===")
    
    # 检查1: 是否有重复顶点
    if len(vertices) != len(analysis_data['unique_vertices']):
        print("⚠️  发现重复顶点，这可能导致边界计算问题")
    
    # 检查2: 边界顶点数量是否合理
    boundary_count = len(analysis_data['boundary_vertices'])
    total_vertices = len(vertices)
    boundary_ratio = boundary_count / total_vertices
    
    print(f"边界顶点比例: {boundary_ratio:.2%}")
    if boundary_ratio > 0.3:
        print("⚠️  边界顶点比例过高，可能存在问题")
    elif boundary_ratio < 0.01:
        print("⚠️  边界顶点比例过低，可能存在问题")
    
    # 检查3: 检查是否有异常的高程值
    z_coords = analysis_data['z_coords']
    z_mean = np.mean(z_coords)
    z_std = np.std(z_coords)
    
    print(f"高程均值: {z_mean:.3f}")
    print(f"高程标准差: {z_std:.3f}")
    
    # 检查4: 检查坐标范围是否合理
    x_range = analysis_data['x_coords'].max() - analysis_data['x_coords'].min()
    y_range = analysis_data['y_coords'].max() - analysis_data['y_coords'].min()
    
    print(f"X坐标范围: {x_range:.3f}")
    print(f"Y坐标范围: {y_range:.3f}")
    
    if x_range < 0.1 or y_range < 0.1:
        print("⚠️  坐标范围过小，可能导致精度问题")
    
    # 检查5: 检查边界边的连接性
    boundary_edges = analysis_data['boundary_edges']
    if boundary_edges:
        # 检查边界边是否能形成闭合路径
        edge_vertices = set()
        for edge in boundary_edges:
            edge_vertices.update(edge)
        
        print(f"边界边涉及的唯一顶点数: {len(edge_vertices)}")
        print(f"边界边数: {len(boundary_edges)}")
        
        if len(boundary_edges) < len(edge_vertices):
            print("⚠️  边界边数少于边界顶点数，可能存在断开")


def main():
    """主函数"""
    obj_filepath = "uploads/terrain.obj"
    
    if not os.path.exists(obj_filepath):
        print(f"❌ OBJ文件不存在: {obj_filepath}")
        return
    
    print("🔍 开始分析OBJ文件结构...")
    print("=" * 50)
    
    # 加载OBJ文件
    vertices, faces = load_obj_file(obj_filepath)
    
    # 分析顶点分布
    vertex_analysis = analyze_vertex_distribution(vertices)
    
    # 分析面结构
    face_analysis = analyze_face_structure(faces, vertices)
    
    # 合并分析数据
    analysis_data = {**vertex_analysis, **face_analysis}
    
    # 检查问题
    check_for_issues(vertices, faces, analysis_data)
    
    # 可视化分析结果
    print("\n📊 生成可视化分析...")
    visualize_analysis(vertices, faces, analysis_data)
    
    print("\n✅ 分析完成！")


if __name__ == "__main__":
    main()
