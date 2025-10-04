#!/usr/bin/env python3
"""
测试不同点数对分布均匀性的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
import csv
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入修复后的模拟器函数
from point_repulsion_simulator import (
    load_parcels_from_txt, init_points, compute_repulsion_forces,
    compute_mean_distance, adaptive_step_size, clip_norm, step
)

def run_density_test(num_points_list, target_dist=2.0):
    """
    测试不同点数对分布均匀性的影响
    
    Args:
        num_points_list: 要测试的点数列表
        target_dist: 目标距离
    """
    print("=== 密度对分布均匀性影响测试 ===")
    
    # 加载地块（只测试前3个地块）
    polygons = load_parcels_from_txt('parcel.txt')[:3]
    print(f"测试地块数: {len(polygons)}")
    
    # 配置参数（基于目标距离重标定）
    base_config = {
        "target_dist": target_dist,
        "target_distance_threshold": target_dist,
        "stop_threshold": min(50, max(num_points_list) // 2),  # 动态调整停止阈值
        "max_iter": 10000,
        "learning_rate": 0.25 * target_dist,
        "min_scale": 0.5,
        "max_scale": 2.0,
        "max_step": 0.2 * target_dist,
        "random_seed": 42,
        "force_cutoff": 3.0 * target_dist,
        "eps": 1e-6,
        "k_neighbors": 6,
    }
    
    results = {}
    
    for num_points in num_points_list:
        print(f"\n--- 测试点数: {num_points} ---")
        
        # 更新配置
        config = base_config.copy()
        config['stop_threshold'] = min(50, num_points // 2)
        
        parcel_results = []
        
        for i, poly in enumerate(polygons):
            print(f"  地块 {i}: 初始点数 {num_points}")
            
            # 初始化点
            points = init_points(poly, num_points, config['random_seed'] + i)
            
            # 运行模拟
            iteration = 0
            removed_count = 0
            max_iter = min(1000, config['max_iter'])  # 限制迭代次数用于测试
            
            while iteration < max_iter and len(points) > 0:
                iteration += 1
                
                new_points, removed_mask, stats = step(points, poly, config)
                points = new_points
                removed_count += np.sum(removed_mask)
                
                # 检查停止条件
                if removed_count >= config['stop_threshold']:
                    print(f"    步骤 {iteration}: 移除点数达到阈值 {config['stop_threshold']}")
                    break
                
                if stats['mean_distance'] >= config['target_distance_threshold']:
                    print(f"    步骤 {iteration}: 达到距离阈值 {config['target_distance_threshold']:.2f}")
                    break
            
            # 计算最终统计
            if len(points) > 0:
                final_mean_dist = compute_mean_distance(points, config['k_neighbors'])
                
                # 计算分布均匀性指标
                uniformity_metrics = calculate_uniformity_metrics(points, poly)
                
                parcel_result = {
                    'initial_points': num_points,
                    'final_points': len(points),
                    'removed_points': removed_count,
                    'iterations': iteration,
                    'final_mean_distance': final_mean_dist,
                    'uniformity_metrics': uniformity_metrics,
                    'final_coords': points.copy()
                }
            else:
                parcel_result = {
                    'initial_points': num_points,
                    'final_points': 0,
                    'removed_points': removed_count,
                    'iterations': iteration,
                    'final_mean_distance': 0,
                    'uniformity_metrics': {},
                    'final_coords': np.array([]).reshape(0, 2)
                }
            
            parcel_results.append(parcel_result)
            print(f"    结果: 剩余 {parcel_result['final_points']} 点, "
                  f"平均距离 {parcel_result['final_mean_distance']:.3f}")
        
        results[num_points] = parcel_results
    
    return results

def calculate_uniformity_metrics(points, polygon):
    """
    计算分布均匀性指标
    
    Args:
        points: 点坐标数组
        polygon: 多边形对象
        
    Returns:
        dict: 均匀性指标
    """
    if len(points) <= 1:
        return {'cv_distance': 0, 'area_coverage': 0, 'edge_distance_ratio': 0}
    
    # 1. 距离变异系数 (Coefficient of Variation)
    tree = cKDTree(points)
    k = min(6, len(points))
    distances, _ = tree.query(points, k=k)
    if k > 1:
        distances = distances[:, 1:]  # 去掉自己
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    cv_distance = std_dist / mean_dist if mean_dist > 0 else 0
    
    # 2. 面积覆盖率
    polygon_area = polygon.area
    # 简化的覆盖率：计算点的凸包面积与多边形面积的比值
    if len(points) >= 3:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        convex_area = hull.volume if len(points) == 2 else hull.volume
        area_coverage = convex_area / polygon_area
    else:
        area_coverage = 0
    
    # 3. 边缘距离比：点到边界距离与内部距离的比值
    centroid = polygon.centroid
    center_distances = np.linalg.norm(points - [centroid.x, centroid.y], axis=1)
    mean_center_dist = np.mean(center_distances)
    
    # 计算到边界的平均距离
    boundary_distances = []
    for point_coords in points:
        point = Point(point_coords)
        boundary_dist = polygon.boundary.distance(point)
        boundary_distances.append(boundary_dist)
    
    mean_boundary_dist = np.mean(boundary_distances)
    edge_distance_ratio = mean_boundary_dist / mean_center_dist if mean_center_dist > 0 else 0
    
    return {
        'cv_distance': cv_distance,           # 越小越均匀
        'area_coverage': area_coverage,       # 越大覆盖越好
        'edge_distance_ratio': edge_distance_ratio  # 适中为好
    }

def visualize_density_comparison(results, output_dir="density_test_outputs"):
    """
    可视化不同密度下的分布效果对比
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_points_list = sorted(results.keys())
    n_densities = len(num_points_list)
    n_parcels = len(results[num_points_list[0]])
    
    # 创建对比图
    fig, axes = plt.subplots(n_densities, n_parcels, 
                            figsize=(5*n_parcels, 4*n_densities))
    
    if n_densities == 1:
        axes = axes.reshape(1, -1)
    if n_parcels == 1:
        axes = axes.reshape(-1, 1)
    
    # 加载地块用于绘制边界
    polygons = load_parcels_from_txt('parcel.txt')[:n_parcels]
    
    for i, num_points in enumerate(num_points_list):
        for j, parcel_result in enumerate(results[num_points]):
            ax = axes[i, j]
            poly = polygons[j]
            
            # 绘制多边形边界
            if hasattr(poly, 'exterior'):
                x, y = poly.exterior.xy
                ax.plot(x, y, 'k-', linewidth=2)
                ax.fill(x, y, alpha=0.1, color='lightblue')
                
                for interior in poly.interiors:
                    ix, iy = interior.xy
                    ax.plot(ix, iy, 'k-', linewidth=1, alpha=0.7)
                    ax.fill(ix, iy, color='white', alpha=0.8)
            
            # 绘制点
            points = parcel_result['final_coords']
            if len(points) > 0:
                ax.scatter(points[:, 0], points[:, 1], c='red', s=8, alpha=0.7)
            
            # 设置标题
            metrics = parcel_result['uniformity_metrics']
            title = f'地块{j} - 初始{num_points}点\n'
            title += f'剩余: {parcel_result["final_points"]}, '
            title += f'平均距离: {parcel_result["final_mean_distance"]:.2f}\n'
            title += f'CV: {metrics.get("cv_distance", 0):.3f}'
            
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
    
    plt.suptitle('不同点数对分布均匀性的影响', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"density_comparison_{timestamp}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"密度对比图已保存: {output_path}")
    
    plt.show()
    
    return output_path

def analyze_density_effects(results):
    """
    分析密度对均匀性的影响
    """
    print("\n=== 密度效果分析 ===")
    
    num_points_list = sorted(results.keys())
    
    # 收集统计数据
    stats = {
        'num_points': [],
        'avg_remaining': [],
        'avg_mean_distance': [],
        'avg_cv_distance': [],
        'avg_area_coverage': [],
        'avg_edge_ratio': []
    }
    
    for num_points in num_points_list:
        parcel_results = results[num_points]
        
        remaining_points = [r['final_points'] for r in parcel_results]
        mean_distances = [r['final_mean_distance'] for r in parcel_results]
        cv_distances = [r['uniformity_metrics'].get('cv_distance', 0) for r in parcel_results]
        area_coverages = [r['uniformity_metrics'].get('area_coverage', 0) for r in parcel_results]
        edge_ratios = [r['uniformity_metrics'].get('edge_distance_ratio', 0) for r in parcel_results]
        
        stats['num_points'].append(num_points)
        stats['avg_remaining'].append(np.mean(remaining_points))
        stats['avg_mean_distance'].append(np.mean(mean_distances))
        stats['avg_cv_distance'].append(np.mean(cv_distances))
        stats['avg_area_coverage'].append(np.mean(area_coverages))
        stats['avg_edge_ratio'].append(np.mean(edge_ratios))
    
    # 创建分析图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 剩余点数 vs 初始点数
    axes[0, 0].plot(stats['num_points'], stats['avg_remaining'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('初始点数')
    axes[0, 0].set_ylabel('平均剩余点数')
    axes[0, 0].set_title('剩余点数 vs 初始点数')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 平均距离 vs 初始点数
    axes[0, 1].plot(stats['num_points'], stats['avg_mean_distance'], 'ro-', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=2.0, color='gray', linestyle='--', alpha=0.7, label='目标距离')
    axes[0, 1].set_xlabel('初始点数')
    axes[0, 1].set_ylabel('平均邻近距离')
    axes[0, 1].set_title('平均距离 vs 初始点数')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 距离变异系数 vs 初始点数
    axes[1, 0].plot(stats['num_points'], stats['avg_cv_distance'], 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('初始点数')
    axes[1, 0].set_ylabel('距离变异系数 (CV)')
    axes[1, 0].set_title('分布均匀性 vs 初始点数\n(越小越均匀)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 面积覆盖率 vs 初始点数
    axes[1, 1].plot(stats['num_points'], stats['avg_area_coverage'], 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('初始点数')
    axes[1, 1].set_ylabel('面积覆盖率')
    axes[1, 1].set_title('覆盖效果 vs 初始点数\n(越大覆盖越好)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存分析图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_path = f"density_test_outputs/density_analysis_{timestamp}.png"
    os.makedirs("density_test_outputs", exist_ok=True)
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    print(f"密度分析图已保存: {analysis_path}")
    
    plt.show()
    
    # 打印分析结论
    print("\n=== 分析结论 ===")
    print(f"测试点数范围: {min(num_points_list)} - {max(num_points_list)}")
    print(f"平均剩余点数: {stats['avg_remaining']}")
    print(f"平均距离: {[f'{d:.3f}' for d in stats['avg_mean_distance']]}")
    print(f"距离变异系数: {[f'{cv:.3f}' for cv in stats['avg_cv_distance']]}")
    print(f"面积覆盖率: {[f'{ac:.3f}' for ac in stats['avg_area_coverage']]}")
    
    # 找出最佳密度
    min_cv_idx = np.argmin(stats['avg_cv_distance'])
    best_density = stats['num_points'][min_cv_idx]
    print(f"\n最佳分布均匀性: {best_density} 点 (CV最小: {stats['avg_cv_distance'][min_cv_idx]:.3f})")
    
    return stats

def main():
    """主函数"""
    # 测试不同点数：50, 100, 200, 300
    num_points_list = [50, 100, 200, 300]
    
    print("开始密度影响测试...")
    results = run_density_test(num_points_list, target_dist=2.0)
    
    print("\n生成可视化对比...")
    visualize_density_comparison(results)
    
    print("\n分析密度效果...")
    stats = analyze_density_effects(results)
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()

