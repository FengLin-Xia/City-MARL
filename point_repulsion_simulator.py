#!/usr/bin/env python3
"""
可配置的点排斥模拟器
在多个多边形地块内进行点排斥模拟，支持自适应步长和边界约束
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
config = {
    "parcel_files": ["parcel.txt"],  # 多边形文件路径
    "num_points": 110,               # 每个地块初始点数
    "target_dist": 2.0,              # 目标平均点距
    "target_distance_threshold": 2.0, # 当平均距离达到此值时停止该地块
    "stop_threshold": 50,            # 被移除点数达到此值时停止该地块
    "max_iter": 10000,               # 最大迭代步
    "learning_rate": 0.5,            # 学习率/步长基数（会在main中按target_dist重标定）
    "min_scale": 0.5,                # 自适应步长下限
    "max_scale": 2.0,                # 自适应步长上限
    "max_step": 1.0,                 # 单步最大位移裁剪（会在main中按target_dist重标定）
    "random_seed": 42,               # 随机种子
    "viz_every": 50,                 # 可视化/快照频率（步）
    "export_dir": "./repulsion_outputs",  # 导出目录
    "force_cutoff": 5.0,             # 力的截断距离（会在main中按target_dist重标定）
    "eps": 1e-6,                     # 数值稳定性参数
    "k_neighbors": 6,                # 用于计算平均距离的近邻数
}

# ==================== 核心函数 ====================

def load_polygon(file_path):
    """
    从文件读取多边形坐标并构造Polygon对象
    
    Args:
        file_path: 文件路径
        
    Returns:
        shapely.geometry.Polygon: 多边形对象
    """
    coords = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and line not in ['[', ']']:
                try:
                    # 支持逗号分隔或空格分隔
                    if ',' in line:
                        parts = [float(x.strip()) for x in line.split(',') if x.strip()]
                    else:
                        parts = [float(x.strip()) for x in line.split() if x.strip()]
                    
                    if len(parts) >= 2:
                        coords.append((parts[0], parts[1]))
                except ValueError:
                    continue
    
    if len(coords) < 3:
        raise ValueError(f"文件 {file_path} 中有效坐标点少于3个")
    
    # 检查是否闭合，如不闭合则自动闭合
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    
    return Polygon(coords)

def load_parcels_from_txt(file_path):
    """
    从parcel.txt格式文件读取多个多边形
    
    Args:
        file_path: 文件路径
        
    Returns:
        list: Polygon对象列表
    """
    polygons = []
    current_parcel = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '[':
                if current_parcel and len(current_parcel) >= 3:
                    try:
                        poly = Polygon(current_parcel)
                        if poly.is_valid:
                            polygons.append(poly)
                    except Exception as e:
                        print(f"警告：地块无效: {e}")
                current_parcel = []
            elif line == ']':
                if current_parcel and len(current_parcel) >= 3:
                    try:
                        poly = Polygon(current_parcel)
                        if poly.is_valid:
                            polygons.append(poly)
                    except Exception as e:
                        print(f"警告：地块无效: {e}")
                current_parcel = []
            elif line and not line.startswith('#') and line not in ['[', ']']:
                try:
                    coords = [float(x.strip()) for x in line.split(',')]
                    if len(coords) >= 2:
                        current_parcel.append((coords[0], coords[1]))
                except ValueError:
                    continue
    
    print(f"成功加载 {len(polygons)} 个地块")
    return polygons

def init_points(polygon, num_points, seed):
    """
    在多边形内均匀初始化点集
    
    Args:
        polygon: 多边形对象
        num_points: 点数
        seed: 随机种子
        
    Returns:
        numpy.ndarray: 形状为[N, 2]的点坐标数组
    """
    np.random.seed(seed)
    
    # 获取多边形边界框
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds
    
    points = []
    attempts = 0
    max_attempts = num_points * 200  # 提高尝试次数
    
    # 在多边形内均匀拒绝采样
    while len(points) < num_points and attempts < max_attempts:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        point = Point(x, y)
        
        if polygon.contains(point):
            points.append([x, y])
        
        attempts += 1
    
    return np.array(points)

def compute_repulsion_forces(points, force_cutoff, eps):
    """
    计算点之间的排斥力
    
    Args:
        points: 点坐标数组 [N, 2]
        force_cutoff: 力截断距离
        eps: 数值稳定性参数
        
    Returns:
        numpy.ndarray: 合力数组 [N, 2]
    """
    n = len(points)
    if n <= 1:
        return np.zeros_like(points)
    
    # 计算所有点对的距离向量
    diff = points[:, None, :] - points[None, :, :]  # [N, N, 2]
    dist = np.linalg.norm(diff, axis=2)  # [N, N]
    
    # 避免除零
    dist = np.maximum(dist, eps)
    
    # 计算排斥力（1/r^2模型）
    mask = dist < force_cutoff  # 只在截断距离内计算力
    force_magnitude = np.where(mask, 1.0 / (dist**2), 0.0)
    
    # 单位方向向量
    direction = diff / dist[:, :, None]  # [N, N, 2]
    
    # 计算合力
    forces = np.sum(force_magnitude[:, :, None] * direction, axis=1)  # [N, 2]
    
    return forces

def compute_mean_distance(points, k_neighbors):
    """
    计算平均邻近距离
    
    Args:
        points: 点坐标数组
        k_neighbors: 近邻数
        
    Returns:
        float: 平均距离
    """
    if len(points) <= 1:
        return 0.0
    
    tree = cKDTree(points)
    
    # 查找k个最近邻（包括自己）
    k = min(k_neighbors + 1, len(points))
    distances, _ = tree.query(points, k=k)
    
    # 排除自己（距离为0）
    if k > 1:
        distances = distances[:, 1:]  # 去掉第一列（自己）
    
    return np.mean(distances)

def adaptive_step_size(mean_dist, target_dist, learning_rate, min_scale, max_scale):
    """
    计算自适应步长
    
    Args:
        mean_dist: 当前平均距离
        target_dist: 目标距离
        learning_rate: 学习率
        min_scale: 最小缩放因子
        max_scale: 最大缩放因子
        
    Returns:
        float: 步长
    """
    if mean_dist <= 0:
        return learning_rate
    
    # 反向缩放：太挤时放大步长，太松时缩小步长
    scale = np.clip(target_dist / mean_dist, min_scale, max_scale)
    return learning_rate * scale

def clip_norm(forces, max_step):
    """
    裁剪力的模长
    
    Args:
        forces: 力数组 [N, 2]
        max_step: 最大步长
        
    Returns:
        numpy.ndarray: 裁剪后的力数组
    """
    norms = np.linalg.norm(forces, axis=1)
    scale = np.minimum(1.0, max_step / (norms + config['eps']))
    return forces * scale[:, None]

def step(points, polygon, config):
    """
    执行一步迭代
    
    Args:
        points: 当前点坐标数组
        polygon: 多边形对象
        config: 配置字典
        
    Returns:
        tuple: (更新后的点坐标, 被移除点的掩码, 统计信息)
    """
    if len(points) == 0:
        return points, np.array([]), {}
    
    # 计算排斥力
    forces = compute_repulsion_forces(points, config['force_cutoff'], config['eps'])
    
    # 计算当前平均距离
    mean_dist = compute_mean_distance(points, config['k_neighbors'])
    
    # 自适应步长
    step_size = adaptive_step_size(
        mean_dist, config['target_dist'], 
        config['learning_rate'], config['min_scale'], config['max_scale']
    )
    
    # 直接使用原始合力的强弱信息，不进行单位化
    displacement = step_size * forces
    
    # 裁剪位移防止发散
    displacement = clip_norm(displacement, config['max_step'])
    
    # 更新位置
    new_points = points + displacement
    
    # 边界检测和移除
    removed_mask = np.zeros(len(new_points), dtype=bool)
    valid_points = []
    
    for i, point_coords in enumerate(new_points):
        point = Point(point_coords)
        if polygon.contains(point):
            valid_points.append(point_coords)
        else:
            removed_mask[i] = True
    
    valid_points = np.array(valid_points) if valid_points else np.array([]).reshape(0, 2)
    
    # 统计信息
    force_norms = np.linalg.norm(forces, axis=1)
    stats = {
        'mean_distance': mean_dist,
        'step_size': step_size,
        'removed_count': np.sum(removed_mask),
        'remaining_count': len(valid_points),
        'force_norm': np.mean(force_norms)
    }
    
    return valid_points, removed_mask, stats

def simulate(polygons, config):
    """
    执行完整的模拟过程 - 每个地块独立运行
    
    Args:
        polygons: 多边形列表
        config: 配置字典
        
    Returns:
        dict: 模拟结果
    """
    print(f"开始模拟，共 {len(polygons)} 个地块")
    print(f"配置: 初始点数={config['num_points']}, 目标距离={config['target_dist']}, "
          f"距离阈值={config['target_distance_threshold']}")
    
    # 初始化
    all_points = []
    all_removed_counts = []
    parcel_stopped = []  # 记录每个地块是否已停止
    
    for i, poly in enumerate(polygons):
        points = init_points(poly, config['num_points'], config['random_seed'] + i)
        all_points.append(points)
        all_removed_counts.append(0)
        parcel_stopped.append(False)
        print(f"地块 {i}: 初始化 {len(points)} 个点")
    
    # 创建输出目录
    os.makedirs(config['export_dir'], exist_ok=True)
    
    # 迭代
    iteration = 0
    total_stopped = 0
    
    while iteration < config['max_iter'] and total_stopped < len(polygons):
        iteration += 1
        
        # 对每个未停止的地块执行一步
        for i, (poly, points, stopped) in enumerate(zip(polygons, all_points, parcel_stopped)):
            if stopped or len(points) == 0:
                continue
            
            new_points, removed_mask, stats = step(points, poly, config)
            all_points[i] = new_points
            all_removed_counts[i] += np.sum(removed_mask)
            
            # 优先检查移除点数阈值
            if all_removed_counts[i] >= config['stop_threshold']:
                parcel_stopped[i] = True
                total_stopped += 1
                print(f"步骤 {iteration}: 地块 {i} 被移除点数达到阈值 {config['stop_threshold']}, "
                      f"当前剩余: {len(new_points)}")
                continue
            
            # 如果点为空，也判停
            if len(new_points) == 0:
                parcel_stopped[i] = True
                total_stopped += 1
                print(f"步骤 {iteration}: 地块 {i} 所有点已被移除，标记为停止")
                continue
            
            # 可选：保留平均距离阈值作为次级停止条件
            if stats['mean_distance'] >= config['target_distance_threshold']:
                parcel_stopped[i] = True
                total_stopped += 1
                print(f"步骤 {iteration}: 地块 {i} 达到平均距离阈值 {config['target_distance_threshold']}, "
                      f"当前平均距离: {stats['mean_distance']:.3f}, 剩余点数: {len(new_points)}")
        
        # 打印进度
        if iteration % 50 == 0:
            remaining_counts = [len(points) for points in all_points]
            stopped_info = [f"已停止" if stopped else f"{remaining_counts[i]}" 
                           for i, stopped in enumerate(parcel_stopped)]
            print(f"步骤 {iteration}: 各地块剩余点数 {stopped_info}, 已停止地块: {total_stopped}/{len(polygons)}")
        
        # 可视化快照
        if iteration % config['viz_every'] == 0:
            visualize_state(polygons, all_points, iteration, config, parcel_stopped)
    
    # 最终结果
    results = {
        'iterations': iteration,
        'total_stopped': total_stopped,
        'parcel_stopped': parcel_stopped,
        'final_points': all_points,
        'removed_counts': all_removed_counts,
        'remaining_counts': [len(points) for points in all_points]
    }
    
    # 导出结果
    export_results(polygons, results, config)
    
    return results

def visualize_state(polygons, all_points, iteration, config, parcel_stopped=None):
    """
    可视化当前状态
    
    Args:
        polygons: 多边形列表
        all_points: 所有地块的点列表
        iteration: 当前迭代步数
        config: 配置字典
        parcel_stopped: 地块停止状态列表
    """
    n_parcels = len(polygons)
    cols = min(3, n_parcels)
    rows = (n_parcels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_parcels == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, (poly, points) in enumerate(zip(polygons, all_points)):
        ax = axes[i]
        
        # 绘制多边形
        if hasattr(poly, 'exterior'):
            x, y = poly.exterior.xy
            ax.plot(x, y, 'k-', linewidth=2, label='边界')
            ax.fill(x, y, alpha=0.1, color='lightblue')
            
            for interior in poly.interiors:
                ix, iy = interior.xy
                ax.plot(ix, iy, 'k-', linewidth=1, alpha=0.7)
                ax.fill(ix, iy, color='white', alpha=0.8)
        
        # 绘制点
        if len(points) > 0:
            # 根据停止状态选择颜色
            if parcel_stopped and parcel_stopped[i]:
                color = 'green'  # 已停止的地块用绿色
                title_suffix = " (已停止)"
            else:
                color = 'red'    # 运行中的地块用红色
                title_suffix = ""
            
            ax.scatter(points[:, 0], points[:, 1], c=color, s=10, alpha=0.7)
            
            # 计算并显示平均距离
            mean_dist = compute_mean_distance(points, config['k_neighbors'])
            ax.set_title(f'地块 {i} (步骤 {iteration}){title_suffix}\n剩余: {len(points)}, 平均距离: {mean_dist:.2f}')
        else:
            status = " (已停止)" if (parcel_stopped and parcel_stopped[i]) else ""
            ax.set_title(f'地块 {i} (步骤 {iteration}){status}\n无剩余点')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # 隐藏多余的子图
    for i in range(n_parcels, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"step_{iteration:04d}.png"
    filepath = os.path.join(config['export_dir'], filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存可视化快照: {filepath}")

def export_results(polygons, results, config):
    """
    导出模拟结果
    
    Args:
        polygons: 多边形列表
        results: 模拟结果
        config: 配置字典
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 导出每个地块的点坐标
    for i, (poly, points) in enumerate(zip(polygons, results['final_points'])):
        if len(points) > 0:
            filename = f"parcel_{i}_final_points_{timestamp}.csv"
            filepath = os.path.join(config['export_dir'], filename)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y'])
                writer.writerows(points)
            
            print(f"已导出地块 {i} 的点坐标: {filepath}")
    
    # 导出统计摘要
    summary_file = os.path.join(config['export_dir'], f"simulation_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== 点排斥模拟结果摘要 ===\n\n")
        f.write(f"配置参数:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\n模拟结果:\n")
        f.write(f"  总迭代步数: {results['iterations']}\n")
        f.write(f"  已停止地块数: {results['total_stopped']}/{len(polygons)}\n")
        f.write(f"\n各地块统计:\n")
        for i, (remaining, removed, stopped) in enumerate(zip(
            results['remaining_counts'], 
            results['removed_counts'],
            results['parcel_stopped']
        )):
            status = "已停止" if stopped else "运行中"
            f.write(f"  地块 {i}: 剩余 {remaining} 点, 移除 {removed} 点, 状态: {status}\n")
            
            # 计算最终平均距离
            if len(results['final_points'][i]) > 0:
                mean_dist = compute_mean_distance(results['final_points'][i], config['k_neighbors'])
                f.write(f"    最终平均邻近距离: {mean_dist:.3f}\n")
    
    print(f"已导出统计摘要: {summary_file}")

def main():
    """主函数"""
    print("=== 点排斥模拟器 ===")
    
    # 根据目标距离重标定参数
    target_dist = config['target_dist']
    config['learning_rate'] = 0.25 * target_dist
    config['max_step'] = 0.2 * target_dist
    config['force_cutoff'] = 3.0 * target_dist
    
    print(f"配置参数: {config}")
    print(f"重标定后的关键参数: learning_rate={config['learning_rate']:.3f}, "
          f"max_step={config['max_step']:.3f}, force_cutoff={config['force_cutoff']:.3f}")
    
    # 加载多边形
    polygons = []
    for file_path in config['parcel_files']:
        if file_path.endswith('parcel.txt'):
            # 特殊处理parcel.txt格式
            polys = load_parcels_from_txt(file_path)
            polygons.extend(polys)
        else:
            # 处理单个多边形文件
            poly = load_polygon(file_path)
            polygons.append(poly)
    
    if not polygons:
        print("错误: 未找到有效的多边形")
        return
    
    print(f"共加载 {len(polygons)} 个地块")
    
    # 执行模拟
    results = simulate(polygons, config)
    
    # 打印最终结果
    print("\n=== 模拟完成 ===")
    print(f"总迭代步数: {results['iterations']}")
    print(f"已停止地块数: {results['total_stopped']}/{len(polygons)}")
    print("各地块最终状态:")
    for i, (remaining, removed, stopped) in enumerate(zip(
        results['remaining_counts'], 
        results['removed_counts'],
        results['parcel_stopped']
    )):
        status = "已停止" if stopped else "运行中"
        print(f"  地块 {i}: 剩余 {remaining} 点, 移除 {removed} 点, 状态: {status}")
        
        # 计算并显示最终平均距离
        if len(results['final_points'][i]) > 0:
            mean_dist = compute_mean_distance(results['final_points'][i], config['k_neighbors'])
            print(f"    最终平均邻近距离: {mean_dist:.3f}")
    
    print(f"\n结果已导出到: {config['export_dir']}")

if __name__ == "__main__":
    main()
