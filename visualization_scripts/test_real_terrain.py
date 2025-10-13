#!/usr/bin/env python3
"""
测试真实地形的三角面填充效果
"""

import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_real_terrain_upload():
    """测试真实地形上传"""
    print("🧪 测试真实地形三角面填充...")
    
    # 检查是否有现有的地形文件
    terrain_files = [
        "data/terrain/terrain.obj",
        "data/terrain/terrain_direct_mesh.json",
        "data/terrain/terrain_direct_mesh_fixed.json"
    ]
    
    obj_file = None
    for file_path in terrain_files:
        if Path(file_path).exists():
            if file_path.endswith('.obj'):
                obj_file = file_path
                break
            elif file_path.endswith('.json'):
                # 如果有JSON文件，我们可以直接分析
                print(f"📊 发现现有地形数据: {file_path}")
                analyze_existing_terrain(file_path)
                return
    
    if not obj_file:
        print("❌ 没有找到OBJ地形文件")
        print("💡 请先在Blender中上传地形，或确保data/terrain/terrain.obj存在")
        return
    
    print(f"✅ 找到地形文件: {obj_file}")
    
    # 先分析OBJ文件的顶点分布
    print("\n🔍 分析OBJ文件顶点分布...")
    analyze_obj_vertices(obj_file)
    
    # 上传到Flask服务器
    try:
        with open(obj_file, 'rb') as f:
            files = {'file': f}
            data = {}  # 不需要有序边界，使用三角面填充
            
            print("🔄 上传到Flask服务器...")
            response = requests.post(
                "http://localhost:5000/upload_terrain",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 上传成功!")
            
            # 分析结果
            analyze_upload_result(result)
            
            return result
        else:
            print(f"❌ 上传失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Flask服务器")
        print("请确保服务器已启动: python main.py")
        return None
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        return None

def analyze_upload_result(result):
    """分析上传结果"""
    terrain_info = result.get('terrain_info', {})
    
    print(f"\n📊 地形分析结果:")
    print(f"   网格大小: {terrain_info.get('grid_size', 'N/A')}")
    print(f"   顶点数量: {terrain_info.get('vertices_count', 'N/A')}")
    print(f"   面数量: {terrain_info.get('faces_count', 'N/A')}")
    
    if 'mask' in terrain_info:
        mask = np.array(terrain_info['mask'])
        print(f"   掩码形状: {mask.shape}")
        print(f"   有效点数: {np.sum(mask)} / {mask.size}")
        print(f"   覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
        
        # 可视化结果
        visualize_terrain_result(terrain_info)
    
    if 'height_map' in terrain_info:
        height_map = np.array(terrain_info['height_map'])
        valid_heights = height_map[mask] if 'mask' in terrain_info else height_map
        print(f"   高程范围: [{np.min(valid_heights):.3f}, {np.max(valid_heights):.3f}]")
        print(f"   平均高程: {np.mean(valid_heights):.3f}")

def analyze_existing_terrain(json_file):
    """分析现有的地形数据"""
    print(f"📊 分析现有地形数据: {json_file}")
    
    with open(json_file, 'r') as f:
        terrain_data = json.load(f)
    
    if 'mask' in terrain_data:
        mask = np.array(terrain_data['mask'])
        print(f"   掩码形状: {mask.shape}")
        print(f"   有效点数: {np.sum(mask)} / {mask.size}")
        print(f"   覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
        
        # 可视化现有数据
        visualize_terrain_result(terrain_data)

def analyze_obj_vertices(obj_file):
    """分析OBJ文件的顶点分布"""
    print(f"🔍 分析OBJ文件顶点分布: {obj_file}")
    
    try:
        vertices = []
        faces = []
        
        with open(obj_file, 'r') as f:
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
        
        if not vertices:
            print("❌ 没有找到顶点数据")
            return
        
        vertices = np.array(vertices)
        print(f"✅ 顶点分析完成:")
        print(f"   顶点数量: {len(vertices)}")
        print(f"   面数量: {len(faces)}")
        print(f"   X范围: [{np.min(vertices[:, 0]):.3f}, {np.max(vertices[:, 0]):.3f}]")
        print(f"   Y范围: [{np.min(vertices[:, 1]):.3f}, {np.max(vertices[:, 1]):.3f}]")
        print(f"   Z范围: [{np.min(vertices[:, 2]):.3f}, {np.max(vertices[:, 2]):.3f}]")
        
        # 可视化顶点分布
        visualize_vertex_distribution(vertices, faces)
        
    except Exception as e:
        print(f"❌ 分析OBJ文件失败: {e}")

def visualize_vertex_distribution(vertices, faces):
    """可视化顶点分布"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. XY平面顶点分布
        axes[0, 0].scatter(vertices[:, 0], vertices[:, 1], c=vertices[:, 2], 
                          cmap='viridis', alpha=0.6, s=1)
        axes[0, 0].set_title('XY平面顶点分布 (颜色表示高程)')
        axes[0, 0].set_xlabel('X坐标')
        axes[0, 0].set_ylabel('Y坐标')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='高程')
        
        # 2. 高程分布直方图
        axes[0, 1].hist(vertices[:, 2], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(np.mean(vertices[:, 2]), color='red', linestyle='--', linewidth=2,
                          label=f'平均值: {np.mean(vertices[:, 2]):.2f}')
        axes[0, 1].set_title('顶点高程分布')
        axes[0, 1].set_xlabel('高程')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 坐标范围分析
        x_range = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        y_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        z_range = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        
        ranges = ['X范围', 'Y范围', 'Z范围']
        values = [x_range, y_range, z_range]
        colors = ['red', 'green', 'blue']
        
        bars = axes[1, 0].bar(ranges, values, color=colors, alpha=0.7)
        axes[1, 0].set_title('坐标范围分析')
        axes[1, 0].set_ylabel('范围大小')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 4. 统计信息
        stats_text = f"""
顶点统计信息:
总顶点数: {len(vertices)}
总面数: {len(faces)}
X范围: [{np.min(vertices[:, 0]):.3f}, {np.max(vertices[:, 0]):.3f}]
Y范围: [{np.min(vertices[:, 1]):.3f}, {np.max(vertices[:, 1]):.3f}]
Z范围: [{np.min(vertices[:, 2]):.3f}, {np.max(vertices[:, 2]):.3f}]
平均高程: {np.mean(vertices[:, 2]):.3f}
高程标准差: {np.std(vertices[:, 2]):.3f}
"""
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='center', fontfamily='monospace')
        axes[1, 1].set_title('统计信息')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig("vertex_analysis_result.png", dpi=300, bbox_inches='tight')
        print("✅ 顶点分析图已保存到: vertex_analysis_result.png")
        plt.show()
        
    except Exception as e:
        print(f"❌ 顶点可视化失败: {e}")

def visualize_terrain_result(terrain_info):
    """可视化地形结果"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 掩码
        if 'mask' in terrain_info:
            mask = np.array(terrain_info['mask'])
            im1 = axes[0, 0].imshow(mask.T, cmap='gray', origin='lower', aspect='equal')
            axes[0, 0].set_title('地形掩码')
            axes[0, 0].set_xlabel('X坐标')
            axes[0, 0].set_ylabel('Y坐标')
        
        # 2. 高度图
        if 'height_map' in terrain_info:
            height_map = np.array(terrain_info['height_map'])
            if 'mask' in terrain_info:
                # 应用掩码
                masked_height = np.where(mask, height_map, np.nan)
            else:
                masked_height = height_map
            
            im2 = axes[0, 1].imshow(masked_height.T, cmap='terrain', origin='lower', aspect='equal')
            axes[0, 1].set_title('地形高程图')
            axes[0, 1].set_xlabel('X坐标')
            axes[0, 1].set_ylabel('Y坐标')
            plt.colorbar(im2, ax=axes[0, 1], label='高程')
        
        # 3. 原始顶点分布
        if 'original_bounds' in terrain_info:
            bounds = terrain_info['original_bounds']
            # 这里我们只能显示边界框，因为原始顶点数据没有保存
            rect = plt.Rectangle((bounds['x_min'], bounds['y_min']), 
                               bounds['x_max'] - bounds['x_min'], 
                               bounds['y_max'] - bounds['y_min'], 
                               fill=False, edgecolor='red', linewidth=2)
            axes[0, 2].add_patch(rect)
            axes[0, 2].set_xlim(bounds['x_min'] - 100, bounds['x_max'] + 100)
            axes[0, 2].set_ylim(bounds['y_min'] - 100, bounds['y_max'] + 100)
            axes[0, 2].set_title('原始边界框')
            axes[0, 2].set_xlabel('X坐标')
            axes[0, 2].set_ylabel('Y坐标')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 高程分布
        if 'height_map' in terrain_info and 'mask' in terrain_info:
            valid_heights = height_map[mask]
            axes[1, 0].hist(valid_heights.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].axvline(np.mean(valid_heights), color='red', linestyle='--', linewidth=2, 
                              label=f'平均值: {np.mean(valid_heights):.2f}')
            axes[1, 0].set_title('高程分布')
            axes[1, 0].set_xlabel('高程')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 坐标范围分析
        if 'original_bounds' in terrain_info:
            bounds = terrain_info['original_bounds']
            x_range = bounds['x_max'] - bounds['x_min']
            y_range = bounds['y_max'] - bounds['y_min']
            z_range = bounds['z_max'] - bounds['z_min']
            
            ranges = ['X范围', 'Y范围', 'Z范围']
            values = [x_range, y_range, z_range]
            colors = ['red', 'green', 'blue']
            
            bars = axes[1, 1].bar(ranges, values, color=colors, alpha=0.7)
            axes[1, 1].set_title('坐标范围分析')
            axes[1, 1].set_ylabel('范围大小')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1f}', ha='center', va='bottom')
        
        # 6. 统计信息
        if 'mask' in terrain_info:
            mask = np.array(terrain_info['mask'])
            stats_text = f"""
地形统计信息:
网格大小: {terrain_info.get('grid_size', 'N/A')}
顶点数量: {terrain_info.get('vertices_count', 'N/A')}
面数量: {terrain_info.get('faces_count', 'N/A')}
有效点数: {np.sum(mask)} / {mask.size}
覆盖率: {np.sum(mask)/mask.size*100:.1f}%
"""
            axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                           fontsize=10, verticalalignment='center', fontfamily='monospace')
            axes[1, 2].set_title('统计信息')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig("terrain_analysis_result.png", dpi=300, bbox_inches='tight')
        print("✅ 地形分析图已保存到: terrain_analysis_result.png")
        plt.show()
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

if __name__ == "__main__":
    print("🚀 测试真实地形三角面填充")
    print("=" * 50)
    
    result = test_real_terrain_upload()
    
    if result:
        print("\n✅ 测试完成")
    else:
        print("\n❌ 测试失败")
