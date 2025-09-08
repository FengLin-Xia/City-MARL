#!/usr/bin/env python3
"""
调试OBJ文件加载器
"""

import numpy as np

def debug_load_obj(obj_filepath: str):
    """调试OBJ文件加载"""
    try:
        vertices = []
        faces = []
        
        print(f"🔍 开始加载OBJ文件: {obj_filepath}")
        
        with open(obj_filepath, 'r') as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line_count % 1000 == 0:
                    print(f"   已处理 {line_count} 行...")
                
                if line.startswith('v '):  # 顶点
                    parts = line.strip().split()[1:]
                    if len(parts) >= 3:
                        try:
                            vertex = [float(parts[0]), float(parts[1]), float(parts[2])]
                            vertices.append(vertex)
                        except ValueError as e:
                            print(f"   第 {line_count} 行顶点数据错误: {line.strip()} - {e}")
                            continue
                            
                elif line.startswith('f '):  # 面
                    parts = line.strip().split()[1:]
                    if len(parts) >= 3:
                        try:
                            # 处理多边形面，只取前三个顶点
                            face = []
                            for part in parts[:3]:  # 只取前3个顶点
                                vertex_part = part.split('/')[0]  # 只取顶点索引
                                vertex_idx = int(vertex_part) - 1  # OBJ索引从1开始
                                face.append(vertex_idx)
                            
                            if len(face) == 3:  # 确保有3个有效顶点
                                faces.append(face)
                            else:
                                print(f"   第 {line_count} 行面数据不完整: {line.strip()}")
                                
                        except ValueError as e:
                            print(f"   第 {line_count} 行面数据错误: {line.strip()} - {e}")
                            continue
        
        print(f"✅ 文件加载完成")
        print(f"   总行数: {line_count}")
        print(f"   顶点数: {len(vertices)}")
        print(f"   面数: {len(faces)}")
        
        if vertices:
            vertices_array = np.array(vertices)
            print(f"   顶点范围: X[{np.min(vertices_array[:, 0]):.2f}, {np.max(vertices_array[:, 0]):.2f}], Y[{np.min(vertices_array[:, 1]):.2f}, {np.max(vertices_array[:, 1]):.2f}]")
            print(f"   高程范围: Z[{np.min(vertices_array[:, 2]):.2f}, {np.max(vertices_array[:, 2]):.2f}]")
        
        return vertices, faces
        
    except Exception as e:
        print(f"❌ 加载OBJ文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    obj_file = "data/terrain/terrain.obj"
    vertices, faces = debug_load_obj(obj_file)
