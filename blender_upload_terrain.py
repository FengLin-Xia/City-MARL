#!/usr/bin/env python3
"""
Blender脚本 - 上传地形OBJ文件到Flask服务器
在Blender中运行此脚本，将当前选中的地形导出并上传
"""

import bpy
import bmesh
import json
import requests
import os
from pathlib import Path

def export_selected_as_obj(filepath):
    """导出选中的对象为OBJ文件"""
    # 确保选中了对象
    if not bpy.context.selected_objects:
        print("❌ 没有选中任何对象")
        return None
    
    print("🔄 尝试导出OBJ文件...")
    
    # 确保Object_1被正确选中
    obj_1 = bpy.data.objects.get("Object_1")
    if obj_1:
        # 取消所有选择
        bpy.ops.object.select_all(action='DESELECT')
        # 选择Object_1
        obj_1.select_set(True)
        bpy.context.view_layer.objects.active = obj_1
        print(f"✅ 确保Object_1被选中: {obj_1.name}")
    
    # 提取有序边界信息
    ordered_boundary = None
    if obj_1:
        print("🔄 提取有序边界信息...")
        ordered_boundary = extract_ordered_boundary(obj_1)
    
    # 方法1：尝试使用标准的OBJ导出操作符
    try:
        bpy.ops.export_scene.obj(
            filepath=filepath,
            use_selection=True,
            use_materials=False,
            use_triangles=True,
            use_normals=True,
            use_uvs=False
        )
        print(f"✅ 标准OBJ导出成功: {filepath}")
        # 计算边界信息
        if obj_1:
            boundary = calculate_boundary(obj_1)
            return filepath, boundary, ordered_boundary
        return filepath
        
    except AttributeError:
        print("⚠️ 标准OBJ导出操作符不可用")
    except Exception as e:
        print(f"❌ 标准OBJ导出失败: {e}")
    
    # 方法2：尝试使用export_mesh.obj
    print("🔄 尝试备用导出方法...")
    result = try_alternative_export(filepath)
    if result:
        if isinstance(result, tuple):
            return result[0], result[1], ordered_boundary
        else:
            return result, None, ordered_boundary
    
    # 方法3：手动导出
    print("🔄 尝试手动导出...")
    result = export_mesh_manually(filepath)
    if result:
        if isinstance(result, tuple):
            return result[0], result[1], ordered_boundary
        else:
            return result, None, ordered_boundary
    
    # 所有方法都失败了
    print("❌ 所有导出方法都失败了")
    print("💡 建议：")
    print("   1. 确保Blender版本支持OBJ导出")
    print("   2. 检查是否有插件冲突")
    print("   3. 尝试重启Blender")
    return None

def export_mesh_manually(filepath):
    """手动导出mesh为OBJ文件，包含边界信息"""
    try:
        # 优先使用Object_1，如果没有选中对象的话
        if not bpy.context.selected_objects:
            obj_1 = bpy.data.objects.get("Object_1")
            if obj_1:
                obj = obj_1
                print(f"✅ 使用Object_1: {obj.name}")
            else:
                print("❌ 没有选中对象且找不到Object_1")
                return None
        else:
            obj = bpy.context.selected_objects[0]
        
        # 确保对象有mesh数据
        if obj.type != 'MESH':
            print(f"❌ 对象 {obj.name} 不是mesh类型")
            return None
        
        # 获取mesh数据
        mesh = obj.data
        
        # 计算边界信息
        vertices_world = []
        for vertex in mesh.vertices:
            world_coord = obj.matrix_world @ vertex.co
            vertices_world.append([world_coord.x, world_coord.y, world_coord.z])
        
        # 计算边界框
        x_coords = [v[0] for v in vertices_world]
        y_coords = [v[1] for v in vertices_world]
        z_coords = [v[2] for v in vertices_world]
        
        boundary = {
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords),
            'z_min': min(z_coords),
            'z_max': max(z_coords)
        }
        
        # 创建OBJ文件内容
        obj_content = []
        obj_content.append(f"# Exported from Blender - {obj.name}")
        obj_content.append(f"# Vertices: {len(mesh.vertices)}")
        obj_content.append(f"# Faces: {len(mesh.polygons)}")
        obj_content.append(f"# Boundary: {boundary}")
        obj_content.append("")
        
        # 写入顶点
        for vertex in mesh.vertices:
            # 应用对象的变换
            world_coord = obj.matrix_world @ vertex.co
            obj_content.append(f"v {world_coord.x:.6f} {world_coord.y:.6f} {world_coord.z:.6f}")
        
        # 写入面
        for polygon in mesh.polygons:
            # OBJ索引从1开始
            face_indices = [str(vertex_index + 1) for vertex_index in polygon.vertices]
            obj_content.append(f"f {' '.join(face_indices)}")
        
        # 写入文件
        with open(filepath, 'w') as f:
            f.write('\n'.join(obj_content))
        
        print(f"✅ 手动导出OBJ文件成功: {filepath}")
        print(f"📊 导出信息: {len(mesh.vertices)} 顶点, {len(mesh.polygons)} 面")
        print(f"📏 边界信息: {boundary}")
        
        # 返回文件路径和边界信息
        return filepath, boundary
        
    except Exception as e:
        print(f"❌ 手动导出失败: {e}")
        return None

def calculate_boundary(obj):
    """计算对象的边界信息"""
    mesh = obj.data
    vertices_world = []
    for vertex in mesh.vertices:
        world_coord = obj.matrix_world @ vertex.co
        vertices_world.append([world_coord.x, world_coord.y, world_coord.z])
    
    x_coords = [v[0] for v in vertices_world]
    y_coords = [v[1] for v in vertices_world]
    z_coords = [v[2] for v in vertices_world]
    
    boundary = {
        'x_min': min(x_coords),
        'x_max': max(x_coords),
        'y_min': min(y_coords),
        'y_max': max(y_coords),
        'z_min': min(z_coords),
        'z_max': max(z_coords)
    }
    return boundary

def extract_ordered_boundary(obj):
    """提取有序的边界轮廓"""
    try:
        # 创建bmesh对象
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)
        
        # 确保bmesh是最新的
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        print(f"🔍 分析mesh: {len(bm.verts)} 顶点, {len(bm.edges)} 边, {len(bm.faces)} 面")
        
        # 找到所有边界边
        boundary_edges = [e for e in bm.edges if e.is_boundary]
        print(f"📏 找到 {len(boundary_edges)} 条边界边")
        
        if not boundary_edges:
            print("❌ 没有找到边界边")
            bm.free()
            return None
        
        # 重建边界循环
        boundary_loops = []
        used_edges = set()
        
        for start_edge in boundary_edges:
            if start_edge in used_edges:
                continue
                
            # 开始一个新的边界循环
            current_loop = []
            current_edge = start_edge
            
            while current_edge and current_edge not in used_edges:
                used_edges.add(current_edge)
                
                # 获取边的顶点
                v1, v2 = current_edge.verts
                
                # 添加到循环中（只添加一个顶点，避免重复）
                if not current_loop:
                    current_loop.append([v1.co.x, v1.co.y, v1.co.z])
                    current_loop.append([v2.co.x, v2.co.y, v2.co.z])
                else:
                    # 只添加新的顶点
                    if len(current_loop) == 0 or (abs(current_loop[-1][0] - v1.co.x) > 1e-6 or 
                                                 abs(current_loop[-1][1] - v1.co.y) > 1e-6 or 
                                                 abs(current_loop[-1][2] - v1.co.z) > 1e-6):
                        current_loop.append([v1.co.x, v1.co.y, v1.co.z])
                    if abs(current_loop[-1][0] - v2.co.x) > 1e-6 or abs(current_loop[-1][1] - v2.co.y) > 1e-6 or abs(current_loop[-1][2] - v2.co.z) > 1e-6:
                        current_loop.append([v2.co.x, v2.co.y, v2.co.z])
                
                # 找到下一个边界边
                next_edge = None
                for edge in v2.link_edges:
                    if edge.is_boundary and edge not in used_edges:
                        next_edge = edge
                        break
                
                current_edge = next_edge
            
            if current_loop:
                boundary_loops.append(current_loop)
                print(f"🔄 边界循环 {len(boundary_loops)}: {len(current_loop)} 个点")
        
        bm.free()
        
        if not boundary_loops:
            print("❌ 没有找到有效的边界循环")
            return None
        
        # 返回所有边界循环
        result = {
            'boundary_loops': boundary_loops,
            'loop_count': len(boundary_loops),
            'total_points': sum(len(loop) for loop in boundary_loops)
        }
        
        print(f"✅ 成功提取 {len(boundary_loops)} 个边界循环，共 {result['total_points']} 个点")
        return result
        
    except Exception as e:
        print(f"❌ 提取边界失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def try_alternative_export(filepath):
    """尝试其他导出方法"""
    try:
        # 确保Object_1被选中
        obj_1 = bpy.data.objects.get("Object_1")
        if obj_1:
            bpy.ops.object.select_all(action='DESELECT')
            obj_1.select_set(True)
            bpy.context.view_layer.objects.active = obj_1
            print(f"✅ 备用方法确保Object_1被选中: {obj_1.name}")
        
        # 尝试使用wavefront_obj导出器
        if hasattr(bpy.ops, 'export_mesh') and hasattr(bpy.ops.export_mesh, 'obj'):
            bpy.ops.export_mesh.obj(
                filepath=filepath,
                use_selection=True,
                use_materials=False,
                use_triangles=True,
                use_normals=True,
                use_uvs=False
            )
            print(f"✅ 使用export_mesh.obj导出成功: {filepath}")
            # 计算边界信息
            if obj_1:
                boundary = calculate_boundary(obj_1)
                return filepath, boundary
            elif bpy.context.selected_objects:
                obj = bpy.context.selected_objects[0]
                boundary = calculate_boundary(obj)
                return filepath, boundary
            return filepath
        else:
            print("❌ export_mesh.obj也不可用")
            return None
            
    except Exception as e:
        print(f"❌ 备用导出方法也失败: {e}")
        return None

def upload_terrain_to_flask(obj_filepath, boundary=None, ordered_boundary=None, flask_url="http://localhost:5000"):
    """上传地形文件到Flask服务器"""
    try:
        # 准备文件上传
        with open(obj_filepath, 'rb') as f:
            files = {'file': f}
            
            # 准备数据
            data = {}
            if boundary:
                data['boundary'] = json.dumps(boundary)
            if ordered_boundary:
                data['ordered_boundary'] = json.dumps(ordered_boundary)
            
            response = requests.post(
                f"{flask_url}/upload_terrain",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 地形上传成功!")
            print(f"📊 地形信息: {result.get('terrain_info', {})}")
            if boundary:
                print(f"📏 边界信息: {boundary}")
            if ordered_boundary:
                print(f"🔄 有序边界: {ordered_boundary['loop_count']} 个循环, {ordered_boundary['total_points']} 个点")
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
        print(f"❌ 上传出错: {e}")
        return None

def get_terrain_info(flask_url="http://localhost:5000"):
    """获取当前地形信息"""
    try:
        response = requests.get(f"{flask_url}/get_terrain")
        
        if response.status_code == 200:
            result = response.json()
            print("📊 当前地形信息:")
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"❌ 获取地形信息失败: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 获取地形信息出错: {e}")
        return None

def auto_select_object_1():
    """自动选择名为Object_1的对象"""
    if "Object_1" in bpy.data.objects:
        # 取消所有选择
        bpy.ops.object.select_all(action='DESELECT')
        # 选择Object_1
        obj = bpy.data.objects["Object_1"]
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        print("✅ 自动选中了 Object_1")
        return True
    return False

def main():
    """主函数"""
    print("🚀 Blender地形上传工具")
    print("=" * 50)
    
    # 检查选中对象
    selected_objects = bpy.context.selected_objects
    if not selected_objects:
        print("❌ 没有选中任何对象")
        print("🔄 尝试自动选择 Object_1...")
        if auto_select_object_1():
            selected_objects = bpy.context.selected_objects
        else:
            print("❌ 未找到名为 'Object_1' 的对象")
            print("💡 请确保场景中有名为 'Object_1' 的地形对象")
            return
    
    print(f"📦 选中的对象: {[obj.name for obj in selected_objects]}")
    
    # 检查是否包含Object_1
    object_1_found = any(obj.name == "Object_1" for obj in selected_objects)
    if not object_1_found:
        print("⚠️ 警告：未找到名为 'Object_1' 的对象")
        print("🔄 尝试自动选择 Object_1...")
        if auto_select_object_1():
            selected_objects = bpy.context.selected_objects
            object_1_found = True
        else:
            print("❌ 无法找到 Object_1，请手动选中正确的地形对象")
            return
    
    # 显示Object_1的详细信息
    if object_1_found:
        obj_1 = bpy.data.objects["Object_1"]
        print(f"\n📊 Object_1 详细信息:")
        print(f"   类型: {obj_1.type}")
        print(f"   位置: {obj_1.location}")
        print(f"   尺寸: {obj_1.dimensions}")
        print(f"   顶点数: {len(obj_1.data.vertices) if hasattr(obj_1.data, 'vertices') else 'N/A'}")
        print(f"   面数: {len(obj_1.data.polygons) if hasattr(obj_1.data, 'polygons') else 'N/A'}")
    
    # 创建临时文件路径
    temp_dir = Path.home() / "temp_terrain"
    temp_dir.mkdir(exist_ok=True)
    obj_filepath = temp_dir / "terrain.obj"
    
    # 导出OBJ文件
    print("\n📤 导出OBJ文件...")
    export_result = export_selected_as_obj(str(obj_filepath))
    if not export_result:
        return
    
    # 处理导出结果
    if isinstance(export_result, tuple):
        if len(export_result) == 3:
            obj_filepath, boundary, ordered_boundary = export_result
        else:
            obj_filepath, boundary = export_result
            ordered_boundary = None
    else:
        obj_filepath = export_result
        boundary = None
        ordered_boundary = None
    
    # 上传到Flask服务器
    print("\n🌐 上传到Flask服务器...")
    result = upload_terrain_to_flask(str(obj_filepath), boundary, ordered_boundary)
    
    if result:
        print("\n✅ 地形上传完成!")
        print("🎯 现在可以在IDE中使用这个地形进行强化学习训练了")
        
        # 显示地形信息
        print("\n📊 地形详细信息:")
        terrain_info = result.get('terrain_info', {})
        if terrain_info:
            print(f"   网格大小: {terrain_info.get('grid_size', 'N/A')}")
            print(f"   顶点数量: {terrain_info.get('vertices_count', 'N/A')}")
            print(f"   面数量: {terrain_info.get('faces_count', 'N/A')}")
    
    # 清理临时文件
    try:
        os.remove(obj_filepath)
        print(f"🧹 临时文件已清理: {obj_filepath}")
    except:
        pass

# 在Blender中运行
if __name__ == "__main__":
    main()
