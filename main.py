#!/usr/bin/env python3
"""
Flask服务器 - 对接Blender多智能体系统
提供/llm_decide接口，接收Blender状态并返回动作决策
"""

from flask import Flask, request, jsonify, send_file
import json
import logging
import random
import os
import numpy as np
from typing import Dict, List, Any
from werkzeug.utils import secure_filename

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 文件上传配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'obj', 'npy', 'json', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 全局状态跟踪
current_state = None
action_history = []
terrain_data = None  # 存储处理后的地形数据

class BlenderAgent:
    """Blender多智能体决策器"""
    
    def __init__(self):
        self.agent_id = "blender_agent"
        self.strategy = "random"
        
    def decide_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """基于当前状态决定下一个动作"""
        planes = state.get("planes", [])
        if not planes:
            return self._get_empty_action()
            
        if self.strategy == "random":
            return self._random_strategy(planes)
        elif self.strategy == "greedy":
            return self._greedy_strategy(planes)
        else:
            return self._random_strategy(planes)
    
    def _get_empty_action(self) -> Dict[str, Any]:
        return {"plane": 0, "layer": 0, "color": 0}
    
    def _random_strategy(self, planes: List[Dict]) -> Dict[str, Any]:
        """随机策略"""
        available_planes = []
        for i, plane in enumerate(planes):
            max_height = plane.get("planesHeight", 4)
            current_height = plane.get("height", 0)
            if current_height < max_height:
                available_planes.append(i)
        
        if not available_planes:
            return self._get_empty_action()
        
        plane_idx = random.choice(available_planes)
        plane = planes[plane_idx]
        layer = plane.get("height", 0)
        color = random.randint(0, 4)
        
        return {
            "plane": plane_idx,
            "layer": layer,
            "color": color
        }
    
    def _greedy_strategy(self, planes: List[Dict]) -> Dict[str, Any]:
        """贪婪策略：优先填满高度较低的plane"""
        plane_heights = [(i, plane.get("height", 0)) for i, plane in enumerate(planes)]
        plane_heights.sort(key=lambda x: x[1])
        
        for plane_idx, current_height in plane_heights:
            plane = planes[plane_idx]
            max_height = plane.get("planesHeight", 4)
            
            if current_height < max_height:
                color = random.randint(0, 4)
                return {
                    "plane": plane_idx,
                    "layer": current_height,
                    "color": color
                }
        
        return self._get_empty_action()

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_obj_file(filepath, boundary=None, ordered_boundary=None):
    """处理OBJ文件，提取地形数据，仅做基本处理，水体识别在IDE端进行"""
    try:
        vertices = []
        faces = []
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '):  # 顶点
                    parts = line.strip().split()[1:]
                    if len(parts) >= 3:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                elif line.startswith('f '):  # 面
                    parts = line.strip().split()[1:]
                    if len(parts) >= 3:
                        face = [int(part.split('/')[0]) - 1 for part in parts]
                        faces.append(face)
        
        if not vertices:
            return None
        
        vertices = np.array(vertices)
        heights = vertices[:, 2]
        
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        
        # 使用边界信息或自动计算边界
        if boundary:
            x_min, x_max = boundary['x_min'], boundary['x_max']
            y_min, y_max = boundary['y_min'], boundary['y_max']
            print(f"✅ 使用提供的边界信息")
        else:
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            print(f"⚠️ 自动计算边界信息")
        
        # 计算原始比例
        x_span = x_max - x_min
        y_span = y_max - y_min
        aspect_ratio = x_span / y_span
        
        # 根据坐标范围动态计算合适的网格大小
        # 目标：让每个像素代表约1-2个世界单位
        coord_range_x = x_max - x_min
        coord_range_y = y_max - y_min
        
        # 计算合适的网格尺寸
        target_pixel_size = 2.0  # 每个像素代表2个世界单位
        grid_x = max(100, int(coord_range_x / target_pixel_size))
        grid_y = max(100, int(coord_range_y / target_pixel_size))
        
        # 限制最大尺寸，避免内存问题
        max_grid_size = 1000
        grid_x = min(grid_x, max_grid_size)
        grid_y = min(grid_y, max_grid_size)
        
        print(f"📏 坐标范围: X={coord_range_x:.1f}, Y={coord_range_y:.1f}")
        print(f"📏 目标像素尺寸: {target_pixel_size}")
        print(f"📏 计算网格尺寸: {grid_x} x {grid_y}")
        
        grid_size = (grid_x, grid_y)
        print(f"📊 原始顶点数: {len(vertices)}, 使用网格大小: {grid_size}")
        print(f"📏 原始比例: {aspect_ratio:.3f}, 边界: X({x_min:.3f}~{x_max:.3f}), Y({y_min:.3f}~{y_max:.3f})")
        
        # 使用三角面填充方法生成高度图和掩码
        height_map, mask = create_triangle_based_terrain(vertices, faces, grid_size, x_min, x_max, y_min, y_max)
        
        print(f"📊 处理后高程范围: {height_map.min():.3f} 到 {height_map.max():.3f}")
        
        # 掩码已经由三角面填充方法生成
        print(f"✅ 掩码已由三角面填充方法生成")
        
        return {
            'height_map': height_map.tolist(),
            'mask': mask.tolist(),
            'grid_size': grid_size,
            'vertices_count': len(vertices),
            'faces_count': len(faces),
            'original_bounds': {
                'x_min': float(x_min),
                'x_max': float(x_max),
                'y_min': float(y_min),
                'y_max': float(y_max),
                'z_min': float(heights.min()),
                'z_max': float(heights.max())
            },
            'scale_factors': {
                'x_scale': float(x_span),
                'y_scale': float(y_span),
                'z_scale': float(heights.max() - heights.min())
            },
            'aspect_ratio': float(aspect_ratio),
            'has_ordered_boundary': ordered_boundary is not None
        }
        
    except Exception as e:
        logger.error(f"处理OBJ文件时出错: {e}")
        return None

def create_triangle_based_terrain(vertices, faces, grid_size, x_min, x_max, y_min, y_max):
    """使用简化的三角面填充方法生成高度图和掩码"""
    try:
        W, H = grid_size
        dx = (x_max - x_min) / W
        dy = (y_max - y_min) / H
        
        print(f"🔍 调试信息:")
        print(f"   网格尺寸: W={W}, H={H}")
        print(f"   坐标范围: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
        print(f"   像素大小: dx={dx:.3f}, dy={dy:.3f}")
        
        # 检查顶点范围是否与给定范围一致
        v_x_min, v_x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        v_y_min, v_y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
        print(f"   顶点范围: x=[{v_x_min:.3f}, {v_x_max:.3f}], y=[{v_y_min:.3f}, {v_y_max:.3f}]")
        
        if abs(v_x_min - x_min) > 1e-6 or abs(v_x_max - x_max) > 1e-6 or \
           abs(v_y_min - y_min) > 1e-6 or abs(v_y_max - y_max) > 1e-6:
            print(f"   ⚠️  警告: 顶点范围与给定范围不一致!")
        
        # 简化的方法：对每个像素，找到包含它的最高三角形
        Z = np.full((H, W), np.nan, dtype=np.float32)
        M = np.zeros((H, W), dtype=bool)
        
        # 像素中心坐标
        i = np.arange(W)
        j = np.arange(H)
        xx = x_min + (i + 0.5) * dx
        yy = y_min + (j + 0.5) * dy  # y方向向上
        XX, YY = np.meshgrid(xx, yy)  # (H,W)
        
        print(f"🔄 开始简化三角面填充处理...")
        print(f"   三角面数量: {len(faces)}")
        
        # 统计变量
        processed_triangles = 0
        total_covered_pixels = 0
        
        # 对每个三角形，找到它覆盖的像素
        for face_idx, (a, b, c) in enumerate(faces):
            if face_idx % 1000 == 0:
                print(f"   处理进度: {face_idx}/{len(faces)}")
            
            xa, ya, za = vertices[a]
            xb, yb, zb = vertices[b]
            xc, yc, zc = vertices[c]
            
            # 计算三角形边界
            minx, maxx = min(xa, xb, xc), max(xa, xb, xc)
            miny, maxy = min(ya, yb, yc), max(ya, yb, yc)
            
            # 找到受影响的像素范围（更保守的估计）
            imin = max(0, int((minx - x_min) / dx))
            imax = min(W - 1, int((maxx - x_min) / dx))
            jmin = max(0, int((miny - y_min) / dy))
            jmax = min(H - 1, int((maxy - y_min) / dy))
            
            if imin > imax or jmin > jmax:
                continue
            
            # 调试：检查包围盒大小
            if face_idx < 5:
                print(f"   三角形{face_idx}: 包围盒 [{imin},{imax}]x[{jmin},{jmax}], 大小 {imax-imin+1}x{jmax-jmin+1}")
            
            # 对包围盒内的每个像素，检查是否在三角形内
            covered_pixels = 0
            for jj in range(jmin, jmax + 1):
                for ii in range(imin, imax + 1):
                    px = xx[ii]
                    py = yy[jj]
                    
                    # 重心坐标计算
                    def crossz(x1, y1, x2, y2):
                        return x1 * y2 - x2 * y1
                    
                    area = crossz(xb - xa, yb - ya, xc - xa, yc - ya)
                    if abs(area) < 1e-12:
                        continue
                    
                    w0 = crossz(xb - px, yb - py, xc - px, yc - py) / area
                    w1 = crossz(xc - px, yc - py, xa - px, ya - py) / area
                    w2 = 1.0 - w0 - w1
                    
                    # 检查是否在三角形内
                    if w0 >= 0 and w1 >= 0 and w2 >= 0:
                        # 计算该点的高程
                        z_val = w0 * za + w1 * zb + w2 * zc
                        
                        # 如果这个三角形更高，就更新
                        if np.isnan(Z[jj, ii]) or z_val > Z[jj, ii]:
                            Z[jj, ii] = z_val
                            M[jj, ii] = True
                            covered_pixels += 1
            
            if covered_pixels > 0:
                processed_triangles += 1
                total_covered_pixels += covered_pixels
                
                if face_idx < 5:
                    print(f"   三角形{face_idx}: 覆盖像素数 {covered_pixels}")
        
        print(f"✅ 简化三角面填充完成")
        print(f"   处理三角形数: {processed_triangles} / {len(faces)}")
        print(f"   总覆盖像素数: {total_covered_pixels}")
        print(f"   最终有效像素数: {np.sum(M)} / {M.size}")
        print(f"   覆盖率: {np.sum(M) / M.size * 100:.1f}%")
        
        # 检查结果
        if np.all(Z == 0) or np.all(np.isnan(Z)):
            print(f"   ⚠️  警告: 高度图全为0或NaN!")
        else:
            valid_z = Z[~np.isnan(Z)]
            print(f"   高度范围: [{np.min(valid_z):.3f}, {np.max(valid_z):.3f}]")
        
        # 转置回原来的格式 (W, H)
        return Z.T, M.T
        
    except Exception as e:
        logger.error(f"简化三角面填充处理时出错: {e}")
        # 返回空的高度图和全True掩码作为后备
        return np.zeros(grid_size), np.ones(grid_size, dtype=bool)

def create_ordered_boundary_mask(ordered_boundary, grid_size, x_min, x_max, y_min, y_max):
    """使用有序边界创建精确的掩码"""
    try:
        from matplotlib.path import Path
        
        grid_x, grid_y = grid_size
        mask = np.zeros(grid_size, dtype=bool)
        
        # 创建网格坐标
        x_coords = np.linspace(x_min, x_max, grid_x)
        y_coords = np.linspace(y_min, y_max, grid_y)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        points = np.column_stack((X.flatten(), Y.flatten()))
        
        # 处理每个边界循环
        boundary_loops = ordered_boundary['boundary_loops']
        
        for i, loop in enumerate(boundary_loops):
            # 只取XY坐标（忽略Z坐标）
            loop_2d = np.array([[point[0], point[1]] for point in loop])
            
            # 创建路径
            path = Path(loop_2d)
            
            # 检查哪些点在路径内
            inside = path.contains_points(points)
            inside = inside.reshape(grid_size)
            
            # 更新掩码（主边界为True，内部空洞为False）
            if i == 0:  # 主边界
                mask = mask | inside
            else:  # 内部空洞
                mask = mask & (~inside)
        
        print(f"✅ 有序边界掩码创建完成")
        print(f"   有效点数: {np.sum(mask)} / {mask.size}")
        print(f"   覆盖率: {np.sum(mask)/mask.size*100:.1f}%")
        
        return mask
        
    except Exception as e:
        logger.error(f"创建有序边界掩码时出错: {e}")
        # 返回全True掩码作为后备
        return np.ones(grid_size, dtype=bool)

# 创建智能体实例
agent = BlenderAgent()

@app.route('/llm_decide', methods=['POST'])
def llm_decide():
    """接收Blender状态并返回动作决策"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        global current_state, action_history
        current_state = data
        
        logger.info(f"收到状态: {len(data.get('planes', []))} 个planes")
        
        action = agent.decide_action(data)
        action_history.append({"timestamp": len(action_history), "action": action})
        
        logger.info(f"返回动作: {action}")
        return jsonify(action)
        
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """获取服务器状态"""
    return jsonify({
        "status": "running",
        "current_state": current_state,
        "action_history_length": len(action_history),
        "agent_strategy": agent.strategy
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "service": "blender_agent",
        "version": "1.0.0"
    })

@app.route('/upload_terrain', methods=['POST'])
def upload_terrain():
    """上传地形文件（OBJ、NPY、JSON等）"""
    global terrain_data
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"文件已保存: {filepath}")
            
            # 获取边界信息（如果提供）
            boundary = None
            ordered_boundary = None
            if 'boundary' in request.form:
                try:
                    boundary = json.loads(request.form['boundary'])
                    logger.info(f"接收到边界信息: {boundary}")
                except json.JSONDecodeError:
                    logger.warning("边界信息格式错误，将使用自动计算")
            
            # 获取有序边界信息（如果提供）
            if 'ordered_boundary' in request.form:
                try:
                    ordered_boundary = json.loads(request.form['ordered_boundary'])
                    logger.info(f"接收到有序边界信息: {ordered_boundary['loop_count']} 个循环, {ordered_boundary['total_points']} 个点")
                except json.JSONDecodeError:
                    logger.warning("有序边界信息格式错误，将使用自动计算")
            
            # 根据文件类型处理
            if filename.endswith('.obj'):
                terrain_data = process_obj_file(filepath, boundary, ordered_boundary)
            elif filename.endswith('.npy'):
                terrain_data = np.load(filepath).tolist()
            elif filename.endswith('.json'):
                with open(filepath, 'r') as f:
                    terrain_data = json.load(f)
            elif filename.endswith('.txt'):
                terrain_data = np.loadtxt(filepath).tolist()
            
            if terrain_data:
                # 保存处理后的地形数据到data/terrain目录
                data_dir = os.path.join(os.getcwd(), 'data', 'terrain')
                os.makedirs(data_dir, exist_ok=True)
                
                # 生成带时间戳的文件名
                import time
                timestamp = int(time.time())
                processed_filename = f"terrain_{timestamp}.json"
                processed_filepath = os.path.join(data_dir, processed_filename)
                
                # 保存地形数据
                with open(processed_filepath, 'w') as f:
                    json.dump(terrain_data, f, indent=2)
                
                logger.info(f"地形数据已保存到: {processed_filepath}")
                
                return jsonify({
                    "status": "success",
                    "message": f"地形文件 {filename} 上传并处理成功",
                    "terrain_info": terrain_data,
                    "saved_file": processed_filename
                })
            else:
                return jsonify({"error": "无法处理地形文件"}), 400
        else:
            return jsonify({"error": "不支持的文件格式"}), 400
            
    except Exception as e:
        logger.error(f"上传文件时出错: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_terrain', methods=['GET'])
def get_terrain():
    """获取当前地形数据"""
    global terrain_data
    
    if terrain_data:
        return jsonify({
            "status": "success",
            "terrain_data": terrain_data
        })
    else:
        return jsonify({"error": "没有可用的地形数据"}), 404

@app.route('/download_terrain', methods=['GET'])
def download_terrain():
    """下载处理后的地形数据"""
    global terrain_data
    
    if not terrain_data:
        return jsonify({"error": "没有可用的地形数据"}), 404
    
    try:
        # 保存为JSON文件供下载
        filename = "processed_terrain.json"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'w') as f:
            json.dump(terrain_data, f, indent=2)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"下载地形数据时出错: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/set_strategy', methods=['POST'])
def set_strategy():
    """设置智能体策略"""
    try:
        data = request.get_json()
        strategy = data.get("strategy", "random")
        
        if strategy in ["random", "greedy"]:
            agent.strategy = strategy
            return jsonify({"status": "success", "strategy": strategy})
        else:
            return jsonify({"error": "Invalid strategy"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("启动Flask服务器...")
    logger.info("Blender接口地址: http://localhost:5000/llm_decide")
    app.run(host='0.0.0.0', port=5000, debug=True)
