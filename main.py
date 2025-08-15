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

def process_obj_file(filepath):
    """处理OBJ文件，提取地形数据"""
    try:
        # 简单的OBJ解析（这里可以根据需要扩展）
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
                        # 提取顶点索引（OBJ索引从1开始）
                        face = [int(part.split('/')[0]) - 1 for part in parts]
                        faces.append(face)
        
        if not vertices:
            return None
        
        # 转换为numpy数组
        vertices = np.array(vertices)
        
        # 提取高程信息（假设Z轴是高度）
        heights = vertices[:, 2]
        
        # 创建2D网格（简化处理）
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        
        # 创建网格
        grid_size = (50, 50)  # 默认网格大小
        height_map = np.zeros(grid_size)
        
        # 简单的插值（这里可以改进）
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x = x_min + (x_max - x_min) * i / (grid_size[0] - 1)
                y = y_min + (y_max - y_min) * j / (grid_size[1] - 1)
                
                # 找到最近的顶点
                distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
                nearest_idx = np.argmin(distances)
                height_map[i, j] = heights[nearest_idx]
        
        # 归一化高度到0-100
        height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min()) * 100
        
        return {
            'height_map': height_map.tolist(),
            'grid_size': grid_size,
            'vertices_count': len(vertices),
            'faces_count': len(faces)
        }
        
    except Exception as e:
        logger.error(f"处理OBJ文件时出错: {e}")
        return None

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
            
            # 根据文件类型处理
            if filename.endswith('.obj'):
                terrain_data = process_obj_file(filepath)
            elif filename.endswith('.npy'):
                terrain_data = np.load(filepath).tolist()
            elif filename.endswith('.json'):
                with open(filepath, 'r') as f:
                    terrain_data = json.load(f)
            elif filename.endswith('.txt'):
                terrain_data = np.loadtxt(filepath).tolist()
            
            if terrain_data:
                return jsonify({
                    "status": "success",
                    "message": f"地形文件 {filename} 上传并处理成功",
                    "terrain_info": terrain_data
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
