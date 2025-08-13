#!/usr/bin/env python3
"""
Flask服务器 - 对接Blender多智能体系统
提供/llm_decide接口，接收Blender状态并返回动作决策
"""

from flask import Flask, request, jsonify
import json
import logging
import random
from typing import Dict, List, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局状态跟踪
current_state = None
action_history = []

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
