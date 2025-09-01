#!/usr/bin/env python3
# 批量简化v3.1建筑位置数据，按月份顺序输出

import json, os, glob

def simplify_building_positions_batch():
    """批量处理v3.1输出文件"""
    input_dir = 'enhanced_simulation_v3_1_output'
    output_dir = 'enhanced_simulation_v3_1_output/simplified'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 类型映射
    type_map = {'residential': 0, 'commercial': 1, 'office': 2, 'public': 3}
    
    # 查找所有建筑位置文件
    pattern = os.path.join(input_dir, 'building_positions_month_*.json')
    files = glob.glob(pattern)
    
    # 按月份排序
    files.sort(key=lambda x: int(x.split('_')[-1].replace('.json', '')))
    
    print(f"🔍 找到 {len(files)} 个建筑位置文件")
    
    for file_path in files:
        # 提取月份
        month = int(file_path.split('_')[-1].replace('.json', ''))
        
        print(f"📝 处理第 {month:02d} 个月...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 格式化建筑数据
            formatted = []
            for building in data.get('buildings', []):
                t = str(building.get('type', 'unknown')).lower()
                mid = type_map.get(t, 4)
                pos = building.get('position', [0.0, 0.0])
                x = float(pos[0]) if len(pos) > 0 else 0.0
                y = float(pos[1]) if len(pos) > 1 else 0.0
                z = 0.0  # 默认高度为0
                formatted.append(f"{mid}({x:.3f}, {y:.3f}, {z:.0f})")
            
            # 生成简化格式的字符串
            simplified_line = ", ".join(formatted)
            
            # 保存到JSON文件
            simplified_data = {
                'month': month,
                'timestamp': f'month_{month:02d}',
                'simplified_format': simplified_line,
                'building_count': len(formatted),
                'source_file': os.path.basename(file_path)
            }
            
            # 保存JSON文件（带顺序编号）
            json_file = os.path.join(output_dir, f'simplified_buildings_{month:02d}.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(simplified_data, f, indent=2, ensure_ascii=False)
            
            # 保存纯文本文件（带顺序编号）
            txt_file = os.path.join(output_dir, f'simplified_buildings_{month:02d}.txt')
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(simplified_line)
            
            print(f"  ✅ 第 {month:02d} 个月：{len(formatted)} 个建筑")
            
        except Exception as e:
            print(f"  ❌ 处理第 {month:02d} 个月时出错：{e}")
    
    print(f"\n🎉 批量处理完成！")
    print(f"📁 简化文件保存在：{output_dir}")

def generate_flask_endpoint():
    """生成Flask API端点"""
    code = '''from flask import Flask, Response
import json, os

app = Flask(__name__)
DATA_DIR = "enhanced_simulation_v3_1_output/simplified"

@app.route("/api/buildings/<month>")
def get_buildings(month):
    """获取指定月份的简化建筑数据"""
    try:
        month_int = int(month)
        json_file = os.path.join(DATA_DIR, f"simplified_buildings_{month_int:02d}.json")
        
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Response(data['simplified_format'], mimetype="text/plain")
        else:
            return Response("", mimetype="text/plain")
    except:
        return Response("", mimetype="text/plain")

@app.route("/api/buildings/<month>/json")
def get_buildings_json(month):
    """获取指定月份的完整JSON数据"""
    try:
        month_int = int(month)
        json_file = os.path.join(DATA_DIR, f"simplified_buildings_{month_int:02d}.json")
        
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        else:
            return {"error": "Month not found"}
    except:
        return {"error": "Invalid month"}

@app.route("/api/buildings/list")
def list_available_months():
    """列出所有可用的月份"""
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.startswith("simplified_buildings_") and f.endswith(".json")]
        months = [int(f.split('_')[-1].replace('.json', '')) for f in files]
        months.sort()
        return {"available_months": months}
    except:
        return {"available_months": []}

if __name__ == "__main__":
    app.run(debug=True, port=5000)
'''
    
    output_dir = 'enhanced_simulation_v3_1_output/simplified'
    os.makedirs(output_dir, exist_ok=True)
    
    flask_file = os.path.join(output_dir, 'flask_api_example.py')
    with open(flask_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"📝 Flask API示例已生成：{flask_file}")

if __name__ == "__main__":
    simplify_building_positions_batch()
    generate_flask_endpoint()
