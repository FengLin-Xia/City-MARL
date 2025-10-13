from flask import Flask, Response
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
