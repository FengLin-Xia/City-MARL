from flask import Flask, Response
app = Flask(__name__)
DATA_FILE = "enhanced_simulation_v2_3_output/simplified_buildings.json"

@app.route("/api/buildings/<month>")
def get_buildings(month):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    return Response(text, mimetype="text/plain")

@app.route("/api/buildings/<month>/flask-format")
def get_buildings_flask_format(month):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    return Response(text, mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
