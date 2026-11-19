from flask import Flask,render_template,request,jsonify,redirect,url_for,flash,send_file
from pathlib import Path
from datetime import datetime
import pandas as pd,joblib,json,sqlite3,io
from features import prepare_features
app=Flask(__name__); app.secret_key="traffic-sense-local-demo-key"
M=Path("models"); DB="traffic_history.db"
def artifacts():
 if not all((M/x).exists() for x in ["best_regressor.joblib","best_classifier.joblib","metrics.json"]): return None
 return joblib.load(M/"best_regressor.joblib"),joblib.load(M/"best_classifier.joblib"),json.loads((M/"metrics.json").read_text())
def connect():
 c=sqlite3.connect(DB); c.execute("CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY,created_at TEXT,volume REAL,level TEXT,weather TEXT,hour INTEGER)"); c.commit(); return c
def input_data(f):
 return pd.DataFrame([{"date_time":f.get("date_time") or datetime.now().strftime("%Y-%m-%dT%H:%M"),"temp":float(f.get("temp",288.15)),"rain_1h":float(f.get("rain_1h",0)),"snow_1h":float(f.get("snow_1h",0)),"clouds_all":float(f.get("clouds_all",40)),"weather_main":f.get("weather_main","Clouds"),"holiday":f.get("holiday","None")}])
@app.get("/")
def home():
 a=artifacts(); return render_template("setup.html") if not a else render_template("index.html",metrics=a[2])
@app.get("/dashboard")
def dashboard():
 a=artifacts()
 if not a:return redirect(url_for("home"))
 c=connect(); rows=c.execute("SELECT created_at,volume,level,weather,hour FROM predictions ORDER BY id DESC LIMIT 30").fetchall(); c.close()
 return render_template("dashboard.html",metrics=a[2],rows=rows)
@app.post("/predict")
def predict():
 a=artifacts()
 if not a:return redirect(url_for("home"))
 try:
  raw=input_data(request.form); x=prepare_features(raw); vol=float(a[0].predict(x)[0]); level=str(a[1].predict(x)[0]); c=connect(); c.execute("INSERT INTO predictions(created_at,volume,level,weather,hour) VALUES(?,?,?,?,?)",(datetime.now().isoformat(timespec="seconds"),vol,level,raw.iloc[0].weather_main,pd.to_datetime(raw.iloc[0].date_time).hour)); c.commit(); c.close()
  return render_template("result.html",volume=round(vol),level=level)
 except Exception as e: flash("Prediction error: "+str(e)); return redirect(url_for("home"))
@app.post("/batch")
def batch():
 a=artifacts(); f=request.files.get("file")
 if not a or not f: flash("Train models and select a CSV file."); return redirect(url_for("home"))
 try:
  d=pd.read_csv(f); x=prepare_features(d); d["predicted_traffic_volume"]=a[0].predict(x).round().astype(int); d["predicted_traffic_level"]=a[1].predict(x); b=io.BytesIO(); d.to_csv(b,index=False); b.seek(0); return send_file(b,mimetype="text/csv",as_attachment=True,download_name="traffic_predictions.csv")
 except Exception as e: flash("CSV error: "+str(e)); return redirect(url_for("home"))
@app.get("/api/health")
def health(): return jsonify({"status":"ok","models_ready":artifacts() is not None})
@app.get("/api/metrics")
def metrics():
 a=artifacts(); return (jsonify(a[2]),200) if a else (jsonify({"error":"Run training first"}),503)
if __name__=="__main__":app.run(debug=True)
