# Smart Traffic Congestion Prediction System (TrafficSense AI)

A third-year Computer Engineering ML project. Predicts hourly traffic volume and classifies it into relative Low/Medium/High traffic-volume bands. Includes four candidate algorithm families, regression/classification evaluation, Flask UI, prediction history, charts and CSV batch prediction.

**Scope honesty:** The UCI dataset is hourly traffic volume for one westbound I-94 corridor in Minnesota (2012–2018). It has no speed/occupancy and is not a live local traffic feed. Categories are quantile-based relative volume bands, not official congestion grades.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_data.py
python train.py
python app.py
```
Open http://127.0.0.1:5000 . If PowerShell activation is blocked, use CMD: `.venv\Scripts\activate.bat`.

## Models
Random Forest, Extra Trees, Histogram Gradient Boosting and XGBoost (if installed). Both regression and classification pipelines are evaluated. Actual scores are generated in `models/metrics.json` after training; no scores are fabricated in this starter package.

## Dataset
UCI Metro Interstate Traffic Volume (ID 492), DOI 10.24432/C5X60B, CC BY 4.0: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

## GitHub
```bash
git init
git add .
git commit -m "Initial Smart Traffic Congestion Prediction System"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/smart-traffic-congestion-prediction.git
git push -u origin main
```
Do not upload `.venv`, raw data, model binaries or private information unless required.
