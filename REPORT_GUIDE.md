# Report Guide — Smart Traffic Congestion Prediction System

## Abstract
This project builds a machine-learning decision-support prototype to estimate hourly traffic volume and classify relative traffic intensity from historical traffic and weather observations. Data is cleaned, transformed into temporal/weather features, split chronologically, and used to compare multiple regression and classification algorithms. A Flask web application supports interactive predictions, batch CSV inference, performance charts and prediction history.

## Objectives
1. Analyze historical traffic volume patterns.
2. Engineer time and weather features.
3. Train and compare at least three ML algorithms.
4. Evaluate regression and classification performance.
5. Deploy the selected models in a web application.

## Dataset
UCI Metro Interstate Traffic Volume (ID 492), 48,204 hourly records for westbound I-94 in Minnesota (2012–2018). DOI 10.24432/C5X60B, CC BY 4.0. Cite the official UCI dataset page.

## Algorithms and metrics
Random Forest, Extra Trees, Histogram Gradient Boosting, optional XGBoost. Regression: MAE, RMSE, R². Classification: accuracy, macro precision, macro recall, macro F1 and confusion matrix. Copy actual results from `models/metrics.json` after running training; do not claim unverified results.

## Limitation
Dataset provides traffic volume, not speed or occupancy. Low/Medium/High are relative quantile bands from training data, not official congestion/Level-of-Service classes. Data is from one Minnesota corridor and is not live traffic or Pune/India traffic.

## Future work
Add local traffic sensors, GPS/IoT, speed and occupancy data, map-based corridor display, real-time APIs and temporal forecasting.
