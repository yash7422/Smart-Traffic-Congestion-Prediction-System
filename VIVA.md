# Viva Questions
1. **Aim?** Predict hourly traffic volume and classify relative traffic intensity.
2. **Dataset?** UCI Metro Interstate Traffic Volume, ID 492.
3. **Target?** Hourly traffic count (`traffic_volume`).
4. **Algorithms?** Random Forest, Extra Trees, Histogram Gradient Boosting and optional XGBoost.
5. **Why feature engineering?** Traffic depends on hour, weekday, month, weekend, rush-hour and weather.
6. **Regression vs classification?** Regression predicts a numeric count; classification predicts a category.
7. **Why chronological split?** It trains on earlier records and tests on later records to respect time order.
8. **MAE?** Mean absolute prediction error; lower is better.
9. **RMSE?** Square root of mean squared error; larger errors are penalized more.
10. **R²?** Fraction of target variance explained by the regression model.
11. **Precision?** Correct positive predictions divided by all positive predictions for a class.
12. **Recall?** Correct positive predictions divided by actual positives.
13. **F1?** Harmonic mean of precision and recall.
14. **How are traffic bands made?** Training target 33rd and 67th percentiles define Low, Medium and High relative-volume groups.
15. **Is this live traffic?** No, it uses historical observations.
16. **Does data contain speed?** No, it contains hourly traffic volume and weather/time variables.
17. **Why Flask?** Lightweight Python web framework to expose model predictions.
18. **Overfitting?** Learning training noise and failing to generalize to unseen data.
19. **Data leakage?** Using information unavailable at prediction time in model training/evaluation.
20. **Future scope?** Integrate local sensor/GPS feeds, speed/occupancy, maps and real-time data.
