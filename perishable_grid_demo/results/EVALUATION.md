# ACN model evaluation summary

**Model:** LightGBM Regressor, selected on chronological validation data.

**Test set:** 4,753 site-hour observations, held out chronologically per site.

| Metric | LightGBM | Previous day (24h) | Previous week (168h) |
|---|---:|---:|---:|
| MAE (kWh) | 5.4403 | 9.1865 | 7.3978 |
| RMSE (kWh) | 18.5827 | 33.9842 | 24.3164 |
| WAPE | 49.41% | 83.44% | 67.19% |

The model beats both seasonal-naive baselines on aggregate test MAE. Its MAE is 26.46% lower than the best seasonal-naive baseline. This is an observed result for the supplied California ACN site data only.

Site-level caution: ACN site `0019` has WAPE of 101.95% because its test period contains very low total recorded energy and many no-arrival hours. Its absolute MAE is 1.02 kWh, lower than the 1.66/1.69 kWh seasonal baselines, but relative error remains large.

Quantile calibration: observed `actual <= forecast` rates were 79.28% for P10, 84.75% for P50 and 93.20% for P90; P10–P90 interval coverage was 90.76% against nominal 80%. These outputs are not calibrated prediction intervals.

See `model_metrics.json`, `test_predictions.csv`, and the PNG diagnostics for the full results.

## Leading final-model features

The final refitted LightGBM gain ranking begins with rolling 24-hour mean demand, hour of day, previous-week lag, rolling 168-hour mean and rolling 168-hour standard deviation. Previous-day and previous-hour demand also contribute materially. See `feature_importance.csv` for the full numeric ranking.
