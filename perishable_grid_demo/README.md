# The Perishable Grid

**Physical energy trading with battery-health- and customer-SLA-aware dispatch**

This repository now keeps two distinct forecasting/data paths:

1. **Original simulator:** five illustrative Indian metro clusters (Delhi NCR, Mumbai MMR, Bengaluru, Hyderabad and Chennai), synthetic demand/supply/price inputs, a previous-day seasonal-naive demand forecast, and transparent heuristic dispatch policies.
2. **ACN-trained forecaster:** a real LightGBM regression model trained on public ACN-Data charging-session records from three California sites. It forecasts **arrival-cohort energy demand** for those ACN sites. It is not calibrated to the five Indian clusters and is not silently wired into their dispatch engine.

The forecasting model improves on the seasonal-naive benchmarks on the supplied data's aggregate chronological test split. Its geographic transfer is unvalidated, and its quantile outputs are not calibrated. See the measured results below and `results/model_metrics.json`.

## 1. Original simulator

Run the existing synthetic simulator:

```bash
python simulate.py
```

It writes the original policy summaries, hourly simulation results and charts. The four policies remain `baseline`, `forecast`, `health` and `full`. The `forecast` policy uses the previous day's same cluster/hour, not the ACN model. The simulation remains runnable if no trained model exists.

The existing simulation is a **heuristic proof of concept**, not a production MILP, live IEX bidding connection, physical pack-repurposing implementation or validated operator business case. Its assumptions and limitations remain in `assumptions.json` and the hackathon brief.

## 2. ACN dataset and observed schema

Place the ACN session JSON at:

```text
data/raw/acn_sessions.json
```

The expected ACN export shape is a JSON object with a top-level `_meta` object and an `_items` array of session objects. The uploaded records were inspected directly. Top-level session fields include:

- `_id`, `sessionID`, `siteID`, `clusterID`, `stationID`, `spaceID`, `timezone`
- `connectionTime`, `disconnectTime`, `doneChargingTime`
- `kWhDelivered`, `userID`, `userInputs`

`kWhRequested` and `requestedDeparture` are nested inside entries in `userInputs`, not top-level fields. `chargingCurrent` does not occur in the supplied exports. No joined weather observations were provided.

### Data supplied for this training run

Three uploaded exports yielded 25,462 complete session objects after recovering complete records from two tail-truncated JSON files. The combined file in `data/raw/acn_sessions.json` is valid JSON and records its recovery provenance in `_meta`; it is **not represented as a complete ACN database dump**.

| Normalized ACN site ID | Sessions | Observed connection-time span | Timezone |
|---|---:|---|---|
| `0002` | 12,199 | 2018-04-25 to 2018-10-20 | America/Los_Angeles |
| `0001` | 11,580 | 2018-09-05 to 2019-05-22 | America/Los_Angeles |
| `0019` | 1,683 | 2019-03-25 to 2021-09-13 | America/Los_Angeles |

The numeric `siteID` value `2` and string value `0002` were normalized to the same ID. The recovered records contained no missing `connectionTime`, missing `kWhDelivered`, invalid timezone, negative energy or duplicate `_id`/`sessionID` values. The preprocessing pipeline still checks and reports these cases for future downloads; records without a usable connection time, site, timezone or non-negative delivered energy are excluded rather than imputed.

**Coverage caveat:** site `0019` has a long observed time span but only 1,683 sessions. The hourly grid therefore contains many hours with no recorded starts. These hours are represented as zero recorded arrival demand between the first and last observed sessions. That assumes the downloaded export is complete during those intervals; completeness of the upstream export cannot be certified from these files. Do not interpret every zero as proof that the physical charging site was continuously monitored and idle.

## 3. Forecasting target and preprocessing

The target is:

```text
arrival_energy_kwh(site, local_hour)
    = sum(kWhDelivered for sessions whose connectionTime falls in that site-local hour)
```

This is **arrival-cohort energy demand**, not exact physical electricity consumption during the same clock hour. A session may continue charging after its connection hour; its full `kWhDelivered` value is assigned to the hour in which it arrived because that is the requested prediction target.

Preprocessing in `forecasting/preprocessing.py`:

1. Reads the actual `_items` JSON structure. If a source export is tail-truncated, recovers only complete session objects and records the truncation in provenance.
2. Deduplicates by `_id` and `sessionID`.
3. Parses `connectionTime` as UTC and converts it with each record's `timezone` before hourly binning. DST repeated hours remain distinct UTC instants with their correct local offset.
4. Uses `connectionTime` as the cohort assignment timestamp. It does not substitute `disconnectTime` or `doneChargingTime`.
5. Excludes unusable timestamp/energy/site/timezone records; no historical values are invented and no missing target energy is replaced with `kWhRequested`.
6. Aggregates site-local hourly arrival-cohort kWh, inserting zero **recorded-arrival** hours inside each site's observed start/end window.
7. Sorts chronologically and constructs lag/rolling variables with a one-row shift.

The weather features Temperature, Solar radiation and Cloud cover are omitted because there are no reliable joined weather observations in the supplied files.

## 4. Features and leakage controls

The fitted model uses:

- Calendar: local hour, day of week, weekend indicator and month.
- Site: categorical ACN `site_id`.
- Historical demand: lag 1 hour, lag 24 hours, lag 168 hours, shifted rolling 24-hour mean, shifted rolling 168-hour mean and shifted rolling 168-hour standard deviation.

Every lag and rolling statistic is calculated from observations strictly before the target hour. The first 168 hours of each site are used as feature warm-up and excluded from supervised training. Session attributes from the target arrival cohort, `disconnectTime`, `doneChargingTime`, `kWhDelivered` for the target hour, nested `kWhRequested` and `requestedDeparture` are not used as predictors.

## 5. Train the model

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train and evaluate:

```bash
python -m forecasting.train --input data/raw/acn_sessions.json
```

The script:

- Uses `LGBMRegressor` when LightGBM is installed.
- Falls back to scikit-learn `HistGradientBoostingRegressor` if LightGBM cannot be imported.
- Pools sites with a categorical site feature (or one-hot site encoding for the fallback).
- Makes a per-site chronological 70% train / 15% validation / 15% final test split. It does not randomly shuffle observations.
- Selects among a small LightGBM/HistGradientBoosting parameter set using validation MAE.
- Refits the selected point model and separate P10/P50/P90 quantile models on train plus validation, then evaluates the untouched test partition.
- Saves deterministic model weights using a fixed random seed.

The trained model is saved to:

```text
models/ev_demand_lightgbm.joblib
```

## 6. Regenerate the next-hour forecast

The training run also creates the next-hour forecast automatically. To regenerate it from saved hourly history:

```bash
python -m forecasting.inference \
  --model models/ev_demand_lightgbm.joblib \
  --history results/hourly_site_demand.csv \
  --output results/forecast_for_simulator.csv
```

The required output columns are:

```text
timestamp,site_id,predicted_demand_kwh,p10_demand_kwh,p50_demand_kwh,p90_demand_kwh
```

This file has one next-hour forecast for each ACN site in the current training data. `timestamp` is site-local ISO-8601 time including its UTC offset. The uploaded records end in 2018–2021, so these are forecasts for the hour after each site's last historical observation, not forecasts for the current date in 2026. They demonstrate inference and are not live-operational inputs. Each site also has a different last observation timestamp. Use `kwh_to_mwh()` in `forecasting/preprocessing.py` when adapting energy values to an MWh interface (`MWh = kWh / 1000`). The dashboard demonstrates the conversion for display.

## 7. Forecast-source dashboard

The uploaded prototype ZIP contained `simulate.py` and report-generation scripts, but **no dashboard application**. A lightweight Streamlit dashboard is therefore added as `dashboard.py`:

```bash
streamlit run dashboard.py
```

Its **Forecast source** selector offers:

1. Existing seasonal-naive forecast — the original synthetic five-cluster forecast (available even when no trained model is present).
2. Trained LightGBM forecast — shown only when both the model and next-hour forecast files exist.

The LightGBM view is explicitly labelled as historical ACN/California data (the supplied observations end in 2021). It is not injected into the existing Indian-cluster dispatch logic because no defensible mapping or Indian site training observations were supplied. The original `python simulate.py` path remains independent of the trained model.

## 8. Evaluation on the uploaded records

The observed final test contains **4,753 site-hour rows**. Hyperparameters were selected on validation; these test metrics are from the final chronological holdout. Energy metrics are in kWh per site-hour.

| Forecast | MAE (kWh) | RMSE (kWh) | WAPE |
|---|---:|---:|---:|
| LightGBM | **5.440** | **18.583** | **49.41%** |
| Previous-day seasonal naive (24h) | 9.186 | 33.984 | 83.44% |
| Previous-week seasonal naive (168h) | 7.398 | 24.316 | 67.19% |

On this aggregate test split, LightGBM's MAE is 26.46% lower than the better of the two seasonal-naive baselines. This is evidence of improvement on these supplied ACN site records, not evidence of production readiness or transfer to India.

### Test performance by site

| ACN site | LightGBM MAE | Previous-day MAE | Previous-week MAE | LightGBM WAPE |
|---|---:|---:|---:|---:|
| `0001` | 14.39 kWh | 27.01 kWh | 20.55 kWh | 43.34% |
| `0002` | 15.37 kWh | 22.32 kWh | 17.88 kWh | 50.14% |
| `0019` | 1.02 kWh | 1.66 kWh | 1.69 kWh | 101.95% |

Site `0019` has very low total test energy and many zero-arrival hours. Its MAE is lower than the seasonal baselines, but its WAPE exceeds 100%; relative error is therefore poor for that sparse site. The aggregate MAE improvement must not conceal this limitation.

### Quantile coverage

Separate quantile models were trained and evaluated. Empirical test coverage is:

| Forecast quantile | Observed `actual <= forecast` | Nominal |
|---|---:|---:|
| P10 | 79.28% | 10% |
| P50 | 84.75% | 50% |
| P90 | 93.20% | 90% |
| P10–P90 interval contains actual | 90.76% | 80% |

The P10/P50/P90 outputs are **not calibrated** on this holdout. The large point mass at zero makes tie-sensitive quantile coverage especially important; these outputs should be treated as experimental ranges, not statistically calibrated prediction intervals. Calibration work on more complete site data is needed before SLA, reserve or trading decisions depend on them.

### Feature importance

The final fitted model's LightGBM gain ranking is led by rolling 24-hour mean, hour of day, previous-week lag, rolling 168-hour mean and rolling 168-hour standard deviation. Previous-day and previous-hour demand also contribute. Calendar and recent/history demand patterns therefore contribute most to this fitted model; the categorical site feature contributes less than the strongest demand-history features. See `results/feature_importance.csv` and `results/feature_importance.png`.

## 9. Generated artifacts

| Artifact | Purpose |
|---|---|
| `models/ev_demand_lightgbm.joblib` | Trained point and quantile models, plus site history for inference |
| `results/model_metrics.json` | Data quality, split sizes, hyperparameter selection, holdout metrics and quantile coverage |
| `results/test_predictions.csv` | Actual, point and quantile predictions with both seasonal-naive baselines |
| `results/forecast_for_simulator.csv` | Next-hour forecasts in the required schema |
| `results/hourly_site_demand.csv` | Auditable hourly per-site target series |
| `results/feature_importance.csv` | Numeric feature-importance ranking |
| `results/actual_vs_predicted.png` | Actual versus predicted test demand by site |
| `results/residual_distribution.png` | Residual distribution |
| `results/prediction_error_by_hour.png` | MAE by site-local hour |
| `results/prediction_error_by_site.png` | MAE by ACN site |
| `results/feature_importance.png` | Feature-importance chart |
| `results/baseline_comparison.png` | MAE comparison against both seasonal-naive baselines |
| `results/error_by_hour.csv`, `results/error_by_site.csv` | Tabular error diagnostics |

## 10. Tests

Run:

```bash
python -m pytest -q
```

Tests cover timezone conversion (including the repeated daylight-saving hour), missing timestamp/energy handling, future-feature leakage, chronological 70/15/15 splitting, deterministic predictions with a fixed seed, and kWh-to-MWh conversion.

## 11. Geographic and behavioural transfer limitations

The ACN sessions provided here are all in `America/Los_Angeles`, spanning a small number of California sites. They do not represent Indian public fast-charging or fleet-depot behaviour by default. Differences include geography, climate, vehicle mix, charger power, user dwell time, fleet operations, tariffs, holidays, charging access and data collection. The five Indian clusters are currently synthetic constructs in the simulator; no crosswalk links ACN IDs `0001`, `0002`, `0019` to the five metros.

Therefore:

- Do not label this as an India-trained demand model.
- Do not map site `0001` to Delhi or infer an Indian demand level by scaling kWh.
- Do not use the model's California forecast directly as Indian cluster demand or dispatch input.
- For a defensible Indian deployment, obtain representative session data from the target Indian sites, retrain and validate using the same chronological protocol, and recalibrate quantiles on held-out local data.

## 12. Keep the system capabilities distinct

The Perishable Grid prototype has separate layers:

- **Forecasting:** the original seasonal-naive synthetic forecast, plus the newly trained site-specific ACN LightGBM forecast.
- **Simulation and dispatch:** the original transparent synthetic heuristic dispatch policies, not production MILP/MPC.
- **Battery health and SLA shaping:** explicit illustrative assumptions and dispatch heuristics, not validated physical repurposing or measured battery degradation.
- **Market participation:** no live IEX bids, VPP revenue or production trading integration.

The trained ML model does not upgrade the heuristic dispatch into an optimisation solver. The forecast-source selector makes the two forecasting sources visible without claiming that the California model is valid for the five Indian metro clusters.
