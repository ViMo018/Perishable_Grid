"""Load the trained model and create leakage-safe next-hour ACN forecasts."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocessing import FEATURE_COLUMNS, TARGET_COLUMN


def _features_for_next_hour(history: pd.DataFrame) -> pd.DataFrame:
    """Construct a single future row from the latest 168+ observed hourly values."""
    from .preprocessing import build_lag_features

    rows = []
    for site_id, group in history.sort_values("timestamp_utc").groupby("site_id", sort=True):
        group = group.sort_values("timestamp_utc").copy()
        if len(group) < 168:
            continue
        latest = group.iloc[-1]
        future_utc = latest["timestamp_utc"] + pd.Timedelta(hours=1)
        timezone = str(latest["timezone"])
        local_future = future_utc.tz_convert(timezone)
        future_target = pd.DataFrame([{
            "timestamp_utc": future_utc,
            "site_id": str(site_id),
            "timezone": timezone,
            "timestamp_local": local_future,
            TARGET_COLUMN: np.nan,
            "session_count": 0,
            "hour_of_day": local_future.hour,
            "day_of_week": local_future.dayofweek,
            "is_weekend": int(local_future.dayofweek >= 5),
            "month": local_future.month,
        }])
        combined = pd.concat([group, future_target], ignore_index=True)
        # For inference only, form shifted features directly, without requiring
        # a known future target value.
        demand = combined[TARGET_COLUMN]
        idx = len(combined) - 1
        feature_row = {
            "hour_of_day": int(local_future.hour),
            "day_of_week": int(local_future.dayofweek),
            "is_weekend": int(local_future.dayofweek >= 5),
            "month": int(local_future.month),
            "site_id": str(site_id),
            "lag_1h_kwh": float(demand.iloc[idx - 1]),
            "lag_24h_kwh": float(demand.iloc[idx - 24]),
            "lag_168h_kwh": float(demand.iloc[idx - 168]),
            "rolling_mean_24h_kwh": float(demand.iloc[idx - 24:idx].mean()),
            "rolling_mean_168h_kwh": float(demand.iloc[idx - 168:idx].mean()),
            "rolling_std_168h_kwh": float(demand.iloc[idx - 168:idx].std(ddof=0)),
            "timestamp_utc": future_utc,
            "timestamp_local": local_future,
            "timezone": timezone,
        }
        rows.append(feature_row)
    return pd.DataFrame(rows)


def _prepare_matrix(frame: pd.DataFrame, package: dict) -> pd.DataFrame:
    x = frame[FEATURE_COLUMNS].copy()
    if package["estimator_type"] == "lightgbm":
        x["site_id"] = pd.Categorical(x["site_id"], categories=package["site_categories"])
    else:
        x["site_id"] = x["site_id"].astype(str)
        x = pd.get_dummies(x, columns=["site_id"], dtype=float)
        x = x.reindex(columns=package["matrix_columns"], fill_value=0.0)
    return x


def predict_next_hour(model_path: str | Path, hourly_history: pd.DataFrame) -> pd.DataFrame:
    """Return one forecast row per site for the next site-local hourly interval."""
    package = joblib.load(model_path)
    feature_rows = _features_for_next_hour(hourly_history)
    if feature_rows.empty:
        raise ValueError("No site has 168 hours of history; cannot build lag features")
    matrix = _prepare_matrix(feature_rows, package)
    predicted = np.maximum(0.0, package["model"].predict(matrix))
    quantiles = {}
    for label, quantile_model in package["quantile_models"].items():
        quantiles[label] = np.maximum(0.0, quantile_model.predict(matrix))
    quantile_matrix = np.column_stack([quantiles["p10"], quantiles["p50"], quantiles["p90"]])
    quantile_matrix.sort(axis=1)
    return pd.DataFrame({
        "timestamp": [ts.isoformat() for ts in feature_rows["timestamp_local"]],
        "site_id": feature_rows["site_id"].astype(str).to_numpy(),
        "predicted_demand_kwh": predicted,
        "p10_demand_kwh": quantile_matrix[:, 0],
        "p50_demand_kwh": quantile_matrix[:, 1],
        "p90_demand_kwh": quantile_matrix[:, 2],
    }).sort_values(["site_id", "timestamp"]).reset_index(drop=True)


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate the next-hour ACN site forecast")
    parser.add_argument("--model", type=Path, default=Path("models/ev_demand_lightgbm.joblib"))
    parser.add_argument("--history", type=Path, default=Path("results/hourly_site_demand.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/forecast_for_simulator.csv"))
    args = parser.parse_args(argv)
    history = pd.read_csv(args.history, dtype={"site_id": str})
    history["timestamp_utc"] = pd.to_datetime(history["timestamp_utc"], utc=True)
    forecast = predict_next_hour(args.model, history)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False)
    print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
