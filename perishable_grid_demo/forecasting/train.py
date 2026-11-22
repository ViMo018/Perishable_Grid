"""Train/evaluate an ACN arrival-cohort EV energy-demand forecaster.

Primary estimator: LightGBM Regressor. Fallback: scikit-learn
HistGradientBoostingRegressor with one-hot encoded site identifiers.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from .preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    aggregate_hourly_arrival_demand,
    build_lag_features,
    chronological_split,
    kwh_to_mwh,
    read_acn_sessions,
)
from .inference import predict_next_hour

SEED = 42

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False


def _metric_values(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    denominator = float(np.abs(actual).sum())
    return {
        "mae_kwh": float(np.mean(np.abs(errors))),
        "rmse_kwh": float(np.sqrt(np.mean(errors ** 2))),
        "wape_pct": float(100.0 * np.abs(errors).sum() / denominator) if denominator > 0 else None,
        "sample_count": int(len(actual)),
        "actual_energy_kwh": float(actual.sum()),
    }


def _make_matrix(frame: pd.DataFrame, estimator_type: str, site_categories: list[str] | None = None,
                 matrix_columns: list[str] | None = None) -> pd.DataFrame:
    x = frame[FEATURE_COLUMNS].copy()
    if estimator_type == "lightgbm":
        x["site_id"] = pd.Categorical(x["site_id"].astype(str), categories=site_categories)
    else:
        x["site_id"] = x["site_id"].astype(str)
        x = pd.get_dummies(x, columns=["site_id"], dtype=float)
        if matrix_columns is not None:
            x = x.reindex(columns=matrix_columns, fill_value=0.0)
    return x


def _fit_estimator(x_train: pd.DataFrame, y_train: pd.Series, estimator_type: str,
                   params: dict, objective: str = "point", alpha: float | None = None):
    if estimator_type == "lightgbm":
        objective_name = "regression_l1" if objective == "point" else "quantile"
        model = LGBMRegressor(
            objective=objective_name,
            alpha=alpha if alpha is not None else 0.5,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params.get("max_depth", -1),
            min_child_samples=params["min_child_samples"],
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=SEED,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(x_train, y_train, categorical_feature=["site_id"])
        return model

    loss = "absolute_error" if objective == "point" else "quantile"
    model = HistGradientBoostingRegressor(
        loss=loss,
        quantile=alpha if objective == "quantile" else None,
        learning_rate=params["learning_rate"],
        max_iter=params["n_estimators"],
        max_leaf_nodes=params["num_leaves"],
        min_samples_leaf=params["min_child_samples"],
        l2_regularization=params.get("reg_lambda", 1.0),
        random_state=SEED,
    )
    model.fit(x_train, y_train)
    return model


def _candidate_parameters() -> list[dict]:
    return [
        {"n_estimators": 250, "learning_rate": 0.04, "num_leaves": 15,
         "max_depth": -1, "min_child_samples": 25, "reg_lambda": 1.0},
        {"n_estimators": 350, "learning_rate": 0.03, "num_leaves": 31,
         "max_depth": -1, "min_child_samples": 40, "reg_lambda": 2.0},
        {"n_estimators": 220, "learning_rate": 0.05, "num_leaves": 7,
         "max_depth": 5, "min_child_samples": 20, "reg_lambda": 1.5},
    ]


def _plot_actual_vs_predicted(test: pd.DataFrame, out_dir: Path) -> None:
    sites = sorted(test["site_id"].astype(str).unique())
    ncols = 2
    nrows = int(math.ceil(len(sites) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows), squeeze=False)
    for ax, site in zip(axes.flat, sites):
        part = test[test.site_id.astype(str) == site].sort_values("timestamp_utc")
        ax.plot(part["timestamp_local"], part[TARGET_COLUMN], label="Actual", linewidth=1.0)
        ax.plot(part["timestamp_local"], part["predicted_demand_kwh"], label="LightGBM", linewidth=1.0)
        ax.set_title(f"ACN site {site}")
        ax.set_ylabel("Arrival-cohort energy (kWh)")
        ax.grid(alpha=0.2)
        ax.legend()
        ax.tick_params(axis="x", rotation=20)
    for ax in axes.flat[len(sites):]:
        ax.set_visible(False)
    fig.suptitle("Actual vs predicted arrival-cohort energy — chronological test", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "actual_vs_predicted.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _make_diagnostic_charts(test: pd.DataFrame, importance: pd.DataFrame, out_dir: Path) -> None:
    residual = test["predicted_demand_kwh"] - test[TARGET_COLUMN]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(residual, bins=50)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title("Test residual distribution (prediction − actual)")
    ax.set_xlabel("Residual (kWh)")
    ax.set_ylabel("Hourly site observations")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_distribution.png", dpi=170)
    plt.close(fig)

    by_hour = test.assign(abs_error=(test["predicted_demand_kwh"] - test[TARGET_COLUMN]).abs()).groupby("hour_of_day").abs_error.mean()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(by_hour.index, by_hour.values)
    ax.set_xticks(range(24))
    ax.set_xlabel("Site-local hour of day")
    ax.set_ylabel("Mean absolute error (kWh)")
    ax.set_title("Prediction error by local hour")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_error_by_hour.png", dpi=170)
    plt.close(fig)

    by_site = test.assign(abs_error=(test["predicted_demand_kwh"] - test[TARGET_COLUMN]).abs()).groupby("site_id").abs_error.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(by_site.index.astype(str), by_site.values)
    ax.set_xlabel("ACN site identifier")
    ax.set_ylabel("Mean absolute error (kWh)")
    ax.set_title("Prediction error by site")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_error_by_site.png", dpi=170)
    plt.close(fig)

    if not importance.empty:
        shown = importance.sort_values("importance", ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(shown.feature, shown.importance)
        ax.set_xlabel("Feature importance (gain or validation permutation score)")
        ax.set_title("Model feature importance")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "feature_importance.png", dpi=170)
        plt.close(fig)

    baseline_mae = {
        "LightGBM": float((test.predicted_demand_kwh - test[TARGET_COLUMN]).abs().mean()),
        "Previous day (24h)": float((test.lag_24h_kwh - test[TARGET_COLUMN]).abs().mean()),
        "Previous week (168h)": float((test.lag_168h_kwh - test[TARGET_COLUMN]).abs().mean()),
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(list(baseline_mae), list(baseline_mae.values()))
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Chronological test MAE: model vs seasonal-naive baselines")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_comparison.png", dpi=170)
    plt.close(fig)


def _feature_importance(model, estimator_type: str, x_validation: pd.DataFrame,
                        y_validation: pd.Series) -> pd.DataFrame:
    if estimator_type == "lightgbm":
        values = model.booster_.feature_importance(importance_type="gain")
        names = model.booster_.feature_name()
        return pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
    result = permutation_importance(model, x_validation, y_validation, n_repeats=4,
                                    random_state=SEED, scoring="neg_mean_absolute_error")
    return pd.DataFrame({"feature": x_validation.columns,
                         "importance": result.importances_mean}).sort_values("importance", ascending=False)


def train(input_path: Path, project_root: Path) -> dict:
    model_dir = project_root / "models"
    results_dir = project_root / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    sessions, data_quality = read_acn_sessions([input_path])
    hourly, hourly_quality = aggregate_hourly_arrival_demand(sessions)
    # Persist auditable preprocessing output and summary.
    hourly.to_csv(results_dir / "hourly_site_demand.csv", index=False)
    features = build_lag_features(hourly)
    split_data = chronological_split(features)
    split_counts = split_data.groupby(["site_id", "split"]).size().unstack(fill_value=0)
    split_summary = {site: {key: int(value) for key, value in row.items()}
                     for site, row in split_counts.to_dict(orient="index").items()}

    train_df = split_data[split_data.split == "train"].copy()
    validation_df = split_data[split_data.split == "validation"].copy()
    test_df = split_data[split_data.split == "test"].copy()
    estimator_type = "lightgbm" if LIGHTGBM_AVAILABLE else "hist_gradient_boosting"
    site_categories = sorted(features.site_id.astype(str).unique().tolist())

    if estimator_type == "lightgbm":
        train_x = _make_matrix(train_df, estimator_type, site_categories)
        validation_x = _make_matrix(validation_df, estimator_type, site_categories)
        test_x = _make_matrix(test_df, estimator_type, site_categories)
        matrix_columns = None
    else:
        train_x = _make_matrix(train_df, estimator_type)
        matrix_columns = train_x.columns.tolist()
        validation_x = _make_matrix(validation_df, estimator_type, matrix_columns=matrix_columns)
        test_x = _make_matrix(test_df, estimator_type, matrix_columns=matrix_columns)

    y_train = train_df[TARGET_COLUMN]
    y_validation = validation_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    candidate_results = []
    best_model = None
    best_params = None
    best_validation_mae = float("inf")
    for params in _candidate_parameters():
        candidate_model = _fit_estimator(train_x, y_train, estimator_type, params, objective="point")
        validation_prediction = np.maximum(0.0, candidate_model.predict(validation_x))
        metrics = _metric_values(y_validation, validation_prediction)
        candidate_results.append({"parameters": params, **metrics})
        if metrics["mae_kwh"] < best_validation_mae:
            best_validation_mae = metrics["mae_kwh"]
            best_model = candidate_model
            best_params = params

    # Refit the selected configuration on train + validation. Test remains unseen.
    train_validation = pd.concat([train_df, validation_df], ignore_index=True)
    if estimator_type == "lightgbm":
        train_validation_x = _make_matrix(train_validation, estimator_type, site_categories)
    else:
        train_validation_x = _make_matrix(train_validation, estimator_type, matrix_columns=matrix_columns)
    point_model = _fit_estimator(train_validation_x, train_validation[TARGET_COLUMN],
                                 estimator_type, best_params, objective="point")
    point_prediction = np.maximum(0.0, point_model.predict(test_x))

    # Train pinball-loss quantile models on train + validation, then report test coverage.
    quantile_models = {}
    quantile_predictions = {}
    for label, alpha in [("p10", 0.10), ("p50", 0.50), ("p90", 0.90)]:
        quantile_models[label] = _fit_estimator(
            train_validation_x, train_validation[TARGET_COLUMN], estimator_type,
            best_params, objective="quantile", alpha=alpha,
        )
        quantile_predictions[label] = np.maximum(0.0, quantile_models[label].predict(test_x))
    quantile_matrix = np.column_stack([
        quantile_predictions["p10"], quantile_predictions["p50"], quantile_predictions["p90"]
    ])
    quantile_matrix.sort(axis=1)

    test_out = test_df.copy()
    test_out["actual_demand_kwh"] = y_test.to_numpy()
    test_out["predicted_demand_kwh"] = point_prediction
    test_out["p10_demand_kwh"] = quantile_matrix[:, 0]
    test_out["p50_demand_kwh"] = quantile_matrix[:, 1]
    test_out["p90_demand_kwh"] = quantile_matrix[:, 2]
    test_out["seasonal_naive_24h_kwh"] = test_out["lag_24h_kwh"]
    test_out["seasonal_naive_168h_kwh"] = test_out["lag_168h_kwh"]

    model_metrics = {
        "objective": "next-hour arrival-cohort energy demand per ACN site",
        "target_column": TARGET_COLUMN,
        "target_units": "kWh",
        "target_interpretation": "sum of kWhDelivered for sessions starting in the hour; not physical consumption during that hour",
        "estimator": "LightGBM Regressor" if estimator_type == "lightgbm" else "HistGradientBoostingRegressor fallback",
        "estimator_type": estimator_type,
        "lightgbm_available": bool(LIGHTGBM_AVAILABLE),
        "chronological_split": "70% train / 15% validation / 15% final test, independently per site, ordered by local-hour instant",
        "split_counts_by_site": split_summary,
        "validation_candidate_results": candidate_results,
        "selected_parameters": best_params,
        "validation": {
            "model": _metric_values(y_validation, np.maximum(0.0, best_model.predict(validation_x))),
            "seasonal_naive_previous_day_24h": _metric_values(y_validation, validation_df.lag_24h_kwh),
            "seasonal_naive_previous_week_168h": _metric_values(y_validation, validation_df.lag_168h_kwh),
        },
        "test": {
            "model": _metric_values(y_test, point_prediction),
            "seasonal_naive_previous_day_24h": _metric_values(y_test, test_out.seasonal_naive_24h_kwh),
            "seasonal_naive_previous_week_168h": _metric_values(y_test, test_out.seasonal_naive_168h_kwh),
        },
        "quantile_test_coverage": {
            "p10_actual_le_forecast": float(np.mean(y_test.to_numpy() <= quantile_matrix[:, 0])),
            "p50_actual_le_forecast": float(np.mean(y_test.to_numpy() <= quantile_matrix[:, 1])),
            "p90_actual_le_forecast": float(np.mean(y_test.to_numpy() <= quantile_matrix[:, 2])),
            "p10_to_p90_interval_coverage": float(np.mean(
                (y_test.to_numpy() >= quantile_matrix[:, 0]) & (y_test.to_numpy() <= quantile_matrix[:, 2])
            )),
            "p10_to_p90_mean_width_kwh": float(np.mean(quantile_matrix[:, 2] - quantile_matrix[:, 0])),
            "nominal_expected_p10_p50_p90_coverage": [0.10, 0.50, 0.90],
            "nominal_expected_p10_p90_interval_coverage": 0.80,
        },
        "weather_features_used": False,
        "weather_note": "No reliable weather observations were included in the uploaded data; weather features were omitted.",
        "data_quality": data_quality,
        "hourly_aggregation_quality": hourly_quality,
        "feature_columns": FEATURE_COLUMNS,
        "feature_engineering": "All lags and rolling statistics use prior observations only (shift(1)); 168-hour warm-up rows excluded.",
        "geographic_transfer_warning": (
            "All supplied sites are in America/Los_Angeles. This fitted model represents those ACN sites only; "
            "it is not calibrated to Delhi NCR, Mumbai MMR, Bengaluru, Hyderabad, or Chennai."
        ),
    }

    best_naive_mae = min(
        model_metrics["test"]["seasonal_naive_previous_day_24h"]["mae_kwh"],
        model_metrics["test"]["seasonal_naive_previous_week_168h"]["mae_kwh"],
    )
    model_mae = model_metrics["test"]["model"]["mae_kwh"]
    model_metrics["test"]["beats_previous_day_baseline"] = model_mae < model_metrics["test"]["seasonal_naive_previous_day_24h"]["mae_kwh"]
    model_metrics["test"]["beats_previous_week_baseline"] = model_mae < model_metrics["test"]["seasonal_naive_previous_week_168h"]["mae_kwh"]
    per_site_metrics = {}
    for site_id, site_rows in test_out.groupby("site_id", sort=True):
        actual_site = site_rows[TARGET_COLUMN]
        model_site = site_rows["predicted_demand_kwh"]
        prev_day_site = site_rows["lag_24h_kwh"]
        prev_week_site = site_rows["lag_168h_kwh"]
        per_site_metrics[str(site_id)] = {
            "model": _metric_values(actual_site, model_site),
            "seasonal_naive_previous_day_24h": _metric_values(actual_site, prev_day_site),
            "seasonal_naive_previous_week_168h": _metric_values(actual_site, prev_week_site),
        }
    model_metrics["test"]["by_site"] = per_site_metrics
    model_metrics["test"]["beats_best_seasonal_naive_baseline"] = model_mae < best_naive_mae
    model_metrics["test"]["mae_improvement_vs_best_naive_pct"] = (
        100.0 * (best_naive_mae - model_mae) / best_naive_mae if best_naive_mae > 0 else None
    )
    model_metrics["conclusion"] = (
        "LightGBM beats both seasonal-naive baselines on test MAE. This supports further site-specific validation, "
        "not transfer to Indian metro clusters."
        if model_metrics["test"]["beats_previous_day_baseline"] and model_metrics["test"]["beats_previous_week_baseline"]
        else "The trained model does not beat both seasonal-naive baselines on test MAE; do not claim it is useful as a replacement."
    )

    # Save test outputs with timestamps expressed in the site's local timezone.
    test_out["timestamp"] = test_out["timestamp_local"].map(lambda value: value.isoformat())
    test_columns = [
        "timestamp", "site_id", TARGET_COLUMN, "actual_demand_kwh", "predicted_demand_kwh",
        "p10_demand_kwh", "p50_demand_kwh", "p90_demand_kwh",
        "seasonal_naive_24h_kwh", "seasonal_naive_168h_kwh", "hour_of_day", "day_of_week",
    ]
    test_out[test_columns].to_csv(results_dir / "test_predictions.csv", index=False)

    # Save feature importance from LightGBM gain (or fallback validation permutation).
    importance = _feature_importance(
        point_model if estimator_type == "lightgbm" else best_model,
        estimator_type,
        train_validation_x if estimator_type == "lightgbm" else validation_x,
        train_validation[TARGET_COLUMN] if estimator_type == "lightgbm" else y_validation,
    )
    importance.to_csv(results_dir / "feature_importance.csv", index=False)
    error_frame = test_out.assign(abs_error=(test_out["predicted_demand_kwh"] - test_out[TARGET_COLUMN]).abs())
    error_by_hour = error_frame.groupby("hour_of_day").agg(
        mae_kwh=("abs_error", "mean"), sample_count=("abs_error", "size")
    ).reset_index()
    error_by_site = error_frame.groupby("site_id").agg(
        mae_kwh=("abs_error", "mean"), sample_count=("abs_error", "size"),
        actual_energy_kwh=(TARGET_COLUMN, "sum"),
        absolute_error_kwh=("abs_error", "sum"),
    ).reset_index()
    error_by_site["wape_pct"] = np.where(
        error_by_site["actual_energy_kwh"] > 0,
        100.0 * error_by_site["absolute_error_kwh"] / error_by_site["actual_energy_kwh"], np.nan
    )
    error_by_hour.to_csv(results_dir / "error_by_hour.csv", index=False)
    error_by_site.to_csv(results_dir / "error_by_site.csv", index=False)

    # Bundle includes estimator(s) and training history for next-hour inference.
    package = {
        "model": point_model,
        "quantile_models": quantile_models,
        "estimator_type": estimator_type,
        "feature_columns": FEATURE_COLUMNS,
        "site_categories": site_categories,
        "matrix_columns": matrix_columns,
        "best_parameters": best_params,
        "random_seed": SEED,
        "training_history": hourly[[
            "timestamp_utc", "timestamp_local", "site_id", "timezone", TARGET_COLUMN, "session_count",
            "hour_of_day", "day_of_week", "is_weekend", "month",
        ]].copy(),
    }
    model_path = model_dir / "ev_demand_lightgbm.joblib"
    joblib.dump(package, model_path, compress=3)

    forecast = predict_next_hour(model_path, hourly)
    forecast.to_csv(results_dir / "forecast_for_simulator.csv", index=False)
    model_metrics["next_hour_forecast_note"] = (
        "One step after the final available observation for each site. These timestamps follow the supplied historical "
        "records (2018-2021), not the current date; forecasts are a pipeline demonstration, not live operational forecasts. "
        "Each site has a different latest observation timestamp."
    )
    model_metrics["next_hour_forecast_rows"] = forecast.to_dict(orient="records")

    model_metrics["artifacts"] = {
        "model": str(model_path.relative_to(project_root)),
        "test_predictions": "results/test_predictions.csv",
        "next_hour_forecast": "results/forecast_for_simulator.csv",
        "hourly_site_demand": "results/hourly_site_demand.csv",
        "feature_importance": "results/feature_importance.csv",
    }
    with (results_dir / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(model_metrics, file, indent=2, allow_nan=False)

    _plot_actual_vs_predicted(test_out, results_dir)
    _make_diagnostic_charts(test_out, importance, results_dir)

    print("\nEstimator:", model_metrics["estimator"])
    print("Test metrics (kWh):")
    for name, metrics in model_metrics["test"].items():
        if isinstance(metrics, dict) and "mae_kwh" in metrics:
            print(f"  {name}: MAE={metrics['mae_kwh']:.3f}, RMSE={metrics['rmse_kwh']:.3f}, WAPE={metrics['wape_pct']:.2f}%")
    print("\n", model_metrics["conclusion"])
    print("Forecast rows for simulator adapter:")
    print(forecast.to_string(index=False))
    return model_metrics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the Perishable Grid ACN demand model")
    parser.add_argument("--input", type=Path, default=Path("data/raw/acn_sessions.json"),
                        help="ACN JSON export containing _items")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Perishable Grid project directory")
    args = parser.parse_args(argv)
    train(args.input, args.project_root)


if __name__ == "__main__":
    main()
