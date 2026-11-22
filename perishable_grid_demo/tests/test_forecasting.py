from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    aggregate_hourly_arrival_demand,
    build_lag_features,
    chronological_split,
    kwh_to_mwh,
    read_acn_sessions,
)
from forecasting.train import LIGHTGBM_AVAILABLE, _fit_estimator, _make_matrix


def _write_sessions(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps({"_meta": {"site": "test"}, "_items": records}), encoding="utf-8")


def test_correct_timestamp_conversion_to_site_timezone(tmp_path):
    source = tmp_path / "sessions.json"
    _write_sessions(source, [{
        "_id": "a", "sessionID": "s1", "siteID": "0001",
        "connectionTime": "Wed, 05 Sep 2018 11:04:13 GMT",
        "timezone": "America/Los_Angeles", "kWhDelivered": 7.5,
    }])
    sessions, _ = read_acn_sessions([source])
    hourly, _ = aggregate_hourly_arrival_demand(sessions)
    assert hourly.iloc[0]["timestamp_local"].isoformat() == "2018-09-05T04:00:00-07:00"
    assert hourly.iloc[0][TARGET_COLUMN] == pytest.approx(7.5)


def test_missing_timestamps_and_energy_are_reported_and_dropped(tmp_path):
    source = tmp_path / "sessions.json"
    records = [
        {"_id": "good", "sessionID": "good", "siteID": "1", "connectionTime": "Wed, 05 Sep 2018 11:04:13 GMT", "timezone": "America/Los_Angeles", "kWhDelivered": 7.5},
        {"_id": "bad_time", "sessionID": "bad_time", "siteID": "1", "connectionTime": None, "timezone": "America/Los_Angeles", "kWhDelivered": 2.0},
        {"_id": "bad_energy", "sessionID": "bad_energy", "siteID": "1", "connectionTime": "Wed, 05 Sep 2018 12:04:13 GMT", "timezone": "America/Los_Angeles", "kWhDelivered": None},
    ]
    _write_sessions(source, records)
    sessions, quality = read_acn_sessions([source])
    assert len(sessions) == 1
    assert quality["missing_connectionTime"] == 1
    assert quality["missing_kWhDelivered"] == 1
    assert quality["records_retained"] == 1


def test_no_future_data_leakage_in_lag_and_rolling_features():
    utc = pd.date_range("2021-01-01", periods=260, freq="h", tz="UTC")
    base = pd.DataFrame({
        "timestamp_utc": utc,
        "timestamp_local": utc.tz_convert("America/Los_Angeles"),
        "site_id": "0001",
        "timezone": "America/Los_Angeles",
        TARGET_COLUMN: np.arange(260, dtype=float),
        "session_count": 1,
        "hour_of_day": utc.tz_convert("America/Los_Angeles").hour,
        "day_of_week": utc.tz_convert("America/Los_Angeles").dayofweek,
        "is_weekend": (utc.tz_convert("America/Los_Angeles").dayofweek >= 5).astype(int),
        "month": utc.tz_convert("America/Los_Angeles").month,
    })
    changed = base.copy()
    changed.loc[changed.index >= 240, TARGET_COLUMN] += 100_000
    f1 = build_lag_features(base).set_index("timestamp_utc")
    f2 = build_lag_features(changed).set_index("timestamp_utc")
    timestamp = utc[240]
    np.testing.assert_allclose(f1.loc[timestamp, FEATURE_COLUMNS[:-1] + [FEATURE_COLUMNS[-1]]].drop(labels=["site_id"]).astype(float),
                               f2.loc[timestamp, FEATURE_COLUMNS].drop(labels=["site_id"]).astype(float))
    assert f1.loc[timestamp, "lag_1h_kwh"] == pytest.approx(239.0)
    assert f1.loc[timestamp, "rolling_mean_24h_kwh"] == pytest.approx(np.mean(np.arange(216, 240)))


def test_chronological_split_is_ordered_and_70_15_15_per_site():
    rows = []
    for site in ["0001", "0002"]:
        for i, ts in enumerate(pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")):
            rows.append({"site_id": site, "timestamp_utc": ts, TARGET_COLUMN: float(i)})
    source = pd.DataFrame(rows)
    split = chronological_split(source)
    for site, group in split.groupby("site_id"):
        labels = group.sort_values("timestamp_utc")["split"].tolist()
        assert labels == ["train"] * 70 + ["validation"] * 15 + ["test"] * 15


def test_reproducible_predictions_with_fixed_seed():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "hour_of_day": rng.integers(0, 24, 180),
        "day_of_week": rng.integers(0, 7, 180),
        "is_weekend": rng.integers(0, 2, 180),
        "month": rng.integers(1, 13, 180),
        "site_id": np.where(np.arange(180) % 2 == 0, "0001", "0002"),
        "lag_1h_kwh": rng.uniform(0, 20, 180),
        "lag_24h_kwh": rng.uniform(0, 20, 180),
        "lag_168h_kwh": rng.uniform(0, 20, 180),
        "rolling_mean_24h_kwh": rng.uniform(0, 20, 180),
        "rolling_mean_168h_kwh": rng.uniform(0, 20, 180),
        "rolling_std_168h_kwh": rng.uniform(0, 10, 180),
    })
    target = frame["lag_1h_kwh"] * 0.6 + frame["hour_of_day"] * 0.2
    estimator_type = "lightgbm" if LIGHTGBM_AVAILABLE else "hist_gradient_boosting"
    categories = ["0001", "0002"]
    matrix = _make_matrix(frame, estimator_type, categories if estimator_type == "lightgbm" else None)
    params = {"n_estimators": 35, "learning_rate": 0.05, "num_leaves": 7,
              "max_depth": 5, "min_child_samples": 8, "reg_lambda": 1.0}
    model_a = _fit_estimator(matrix, target, estimator_type, params)
    model_b = _fit_estimator(matrix, target, estimator_type, params)
    np.testing.assert_allclose(model_a.predict(matrix), model_b.predict(matrix), rtol=0, atol=1e-10)


def test_kwh_to_mwh_conversion():
    assert kwh_to_mwh(1_500.0).item() == pytest.approx(1.5)
    series = pd.Series([0.0, 250.0, 1_000.0])
    converted = kwh_to_mwh(series)
    np.testing.assert_allclose(converted.to_numpy(), [0.0, 0.25, 1.0])


def test_site_ids_are_normalized_and_dst_repeated_hour_is_distinguished(tmp_path):
    source = tmp_path / "dst.json"
    _write_sessions(source, [
        {"_id": "a", "sessionID": "a", "siteID": 2, "connectionTime": "Sun, 01 Nov 2020 08:30:00 GMT", "timezone": "America/Los_Angeles", "kWhDelivered": 2.0},
        {"_id": "b", "sessionID": "b", "siteID": "0002", "connectionTime": "Sun, 01 Nov 2020 09:30:00 GMT", "timezone": "America/Los_Angeles", "kWhDelivered": 3.0},
    ])
    sessions, _ = read_acn_sessions([source])
    hourly, _ = aggregate_hourly_arrival_demand(sessions)
    assert sessions.site_id.unique().tolist() == ["0002"]
    assert len(hourly) == 2
    assert hourly.timestamp_local.map(lambda value: value.hour).tolist() == [1, 1]
    assert hourly.timestamp_local.map(lambda value: value.utcoffset().total_seconds()).tolist() == [-25200.0, -28800.0]
