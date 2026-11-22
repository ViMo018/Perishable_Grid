"""Schema-aware ACN-Data preprocessing for arrival-cohort demand forecasting.

The target is the sum of kWhDelivered for sessions whose connectionTime falls
inside a site-local hourly bin. It is not physical kWh consumed during that
same clock hour: sessions may continue charging after arrival.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "site_id",
    "lag_1h_kwh",
    "lag_24h_kwh",
    "lag_168h_kwh",
    "rolling_mean_24h_kwh",
    "rolling_mean_168h_kwh",
    "rolling_std_168h_kwh",
]
TARGET_COLUMN = "arrival_energy_kwh"


def normalize_site_id(value: object) -> str | None:
    """Normalize integer-like ACN site IDs without conflating other strings."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    value_str = str(value).strip()
    if not value_str:
        return None
    if re.fullmatch(r"\d+", value_str) and len(value_str) <= 4:
        return value_str.zfill(4)
    return value_str


def _decode_export(path: Path) -> tuple[list[dict], dict]:
    """Read a normal ACN JSON export, recovering complete _items if tail-truncated.

    A malformed/incomplete final object is not recovered. Earlier complete
    objects are retained and the truncation is recorded in provenance.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("_items"), list):
            return parsed["_items"], {
                "file": str(path), "parse_mode": "strict_json",
                "complete_items": len(parsed["_items"]), "truncated": False,
                "meta": parsed.get("_meta", {}),
            }
        if isinstance(parsed, list):
            return parsed, {
                "file": str(path), "parse_mode": "strict_json_list",
                "complete_items": len(parsed), "truncated": False, "meta": {},
            }
        raise ValueError(f"Unsupported JSON root structure in {path}; expected _items list")
    except json.JSONDecodeError as exc:
        marker = re.search(r'"_items"\s*:\s*\[', raw)
        if marker is None:
            raise ValueError(f"Invalid JSON and no _items array found in {path}: {exc}") from exc
        decoder = json.JSONDecoder()
        position = marker.end()
        recovered: list[dict] = []
        stopped_on_error = False
        while position < len(raw):
            while position < len(raw) and raw[position] in " \r\n\t,":
                position += 1
            if position >= len(raw) or raw[position] == "]":
                break
            try:
                item, next_position = decoder.raw_decode(raw, position)
            except json.JSONDecodeError:
                stopped_on_error = True
                break
            if not isinstance(item, dict):
                stopped_on_error = True
                break
            recovered.append(item)
            position = next_position
        return recovered, {
            "file": str(path), "parse_mode": "recovered_complete_items",
            "complete_items": len(recovered), "truncated": True,
            "recovery_stopped_at_byte": len(raw[:position].encode("utf-8")),
            "original_json_error": str(exc), "meta": {},
        }


def read_acn_sessions(paths: Iterable[str | Path]) -> tuple[pd.DataFrame, dict]:
    """Load one or more ACN exports, deduplicate, and report source data quality."""
    all_items: list[dict] = []
    provenance: list[dict] = []
    for value in paths:
        path = Path(value)
        items, source_info = _decode_export(path)
        provenance.append(source_info)
        all_items.extend(items)

    key_counts = Counter(key for item in all_items for key in item.keys())
    nested_input_keys = Counter(
        key
        for item in all_items
        for user_input in (item.get("userInputs") or [])
        if isinstance(user_input, dict)
        for key in user_input.keys()
    )

    # ACN exports use _id and sessionID. Use both for defensive de-duplication.
    seen_ids: set[str] = set()
    seen_sessions: set[str] = set()
    unique_items: list[dict] = []
    duplicate_count = 0
    for item in all_items:
        record_id = item.get("_id")
        session_id = item.get("sessionID")
        record_id = str(record_id) if record_id is not None else None
        session_id = str(session_id) if session_id is not None else None
        duplicate = ((record_id is not None and record_id in seen_ids) or
                     (session_id is not None and session_id in seen_sessions))
        if duplicate:
            duplicate_count += 1
            continue
        if record_id is not None:
            seen_ids.add(record_id)
        if session_id is not None:
            seen_sessions.add(session_id)
        unique_items.append(item)

    raw_df = pd.DataFrame(unique_items)
    if raw_df.empty:
        raise ValueError("No ACN session records were recovered from the supplied files")

    raw_count = len(raw_df)
    missing_connection_time = int(raw_df.get("connectionTime", pd.Series(index=raw_df.index, dtype=object)).isna().sum())
    missing_energy = int(raw_df.get("kWhDelivered", pd.Series(index=raw_df.index, dtype=object)).isna().sum())
    missing_timezone = int(raw_df.get("timezone", pd.Series(index=raw_df.index, dtype=object)).isna().sum())
    missing_site = int(raw_df.get("siteID", pd.Series(index=raw_df.index, dtype=object)).isna().sum())

    # No fallback to disconnectTime/doneChargingTime or kWhRequested: those do
    # not define the arrival cohort's delivered energy target.
    raw_df["connection_utc"] = pd.to_datetime(raw_df.get("connectionTime"), utc=True, errors="coerce")
    raw_df["energy_kwh"] = pd.to_numeric(raw_df.get("kWhDelivered"), errors="coerce")
    raw_df["site_id"] = raw_df.get("siteID").map(normalize_site_id)
    raw_df["timezone"] = raw_df.get("timezone").astype("string").str.strip()

    invalid_timestamp = int(raw_df["connection_utc"].isna().sum())
    invalid_energy = int((raw_df["energy_kwh"].isna() | ~np.isfinite(raw_df["energy_kwh"])).sum())
    negative_energy = int((raw_df["energy_kwh"] < 0).fillna(False).sum())
    missing_or_invalid_site = int(raw_df["site_id"].isna().sum())

    valid = (
        raw_df["connection_utc"].notna()
        & raw_df["energy_kwh"].notna()
        & np.isfinite(raw_df["energy_kwh"])
        & (raw_df["energy_kwh"] >= 0)
        & raw_df["site_id"].notna()
        & raw_df["timezone"].notna()
        & (raw_df["timezone"] != "")
    )
    clean = raw_df.loc[valid, ["_id", "sessionID", "site_id", "timezone", "connection_utc", "energy_kwh"]].copy()

    invalid_timezone_rows = []
    for timezone in clean["timezone"].dropna().unique():
        try:
            pd.Timestamp("2020-01-01", tz=str(timezone))
        except Exception:
            invalid_timezone_rows.append(str(timezone))
    if invalid_timezone_rows:
        clean = clean[~clean["timezone"].isin(invalid_timezone_rows)].copy()

    data_quality = {
        "source_files": provenance,
        "source_record_count": len(all_items),
        "duplicate_records_removed": duplicate_count,
        "unique_record_count_before_validation": raw_count,
        "records_retained": len(clean),
        "missing_connectionTime": missing_connection_time,
        "invalid_connectionTime": invalid_timestamp,
        "missing_kWhDelivered": missing_energy,
        "invalid_or_nonfinite_kWhDelivered": invalid_energy,
        "negative_kWhDelivered": negative_energy,
        "missing_timezone": missing_timezone,
        "invalid_timezone_values": invalid_timezone_rows,
        "missing_siteID": missing_site,
        "invalid_or_missing_siteID": missing_or_invalid_site,
        "top_level_fields": sorted(str(x) for x in key_counts),
        "nested_userInputs_fields": sorted(str(x) for x in nested_input_keys),
        "top_level_field_counts": dict(key_counts),
        "nested_userInputs_field_counts": dict(nested_input_keys),
        "site_ids_after_normalization": sorted(clean["site_id"].unique().tolist()),
        "site_timezones": {
            site: sorted(group["timezone"].unique().tolist())
            for site, group in clean.groupby("site_id")
        },
        "timezone_handling": "connectionTime parsed as UTC, converted to each record's timezone before local-hour binning",
        "target_definition": "sum(kWhDelivered) for sessions whose connectionTime falls in the site-local hour",
        "target_interpretation": "arrival-cohort energy demand; not physical consumption during the same clock hour",
        "features_excluded_to_avoid_leakage": [
            "disconnectTime", "doneChargingTime", "kWhDelivered from the target hour",
            "nested userInputs.kWhRequested", "nested userInputs.requestedDeparture",
        ],
        "weather_available": False,
        "weather_note": "No joined weather observations are present in the uploaded session exports.",
    }
    return clean.reset_index(drop=True), data_quality


def aggregate_hourly_arrival_demand(sessions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aggregate sessions to a complete site-local hourly grid.

    Hours between each site's first and last observed session are retained;
    hours with no recorded arrivals receive 0 kWh. Each site's actual timezone
    offset is used to compute the local-hour boundary without DST ambiguity.
    """
    rows: list[pd.DataFrame] = []
    inserted_zero_hours = 0
    per_site: dict = {}
    timezone_counts = sessions.groupby("site_id")["timezone"].nunique()
    inconsistent_sites = timezone_counts[timezone_counts > 1].index.tolist()
    if inconsistent_sites:
        raise ValueError(
            "A site_id is associated with multiple timezones; resolve the source site mapping before hourly aggregation: "
            + ", ".join(map(str, inconsistent_sites))
        )

    for (site_id, timezone), group in sessions.groupby(["site_id", "timezone"], sort=True):
        group = group.sort_values("connection_utc").copy()
        local = group["connection_utc"].dt.tz_convert(str(timezone))
        # Convert the local wall-clock hour boundary back to UTC. Subtracting
        # local minute/second components preserves the correct DST fold offset.
        elapsed_in_local_hour = (
            local.dt.minute.astype(float) * 60.0
            + local.dt.second.astype(float)
            + local.dt.microsecond.astype(float) / 1_000_000.0
        )
        group["hour_start_utc"] = group["connection_utc"] - pd.to_timedelta(elapsed_in_local_hour, unit="s")

        observed = group.groupby("hour_start_utc", as_index=True).agg(
            arrival_energy_kwh=("energy_kwh", "sum"),
            session_count=("energy_kwh", "size"),
        )
        observed.index.name = "timestamp_utc"
        start = observed.index.min()
        end = observed.index.max()
        full_hours = pd.date_range(start=start, end=end, freq="h", tz="UTC", name="timestamp_utc")
        complete = observed.reindex(full_hours)
        zero_mask = complete["arrival_energy_kwh"].isna()
        inserted_zero_hours += int(zero_mask.sum())
        complete["arrival_energy_kwh"] = complete["arrival_energy_kwh"].fillna(0.0)
        complete["session_count"] = complete["session_count"].fillna(0).astype(int)
        complete = complete.reset_index()
        complete["site_id"] = str(site_id)
        complete["timezone"] = str(timezone)
        complete["timestamp_local"] = complete["timestamp_utc"].dt.tz_convert(str(timezone))
        rows.append(complete)
        per_site[str(site_id)] = {
            "timezone": str(timezone),
            "first_hour_local": complete["timestamp_local"].iloc[0].isoformat(),
            "last_hour_local": complete["timestamp_local"].iloc[-1].isoformat(),
            "hour_rows": int(len(complete)),
            "hours_with_recorded_arrivals": int((complete["session_count"] > 0).sum()),
            "zero_recorded_arrival_hours": int((complete["session_count"] == 0).sum()),
            "session_count": int(complete["session_count"].sum()),
            "arrival_energy_kwh": float(complete["arrival_energy_kwh"].sum()),
        }

    hourly = pd.concat(rows, ignore_index=True)
    hourly = hourly.sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)
    local = hourly["timestamp_local"]
    hourly["hour_of_day"] = local.map(lambda ts: ts.hour).astype(int)
    hourly["day_of_week"] = local.map(lambda ts: ts.dayofweek).astype(int)
    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype(int)
    hourly["month"] = local.map(lambda ts: ts.month).astype(int)

    quality = {
        "hourly_rows": int(len(hourly)),
        "zero_recorded_arrival_hours_inserted": inserted_zero_hours,
        "site_summary": per_site,
        "zero_hour_interpretation": (
            "No session start was recorded in that site-local hour between the first and last observed session. "
            "This assumes the downloaded export is complete over those intervals; source completeness is not certified."
        ),
        "dst_note": "Hourly bins are formed from site-local wall-clock boundaries with UTC offsets preserved; repeated DST hours remain distinct instants.",
    }
    return hourly, quality


def build_lag_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Build one-step-ahead features using only demand observed before each hour."""
    frame = hourly.copy().sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)
    grouped = frame.groupby("site_id", sort=False)[TARGET_COLUMN]
    frame["lag_1h_kwh"] = grouped.shift(1)
    frame["lag_24h_kwh"] = grouped.shift(24)
    frame["lag_168h_kwh"] = grouped.shift(168)
    frame["rolling_mean_24h_kwh"] = grouped.transform(
        lambda series: series.shift(1).rolling(window=24, min_periods=24).mean()
    )
    frame["rolling_mean_168h_kwh"] = grouped.transform(
        lambda series: series.shift(1).rolling(window=168, min_periods=168).mean()
    )
    frame["rolling_std_168h_kwh"] = grouped.transform(
        lambda series: series.shift(1).rolling(window=168, min_periods=168).std(ddof=0)
    )
    # The 168-hour warm-up is intentionally excluded from supervised training.
    frame = frame.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    frame["site_id"] = frame["site_id"].astype(str)
    return frame.reset_index(drop=True)


def chronological_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign 70/15/15 chronological partitions independently within each site."""
    split = pd.Series(index=frame.index, dtype="object")
    for _, group in frame.sort_values(["site_id", "timestamp_utc"]).groupby("site_id", sort=True):
        indices = group.index.to_list()
        n = len(indices)
        train_end = int(np.floor(0.70 * n))
        validation_end = int(np.floor(0.85 * n))
        if train_end < 1 or validation_end <= train_end or validation_end >= n:
            raise ValueError(f"Not enough chronological samples for 70/15/15 split (site rows={n})")
        split.loc[indices[:train_end]] = "train"
        split.loc[indices[train_end:validation_end]] = "validation"
        split.loc[indices[validation_end:]] = "test"
    result = frame.copy()
    result["split"] = split
    return result


def kwh_to_mwh(values):
    """Convert energy values from kWh to MWh (1 MWh = 1,000 kWh)."""
    if isinstance(values, (pd.Series, pd.DataFrame)):
        return values / 1000.0
    return np.asarray(values) / 1000.0
