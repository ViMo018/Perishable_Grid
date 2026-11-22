"""Transparent forecast helpers used by The Perishable Grid prototype.

The only implemented forecast is the one-day seasonal-naive baseline. This is
not a trained machine-learning model.
"""
from __future__ import annotations
import pandas as pd


def add_seasonal_naive_forecasts(data: pd.DataFrame) -> pd.DataFrame:
    """Forecast each cluster/hour from the same cluster's value 24 rows earlier."""
    out = data.copy()
    for source, target in (
        ("demand_mwh", "forecast_demand_mwh"),
        ("renewable_mwh", "forecast_renewable_mwh"),
        ("iex_price_rs_kwh", "forecast_iex_price_rs_kwh"),
    ):
        out[target] = out.groupby("cluster")[source].shift(24)
    return out
