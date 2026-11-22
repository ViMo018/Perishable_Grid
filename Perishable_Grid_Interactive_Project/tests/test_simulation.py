from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scenario_engine import (  # noqa: E402
    CLUSTERS, DEFAULT_PARAMS, POLICIES, SCENARIOS, _merge_params,
    generate_scenario_data, run_all_policies,
)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_scenarios_produce_four_comparable_policies(scenario):
    data, hourly, summary = run_all_policies(scenario, DEFAULT_PARAMS)
    assert set(data.cluster.unique()) == set(CLUSTERS)
    assert set(hourly.cluster.unique()) == set(CLUSTERS)
    assert set(summary.policy) == set(POLICIES)
    assert len(summary) == 4
    assert len(hourly) == 14 * 24 * len(CLUSTERS) * 4
    required = ["reliability_pct", "gross_contribution_rs_lakh", "wear_cost_rs_lakh",
                "unserved_mwh", "renewable_utilisation_pct", "procurement_cost_rs_lakh"]
    assert summary[required].notna().all().all()



def test_scenario_overlays_apply_distinct_stress_assumptions():
    normal = generate_scenario_data(SCENARIOS[0], DEFAULT_PARAMS)
    monsoon = generate_scenario_data(SCENARIOS[1], DEFAULT_PARAMS)
    heatwave = generate_scenario_data(SCENARIOS[2], DEFAULT_PARAMS)
    surplus = generate_scenario_data(SCENARIOS[3], DEFAULT_PARAMS)
    assert np.allclose(heatwave.demand_mwh.to_numpy(), normal.demand_mwh.to_numpy() * 1.18)
    assert np.allclose(surplus.demand_mwh.to_numpy(), normal.demand_mwh.to_numpy() * 0.90)
    assert monsoon.renewable_mwh.mean() < normal.renewable_mwh.mean()
    assert surplus.renewable_mwh.mean() > normal.renewable_mwh.mean()
    evening = normal.hour.isin([17, 18, 19, 20, 21])
    assert monsoon.loc[evening, "iex_price_rs_kwh"].mean() > normal.loc[evening, "iex_price_rs_kwh"].mean()

def test_battery_soc_within_capacity_and_above_reserve():
    params = _merge_params(DEFAULT_PARAMS)
    _, hourly, _ = run_all_policies(SCENARIOS[0], params)
    nominal = params["battery_mwh_total"] / len(CLUSTERS)
    rapid_cap = nominal * params["rapid_share"] * params["rapid_soh"]
    second_cap = nominal * (1 - params["rapid_share"]) * params["second_life_soh"]
    total_cap = rapid_cap + second_cap
    reserve = params["reserve_fraction"]
    assert hourly.battery_soc_mwh.max() <= total_cap + 1e-6
    assert (hourly.battery_soc_mwh >= hourly.battery_soc_min_mwh - 1e-6).all()
    health = hourly[hourly.policy.isin(["health", "full"])]
    assert health.rapid_soc_mwh.max() <= rapid_cap + 1e-6
    assert health.second_life_soc_mwh.max() <= second_cap + 1e-6
    assert (health.rapid_soc_mwh >= rapid_cap * reserve - 1e-6).all()
    assert (health.second_life_soc_mwh >= second_cap * reserve - 1e-6).all()


def test_grid_import_and_battery_power_limits():
    params = _merge_params(DEFAULT_PARAMS)
    _, hourly, _ = run_all_policies(SCENARIOS[1], params)
    assert hourly.grid_mwh.max() <= params["grid_import_mw_per_cluster"] + 1e-6
    combined_battery_power = hourly.battery_charge_mwh + hourly.battery_discharge_mwh
    assert combined_battery_power.max() <= params["battery_mw_per_cluster"] + 1e-6


def test_customer_tariffs_stay_within_case_band_after_discount():
    cases = [
        {"retail_tariff_rs_kwh": 11.2, "shift_discount": 0.20},
        {"retail_tariff_rs_kwh": 14.0, "shift_discount": 0.20},
        {"retail_tariff_rs_kwh": 16.8, "shift_discount": 0.20},
    ]
    for raw in cases:
        p = _merge_params(raw)
        effective = p["retail_tariff_rs_kwh"] * (1 - p["shift_discount"])
        assert 11.2 - 1e-9 <= p["retail_tariff_rs_kwh"] <= 16.8 + 1e-9
        assert 11.2 - 1e-9 <= effective <= 16.8 + 1e-9


def test_five_metro_identities_preserved_and_no_cross_metro_shift():
    data = generate_scenario_data(SCENARIOS[0], DEFAULT_PARAMS)
    assert sorted(data.cluster.unique()) == sorted(CLUSTERS)
    _, hourly, _ = run_all_policies(SCENARIOS[0], DEFAULT_PARAMS)
    full = hourly[hourly.policy == "full"]
    # Signed load shift is balanced within every metro/day; no energy is moved across metros.
    per_metro_day = full.groupby(["cluster", "day"]).shifted_mwh.sum()
    assert np.allclose(per_metro_day.to_numpy(), 0.0, atol=1e-6)
    # Priority fleet demand never receives a shift allocation.
    assert (full.priority_demand_mwh >= 0).all()
    assert full.shifted_from_peak_mwh.sum() > 0


def test_no_negative_or_impossible_energy_quantities():
    _, hourly, _ = run_all_policies(SCENARIOS[2], DEFAULT_PARAMS)
    nonnegative = ["demand_mwh", "delivered_mwh", "unmet_mwh", "renewable_available_mwh",
                   "renewable_used_mwh", "grid_mwh", "battery_discharge_mwh",
                   "battery_charge_mwh", "battery_soc_mwh", "wear_cost_rs",
                   "grid_cost_rs", "renewable_cost_rs", "priority_unmet_mwh"]
    assert (hourly[nonnegative] >= -1e-8).all().all()
    assert (hourly.delivered_mwh <= hourly.demand_mwh + 1e-6).all()
    assert np.allclose(hourly.grid_mwh, hourly.grid_iex_mwh + hourly.grid_discom_mwh, atol=1e-6)
    assert (hourly.renewable_used_mwh <= hourly.renewable_available_mwh + 1e-6).all()
    assert (hourly.demand_mwh <= 240.0 + 1e-6).all()  # 1,200 MW / five independent metro clusters


def test_reserve_floor_cannot_be_set_below_five_percent():
    params = _merge_params({"reserve_fraction": 0.01})
    assert params["reserve_fraction"] == pytest.approx(0.05)


def test_triple_scarcity_thresholds_change_battery_and_customer_decisions():
    economic_battery = dict(DEFAULT_PARAMS)
    economic_battery.update({"grid_import_mw_per_cluster": 70.0,
                             "rapid_wear_rs_kwh": 0.2,
                             "second_life_wear_rs_kwh": 0.1})
    expensive_battery = dict(economic_battery)
    expensive_battery.update({"rapid_wear_rs_kwh": 10.0,
                              "second_life_wear_rs_kwh": 10.0})
    _, low_wear_hourly, _ = run_all_policies(SCENARIOS[0], economic_battery)
    _, high_wear_hourly, _ = run_all_policies(SCENARIOS[0], expensive_battery)
    low_discharge = low_wear_hourly.loc[low_wear_hourly.policy == "health", "battery_discharge_mwh"].sum()
    high_discharge = high_wear_hourly.loc[high_wear_hourly.policy == "health", "battery_discharge_mwh"].sum()
    assert high_discharge < low_discharge

    low_shift_cost = dict(DEFAULT_PARAMS)
    low_shift_cost.update({"discom_tariff_rs_kwh": 4.0, "customer_delay_cost_rs_kwh": 0.0})
    high_shift_cost = dict(low_shift_cost)
    high_shift_cost.update({"customer_delay_cost_rs_kwh": 5.0})
    _, _, low_shift_summary = run_all_policies(SCENARIOS[0], low_shift_cost)
    _, _, high_shift_summary = run_all_policies(SCENARIOS[0], high_shift_cost)
    low_shifted = low_shift_summary.loc[low_shift_summary.policy == "full", "shifted_peak_mwh"].iloc[0]
    high_shifted = high_shift_summary.loc[high_shift_summary.policy == "full", "shifted_peak_mwh"].iloc[0]
    assert high_shifted < low_shifted
