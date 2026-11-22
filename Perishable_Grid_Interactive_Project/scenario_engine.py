"""Scenario-based extension of the original Perishable Grid heuristic.

The original simulate.py implementation remains intact and is still the
reproducible reference run. This module parameterises that same transparent
hourly dispatch structure for interactive scenarios; it is not a MILP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from forecast import add_seasonal_naive_forecasts

CLUSTERS = ["Delhi NCR", "Mumbai MMR", "Bengaluru", "Hyderabad", "Chennai"]
POLICIES = ["baseline", "forecast", "health", "full"]
POLICY_LABELS = {
    "baseline": "Rule-based baseline",
    "forecast": "Price-aware dispatch",
    "health": "Health-aware dispatch",
    "full": "Health + SLA strategy",
}
SCENARIOS = [
    "Normal operating day",
    "Monsoon cloud-cover and price shock",
    "Heatwave with customer demand surge",
    "Renewable generation surplus during low demand",
]

DEFAULT_PARAMS: dict[str, Any] = {
    "seed": 42,
    "test_days": 14,
    "battery_mwh_total": 600.0,
    "battery_mw_per_cluster": 30.0,
    "grid_import_mw_per_cluster": 38.0,
    "charging_capacity_mw_total": 1200.0,
    "reserve_fraction": 0.05,
    "line_efficiency": 0.975,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "discom_tariff_rs_kwh": 8.50,
    "renewable_cost_rs_kwh": 3.50,
    "retail_tariff_rs_kwh": 14.00,
    "iex_price_multiplier": 1.0,
    "rapid_share": 0.70,
    "rapid_soh": 0.90,
    "second_life_soh": 0.80,
    "rapid_wear_rs_kwh": 1.20,
    "second_life_wear_rs_kwh": 0.65,
    "uniform_wear_rs_kwh": 1.00,
    "flexible_demand_share": 0.40,
    "shift_share_of_flexible": 0.15,
    "shift_discount": 0.10,
    "price_elasticity": 1.0,
    "customer_delay_cost_rs_kwh": 0.50,
    "priority_segment_share": 0.10,
    "logistics_segment_share": 0.30,
    "ridehail_segment_share": 0.35,
    "retail_segment_share": 0.25,
}


def _merge_params(params: dict[str, Any] | None) -> dict[str, Any]:
    cfg = DEFAULT_PARAMS.copy()
    if params:
        cfg.update(params)
    # Constrain exposed decision variables to the physical / commercial limits.
    cfg["rapid_share"] = float(np.clip(cfg["rapid_share"], 0.0, 1.0))
    cfg["flexible_demand_share"] = float(np.clip(cfg["flexible_demand_share"], 0.0, 1.0))
    cfg["shift_share_of_flexible"] = float(np.clip(cfg["shift_share_of_flexible"], 0.0, 0.30))
    cfg["shift_discount"] = float(np.clip(cfg["shift_discount"], 0.0, 0.20))
    cfg["reserve_fraction"] = float(np.clip(cfg["reserve_fraction"], 0.05, 0.30))
    cfg["retail_tariff_rs_kwh"] = float(np.clip(
        cfg["retail_tariff_rs_kwh"], 0.8 * DEFAULT_PARAMS["retail_tariff_rs_kwh"],
        1.2 * DEFAULT_PARAMS["retail_tariff_rs_kwh"]))
    # The discounted, shifted-customer tariff must also stay within the case's
    # +/-20% band around ₹14/kWh, not merely the undiscounted tariff.
    minimum_customer_tariff = 0.8 * DEFAULT_PARAMS["retail_tariff_rs_kwh"]
    max_discount_at_tariff = max(0.0, 1.0 - minimum_customer_tariff / cfg["retail_tariff_rs_kwh"])
    cfg["shift_discount"] = min(cfg["shift_discount"], max_discount_at_tariff)
    cfg["rapid_soh"] = float(np.clip(cfg["rapid_soh"], 0.70, 1.00))
    cfg["second_life_soh"] = float(np.clip(cfg["second_life_soh"], 0.50, 0.95))
    cfg["charge_efficiency"] = float(np.clip(cfg["charge_efficiency"], 0.50, 1.00))
    cfg["discharge_efficiency"] = float(np.clip(cfg["discharge_efficiency"], 0.50, 1.00))
    cfg["line_efficiency"] = float(np.clip(cfg["line_efficiency"], 0.80, 1.00))
    cfg["iex_price_multiplier"] = float(np.clip(cfg["iex_price_multiplier"], 0.50, 1.75))
    cfg["price_elasticity"] = float(np.clip(cfg["price_elasticity"], 0.0, 2.0))
    cfg["customer_delay_cost_rs_kwh"] = float(np.clip(cfg["customer_delay_cost_rs_kwh"], 0.0, 5.0))
    return cfg


def generate_scenario_data(scenario: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Generate a reproducible 1-day warm-up plus 14 synthetic operating days.

    Scenario multipliers are explicit stress-test assumptions, not weather or
    exchange observations. The same seed is used for reproducible comparisons.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    cfg = _merge_params(params)
    rng = np.random.default_rng(int(cfg["seed"]))
    per_metro_charging_cap = float(cfg["charging_capacity_mw_total"]) / len(CLUSTERS)
    rows: list[dict[str, Any]] = []
    scale = np.array([1.05, 1.08, 0.95, 0.90, 0.88])
    for h in range(24 * (int(cfg["test_days"]) + 1)):
        day, hour = divmod(h, 24)
        morning = 14.0 * np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
        evening = 25.0 * np.exp(-0.5 * ((hour - 19) / 2.4) ** 2)
        midday_dip = -4.0 * np.exp(-0.5 * ((hour - 13) / 2.5) ** 2)
        for ci, cluster in enumerate(CLUSTERS):
            demand = (57.0 + morning + evening + midday_dip) * scale[ci]
            demand *= 1.0 + rng.normal(0, 0.035)
            solar_shape = max(0.0, np.sin(np.pi * (hour - 6) / 12.0))
            solar = 55.0 * solar_shape * (0.9 + rng.normal(0, 0.04))
            wind = max(2.0, 9.0 + rng.normal(0, 1.5))
            price = 5.0 + 2.8 * np.exp(-0.5 * ((hour - 19) / 2.2) ** 2)
            price += 1.1 * np.exp(-0.5 * ((hour - 9) / 1.8) ** 2)
            price += rng.normal(0, 0.45)
            if scenario == "Monsoon cloud-cover and price shock":
                solar *= 0.55
                if hour in range(17, 22):
                    price += 2.2
            elif scenario == "Heatwave with customer demand surge":
                demand *= 1.18
            elif scenario == "Renewable generation surplus during low demand":
                solar *= 1.22
                demand *= 0.90
            renewable = max(0.0, solar + wind)
            price = float(np.clip(price * cfg["iex_price_multiplier"], 3.5, 18.0))
            rows.append({
                "hour_index": h, "day": day + 1, "hour": hour,
                "cluster": cluster, "demand_mwh": min(per_metro_charging_cap, max(15.0, demand)),
                "renewable_mwh": renewable,
                "iex_price_rs_kwh": price,
                "scenario": scenario,
            })
    data = pd.DataFrame(rows)
    data = add_seasonal_naive_forecasts(data)
    return data[data.day > 1].copy().reset_index(drop=True)


def shift_flexible_demand(data: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Shift eligible same-metro, non-priority demand into renewable-rich hours.

    Segment shares are synthetic. Priority/emergency fleets are never shifted.
    The price-elasticity term is a simple transparent uptake proxy, not an
    empirically estimated demand-response curve.
    """
    cfg = _merge_params(params)
    x = data.copy()
    x["demand_adjusted_mwh"] = x["demand_mwh"].astype(float)
    x["shifted_mwh"] = 0.0
    x["shifted_from_peak_mwh"] = 0.0
    segment_cols = ["priority_demand_mwh", "logistics_demand_mwh",
                    "ridehail_demand_mwh", "retail_demand_mwh"]
    shares = [cfg["priority_segment_share"], cfg["logistics_segment_share"],
              cfg["ridehail_segment_share"], cfg["retail_segment_share"]]
    if not np.isclose(sum(shares), 1.0, atol=1e-6):
        raise ValueError("Customer segment shares must sum to 1.0")
    for col, share in zip(segment_cols, shares):
        x[col] = x["demand_mwh"] * share
    for col in ["shifted_logistics_mwh", "shifted_ridehail_mwh", "shifted_retail_mwh"]:
        x[col] = 0.0

    uptake = (cfg["shift_share_of_flexible"]
              * (cfg["shift_discount"] / 0.10)
              * cfg["price_elasticity"])
    effective_shift_share = float(np.clip(uptake, 0.0, 0.30))
    nonpriority_total = sum(shares[1:])
    segment_fractions = np.array(shares[1:], dtype=float) / max(nonpriority_total, 1e-9)

    for cluster in CLUSTERS:
        for day in sorted(x.loc[x.cluster == cluster, "day"].unique()):
            peak_idx = x.index[(x.cluster == cluster) & (x.day == day) & (x.hour.isin([17, 18, 19, 20, 21]))]
            valley_idx = x.index[(x.cluster == cluster) & (x.day == day) & (x.hour.isin([11, 12, 13, 14, 15]))]
            if len(peak_idx) == 0 or len(valley_idx) == 0:
                continue
            peak_grid_cost = min(float(x.loc[peak_idx, "forecast_iex_price_rs_kwh"].mean()),
                                 cfg["discom_tariff_rs_kwh"])
            shift_cost = (cfg["retail_tariff_rs_kwh"] * cfg["shift_discount"]
                          + cfg["customer_delay_cost_rs_kwh"])
            # Shift only when incentive plus assumed customer delay/detour proxy
            # is cheaper than the forecast marginal grid unit.
            if shift_cost >= peak_grid_cost:
                continue
            weights = x.loc[valley_idx, "renewable_mwh"].to_numpy(dtype=float) + 1.0
            weights = weights / weights.sum()
            for seg_col, shift_col, seg_fraction in zip(segment_cols[1:],
                    ["shifted_logistics_mwh", "shifted_ridehail_mwh", "shifted_retail_mwh"],
                    segment_fractions):
                moved = x.loc[peak_idx, seg_col].to_numpy(dtype=float) * cfg["flexible_demand_share"] * effective_shift_share
                x.loc[peak_idx, seg_col] -= moved
                x.loc[peak_idx, "shifted_mwh"] -= moved
                x.loc[peak_idx, "shifted_from_peak_mwh"] += moved
                total_moved = float(moved.sum())
                x.loc[valley_idx, seg_col] += total_moved * weights
                x.loc[valley_idx, shift_col] += total_moved * weights
                x.loc[valley_idx, "shifted_mwh"] += total_moved * weights
            x.loc[peak_idx, "demand_adjusted_mwh"] = x.loc[peak_idx, segment_cols].sum(axis=1)
            x.loc[valley_idx, "demand_adjusted_mwh"] = x.loc[valley_idx, segment_cols].sum(axis=1)
    return x


def _energy_capacity_per_cluster(cfg: dict[str, Any]) -> tuple[float, float, float]:
    nominal = cfg["battery_mwh_total"] / len(CLUSTERS)
    rapid_share = cfg["rapid_share"]
    second_share = 1.0 - rapid_share
    # SoH reduces effective energy capacity; the nameplate system remains 600 MWh.
    rapid_cap = nominal * rapid_share * cfg["rapid_soh"]
    second_cap = nominal * second_share * cfg["second_life_soh"]
    return rapid_cap + second_cap, rapid_cap, second_cap


def run_policy(data: pd.DataFrame, policy: str,
               params: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one of the four transparent heuristic policies with UI assumptions."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    cfg = _merge_params(params)
    df = shift_flexible_demand(data, cfg) if policy == "full" else data.copy()
    demand_col = "demand_adjusted_mwh" if policy == "full" else "demand_mwh"
    rapid_share = cfg["rapid_share"]
    second_share = 1.0 - rapid_share
    nominal_per_cluster = cfg["battery_mwh_total"] / len(CLUSTERS)
    effective_total, rapid_cap, second_cap = _energy_capacity_per_cluster(cfg)
    battery_power = float(cfg["battery_mw_per_cluster"])
    grid_limit = float(cfg["grid_import_mw_per_cluster"])
    reserve_frac = float(cfg["reserve_fraction"])
    line_eff = float(cfg["line_efficiency"])
    charge_eff = float(cfg["charge_efficiency"])
    discharge_eff = float(cfg["discharge_efficiency"])

    # The category power split is a heuristic based on the 70:30 reference mix.
    # It retains the original 21 MW Rapid / 6 MW second-life dispatch caps at
    # the default 30 MW inverter setting, without pretending to reassign packs.
    rapid_power = min(battery_power * rapid_share, battery_power * (21.0 / 30.0) * (rapid_share / 0.70))
    second_power = min(battery_power * second_share, battery_power * (6.0 / 30.0) * (second_share / 0.30))

    rows: list[dict[str, Any]] = []
    for cluster in CLUSTERS:
        soc_total = effective_total * 0.50
        soc_rapid, soc_second = rapid_cap * 0.50, second_cap * 0.50
        cluster_df = df[df.cluster == cluster].sort_values("hour_index")
        for r in cluster_df.itertuples(index=False):
            demand = float(getattr(r, demand_col))
            renewable = float(r.renewable_mwh)
            iex_price = float(r.iex_price_rs_kwh)
            forecast_demand = float(r.forecast_demand_mwh)
            forecast_renewable = float(r.forecast_renewable_mwh)
            forecast_price = float(r.forecast_iex_price_rs_kwh)
            forecast_gap = max(0.0, forecast_demand / line_eff - forecast_renewable - grid_limit)
            load_requirement = demand / line_eff
            renewable_to_load = min(load_requirement, renewable)
            remaining = max(0.0, load_requirement - renewable_to_load)
            grid_load = battery_out = charge_input = grid_charge = wear_cost = 0.0
            rapid_out = second_out = 0.0
            rapid_charge = sec_charge = sec_grid = rapid_grid = 0.0
            grid_marginal_cost = min(forecast_price, cfg["discom_tariff_rs_kwh"])
            rapid_wear = cfg["rapid_wear_rs_kwh"] * (0.90 / cfg["rapid_soh"])
            second_wear = cfg["second_life_wear_rs_kwh"] * (0.80 / cfg["second_life_soh"])
            battery_cost_gate_open = False

            if policy in ("baseline", "forecast"):
                reserve = effective_total * reserve_frac
                available_out = max(0.0, (soc_total - reserve) * discharge_eff)
                if policy == "baseline":
                    desired = min(18.0, remaining) if r.hour in [18, 19, 20, 21] else 0.0
                else:
                    peak_signal = forecast_price >= 7.5 and r.hour in [17, 18, 19, 20, 21]
                    planned_gap = max(forecast_gap, max(0.0, remaining - grid_limit))
                    desired = min(15.0, remaining) if peak_signal else min(battery_power, planned_gap, remaining)
                battery_out = min(battery_power, desired, available_out)
                residual = max(0.0, remaining - battery_out)
                grid_load = min(residual, grid_limit)
                shortage = max(0.0, residual - grid_load)
                extra = min(max(0.0, battery_power - battery_out), shortage,
                            max(0.0, available_out - battery_out))
                battery_out += extra
                soc_total -= battery_out / discharge_eff
                wear_cost = battery_out * 1000.0 * cfg["uniform_wear_rs_kwh"]
            else:
                rapid_reserve = rapid_cap * reserve_frac
                second_reserve = second_cap * reserve_frac
                peak_signal = forecast_price >= 7.5 and r.hour in [17, 18, 19, 20, 21]
                second_available = max(0.0, (soc_second - second_reserve) * discharge_eff)
                rapid_available = max(0.0, (soc_rapid - rapid_reserve) * discharge_eff)
                battery_cost_gate_open = bool(peak_signal and min(rapid_wear, second_wear) <= grid_marginal_cost)

                # Triple-scarcity heuristic: use a battery category for economic
                # peak displacement only when its marginal wear proxy is no higher
                # than the forecast grid unit. Grid import and reserve caps remain
                # binding. Any residual beyond the grid connection is served by
                # remaining feasible battery headroom regardless of relative cost.
                second_target = min(second_power, remaining, second_available)                     if peak_signal and second_wear <= grid_marginal_cost else 0.0
                after_second = max(0.0, remaining - second_target)
                rapid_target = min(rapid_power, after_second, rapid_available)                     if peak_signal and rapid_wear <= grid_marginal_cost else 0.0
                after_economic_battery = max(0.0, after_second - rapid_target)
                grid_load = min(after_economic_battery, grid_limit)
                shortage = max(0.0, after_economic_battery - grid_load)

                second_extra = min(max(0.0, second_power - second_target), shortage,
                                   max(0.0, second_available - second_target))
                rapid_extra = min(max(0.0, rapid_power - rapid_target),
                                  max(0.0, shortage - second_extra),
                                  max(0.0, rapid_available - rapid_target))
                second_out = second_target + second_extra
                rapid_out = rapid_target + rapid_extra
                battery_out = rapid_out + second_out
                soc_rapid -= rapid_out / discharge_eff
                soc_second -= second_out / discharge_eff
                wear_cost = rapid_out * 1000.0 * rapid_wear + second_out * 1000.0 * second_wear

            renewable_surplus = max(0.0, renewable - renewable_to_load)
            charge_power_budget = max(0.0, battery_power - battery_out)
            if policy in ("baseline", "forecast"):
                room_input = max(0.0, (effective_total - soc_total) / charge_eff)
                charge_input = min(charge_power_budget, renewable_surplus, room_input)
                if policy == "forecast" and forecast_price <= 5.7 and r.hour in range(10, 16):
                    spare_grid = max(0.0, grid_limit - grid_load)
                    total_room = max(0.0, room_input - charge_input)
                    grid_charge = min(max(0.0, battery_power - battery_out - charge_input),
                                      spare_grid, total_room)
                soc_total += (charge_input + grid_charge) * charge_eff
                soc_total = min(effective_total, soc_total)
                wear_cost += (charge_input + grid_charge) * 1000.0 * cfg["uniform_wear_rs_kwh"] * 0.35
            else:
                sec_room = max(0.0, (second_cap - soc_second) / charge_eff)
                rapid_room = max(0.0, (rapid_cap - soc_rapid) / charge_eff)
                sec_charge = min(second_power, charge_power_budget, renewable_surplus, sec_room)
                rapid_charge = min(max(0.0, charge_power_budget - sec_charge),
                                   max(0.0, renewable_surplus - sec_charge), rapid_room, rapid_power)
                charge_input = sec_charge + rapid_charge
                if forecast_price <= 5.7 and r.hour in range(10, 16):
                    spare_grid = max(0.0, grid_limit - grid_load)
                    total_room = max(0.0, (effective_total - soc_rapid - soc_second) / charge_eff - charge_input)
                    category_power_room = (max(0.0, second_power - sec_charge)
                                           + max(0.0, rapid_power - rapid_charge))
                    grid_charge = min(max(0.0, battery_power - battery_out - charge_input),
                                      spare_grid, total_room, category_power_room)
                sec_grid = min(max(0.0, second_power - sec_charge), grid_charge)
                rapid_grid = min(max(0.0, rapid_power - rapid_charge), max(0.0, grid_charge - sec_grid))
                grid_charge = sec_grid + rapid_grid
                soc_second += (sec_charge + sec_grid) * charge_eff
                soc_rapid += (rapid_charge + rapid_grid) * charge_eff
                soc_second = min(second_cap, soc_second)
                soc_rapid = min(rapid_cap, soc_rapid)
                wear_cost += (sec_charge + sec_grid) * 1000.0 * cfg["second_life_wear_rs_kwh"] * (0.80 / cfg["second_life_soh"]) * 0.20
                wear_cost += (rapid_charge + rapid_grid) * 1000.0 * cfg["rapid_wear_rs_kwh"] * (0.90 / cfg["rapid_soh"]) * 0.20
                soc_total = soc_rapid + soc_second

            # Forecast selects the market route; actual synthetic IEX price is settled.
            use_iex = policy != "baseline" and forecast_price < cfg["discom_tariff_rs_kwh"]
            grid_energy = grid_load + grid_charge
            grid_iex_mwh = grid_energy if use_iex else 0.0
            grid_discom_mwh = 0.0 if use_iex else grid_energy
            settlement_price = iex_price if use_iex else cfg["discom_tariff_rs_kwh"]
            grid_cost = grid_energy * 1000.0 * settlement_price
            renewable_charge = min(charge_input, renewable_surplus)
            renewable_used = renewable_to_load + renewable_charge
            renewable_cost = renewable_used * 1000.0 * cfg["renewable_cost_rs_kwh"]
            supplied_at_bus = renewable_to_load + grid_load + battery_out
            delivered = min(demand, supplied_at_bus * line_eff)
            unmet = max(0.0, demand - delivered)
            shifted_here = max(0.0, float(getattr(r, "shifted_mwh", 0.0))) if policy == "full" else 0.0
            discount_value = min(delivered, shifted_here) * 1000.0 * cfg["retail_tariff_rs_kwh"] * cfg["shift_discount"]
            revenue = delivered * 1000.0 * cfg["retail_tariff_rs_kwh"] - discount_value
            contribution = revenue - grid_cost - renewable_cost - wear_cost
            losses = (renewable_to_load + grid_load + battery_out) * (1.0 - line_eff)
            losses += (charge_input + grid_charge) * (1.0 - charge_eff)
            losses += battery_out * (1.0 / discharge_eff - 1.0)

            # Priority service is allocated first after aggregate dispatch; it is never shifted.
            if hasattr(r, "priority_demand_mwh"):
                priority_demand = float(r.priority_demand_mwh)
                logistics_demand = float(r.logistics_demand_mwh)
                ridehail_demand = float(r.ridehail_demand_mwh)
                retail_demand = float(r.retail_demand_mwh)
            else:
                priority_demand = demand * cfg["priority_segment_share"]
                logistics_demand = demand * cfg["logistics_segment_share"]
                ridehail_demand = demand * cfg["ridehail_segment_share"]
                retail_demand = demand * cfg["retail_segment_share"]
            priority_delivered = min(priority_demand, delivered)
            nonpriority_demand = max(0.0, demand - priority_demand)
            nonpriority_delivered = min(nonpriority_demand, max(0.0, delivered - priority_delivered))
            congestion_pct = 100.0 * min(grid_energy, grid_limit) / max(grid_limit, 1e-9)

            rows.append({
                "day": int(r.day), "hour": int(r.hour), "hour_index": int(r.hour_index),
                "cluster": cluster, "policy": policy, "scenario": str(r.scenario),
                "demand_mwh": demand, "delivered_mwh": delivered, "unmet_mwh": unmet,
                "priority_demand_mwh": priority_demand,
                "priority_delivered_mwh": priority_delivered,
                "priority_unmet_mwh": max(0.0, priority_demand - priority_delivered),
                "logistics_demand_mwh": logistics_demand,
                "ridehail_demand_mwh": ridehail_demand,
                "retail_demand_mwh": retail_demand,
                "shifted_logistics_mwh": float(getattr(r, "shifted_logistics_mwh", 0.0)) if policy == "full" else 0.0,
                "shifted_ridehail_mwh": float(getattr(r, "shifted_ridehail_mwh", 0.0)) if policy == "full" else 0.0,
                "shifted_retail_mwh": float(getattr(r, "shifted_retail_mwh", 0.0)) if policy == "full" else 0.0,
                "nonpriority_demand_mwh": nonpriority_demand,
                "nonpriority_delivered_mwh": nonpriority_delivered,
                "forecast_demand_mwh": forecast_demand,
                "renewable_available_mwh": renewable,
                "renewable_used_mwh": renewable_used,
                "grid_mwh": grid_energy, "grid_iex_mwh": grid_iex_mwh,
                "grid_discom_mwh": grid_discom_mwh,
                "battery_discharge_mwh": battery_out,
                "battery_charge_mwh": charge_input + grid_charge,
                "rapid_discharge_mwh": rapid_out, "second_life_discharge_mwh": second_out,
                "rapid_charge_mwh": rapid_charge,
                "second_life_charge_mwh": sec_charge,
                "battery_soc_mwh": soc_total,
                "rapid_soc_mwh": soc_rapid if policy in ("health", "full") else np.nan,
                "second_life_soc_mwh": soc_second if policy in ("health", "full") else np.nan,
                "battery_soc_min_mwh": effective_total * reserve_frac,
                "wear_cost_rs": wear_cost, "grid_cost_rs": grid_cost,
                "renewable_cost_rs": renewable_cost, "revenue_rs": revenue,
                "discount_value_rs": discount_value, "contribution_rs": contribution,
                "losses_mwh": losses, "external_energy_mwh": renewable_used + grid_energy,
                "shifted_mwh": float(getattr(r, "shifted_mwh", 0.0)) if policy == "full" else 0.0,
                "shifted_from_peak_mwh": float(getattr(r, "shifted_from_peak_mwh", 0.0)) if policy == "full" else 0.0,
                "grid_utilisation_pct": congestion_pct,
                "grid_marginal_cost_rs_kwh": grid_marginal_cost,
                "rapid_wear_proxy_rs_kwh": rapid_wear,
                "second_life_wear_proxy_rs_kwh": second_wear,
                "customer_shift_incentive_rs_kwh": cfg["retail_tariff_rs_kwh"] * cfg["shift_discount"],
                "customer_delay_cost_proxy_rs_kwh": cfg["customer_delay_cost_rs_kwh"],
                "customer_shift_total_cost_proxy_rs_kwh": cfg["retail_tariff_rs_kwh"] * cfg["shift_discount"] + cfg["customer_delay_cost_rs_kwh"],
                "grid_source": "IEX-style synthetic" if use_iex else "DISCOM",
                "battery_cost_gate_open": battery_cost_gate_open,
            })

    result = pd.DataFrame(rows)
    total_demand = float(result.demand_mwh.sum())
    total_delivered = float(result.delivered_mwh.sum())
    total_external = float(result.external_energy_mwh.sum())
    summary = {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "reliability_pct": 100.0 * total_delivered / max(total_demand, 1e-9),
        "priority_reliability_pct": 100.0 * result.priority_delivered_mwh.sum() / max(result.priority_demand_mwh.sum(), 1e-9),
        "modelled_loss_proxy_pct": 100.0 * result.losses_mwh.sum() / max(1e-9, result.losses_mwh.sum() + total_delivered),
        "procurement_cost_rs_lakh": (result.grid_cost_rs.sum() + result.renewable_cost_rs.sum()) / 100000.0,
        "grid_procurement_cost_rs_lakh": result.grid_cost_rs.sum() / 100000.0,
        "renewable_procurement_cost_rs_lakh": result.renewable_cost_rs.sum() / 100000.0,
        "wear_cost_rs_lakh": result.wear_cost_rs.sum() / 100000.0,
        "discount_value_rs_lakh": result.discount_value_rs.sum() / 100000.0,
        "gross_contribution_rs_lakh": result.contribution_rs.sum() / 100000.0,
        "unserved_mwh": result.unmet_mwh.sum(),
        "priority_unserved_mwh": result.priority_unmet_mwh.sum(),
        "battery_throughput_mwh": result.battery_discharge_mwh.sum() + result.battery_charge_mwh.sum(),
        "renewable_utilisation_pct": 100.0 * result.renewable_used_mwh.sum() / max(1e-9, result.renewable_available_mwh.sum()),
        "renewable_used_mwh": result.renewable_used_mwh.sum(),
        "grid_iex_mwh": result.grid_iex_mwh.sum(),
        "grid_discom_mwh": result.grid_discom_mwh.sum(),
        "grid_and_renewable_input_mwh": total_external,
        "delivered_energy_mwh": total_delivered,
        "shifted_peak_mwh": result.shifted_from_peak_mwh.sum(),
        "peak_grid_utilisation_pct": result.grid_utilisation_pct.max(),
        "average_grid_utilisation_pct": result.grid_utilisation_pct.mean(),
    }
    return result, summary


def run_all_policies(scenario: str, params: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return scenario input data, hourly results and the four-policy summary."""
    cfg = _merge_params(params)
    data = generate_scenario_data(scenario, cfg)
    hourly_parts = []
    summaries = []
    for policy in POLICIES:
        hourly, summary = run_policy(data, policy, cfg)
        hourly_parts.append(hourly)
        summaries.append(summary)
    return data, pd.concat(hourly_parts, ignore_index=True), pd.DataFrame(summaries)
