"""Perishable Grid: small, reproducible 5-cluster simulation.

All generated operating data and all parameters labelled ASSUMED are illustrative.
This prototype is a transparent dispatch heuristic, NOT a production MILP or a
claim about real operator performance. Requires numpy, pandas, matplotlib.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
OUT = Path(__file__).resolve().parent
CLUSTERS = ["Delhi NCR", "Mumbai MMR", "Bengaluru", "Hyderabad", "Chennai"]
HOURS = 24 * 15  # 1 warm-up day + 14 synthetic test days

# Case-provided values: 5 clusters, 1,200 MW charging capacity, 600 MWh storage.
# ASSUMED for this prototype: equal cluster allocation, 120 MWh/cluster,
# 30 MW/cluster battery inverter rating (0.25C), 38 MW/cluster grid import cap.
BATTERY_MWH = 120.0
BATTERY_MW = 30.0
GRID_MW = 38.0
RESERVE_FRAC = 0.05
LINE_EFF = 0.975
CHARGE_EFF = 0.95
DISCHARGE_EFF = 0.95
DISCOM_RS_KWH = 8.50
RENEWABLE_RS_KWH = 3.50
RETAIL_RS_KWH = 14.00
RAPID_WEAR_RS_KWH = 1.20
SECOND_LIFE_WEAR_RS_KWH = 0.65
UNIFORM_WEAR_RS_KWH = 1.00
FLEX_SHARE = 0.40
SHIFT_SHARE_OF_FLEX = 0.15
SHIFT_DISCOUNT = 0.10


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    scale = np.array([1.05, 1.08, 0.95, 0.90, 0.88])
    for h in range(HOURS):
        day, hour = divmod(h, 24)
        # Common daily demand shape; cluster noise keeps hubs distinct.
        morning = 14.0 * np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
        evening = 25.0 * np.exp(-0.5 * ((hour - 19) / 2.4) ** 2)
        midday_dip = -4.0 * np.exp(-0.5 * ((hour - 13) / 2.5) ** 2)
        for ci, cluster in enumerate(CLUSTERS):
            demand = (57.0 + morning + evening + midday_dip) * scale[ci]
            demand *= 1.0 + rng.normal(0, 0.035)
            # Stress windows: heatwave days 5-6; monsoon days 8-9; surplus days 11-12.
            if day in (5, 6):
                demand *= 1.18
            solar_shape = max(0.0, np.sin(np.pi * (hour - 6) / 12.0))
            solar = 55.0 * solar_shape * (0.9 + rng.normal(0, 0.04))
            wind = max(2.0, 9.0 + rng.normal(0, 1.5))
            if day in (8, 9):
                solar *= 0.55  # monsoon cloud-cover shock
            if day in (11, 12):
                solar *= 1.22  # renewable-surplus days
            renewable = max(0.0, solar + wind)
            # Illustrative day-ahead price shape, not observed IEX prices.
            price = 5.0 + 2.8 * np.exp(-0.5 * ((hour - 19) / 2.2) ** 2)
            price += 1.1 * np.exp(-0.5 * ((hour - 9) / 1.8) ** 2)
            price += rng.normal(0, 0.45)
            if day in (8, 9) and hour in range(17, 22):
                price += 2.2  # stress price spike
            price = float(np.clip(price, 3.5, 11.5))
            rows.append({"hour_index": h, "day": day + 1, "hour": hour,
                         "cluster": cluster, "demand_mwh": max(15.0, demand),
                         "renewable_mwh": renewable, "iex_price_rs_kwh": price})
    data = pd.DataFrame(rows)
    # Seasonal-naive one-day-ahead forecast: same cluster and hour from the
    # previous day. The first day is a warm-up day and is excluded from scoring.
    for col, forecast_col in [("demand_mwh", "forecast_demand_mwh"),
                              ("renewable_mwh", "forecast_renewable_mwh"),
                              ("iex_price_rs_kwh", "forecast_iex_price_rs_kwh")]:
        data[forecast_col] = data.groupby("cluster")[col].shift(24)
    return data


def shift_flexible_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Move 15% of flexible (40%) evening demand into solar-rich midday hours.
    Movement is within the same metro cluster. This is a scenario assumption.
    """
    x = df.copy()
    x["demand_adjusted_mwh"] = x["demand_mwh"]
    x["shifted_mwh"] = 0.0
    for cluster in CLUSTERS:
        idx = x.index[x.cluster == cluster]
        for d in sorted(x.loc[idx, "day"].unique()):
            day_idx = x.index[(x.cluster == cluster) & (x.day == d)]
            peak_idx = x.index[(x.cluster == cluster) & (x.day == d) & (x.hour.isin([17,18,19,20,21]))]
            valley_idx = x.index[(x.cluster == cluster) & (x.day == d) & (x.hour.isin([11,12,13,14,15]))]
            amount = x.loc[peak_idx, "demand_mwh"].sum() * FLEX_SHARE * SHIFT_SHARE_OF_FLEX
            # Spread deferred load across the five destination hours.
            x.loc[peak_idx, "demand_adjusted_mwh"] -= x.loc[peak_idx, "demand_mwh"] * FLEX_SHARE * SHIFT_SHARE_OF_FLEX
            x.loc[peak_idx, "shifted_mwh"] -= x.loc[peak_idx, "demand_mwh"] * FLEX_SHARE * SHIFT_SHARE_OF_FLEX
            weights = x.loc[valley_idx, "renewable_mwh"].to_numpy() + 1.0
            weights = weights / weights.sum()
            x.loc[valley_idx, "demand_adjusted_mwh"] += amount * weights
            x.loc[valley_idx, "shifted_mwh"] += amount * weights
    return x


def run_policy(data: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict]:
    """Small dispatch policy used for the hackathon proof-of-concept.

    This is intentionally a transparent heuristic, not a claim that a full MILP
    has been implemented. It separates four ablation stages and enforces the
    reserve, grid-import and battery power/energy limits.
    """
    df = shift_flexible_demand(data) if policy == "full" else data.copy()
    demand_col = "demand_adjusted_mwh" if policy == "full" else "demand_mwh"
    rapid_cap, second_cap = BATTERY_MWH * 0.70, BATTERY_MWH * 0.30
    rows = []
    initial_soc_network = 0.0
    final_soc_network = 0.0
    for cluster in CLUSTERS:
        soc_total = BATTERY_MWH * 0.50
        soc_rapid, soc_second = rapid_cap * 0.50, second_cap * 0.50
        initial_soc_network += soc_total
        for r in df[df.cluster == cluster].itertuples(index=False):
            demand = float(getattr(r, demand_col))
            renewable = float(r.renewable_mwh)
            iex_price = float(r.iex_price_rs_kwh)
            forecast_demand = float(r.forecast_demand_mwh)
            forecast_renewable = float(r.forecast_renewable_mwh)
            forecast_price = float(r.forecast_iex_price_rs_kwh)
            forecast_gap = max(0.0, forecast_demand/LINE_EFF - forecast_renewable - GRID_MW)
            load_requirement = demand / LINE_EFF
            renewable_to_load = min(load_requirement, renewable)
            remaining = max(0.0, load_requirement - renewable_to_load)
            grid_load = 0.0
            battery_out = 0.0
            charge_input = 0.0
            grid_charge = 0.0
            wear_cost = 0.0
            rapid_out = second_out = 0.0

            if policy in ("baseline", "forecast"):
                reserve = BATTERY_MWH * RESERVE_FRAC
                available_out = max(0.0, (soc_total - reserve) * DISCHARGE_EFF)
                if policy == "baseline":
                    # Fixed peak-window dispatch, no price-aware procurement.
                    desired = min(18.0, remaining) if r.hour in [18, 19, 20, 21] else 0.0
                    battery_out = min(desired, available_out)
                else:
                    # Forecast-informed dispatch: use storage during expensive peak periods,
                    # and cover any remaining deficit above the grid connection limit.
                    peak_signal = forecast_price >= 7.5 and r.hour in [17, 18, 19, 20, 21]
                    planned_gap = max(forecast_gap, max(0.0, remaining-GRID_MW))
                    desired = min(15.0, remaining) if peak_signal else min(BATTERY_MW, planned_gap, remaining)
                    battery_out = min(BATTERY_MW, desired, available_out)
                residual = max(0.0, remaining - battery_out)
                grid_load = min(residual, GRID_MW)
                # If the grid limit still leaves a deficit, use remaining battery power.
                shortage = max(0.0, residual - grid_load)
                extra = min(max(0.0, BATTERY_MW-battery_out), shortage,
                            max(0.0, available_out-battery_out))
                battery_out += extra
                soc_total -= battery_out / DISCHARGE_EFF
                wear_cost = battery_out * 1000.0 * UNIFORM_WEAR_RS_KWH
            else:
                # Explicit asymmetric battery allocation:
                # 70% Rapid tier (assumed SoH 90%), 30% second-life tier (assumed SoH 80%).
                # Second-life pool performs shallow evening arbitrage first; Rapid pool
                # supplies the remaining high-power shortfall.
                rapid_reserve = rapid_cap * RESERVE_FRAC
                second_reserve = second_cap * RESERVE_FRAC
                peak_signal = forecast_price >= 7.5 and r.hour in [17,18,19,20,21]
                second_available = max(0.0, (soc_second-second_reserve)*DISCHARGE_EFF)
                rapid_available = max(0.0, (soc_rapid-rapid_reserve)*DISCHARGE_EFF)
                second_target = min(6.0, remaining, second_available) if peak_signal else 0.0
                remaining_after_second = max(0.0, remaining-second_target)
                # Rapid batteries are held for the remaining deficit; the grid supplies
                # the balance up to its connection limit.
                grid_load = min(remaining_after_second, GRID_MW)
                shortage = max(0.0, remaining_after_second-grid_load)
                rapid_target = min(21.0, shortage, rapid_available)
                # If a severe shortage remains, use any unused second-life capability.
                residual_shortage = max(0.0, shortage-rapid_target)
                extra_second = min(max(0.0, 6.0-second_target), residual_shortage,
                                   max(0.0, second_available-second_target))
                second_out = second_target + extra_second
                rapid_out = rapid_target
                battery_out = rapid_out + second_out
                soc_rapid -= rapid_out / DISCHARGE_EFF
                soc_second -= second_out / DISCHARGE_EFF
                wear_cost = rapid_out*1000.0*RAPID_WEAR_RS_KWH + second_out*1000.0*SECOND_LIFE_WEAR_RS_KWH

            residual_after_dispatch = max(0.0, remaining-grid_load-battery_out)
            # Charge on renewable surplus first. In health-aware modes, charge the
            # second-life pool gently before using remaining power for Rapid packs.
            renewable_surplus = max(0.0, renewable-renewable_to_load)
            if policy in ("baseline", "forecast"):
                room_input = max(0.0, (BATTERY_MWH-soc_total)/CHARGE_EFF)
                charge_input = min(BATTERY_MW, renewable_surplus, room_input)
                if policy == "forecast" and forecast_price <= 5.7 and r.hour in range(10,16):
                    spare_grid = max(0.0, GRID_MW-grid_load)
                    grid_charge = min(BATTERY_MW-charge_input, spare_grid,
                                      max(0.0, room_input-charge_input))
                soc_total += (charge_input+grid_charge)*CHARGE_EFF
                soc_total = min(BATTERY_MWH, soc_total)
                wear_cost += (charge_input+grid_charge)*1000.0*UNIFORM_WEAR_RS_KWH*0.35
            else:
                sec_room = max(0.0, (second_cap-soc_second)/CHARGE_EFF)
                rapid_room = max(0.0, (rapid_cap-soc_rapid)/CHARGE_EFF)
                sec_charge = min(6.0, renewable_surplus, sec_room)
                rapid_charge = min(max(0.0, BATTERY_MW-sec_charge),
                                   max(0.0, renewable_surplus-sec_charge), rapid_room)
                charge_input = sec_charge+rapid_charge
                if forecast_price <= 5.7 and r.hour in range(10,16):
                    spare_grid = max(0.0, GRID_MW-grid_load)
                    total_room = max(0.0, (BATTERY_MWH-soc_rapid-soc_second)/CHARGE_EFF-charge_input)
                    grid_charge = min(BATTERY_MW-charge_input, spare_grid, total_room)
                # Allocate any low-price grid charging to the second-life pool first.
                sec_grid = min(max(0.0, 6.0-sec_charge), grid_charge)
                rapid_grid = max(0.0, grid_charge-sec_grid)
                soc_second += (sec_charge+sec_grid)*CHARGE_EFF
                soc_rapid += (rapid_charge+rapid_grid)*CHARGE_EFF
                soc_second = min(second_cap, soc_second)
                soc_rapid = min(rapid_cap, soc_rapid)
                wear_cost += (sec_charge+sec_grid)*1000.0*SECOND_LIFE_WEAR_RS_KWH*0.20
                wear_cost += (rapid_charge+rapid_grid)*1000.0*RAPID_WEAR_RS_KWH*0.20
                soc_total = soc_rapid+soc_second

            # Price choice for grid energy: buy from IEX only when cheaper than DISCOM.
            use_iex = policy != "baseline" and forecast_price < DISCOM_RS_KWH
            grid_price_load = iex_price if use_iex else DISCOM_RS_KWH
            grid_price_charge = iex_price if use_iex else DISCOM_RS_KWH
            grid_cost = (grid_load*1000.0*grid_price_load
                         + grid_charge*1000.0*grid_price_charge)
            grid_energy = grid_load + grid_charge
            renewable_charge = min(charge_input, renewable_surplus)
            renewable_used = renewable_to_load+renewable_charge
            renewable_cost = renewable_used*1000.0*RENEWABLE_RS_KWH
            supplied_at_bus = renewable_to_load+grid_load+battery_out
            delivered = min(demand, supplied_at_bus*LINE_EFF)
            unmet = max(0.0, demand-delivered)
            shifted_here = max(0.0, float(getattr(r, "shifted_mwh", 0.0))) if policy == "full" else 0.0
            discount_value = min(delivered, shifted_here)*1000.0*RETAIL_RS_KWH*SHIFT_DISCOUNT
            revenue = delivered*1000.0*RETAIL_RS_KWH-discount_value
            contribution = revenue-grid_cost-renewable_cost-wear_cost
            losses = (renewable_to_load+grid_load+battery_out)* (1.0-LINE_EFF)
            losses += (charge_input+grid_charge)*(1.0-CHARGE_EFF)
            losses += battery_out*(1.0/DISCHARGE_EFF-1.0)
            rows.append({"day":r.day,"hour":r.hour,"cluster":cluster,"policy":policy,
                         "demand_mwh":demand,"delivered_mwh":delivered,"unmet_mwh":unmet,
                         "renewable_available_mwh":renewable,"renewable_used_mwh":renewable_used,
                         "grid_mwh":grid_energy,"battery_discharge_mwh":battery_out,
                         "battery_charge_mwh":charge_input+grid_charge,
                         "battery_soc_mwh":soc_total,"wear_cost_rs":wear_cost,
                         "grid_cost_rs":grid_cost,"renewable_cost_rs":renewable_cost,
                         "revenue_rs":revenue,"discount_value_rs":discount_value,
                         "contribution_rs":contribution,"losses_mwh":losses,
                         "external_energy_mwh":renewable_used+grid_energy})
        final_soc_network += soc_total
    result = pd.DataFrame(rows)
    total_demand = result.demand_mwh.sum()
    total_delivered = result.delivered_mwh.sum()
    total_external = result.external_energy_mwh.sum()
    summary = {
        "policy": policy,
        "reliability_pct": 100*total_delivered/total_demand,
        "modelled_loss_proxy_pct": 100*result.losses_mwh.sum()/max(1e-9,(result.losses_mwh.sum()+total_delivered)),
        "procurement_cost_rs_lakh": (result.grid_cost_rs.sum()+result.renewable_cost_rs.sum())/100000,
        "wear_cost_rs_lakh": result.wear_cost_rs.sum()/100000,
        "discount_value_rs_lakh": result.discount_value_rs.sum()/100000,
        "gross_contribution_rs_lakh": result.contribution_rs.sum()/100000,
        "unserved_mwh": result.unmet_mwh.sum(),
        "battery_throughput_mwh": result.battery_discharge_mwh.sum()+result.battery_charge_mwh.sum(),
        "renewable_utilisation_pct": 100*result.renewable_used_mwh.sum()/max(1e-9,result.renewable_available_mwh.sum()),
        "grid_and_renewable_input_mwh": total_external,
        "delivered_energy_mwh": total_delivered,
    }
    return result, summary


def make_charts(all_results: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    order = ["baseline", "forecast", "health", "full"]
    labels = ["Rule-based", "Forecast", "Health-aware", "Full system"]
    s = summary_df.set_index("policy").loc[order]

    # Chart 1: reliability with case reference lines. Simulated baseline is NOT calibrated
    # to the case's reported 78%; these are separate references.
    fig, ax = plt.subplots(figsize=(8.2,4.6))
    bars=ax.bar(labels,s.reliability_pct)
    ax.axhline(78.0,linestyle="--",linewidth=1.4,label="Case-reported current reliability: 78%")
    ax.axhline(89.7,linestyle=":",linewidth=1.8,label="Case target: 89.7%")
    ax.set_ylim(70,95); ax.set_ylabel("Service reliability (%)")
    ax.set_title("Ablation: service reliability across four policies")
    ax.tick_params(axis="x",rotation=12); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.22)
    for bar,val in zip(bars,s.reliability_pct): ax.text(bar.get_x()+bar.get_width()/2,val+.25,f"{val:.1f}%",ha="center",fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/"chart_reliability.png",dpi=180); plt.close(fig)

    # Chart 2: gross contribution, excluding fixed O&M/capex/tax and therefore NOT EBITDA.
    fig, ax=plt.subplots(figsize=(8.2,4.6))
    vals=s.gross_contribution_rs_lakh/100.0
    bars=ax.bar(labels,vals)
    ax.set_ylabel("₹ crore over 14 synthetic days")
    ax.set_ylim(0, 100)
    ax.set_title("Simulated gross energy contribution (not EBITDA)")
    ax.tick_params(axis='x',rotation=12); ax.grid(axis="y",alpha=.22)
    for bar,val in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,val+.3,f"₹{val:.2f} cr",ha="center",fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/"chart_contribution.png",dpi=180); plt.close(fig)

    # Chart 3: wear-cost proxy, based on explicitly assumed Rs/kWh throughput rates.
    fig, ax=plt.subplots(figsize=(8.2,4.6))
    vals=s.wear_cost_rs_lakh
    bars=ax.bar(labels,vals)
    ax.set_ylabel("₹ lakh over 14 synthetic days")
    ax.set_ylim(0, 110)
    ax.set_title("Battery wear-cost proxy by policy")
    ax.tick_params(axis='x',rotation=12); ax.grid(axis="y",alpha=.22)
    for bar,val in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,val+.8,f"{val:.1f}",ha="center",fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/"chart_wear.png",dpi=180); plt.close(fig)

    # Chart 4: one pre-defined monsoon/price-spike day, aggregated across clusters.
    sub=all_results[(all_results.policy=="full") & (all_results.day==9)].groupby("hour")[["demand_mwh","delivered_mwh"]].sum()
    fig, ax=plt.subplots(figsize=(8.2,4.6))
    ax.plot(sub.index,sub.demand_mwh,marker="o",label="Demand after SLA shifting")
    ax.plot(sub.index,sub.delivered_mwh,marker="s",label="Delivered energy")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("MWh across five clusters")
    ax.set_title("Stress-day profile: monsoon cloud cover and price spike")
    ax.legend(fontsize=8); ax.grid(alpha=.22); fig.tight_layout()
    fig.savefig(OUT/"chart_stress_day.png",dpi=180); plt.close(fig)


def main():
    all_data=make_data()
    data=all_data[all_data.day > 1].copy()  # exclude the warm-up day
    all_parts=[]; summaries=[]
    for policy in ["baseline","forecast","health","full"]:
        res, summ=run_policy(data,policy)
        all_parts.append(res); summaries.append(summ)
    all_results=pd.concat(all_parts,ignore_index=True)
    summary_df=pd.DataFrame(summaries)
    all_results.to_csv(OUT/"hourly_results.csv",index=False)
    summary_df.to_csv(OUT/"summary_results.csv",index=False)
    make_charts(all_results,summary_df)
    with open(OUT/"assumptions.json","w") as f:
        json.dump({"status":"illustrative synthetic assumptions; not real operator data",
          "seed":SEED,"test_days":14,"warmup_days":1,"forecast_method":"one-day seasonal naive baseline (same cluster/hour, previous day)","clusters":CLUSTERS,"battery_mwh_per_cluster":BATTERY_MWH,
          "battery_mw_per_cluster_assumed":BATTERY_MW,"grid_import_mw_per_cluster_assumed":GRID_MW,
          "reserve_fraction":RESERVE_FRAC,"line_efficiency_assumed":LINE_EFF,
          "charge_efficiency_assumed":CHARGE_EFF,"discharge_efficiency_assumed":DISCHARGE_EFF,
          "DISCOM_tariff_rs_kwh_assumed":DISCOM_RS_KWH,"renewable_cost_rs_kwh_assumed":RENEWABLE_RS_KWH,
          "retail_tariff_rs_kwh_assumed":RETAIL_RS_KWH,"rapid_battery_share_assumed":0.70,
          "rapid_SoH_assumed":0.90,"second_life_share_assumed":0.30,"second_life_SoH_assumed":0.80,
          "flexible_demand_share_assumed":FLEX_SHARE,"shift_share_of_flexible_peak_demand_assumed":SHIFT_SHARE_OF_FLEX,
          "shift_discount_assumed":SHIFT_DISCOUNT,
          "important_limitations":["Heuristic dispatch, not MILP solver output","No actual five-year operator data",
          "No VPP revenue included","No inter-hub pack transport implemented","No fixed O&M, capex, taxes or depreciation; gross contribution is not EBITDA",
          "Loss metric is a simplified model proxy and should be reconciled to the operator's metering boundary"]},f,indent=2)
    print(summary_df.to_string(index=False,float_format=lambda x:f"{x:,.2f}"))
    print(f"\nWrote outputs to {OUT}")

if __name__=="__main__":
    main()
