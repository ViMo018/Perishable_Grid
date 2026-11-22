"""The Perishable Grid — interactive Streamlit decision-support prototype."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from scenario_engine import (
    CLUSTERS, DEFAULT_PARAMS, POLICY_LABELS, POLICIES, SCENARIOS,
    _merge_params, run_all_policies,
)
from visualizations import (
    battery_soc_chart, cluster_overview_chart, customer_shift_chart,
    hourly_supply_chart, policy_bar, price_chart, procurement_mix_chart,
    triple_scarcity_chart,
)

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="The Perishable Grid | Energy Control Room",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --pg-bg:#071521;
  --pg-sidebar:#0A1925;
  --pg-card:#102433;
  --pg-surface:#142B3A;
  --pg-teal:#25C8BD;
  --pg-green:#65D6A2;
  --pg-text:#F0F5F8;
  --pg-muted:#9BAFBD;
  --pg-border:#24404F;
}
.stApp { background:var(--pg-bg); color:var(--pg-text); }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.block-container { padding:1.35rem 2rem 2.1rem; max-width:1540px; }
section[data-testid="stSidebar"] { background:var(--pg-sidebar); border-right:1px solid var(--pg-border); }
section[data-testid="stSidebar"] > div { padding-top:1.1rem; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color:#DCE7ED !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--pg-muted); }
section[data-testid="stSidebar"] [data-testid="stExpander"] {
  border:1px solid var(--pg-border); border-radius:10px; background:#0D202D;
  margin-bottom:7px;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details summary { padding:0.55rem 0.75rem; }
section[data-testid="stSidebar"] [data-testid="stExpander"] details summary p { font-size:12px; font-weight:700; }
section[data-testid="stSidebar"] [data-testid="stSlider"] { padding-top:0.1rem; padding-bottom:0.35rem; }
section[data-testid="stSidebar"] [data-testid="stSlider"] label p { font-size:11px; line-height:1.35; }
section[data-testid="stSidebar"] [data-testid="stSelectbox"] label p { font-size:11px; }
.pg-brand { display:flex; align-items:center; gap:10px; margin:0 0 8px; }
.pg-orb {
  width:36px; height:36px; border-radius:10px; background:var(--pg-teal);
  display:flex; align-items:center; justify-content:center; color:#06141D;
  font-size:20px; font-weight:900;
}
.pg-eyebrow { color:#79C9C2; letter-spacing:1.6px; text-transform:uppercase; font-size:9px; font-weight:800; }
.pg-title { font-size:29px; line-height:1.15; font-weight:800; letter-spacing:-0.8px; color:var(--pg-text); margin:2px 0 4px; }
.pg-subtitle { color:var(--pg-muted); font-size:12px; line-height:1.5; margin-bottom:12px; }
.pg-strip {
  display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:7px 14px;
  border:1px solid var(--pg-border); background:#0E2230; border-radius:9px;
  padding:8px 12px; color:#C3D3DC; font-size:11px; line-height:1.45; margin:0 0 14px;
}
.pg-strip strong { color:#DDF7F3; font-size:10px; letter-spacing:.45px; }
.pg-strip-status { color:var(--pg-green); font-size:10px; font-weight:800; letter-spacing:.5px; white-space:nowrap; }
.pg-section { font-size:20px; font-weight:800; letter-spacing:-.35px; color:var(--pg-text); margin:2px 0 4px; }
[data-testid="stCaptionContainer"] { color:var(--pg-muted); font-size:11px; line-height:1.55; }
.pg-kpi {
  background:var(--pg-card); border:1px solid var(--pg-border); border-top:2px solid #278F90;
  border-radius:10px; padding:12px 13px 11px; min-height:98px;
}
.pg-kpi-label { color:var(--pg-muted); font-size:9px; text-transform:uppercase; letter-spacing:1px; font-weight:800; line-height:1.4; }
.pg-kpi-value { color:var(--pg-text); font-size:22px; font-weight:800; margin-top:7px; letter-spacing:-.4px; font-variant-numeric:tabular-nums; white-space:nowrap; }
.pg-kpi-delta { color:#83CDBB; font-size:9px; line-height:1.45; margin-top:3px; }
.pg-card { border:1px solid var(--pg-border); background:var(--pg-card); border-radius:10px; padding:13px 15px; }
.pg-card-title { font-size:11px; font-weight:800; color:#DCEAF0; letter-spacing:.45px; margin:8px 0 7px; }
.pg-note { font-size:10px; color:var(--pg-muted); line-height:1.55; }
.pg-good {
  border-left:2px solid var(--pg-teal); background:#0D222D; border-radius:0 7px 7px 0;
  padding:8px 11px; color:#BBD4D7; font-size:10px; line-height:1.55;
}
.pg-mini { color:var(--pg-muted); font-size:9px; }
div[data-testid="stMetric"] { background:var(--pg-card); border:1px solid var(--pg-border); border-radius:9px; padding:9px 11px; }
div[data-testid="stMetricLabel"] { color:var(--pg-muted) !important; font-size:10px !important; }
div[data-testid="stMetricValue"] { color:var(--pg-text) !important; font-size:20px !important; }
.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--pg-border); flex-wrap:wrap; }
.stTabs [data-baseweb="tab"] {
  color:var(--pg-muted); background:transparent; border:0; border-bottom:2px solid transparent;
  border-radius:0; padding:9px 11px; font-size:11px; font-weight:600;
}
.stTabs [aria-selected="true"] { color:var(--pg-teal) !important; background:transparent !important; border-bottom:2px solid var(--pg-teal) !important; }
.stButton button, .stDownloadButton button {
  background:#123B46; color:#E4F8F5; border:1px solid #2B6870; border-radius:8px;
  min-height:38px; font-size:11px; font-weight:700;
}
.stButton button:hover, .stDownloadButton button:hover { background:#174B54; color:white; border-color:var(--pg-teal); }
[data-testid="stDataFrame"] { border:1px solid var(--pg-border); border-radius:9px; overflow:hidden; }
div[data-testid="stExpander"] { border:1px solid var(--pg-border); border-radius:9px; background:#0D202D; }
div[data-testid="stExpander"] details summary p { color:#DCEAF0; font-size:12px; font-weight:700; }
hr { border-color:var(--pg-border); }
@media (max-width: 1100px) {
  .block-container { padding-left:1.1rem; padding-right:1.1rem; }
  .pg-title { font-size:26px; }
  .pg-kpi-value { font-size:19px; }
}
@media (max-width: 700px) {
  .block-container { padding:1rem .8rem 1.5rem; }
  .pg-title { font-size:24px; }
  .stTabs [data-baseweb="tab"] { padding:8px 7px; font-size:10px; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=24)
def calculate(scenario: str, params: dict):
    return run_all_policies(scenario, params)


def kpi(label: str, value: str, note: str = ""):
    st.markdown(
        f'<div class="pg-kpi"><div class="pg-kpi-label">{label}</div>'
        f'<div class="pg-kpi-value">{value}</div><div class="pg-kpi-delta">{note}</div></div>',
        unsafe_allow_html=True,
    )


def section_title(title: str, sub: str | None = None):
    st.markdown(f'<div class="pg-section">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.caption(sub)


def money_cr(value_lakh: float) -> str:
    return f"₹{value_lakh / 100:,.2f} cr"


def rupees(value: float) -> str:
    return f"₹{value:,.2f}"


# Sidebar controls ----------------------------------------------------------------
st.sidebar.markdown("<div class='pg-brand'><div class='pg-orb'>ϟ</div><div><div class='pg-eyebrow'>Energy decision system</div><div style='font-size:18px;font-weight:800;color:#eef7fa'>PERISHABLE GRID</div></div></div>", unsafe_allow_html=True)
st.sidebar.caption("SIMULATION CONTROLS · synthetic inputs")
with st.sidebar.expander("01 · Operating scenario", expanded=True):
    scenario = st.selectbox("Select operating scenario", SCENARIOS, index=0, key="operating_scenario")

with st.sidebar.expander("02 · Energy trading & procurement", expanded=False):
    discom_tariff = st.slider("DISCOM tariff (₹/kWh)", 4.0, 14.0, 8.5, 0.1)
    iex_multiplier = st.slider("Synthetic IEX-style price multiplier", 0.50, 1.75, 1.00, 0.05)
    renewable_cost = st.slider("Renewable supply cost (₹/kWh)", 1.5, 8.0, 3.5, 0.1)
    grid_limit = st.slider("Grid import limit per metro (MW)", 15, 70, 38, 1)

with st.sidebar.expander("03 · Battery health & power", expanded=False):
    rapid_share_pct = st.slider("Rapid allocation (%)", 50, 90, 70, 1)
    rapid_soh_pct = st.slider("Rapid battery SoH (%)", 70, 100, 90, 1)
    second_soh_pct = st.slider("Second-life battery SoH (%)", 50, 95, 80, 1)
    battery_power = st.slider("Battery inverter power per metro (MW)", 10, 60, 30, 1)
    reserve_pct = st.slider("Minimum storage reserve (%)", 5, 20, 5, 1)
    charge_eff_pct = st.slider("Charging efficiency (%)", 85, 99, 95, 1)
    discharge_eff_pct = st.slider("Discharging efficiency (%)", 85, 99, 95, 1)
    rapid_wear = st.slider("Rapid wear-cost proxy (₹/kWh)", 0.2, 10.0, 1.2, 0.1)
    second_wear = st.slider("Second-life wear-cost proxy (₹/kWh)", 0.1, 10.0, 0.65, 0.05)

with st.sidebar.expander("04 · Customer demand & incentives", expanded=False):
    flexible_pct = st.slider("Flexible demand share (%)", 0, 80, 40, 1)
    shift_pct = st.slider("Flexible peak demand shifted (%)", 0, 30, 15, 1)
    discount_pct = st.slider("Shift incentive / discount (%)", 0, 20, 10, 1)
    elasticity = st.slider("Customer price-elasticity assumption", 0.0, 2.0, 1.0, 0.1)
    delay_cost = st.slider("Customer delay / detour cost proxy (₹/kWh)", 0.0, 5.0, 0.5, 0.1)
    retail_tariff = st.slider("Base customer tariff (₹/kWh)", 11.2, 16.8, 14.0, 0.1)

with st.sidebar.expander("05 · Simulation & policy", expanded=False):
    st.caption("Seed 42 · 14 synthetic test days · five independent metro clusters")
    selected_policy = st.selectbox(
        "Hourly trace policy", POLICIES, index=3,
        format_func=lambda policy: POLICY_LABELS[policy], key="results_policy",
    )

params = {
    "seed": 42,
    "test_days": 14,
    "battery_mwh_total": 600.0,
    "battery_mw_per_cluster": float(battery_power),
    "grid_import_mw_per_cluster": float(grid_limit),
    "charging_capacity_mw_total": 1200.0,
    "reserve_fraction": reserve_pct / 100.0,
    "line_efficiency": 0.975,
    "charge_efficiency": charge_eff_pct / 100.0,
    "discharge_efficiency": discharge_eff_pct / 100.0,
    "discom_tariff_rs_kwh": float(discom_tariff),
    "renewable_cost_rs_kwh": float(renewable_cost),
    "retail_tariff_rs_kwh": float(retail_tariff),
    "iex_price_multiplier": float(iex_multiplier),
    "rapid_share": rapid_share_pct / 100.0,
    "rapid_soh": rapid_soh_pct / 100.0,
    "second_life_soh": second_soh_pct / 100.0,
    "rapid_wear_rs_kwh": float(rapid_wear),
    "second_life_wear_rs_kwh": float(second_wear),
    "uniform_wear_rs_kwh": 1.0,
    "flexible_demand_share": flexible_pct / 100.0,
    "shift_share_of_flexible": shift_pct / 100.0,
    "shift_discount": discount_pct / 100.0,
    "price_elasticity": float(elasticity),
    "customer_delay_cost_rs_kwh": float(delay_cost),
    "priority_segment_share": 0.10,
    "logistics_segment_share": 0.30,
    "ridehail_segment_share": 0.35,
    "retail_segment_share": 0.25,
}
params = _merge_params(params)

with st.spinner("Recomputing synthetic demand, renewable, price and dispatch traces..."):
    data, hourly, summary = calculate(scenario, params)

# Derived tables ------------------------------------------------------------------
full = hourly[hourly.policy == "full"].copy()
base = hourly[hourly.policy == "baseline"].copy()
full_summary = summary.loc[summary.policy == "full"].iloc[0]
base_summary = summary.loc[summary.policy == "baseline"].iloc[0]

cluster_summary = full.groupby("cluster", as_index=False).agg(
    demand_mwh=("demand_mwh", "sum"),
    delivered_mwh=("delivered_mwh", "sum"),
    renewable_available_mwh=("renewable_available_mwh", "sum"),
    renewable_used_mwh=("renewable_used_mwh", "sum"),
    grid_mwh=("grid_mwh", "sum"),
    mean_soc_mwh=("battery_soc_mwh", "mean"),
    min_soc_mwh=("battery_soc_mwh", "min"),
    peak_grid_utilisation_pct=("grid_utilisation_pct", "max"),
)
forecast_cluster = data.groupby("cluster", as_index=False).agg(
    forecast_demand_mwh=("forecast_demand_mwh", "sum"),
    forecast_peak_mwh=("forecast_demand_mwh", "max"),
)
cluster_summary = cluster_summary.merge(forecast_cluster, on="cluster", how="left")
cluster_summary["reliability_pct"] = 100 * cluster_summary.delivered_mwh / cluster_summary.demand_mwh.clip(lower=1e-9)
cluster_summary["congestion"] = np.select(
    [cluster_summary.peak_grid_utilisation_pct >= 95, cluster_summary.peak_grid_utilisation_pct >= 80],
    ["High", "Elevated"], default="Normal")

# Header --------------------------------------------------------------------------
st.markdown("<div class='pg-eyebrow'>Physical energy trading · battery health · customer SLA</div>", unsafe_allow_html=True)
st.markdown("<div class='pg-title'>The Perishable Grid</div>", unsafe_allow_html=True)
st.markdown("<div class='pg-subtitle'>Five-metro energy command centre · 14-day synthetic operating window · hourly heuristic dispatch</div>", unsafe_allow_html=True)
scenario_notes = {
    SCENARIOS[0]: "Baseline synthetic operating shape; no scenario shock overlay.",
    SCENARIOS[1]: "Solar availability reduced by 45%; evening synthetic market price receives a ₹2.20/kWh shock.",
    SCENARIOS[2]: "Synthetic customer demand is increased by 18% across the test window.",
    SCENARIOS[3]: "Solar generation is increased by 22% while demand is reduced by 10% to represent a low-load surplus case.",
}
st.markdown(
    f"<div class='pg-strip'><span><strong>SCENARIO · {scenario.upper()}</strong> &nbsp; {scenario_notes[scenario]}</span>"
    "<span class='pg-strip-status'>● SIMULATION RECOMPUTED</span></div>",
    unsafe_allow_html=True,
)

# Main section tabs ----------------------------------------------------------------
tabs = st.tabs([
    "A · Network Command Centre",
    "B · Energy Trading Desk",
    "C · Battery Intelligence",
    "D · Customer Demand Intelligence",
    "E · Results & Comparison",
])

# A. Network Command Centre --------------------------------------------------------
with tabs[0]:
    section_title("Network command centre", "Five independent metro clusters · one-hour energy balance, no cross-metro transfers")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Installed charging capacity", "1,200 MW", "Case: 1,200 MW · 240 MW equal-share assumption")
    with c2: kpi("Nameplate storage", "600 MWh", "600 MWh case value · 120 MWh equal-share assumption")
    with c3: kpi("Forecast-window reliability", f"{full_summary.reliability_pct:.2f}%", "Full health + SLA policy")
    with c4: kpi("Renewable utilisation", f"{full_summary.renewable_utilisation_pct:.1f}%", "Synthetic renewable supply used")
    with c5: kpi("Peak grid utilisation", f"{full_summary.peak_grid_utilisation_pct:.1f}%", "Full policy · grid cap per metro")

    st.plotly_chart(
        cluster_overview_chart(cluster_summary), use_container_width=True,
        config={"displayModeBar": False}, key="chart_network_cluster_overview",
    )
    st.markdown("<div class='pg-card-title'>CLUSTER OPERATING SNAPSHOT · FULL POLICY</div>", unsafe_allow_html=True)
    display_cluster = cluster_summary[[
        "cluster", "forecast_demand_mwh", "grid_mwh", "reliability_pct",
        "mean_soc_mwh", "peak_grid_utilisation_pct", "congestion",
    ]].copy()
    display_cluster.columns = [
        "Metro", "Forecast MWh", "Grid MWh", "Reliability %",
        "Mean SoC MWh", "Peak grid %", "Congestion",
    ]
    display_cluster["Forecast MWh"] = display_cluster["Forecast MWh"].map(lambda x: f"{x:,.0f}")
    display_cluster["Grid MWh"] = display_cluster["Grid MWh"].map(lambda x: f"{x:,.0f}")
    display_cluster["Reliability %"] = display_cluster["Reliability %"].map(lambda x: f"{x:.1f}%")
    display_cluster["Mean SoC MWh"] = display_cluster["Mean SoC MWh"].map(lambda x: f"{x:.1f}")
    display_cluster["Peak grid %"] = display_cluster["Peak grid %"].map(lambda x: f"{x:.0f}%")
    st.dataframe(display_cluster, use_container_width=True, hide_index=True, height=225)
    st.markdown(
        "<div class='pg-note'>Congestion uses peak grid-import utilisation: Normal &lt;80%, Elevated 80–94%, High ≥95%. "
        "These are dashboard thresholds, not grid-code limits.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='pg-card-title'>TRIPLE-SCARCITY DECISION LENS</div>", unsafe_allow_html=True)
    median_grid_price = min(float(data.forecast_iex_price_rs_kwh.median()), params["discom_tariff_rs_kwh"])
    rapid_wear_cost = params["rapid_wear_rs_kwh"] * (0.90 / params["rapid_soh"])
    second_wear_cost = params["second_life_wear_rs_kwh"] * (0.80 / params["second_life_soh"])
    effective_discount = params["shift_discount"]
    customer_incentive_cost = params["retail_tariff_rs_kwh"] * effective_discount
    customer_shift_cost = customer_incentive_cost + params["customer_delay_cost_rs_kwh"]
    cost_chart = triple_scarcity_chart(median_grid_price, rapid_wear_cost, second_wear_cost, customer_shift_cost)
    st.plotly_chart(cost_chart, use_container_width=True, config={"displayModeBar": False}, key="chart_triple_scarcity_costs")
    st.caption(f"All four bars are marginal ₹/kWh proxies. Customer shift cost combines the tariff incentive (₹{customer_incentive_cost:.2f}) and a user-set synthetic delay/detour proxy (₹{params['customer_delay_cost_rs_kwh']:.2f}); neither is calibrated to field behaviour. This is a transparent heuristic, not a unified MILP argmin.")

# B. Energy Trading Desk -----------------------------------------------------------
with tabs[1]:
    section_title("Energy trading desk", "Synthetic market-price signal · renewable supply procurement · forecast-guided grid route")
    selected_metro = st.selectbox("Inspect metro price signal", CLUSTERS, key="trading_metro")
    p1, p2 = st.columns([1.2, 1])
    with p1:
        st.plotly_chart(price_chart(data, selected_metro, params["discom_tariff_rs_kwh"]), use_container_width=True, config={"displayModeBar": False}, key="chart_trading_price_signal")
    with p2:
        st.plotly_chart(procurement_mix_chart(summary), use_container_width=True, config={"displayModeBar": False}, key="chart_trading_procurement_mix")
    t1, t2, t3, t4 = st.columns(4)
    with t1: st.metric("Grid procurement · full", f"{full_summary.grid_iex_mwh + full_summary.grid_discom_mwh:,.0f} MWh")
    with t2: st.metric("IEX-style route", f"{full_summary.grid_iex_mwh:,.0f} MWh")
    with t3: st.metric("DISCOM route", f"{full_summary.grid_discom_mwh:,.0f} MWh")
    with t4: st.metric("Renewable used", f"{full_summary.renewable_used_mwh:,.0f} MWh")

    st.plotly_chart(hourly_supply_chart(hourly, "full", selected_metro), use_container_width=True, config={"displayModeBar": False}, key="chart_trading_hourly_supply")
    st.markdown("<div class='pg-note'>The non-baseline policies select the synthetic IEX-style route when the seasonal-naive forecast price is below the configured DISCOM tariff. The realised synthetic price is used for settlement. This is a procurement-rule demonstration only; it does not submit exchange bids or integrate with a live market.</div>", unsafe_allow_html=True)

# C. Battery Intelligence ----------------------------------------------------------
with tabs[2]:
    section_title("Battery intelligence", "600 MWh nameplate system · configurable health-tier allocation · reserve and inverter constraints")
    rapid_nameplate = 600.0 * params["rapid_share"]
    second_nameplate = 600.0 * (1 - params["rapid_share"])
    rapid_effective = rapid_nameplate * params["rapid_soh"]
    second_effective = second_nameplate * params["second_life_soh"]
    total_effective = rapid_effective + second_effective
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1: kpi("Rapid nameplate", f"{rapid_nameplate:.0f} MWh", f"{params['rapid_share']*100:.0f}% allocation")
    with b2: kpi("Second-life nameplate", f"{second_nameplate:.0f} MWh", f"{(1-params['rapid_share'])*100:.0f}% allocation")
    with b3: kpi("SoH-adjusted capacity", f"{total_effective:.1f} MWh", "Rapid + second-life")
    with b4: kpi("Available above reserve", f"{total_effective*(1-params['reserve_fraction']):.1f} MWh", f"{params['reserve_fraction']*100:.0f}% held back")
    with b5: kpi("Inverter power", f"{params['battery_mw_per_cluster']*5:.0f} MW", "Aggregate across five metros")

    battery_metro = st.selectbox("Battery SoC trace", CLUSTERS, index=0, key="battery_metro")
    batt_left, batt_right = st.columns([1.25, 1])
    with batt_left:
        st.plotly_chart(battery_soc_chart(hourly, "full", battery_metro), use_container_width=True, config={"displayModeBar": False}, key="chart_battery_soc_detail")
    with batt_right:
        full_batt = full[full.cluster == battery_metro]
        st.markdown("<div class='pg-card-title'>FULL POLICY BATTERY FLOWS</div>", unsafe_allow_html=True)
        st.metric("Rapid discharge", f"{full_batt.rapid_discharge_mwh.sum():,.1f} MWh")
        st.metric("Second-life discharge", f"{full_batt.second_life_discharge_mwh.sum():,.1f} MWh")
        st.metric("Battery throughput", f"{full_batt.battery_charge_mwh.sum()+full_batt.battery_discharge_mwh.sum():,.1f} MWh")
        st.metric("Wear-cost proxy", f"₹{full_batt.wear_cost_rs.sum()/100000:,.2f} lakh")
        st.markdown("<div class='pg-note'>SoH reduces effective energy capacity. The 70:30 split is a configurable simulation assumption, not an optimiser-selected allocation or individual pack reassignment. Category-specific inverter limits are heuristic.</div>", unsafe_allow_html=True)

    st.markdown("<div class='pg-card-title'>OPERATING CONSTRAINTS</div>", unsafe_allow_html=True)
    constraints = pd.DataFrame([
        ["Nameplate battery energy", "600 MWh", "Case-provided"],
        ["Effective energy at current SoH", f"{total_effective:.1f} MWh", "Derived from assumed SoH"],
        ["Inverter rating", f"{params['battery_mw_per_cluster']:.0f} MW / metro", "Synthetic assumption / control"],
        ["Grid import cap", f"{params['grid_import_mw_per_cluster']:.0f} MW / metro", "Synthetic assumption / control"],
        ["Charge / discharge efficiency", f"{params['charge_efficiency']*100:.0f}% / {params['discharge_efficiency']*100:.0f}%", "Synthetic assumption / control"],
        ["Minimum energy reserve", f"{params['reserve_fraction']*100:.0f}%", "Case-relevant floor; applied to storage pools"],
    ], columns=["Constraint", "Configured value", "Basis"])
    st.dataframe(constraints, use_container_width=True, hide_index=True)

# D. Customer Demand Intelligence --------------------------------------------------
with tabs[3]:
    section_title("Customer demand intelligence", "Same-metro time shifting · priority segment protected from incentives · tariff-bounded flexibility")
    customer_cols = st.columns(4)
    with customer_cols[0]: kpi("Priority / emergency", "10%", "Never shifted · synthetic segment mix")
    with customer_cols[1]: kpi("Flexible demand share", f"{params['flexible_demand_share']*100:.0f}%", "Assumed across non-priority groups")
    with customer_cols[2]: kpi("Peak load shifted", f"{full_summary.shifted_peak_mwh:,.0f} MWh", "Across all metros / 14 days")
    discounted_tariff = params["retail_tariff_rs_kwh"] * (1 - params["shift_discount"])
    with customer_cols[3]: kpi("Shifted customer tariff", f"₹{discounted_tariff:.2f}/kWh", f"Effective incentive {params['shift_discount']*100:.1f}% · tariff band enforced")

    segment_rows = [
        ["Priority / emergency fleets", "10%", full.priority_demand_mwh.sum(), 0.0, "Protected; no shift"],
        ["Logistics fleets", "30%", full.logistics_demand_mwh.sum(), full.shifted_logistics_mwh.clip(lower=0).sum(), "Flexible portion eligible"],
        ["Ride-hailing fleets", "35%", full.ridehail_demand_mwh.sum(), full.shifted_ridehail_mwh.clip(lower=0).sum(), "Flexible portion eligible"],
        ["Retail drivers", "25%", full.retail_demand_mwh.sum(), full.shifted_retail_mwh.clip(lower=0).sum(), "Flexible portion eligible"],
    ]
    segment_table = pd.DataFrame(segment_rows, columns=["Customer segment", "Synthetic share", "Demand MWh", "Shifted into solar hours MWh", "Treatment"])
    left, right = st.columns([1, 1.1])
    with left:
        st.dataframe(segment_table.style.format({"Demand MWh": "{:,.0f}", "Shifted into solar hours MWh": "{:,.1f}"}), use_container_width=True, hide_index=True, height=240)
        st.metric("Priority reliability", f"{full_summary.priority_reliability_pct:.2f}%", f"Priority unserved {full_summary.priority_unserved_mwh:,.1f} MWh")
        st.markdown("<div class='pg-note'>Segment shares are illustrative assumptions (10/30/35/25), not observed customer composition. Priority demand receives first allocation in the served-energy accounting; the model does not have customer-level charging sessions.</div>", unsafe_allow_html=True)
    with right:
        st.plotly_chart(customer_shift_chart(hourly), use_container_width=True, config={"displayModeBar": False}, key="chart_customer_demand_shift")
    effective_tariff_band = (0.8 * 14.0, 1.2 * 14.0)
    st.markdown(f"<div class='pg-good'><b>Tariff guardrail active.</b> The effective shifted tariff is ₹{discounted_tariff:.2f}/kWh. Both the base tariff and the post-discount price are kept within the case's ±20% band around ₹14/kWh (₹{effective_tariff_band[0]:.2f}–₹{effective_tariff_band[1]:.2f}). The demand-response uptake proxy scales with discount and elasticity, and shifting remains inside each metro.</div>", unsafe_allow_html=True)

# E. Results & Comparison ----------------------------------------------------------
with tabs[4]:
    section_title("Results & comparison", "Four comparable policies · same scenario data and parameter set · 14 synthetic test days")
    results_display = summary[[
        "policy_label", "reliability_pct", "priority_reliability_pct", "gross_contribution_rs_lakh",
        "procurement_cost_rs_lakh", "wear_cost_rs_lakh", "unserved_mwh", "renewable_utilisation_pct"
    ]].copy()
    results_display.columns = ["Policy", "Reliability %", "Priority reliability %", "Gross contribution ₹ lakh",
                               "Procurement ₹ lakh", "Wear proxy ₹ lakh", "Unserved MWh", "Renewable utilisation %"]
    st.dataframe(results_display.style.format({
        "Reliability %": "{:.2f}", "Priority reliability %": "{:.2f}",
        "Gross contribution ₹ lakh": "{:,.2f}", "Procurement ₹ lakh": "{:,.2f}",
        "Wear proxy ₹ lakh": "{:,.2f}", "Unserved MWh": "{:,.1f}", "Renewable utilisation %": "{:.2f}"
    }), use_container_width=True, hide_index=True)

    reference_path = ROOT / "summary_results.csv"
    if reference_path.exists():
        ref = pd.read_csv(reference_path)
        ref_labels = {"baseline": "Rule-based baseline", "forecast": "Price-aware dispatch",
                      "health": "Health-aware dispatch", "full": "Health + SLA strategy"}
        ref["Original reference policy"] = ref.policy.map(ref_labels)
        st.markdown("<div class='pg-card-title'>UPLOADED ORIGINAL REFERENCE RUN · MIXED-STRESS 14-DAY SIMULATION</div>", unsafe_allow_html=True)
        st.caption("Preserved from the supplied project. This is not the same input scenario as the selected dashboard run, so compare it as provenance rather than a like-for-like scenario result.")
        ref_view = ref[["Original reference policy", "reliability_pct", "gross_contribution_rs_lakh", "wear_cost_rs_lakh"]].copy()
        ref_view.columns = ["Original reference policy", "Reliability %", "Gross contribution ₹ lakh", "Wear proxy ₹ lakh"]
        st.dataframe(ref_view.style.format({"Reliability %": "{:.2f}", "Gross contribution ₹ lakh": "{:,.2f}", "Wear proxy ₹ lakh": "{:,.2f}"}),
                     use_container_width=True, hide_index=True)

    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.plotly_chart(policy_bar(summary, "reliability_pct", "Service reliability by policy", "Reliability (%)", target=89.7, reference=78.0, suffix="%"), use_container_width=True, config={"displayModeBar": False}, key="chart_results_reliability")
    with chart_b:
        st.plotly_chart(policy_bar(summary, "gross_contribution_rs_lakh", "Gross energy contribution · not EBITDA", "₹ lakh over 14 days"), use_container_width=True, config={"displayModeBar": False}, key="chart_results_gross_contribution")
    chart_c, chart_d = st.columns(2)
    with chart_c:
        st.plotly_chart(policy_bar(summary, "wear_cost_rs_lakh", "Battery wear-cost proxy", "₹ lakh over 14 days"), use_container_width=True, config={"displayModeBar": False}, key="chart_results_battery_wear")
    with chart_d:
        st.plotly_chart(policy_bar(summary, "unserved_mwh", "Unserved demand", "MWh over 14 days"), use_container_width=True, config={"displayModeBar": False}, key="chart_results_unserved_demand")
    chart_e, chart_f = st.columns(2)
    with chart_e:
        st.plotly_chart(policy_bar(summary, "procurement_cost_rs_lakh", "Energy procurement expenditure", "₹ lakh over 14 days"), use_container_width=True, config={"displayModeBar": False}, key="chart_results_procurement_cost")
    with chart_f:
        st.plotly_chart(policy_bar(summary, "renewable_utilisation_pct", "Renewable utilisation", "Utilisation (%)", suffix="%"), use_container_width=True, config={"displayModeBar": False}, key="chart_results_renewable_utilisation")

    st.caption(f"Hourly trace policy: {POLICY_LABELS[selected_policy]} · change it in the sidebar under Simulation & policy.")
    hourly_metro = st.selectbox("Hourly trace metro", ["All metros"] + CLUSTERS, key="results_metro")
    st.plotly_chart(hourly_supply_chart(hourly, selected_policy, hourly_metro), use_container_width=True, config={"displayModeBar": False}, key="chart_results_hourly_supply")
    soc_metro = st.selectbox("Battery SoC metro", CLUSTERS, index=0, key="results_soc_metro")
    st.plotly_chart(battery_soc_chart(hourly, selected_policy, soc_metro), use_container_width=True, config={"displayModeBar": False}, key="chart_results_battery_soc")

    # Decision explanation panel --------------------------------------------------
    section_title("Decision trace · what changed and why")
    forecast_avg = float(data.forecast_demand_mwh.mean())
    forecast_peak = float(data.forecast_demand_mwh.max())
    scarce_hour = full.loc[full.grid_utilisation_pct.idxmax()]
    rapid_used = float(full.rapid_discharge_mwh.sum())
    second_used = float(full.second_life_discharge_mwh.sum())
    grid_delta = float(full.grid_mwh.sum() - base.grid_mwh.sum())
    shift_mwh = float(full.shifted_from_peak_mwh.sum())
    reliability_delta = float(full_summary.reliability_pct - base_summary.reliability_pct)
    contribution_delta = float(full_summary.gross_contribution_rs_lakh - base_summary.gross_contribution_rs_lakh)
    if full_summary.peak_grid_utilisation_pct >= 95:
        scarce_resource = "grid import headroom" 
    elif full.battery_soc_mwh.min() <= full.battery_soc_min_mwh.min() + 0.5:
        scarce_resource = "battery energy above the minimum reserve"
    else:
        scarce_resource = "available flexible customer time / renewable timing"
    grid_direction = "increased" if grid_delta > 0 else "decreased" if grid_delta < 0 else "was unchanged"
    st.markdown(f"""
<div class='pg-card'>
  <div class='pg-card-title'>SCENARIO READOUT · {scenario.upper()}</div>
  <div style='color:#8fd9cf;font-size:11px;margin:0 0 6px'>TRIPLE-COST GATE · Grid ₹{median_grid_price:.2f}/kWh · Rapid wear ₹{rapid_wear_cost:.2f}/kWh · Second-life wear ₹{second_wear_cost:.2f}/kWh · Shift cost ₹{customer_shift_cost:.2f}/kWh</div>
  <ul style='color:#d5e5eb;font-size:12px;line-height:1.75;margin:4px 0 4px 18px;padding:0'>
    <li><b>Forecast:</b> average one-day seasonal-naive demand was {forecast_avg:.1f} MWh per metro-hour; forecast peak was {forecast_peak:.1f} MWh.</li>
    <li><b>Scarcity signal:</b> the strongest operating pressure was {scarce_resource}; maximum full-policy grid utilisation was {full_summary.peak_grid_utilisation_pct:.1f}% at {scarce_hour.cluster}, hour {int(scarce_hour.hour):02d}:00.</li>
    <li><b>Battery category used:</b> Rapid delivered {rapid_used:,.1f} MWh and second-life delivered {second_used:,.1f} MWh under the health + SLA policy. Health-aware policies use each wear proxy against forecast grid marginal cost before discretionary peak discharge; reserve and power caps still bind.</li>
    <li><b>Grid procurement:</b> full-policy grid energy {grid_direction} by {abs(grid_delta):,.1f} MWh versus the rule-based baseline. Synthetic market route selection remains forecast-threshold based.</li>
    <li><b>Customer response:</b> {shift_mwh:,.1f} MWh was moved out of peak windows and back into solar-rich hours within the same metro when the incentive proxy was below the forecast grid marginal price; priority fleets were not shifted.</li>
    <li><b>Outcome:</b> reliability changed by {reliability_delta:+.2f} percentage points and gross contribution by ₹{contribution_delta/100:,.2f} crore versus the rule-based baseline in this selected scenario.</li>
  </ul>
  <div class='pg-mini'>This is an interpretation of the modelled hourly traces, not a promise of field performance.</div>
</div>
""", unsafe_allow_html=True)

    # Downloads ------------------------------------------------------------------
    st.markdown("<div class='pg-card-title'>EXPORT SCENARIO RESULTS</div>", unsafe_allow_html=True)
    hourly_csv = hourly.to_csv(index=False).encode("utf-8")
    summary_csv = summary.to_csv(index=False).encode("utf-8")
    data_csv = data.to_csv(index=False).encode("utf-8")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("↓ Hourly results CSV", hourly_csv, file_name="perishable_grid_hourly_results.csv", mime="text/csv", use_container_width=True)
    with d2:
        st.download_button("↓ Policy summary CSV", summary_csv, file_name="perishable_grid_summary_results.csv", mime="text/csv", use_container_width=True)
    with d3:
        st.download_button("↓ Forecast & scenario inputs CSV", data_csv, file_name="perishable_grid_scenario_inputs.csv", mime="text/csv", use_container_width=True)

    with st.expander("Model Assumptions & Limitations", expanded=False):
        st.markdown(
            """
            **Dispatch and forecasting**
            - Dispatch is a transparent hourly heuristic, not a solved MILP or production MPC.
            - The forecast is a one-day seasonal-naive baseline (same metro and hour from the previous day), not a trained TFT, XGBoost or other machine-learning model.

            **Markets and energy flows**
            - IEX-style prices, renewable profiles and scenario shocks are synthetic assumptions. No live IEX connection or bids are placed.
            - VPP and ancillary-service revenue are excluded.
            - Energy and demand shifting remain within each metro. No inter-metro energy transfer or physical pack transportation is modelled.

            **Battery representation**
            - The Rapid / second-life allocation is a configurable simulation assumption, not an optimiser-selected assignment or individual pack reassignment.
            - SoH-tier capacities, inverter limits and the 5% energy reserve are represented by the prototype's simplified storage model.

            **Financial and target interpretation**
            - Gross contribution is not EBITDA; fixed O&M, capex, taxes and depreciation are excluded.
            - The simplified loss proxy is not directly comparable with the case's 10% loss reference or 8% target because the metering boundary is not provided.
            - Reliability (89.7%), loss (8%) and EBITDA (above 12%) are comparison targets, not optimisation constraints.
            - The synthetic rule-based baseline is not calibrated to the case-reported 78% reliability.
            """
        )

# Footer ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("<div style='display:flex;justify-content:space-between;color:#7893a2;font-size:10px'><span>THE PERISHABLE GRID · HACKATHON DECISION-SUPPORT PROTOTYPE</span><span>SEED 42 · 14 SYNTHETIC DAYS · 5 INDEPENDENT METROS</span></div>", unsafe_allow_html=True)
