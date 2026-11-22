"""Plotly views for The Perishable Grid Streamlit control room."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAVY = "#091521"
PANEL = "#102434"
TEAL = "#39D0C3"
GREEN = "#76D7A6"
AMBER = "#E8B45B"
RED = "#F07878"
MUTED = "#91A8B8"
TEXT = "#EAF2F7"
GRID = "rgba(151,180,196,0.16)"


def _layout(fig: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=12),
        title_font=dict(size=15, color=TEXT),
        margin=dict(l=12, r=12, t=48 if title else 18, b=28),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        hoverlabel=dict(bgcolor="#142B3C", font_color=TEXT),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def policy_bar(summary: pd.DataFrame, metric: str, title: str, y_title: str,
               *, target: float | None = None, reference: float | None = None,
               suffix: str = "") -> go.Figure:
    frame = summary.copy()
    labels = frame["policy_label"]
    values = frame[metric]
    fig = go.Figure(go.Bar(
        x=labels, y=values, text=[f"{v:,.2f}{suffix}" for v in values],
        textposition="outside", marker=dict(color=[TEAL, "#4BA8B8", GREEN, AMBER]),
        hovertemplate="%{x}<br>%{y:,.3f}<extra></extra>",
    ))
    if target is not None:
        fig.add_hline(y=target, line_dash="dot", line_color=AMBER,
                      annotation_text=f"Case target {target:g}{suffix}", annotation_position="top left")
    if reference is not None:
        fig.add_hline(y=reference, line_dash="dash", line_color=MUTED,
                      annotation_text=f"Case reference {reference:g}{suffix}", annotation_position="bottom left")
    fig.update_layout(yaxis_title=y_title, xaxis_title=None)
    return _layout(fig, 340, title)


def hourly_supply_chart(hourly: pd.DataFrame, policy: str, cluster: str = "All metros") -> go.Figure:
    frame = hourly[hourly.policy == policy].copy()
    if cluster != "All metros":
        frame = frame[frame.cluster == cluster]
    frame = frame.groupby("hour", as_index=False)[[
        "demand_mwh", "delivered_mwh", "renewable_used_mwh", "grid_mwh", "battery_discharge_mwh", "battery_charge_mwh"
    ]].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.hour, y=frame.demand_mwh, name="Demand", mode="lines+markers",
                             line=dict(color=TEXT, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=frame.hour, y=frame.delivered_mwh, name="Delivered", mode="lines+markers",
                             line=dict(color=TEAL, width=2.5)))
    fig.add_trace(go.Scatter(x=frame.hour, y=frame.renewable_used_mwh, name="Renewable used", stackgroup="supply",
                             line=dict(color=GREEN, width=1)))
    fig.add_trace(go.Scatter(x=frame.hour, y=frame.grid_mwh, name="Grid procurement", stackgroup="supply",
                             line=dict(color="#4A94C6", width=1)))
    fig.add_trace(go.Scatter(x=frame.hour, y=frame.battery_discharge_mwh, name="Battery discharge", stackgroup="supply",
                             line=dict(color=AMBER, width=1)))
    fig.add_trace(go.Scatter(x=frame.hour, y=-frame.battery_charge_mwh, name="Battery charging (load)",
                             mode="lines", line=dict(color="#B28AE6", width=1.6, dash="dash")))
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Average MWh per metro-hour (charging shown negative)",
                      hovermode="x unified")
    return _layout(fig, 390, "Average hourly demand and energy contributions")


def battery_soc_chart(hourly: pd.DataFrame, policy: str, cluster: str = "Delhi NCR") -> go.Figure:
    frame = hourly[(hourly.policy == policy) & (hourly.cluster == cluster)].sort_values("hour_index")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.hour_index, y=frame.battery_soc_mwh, name="Total SoC",
                             line=dict(color=TEAL, width=2)))
    if policy in ("health", "full"):
        fig.add_trace(go.Scatter(x=frame.hour_index, y=frame.rapid_soc_mwh, name="Rapid SoC",
                                 line=dict(color=GREEN, width=1.6)))
        fig.add_trace(go.Scatter(x=frame.hour_index, y=frame.second_life_soc_mwh, name="Second-life SoC",
                                 line=dict(color=AMBER, width=1.6)))
    fig.add_trace(go.Scatter(x=frame.hour_index, y=frame.battery_soc_min_mwh, name="Minimum reserve",
                             line=dict(color=RED, width=1.2, dash="dash")))
    fig.update_layout(xaxis_title="Simulation hour (14 test days)", yaxis_title="Stored energy (MWh)",
                      hovermode="x unified")
    return _layout(fig, 360, f"Battery state of charge · {cluster}")


def cluster_overview_chart(cluster_summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cluster_summary.cluster, y=cluster_summary.demand_mwh,
                         name="Demand", marker_color=TEXT))
    fig.add_trace(go.Bar(x=cluster_summary.cluster, y=cluster_summary.renewable_available_mwh,
                         name="Renewable availability", marker_color=GREEN))
    fig.update_layout(barmode="group", xaxis_title=None, yaxis_title="MWh over test window")
    return _layout(fig, 340, "Demand and renewable availability by metro")


def price_chart(data: pd.DataFrame, cluster: str = "Delhi NCR", discom_tariff: float = 8.5) -> go.Figure:
    frame = data[data.cluster == cluster].sort_values("hour_index")
    day = int(frame.day.min())
    frame = frame[frame.day == day]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.hour, y=frame.iex_price_rs_kwh, name="Synthetic IEX-style price",
                             mode="lines+markers", line=dict(color=TEAL, width=2.2)))
    fig.add_hline(y=discom_tariff, line_dash="dash", line_color=AMBER,
                  annotation_text=f"DISCOM ₹{discom_tariff:.2f}/kWh")
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="₹ / kWh", hovermode="x unified")
    return _layout(fig, 330, f"Synthetic price signal · {cluster} · day {day}")


def procurement_mix_chart(summary: pd.DataFrame) -> go.Figure:
    labels = summary.policy_label.tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=summary.grid_iex_mwh, name="IEX-style route", marker_color=TEAL))
    fig.add_trace(go.Bar(x=labels, y=summary.grid_discom_mwh, name="DISCOM route", marker_color="#4A94C6"))
    fig.update_layout(barmode="stack", yaxis_title="Grid energy (MWh)", xaxis_title=None)
    return _layout(fig, 330, "Synthetic grid-procurement route split")


def customer_shift_chart(hourly: pd.DataFrame) -> go.Figure:
    frame = hourly[hourly.policy == "full"].groupby("hour", as_index=False)[
        ["shifted_mwh", "shifted_from_peak_mwh"]].sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=frame.hour, y=frame.shifted_from_peak_mwh, name="Moved out of peak",
                         marker_color=AMBER))
    fig.add_trace(go.Bar(x=frame.hour, y=frame.shifted_mwh.clip(lower=0), name="Moved into solar hours",
                         marker_color=GREEN))
    fig.update_layout(barmode="overlay", xaxis_title="Hour of day", yaxis_title="MWh shifted (average day)")
    return _layout(fig, 320, "Same-metro flexible demand movement")


def triple_scarcity_chart(grid_cost: float, rapid_wear: float, second_wear: float,
                          customer_incentive: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Grid marginal price", "Rapid wear proxy", "Second-life wear proxy", "Customer shift cost"],
        y=[grid_cost, rapid_wear, second_wear, customer_incentive],
        marker_color=["#4A94C6", GREEN, AMBER, TEAL],
        text=[f"₹{x:.2f}" for x in [grid_cost, rapid_wear, second_wear, customer_incentive]],
        textposition="outside",
    ))
    fig.update_layout(yaxis_title="₹ / kWh (proxy)", xaxis_title=None)
    return _layout(fig, 310, "Triple-scarcity marginal-cost lens")
