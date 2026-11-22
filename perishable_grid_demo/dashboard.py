"""Small dashboard wrapper for the existing simulator and ACN-trained model.

The ACN model is displayed as a separate U.S.-site forecast source. It is not
silently substituted into the five synthetic Indian metro-cluster dispatches.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from simulate import CLUSTERS, make_data
from forecasting.preprocessing import kwh_to_mwh

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ev_demand_lightgbm.joblib"
FORECAST_PATH = ROOT / "results" / "forecast_for_simulator.csv"


def _forecast_markdown_table(frame: pd.DataFrame) -> str:
    """Render small forecast outputs without Streamlit's interactive DataFrame grid.

    Community Cloud can fail while fetching the lazily-loaded DataFrame JS chunk.
    This static Markdown table avoids requesting that frontend bundle entirely.
    """
    columns = [
        ("timestamp", "Timestamp (site local)"),
        ("site_id", "ACN site"),
        ("predicted_demand_kwh", "Forecast (kWh)"),
        ("predicted_demand_mwh", "Forecast (MWh)"),
        ("p10_demand_kwh", "P10 (kWh)"),
        ("p50_demand_kwh", "P50 (kWh)"),
        ("p90_demand_kwh", "P90 (kWh)"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]

    for _, record in frame.iterrows():
        cells = []
        for key, _ in columns:
            value = record[key]
            if key == "timestamp":
                rendered = str(value).replace("|", "\\|")
            elif key == "site_id":
                rendered = str(value).replace("|", "\\|")
            else:
                rendered = "—" if pd.isna(value) else f"{float(value):,.3f}"
            cells.append(rendered)
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Streamlit is not installed. Run: pip install -r requirements.txt") from exc

    st.set_page_config(page_title="The Perishable Grid", layout="wide")
    st.title("The Perishable Grid")
    st.caption("Heuristic dispatch simulation and a separately trained ACN-Data demand forecaster")

    model_available = MODEL_PATH.exists() and FORECAST_PATH.exists()
    forecast_sources = ["Existing seasonal-naive forecast"]
    if model_available:
        forecast_sources.append("Trained LightGBM forecast")
    selected_source = st.sidebar.selectbox("Forecast source", forecast_sources)

    if selected_source == "Existing seasonal-naive forecast":
        st.subheader("Existing simulator: seasonal-naive demand forecast")
        st.info("This is the original synthetic five-cluster forecast. It uses the previous day's same cluster/hour and remains available without a trained model.")
        selected_cluster = st.selectbox("Indian metro cluster", CLUSTERS)
        sim_data = make_data()
        sim_data = sim_data[sim_data.day > 1].copy()
        chart_data = sim_data[sim_data.cluster == selected_cluster].set_index("hour_index")[[
            "demand_mwh", "forecast_demand_mwh"
        ]].rename(columns={
            "demand_mwh": "Synthetic demand (MWh)",
            "forecast_demand_mwh": "Seasonal-naive forecast (MWh)",
        })
        st.line_chart(chart_data)
        st.caption("Illustrative synthetic simulator inputs, not measured demand from an Indian charging operator.")
    else:
        st.subheader("Trained LightGBM: next-hour ACN site forecast")
        st.warning(
            "The supplied ACN sites are in California (America/Los_Angeles), and their last observations are from 2018–2021. "
            "The displayed next-hour rows are historical-origin pipeline examples, not current live forecasts. They are not "
            "mapped to Delhi NCR, Mumbai MMR, Bengaluru, Hyderabad or Chennai and are not fed into the Indian-cluster dispatch engine."
        )
        forecast = pd.read_csv(FORECAST_PATH, dtype={"site_id": str})
        required = ["timestamp", "site_id", "predicted_demand_kwh", "p10_demand_kwh", "p50_demand_kwh", "p90_demand_kwh"]
        missing = [column for column in required if column not in forecast.columns]
        if missing:
            st.error(f"Forecast file is missing columns: {missing}")
            return
        forecast["predicted_demand_mwh"] = kwh_to_mwh(forecast["predicted_demand_kwh"])
        forecast["p10_demand_mwh"] = kwh_to_mwh(forecast["p10_demand_kwh"])
        forecast["p50_demand_mwh"] = kwh_to_mwh(forecast["p50_demand_kwh"])
        forecast["p90_demand_mwh"] = kwh_to_mwh(forecast["p90_demand_kwh"])

        # The CSV is only three site rows in this supplied run. A static Markdown
        # table is clearer here and avoids Streamlit's lazily imported DataFrame
        # frontend chunk, which is the asset named in the reported browser error.
        st.markdown(_forecast_markdown_table(forecast))
        st.caption(
            "The model forecasts session-arrival cohort energy (kWh), not exact physical "
            "consumption during that clock hour. MWh values are kWh / 1,000."
        )

    st.divider()
    st.caption("The dispatch simulator remains heuristic. Forecasting, dispatch, battery-health assumptions and market participation are separate capabilities.")


if __name__ == "__main__":
    main()
