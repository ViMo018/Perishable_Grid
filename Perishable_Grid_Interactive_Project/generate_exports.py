"""Generate a reproducible default dashboard scenario export without Streamlit."""
from pathlib import Path
import json
from scenario_engine import DEFAULT_PARAMS, SCENARIOS, run_all_policies

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exports"
OUT.mkdir(exist_ok=True)
scenario = SCENARIOS[0]
data, hourly, summary = run_all_policies(scenario, DEFAULT_PARAMS)
data.to_csv(OUT / "normal_operating_day_inputs.csv", index=False)
hourly.to_csv(OUT / "normal_operating_day_hourly_results.csv", index=False)
summary.to_csv(OUT / "normal_operating_day_summary_results.csv", index=False)
with open(OUT / "normal_operating_day_parameters.json", "w", encoding="utf-8") as f:
    json.dump({"scenario": scenario, "parameters": DEFAULT_PARAMS}, f, indent=2)
print(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print(f"\nWrote dashboard exports to {OUT}")
