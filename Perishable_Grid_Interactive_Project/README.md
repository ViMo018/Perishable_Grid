# The Perishable Grid — interactive energy decision simulator

A Streamlit control-room prototype for five Indian metro clusters: Delhi NCR, Mumbai MMR, Bengaluru, Hyderabad and Chennai. It connects synthetic demand, renewable availability, electricity procurement, battery health proxies and same-metro customer time shifting in one inspectable hourly simulation.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS / Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

Run the automated checks:

```bash
pytest -q
```

Regenerate the original uploaded reference run:

```bash
python simulate.py
```

Generate the dashboard's default normal-day CSV exports without starting Streamlit:

```bash
python generate_exports.py
```

## Dashboard sections

1. **Network Command Centre** — five-cluster demand, renewable availability, grid draw, storage state of charge, reliability and grid-congestion indicators. Each metro is simulated independently; no inter-metro energy transfer occurs.
2. **Energy Trading Desk** — synthetic hourly market-price signals, a DISCOM-versus-IEX-style route decision, renewable supply costs, procurement mix and battery schedule traces. The interface labels all IEX-style prices as synthetic.
3. **Battery Intelligence** — 600 MWh nameplate storage, configurable Rapid / second-life allocation, state of health, effective energy, inverter power, efficiency, reserve and wear-cost proxy. The allocation remains a configurable assumption; no pack-level optimisation or physical reassignment is claimed.
4. **Customer Demand Intelligence** — priority/emergency, logistics, ride-hailing and retail-driver segments; same-metro shifting of flexible non-priority demand; price elasticity as a simple uptake proxy; and a customer tariff guardrail.
5. **Results & Comparison** — four policies, reliability, gross contribution, procurement, battery wear, unserved demand, renewable use, hourly supply traces, battery SoC traces and CSV downloads.

## Four operating scenarios

| Scenario | Synthetic overlay |
|---|---|
| Normal operating day | No shock overlay; normal synthetic demand, solar, wind and price shapes |
| Monsoon cloud-cover and electricity-price shock | Solar availability ×0.55; evening synthetic price +₹2.20/kWh |
| Heatwave with customer demand surge | Demand ×1.18 |
| Renewable generation surplus during low demand | Solar generation ×1.22; demand ×0.90 |

These are deterministic stress-test assumptions using seed 42, not weather forecasts or observed market data. The dashboard generates one warm-up day, then evaluates 14 synthetic test days for each scenario.

## Four policies

- `baseline` — fixed peak-window battery dispatch and DISCOM-first procurement.
- `forecast` — one-day seasonal-naive forecast, price-aware procurement and storage scheduling.
- `health` — forecast logic plus separate Rapid / second-life energy pools and different wear-cost proxies.
- `full` — health-aware dispatch plus SLA-style same-metro shifting of a share of flexible non-priority evening demand into solar-rich hours.

All four policies receive the same synthetic scenario inputs and configured parameters. The original `simulate.py` is retained. It continues to regenerate the initial 14-day mixed-stress reference run and its original `summary_results.csv`, `hourly_results.csv` and chart files. The Streamlit scenario engine is a parameterised extension; its outputs should not be confused with the original mixed-stress benchmark because each dashboard scenario is a distinct synthetic case.

## Core assumptions and constraints

### Case-provided values

- Five clusters.
- 1,200 MW total installed charging capacity (represented as a 240 MW per-metro demand envelope for this prototype).
- 600 MWh nameplate storage (represented as 120 MWh per metro).
- Customer tariff variation limited to ±20% around ₹14/kWh, i.e. ₹11.20–₹16.80/kWh.
- Case-reported current reliability: 78%.
- Case comparison targets: 89.7% reliability, 8% losses and above 12% EBITDA.

### Synthetic assumptions and UI controls

- 30 MW battery inverter and 38 MW grid-import cap per metro by default.
- 5% minimum storage reserve (the UI does not allow a lower reserve setting).
- 95% charge and discharge efficiency; 97.5% simplified delivery efficiency.
- 70% Rapid allocation at 90% SoH; 30% second-life allocation at 80% SoH.
- DISCOM ₹8.50/kWh; renewable supply cost ₹3.50/kWh; customer tariff ₹14/kWh.
- Synthetic IEX-style prices; no historical exchange data is loaded.
- Wear-cost proxies: Rapid ₹1.20/kWh and second-life ₹0.65/kWh at default SoH; baseline pooled proxy ₹1.00/kWh.
- Flexible-demand share 40%; shifted share of flexible peak demand 15%; incentive 10%; price-elasticity multiplier 1.0; synthetic customer delay/detour cost proxy ₹0.50/kWh.
- Synthetic segment split: priority/emergency 10%, logistics 30%, ride-hailing 35%, retail drivers 25%.

The battery SoH assumptions are applied to effective energy capacity in the dashboard scenario engine. The nameplate system remains 600 MWh. The original `simulate.py` implementation is preserved unchanged as the reference run; it does not reduce capacity by SoH.

The customer price guardrail applies to both base and discounted shifted tariffs. Discount and elasticity combine in a transparent uptake proxy, capped at 30% of the eligible flexible segment. Priority demand is not shifted and receives first allocation in the delivered-energy accounting. Segment shares and elasticities are synthetic, not empirical customer data.

The 1,200 MW charging-system constraint is represented through a 240 MW per-metro demand ceiling. Grid imports and battery inverter power are enforced per metro and per hour. Battery energy capacity and reserve are checked in every interval. The separate swap-pack inventory reserve and customer-level charging-session inventory are outside the source model because pack inventory and session data are not supplied.

## Mathematical model and decision logic

The forecast is a **one-day seasonal-naive baseline**: each cluster/hour forecast takes the same cluster/hour value from the previous day. It is not a trained TFT, XGBoost or other machine-learning model.

Dispatch is a transparent hourly heuristic, not a solved MILP or production MPC. Baseline and forecast policies use a pooled battery state. Health-aware policies use distinct Rapid and second-life pools, with separate energy caps, reserve checks, heuristic power shares and wear-cost proxies. Grid procurement uses a forecast-price threshold to select an IEX-style synthetic route when forecast price is below the configured DISCOM tariff. The realised synthetic market price is used for settlement.

The triple-scarcity panel displays marginal ₹/kWh proxies for grid procurement, Rapid wear, second-life wear and customer shift cost (tariff incentive plus a configurable synthetic delay/detour proxy). In health-aware modes, a battery category is used for discretionary peak displacement when its wear proxy is no higher than the forecast grid marginal price; in the full policy, flexible demand is shifted when its customer shift-cost proxy is below the forecast grid marginal price. Grid connection, inverter and reserve limits remain binding. This is a transparent heuristic, not a unified MILP objective, and the delay/detour proxy is not empirically calibrated.

Demand shifting is time-only and stays inside each metro. The model does not transfer batteries, packs or energy between metros. It does not physically repurpose packs, optimise individual pack assignment or provide customer-level charging-session control.

## Metric definitions

The dashboard retains the source model's units and financial conventions:

- **Reliability (%)** = total delivered energy ÷ total demand energy × 100.
- **Unserved demand (MWh)** = demand energy less delivered energy.
- **Procurement expenditure (₹ lakh)** = synthetic grid procurement cost plus renewable supply cost, divided by 100,000.
- **Battery wear-cost proxy (₹ lakh)** = modelled charge/discharge throughput wear proxy, divided by 100,000.
- **Gross contribution (₹ lakh)** = simulated customer revenue less simulated grid procurement, renewable supply cost, incentive discounts and battery wear-cost proxy, divided by 100,000.
- **Renewable utilisation (%)** = renewable energy used ÷ renewable energy available × 100.
- **Modelled loss proxy (%)** = the source's simplified losses ÷ (simplified losses + delivered energy) × 100.

Gross contribution is **not EBITDA**: fixed O&M, capex, taxes and depreciation are excluded. The modelled loss proxy must not be compared directly with the case's 10% loss reference or 8% target because the real metering boundary is not supplied. The dashboard does not force 89.7% reliability, 8% losses or above 12% EBITDA as optimisation constraints.

The uploaded original synthetic mixed-stress run reported 81.70% rule-based reliability and 87.58% full-policy reliability. It does not reproduce the case-reported 78% current reliability and was not calibrated to operator data. Dashboard scenario metrics vary with the chosen scenario and UI assumptions; no performance gain is guaranteed.

## Project structure

```text
perishable_grid_demo/
├── app.py                         # Streamlit application
├── simulate.py                    # Original reference simulation (preserved)
├── scenario_engine.py             # Interactive scenario and configurable dispatch engine
├── forecast.py                    # Seasonal-naive forecast helper
├── visualizations.py              # Plotly chart builders
├── generate_exports.py             # Generate default dashboard CSVs
├── assumptions.json               # Case values, synthetic assumptions and limitations
├── summary_results.csv            # Original reference policy summary
├── hourly_results.csv             # Original reference hourly results
├── exports/                       # Default dashboard scenario export files
├── tests/test_simulation.py        # Physics, tariff, cluster and comparability checks
├── sources.md                     # Source register
├── requirements.txt
└── README.md
```

## Limitations to state in the pitch

- Heuristic dispatch, not MILP output; no production MPC.
- Seasonal-naive forecast, not trained machine learning.
- Synthetic energy prices and renewable profiles; no live IEX market integration or bid submission.
- No VPP or ancillary-service revenue.
- No individual pack reassignment, physical repurposing or inter-hub pack transportation.
- No inter-metro energy exchange.
- Gross contribution is not EBITDA.
- Simplified loss proxy is not comparable to the case loss percentage without a defined meter boundary.
- Synthetic rule-based baseline is not calibrated to the case-reported 78% reliability.
- 5% storage energy reserve is enforced; separate swap-inventory reserve accounting is not implemented.
- Customer segment composition and price elasticity are illustrative; the model does not use actual customer/session data.

## References

See [`sources.md`](sources.md) for the references used to motivate physical market framing, storage degradation, second-life batteries and smart charging. Those sources do not imply that this prototype has live market, VPP or production bidding capability.
