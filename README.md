# The Perishable Grid
### AI-Powered Physical Energy Trading and Health-Aware EV Energy Optimisation

The Perishable Grid is a decision-support prototype designed to optimise energy allocation, battery utilisation and customer demand management across a network of electric vehicle charging and battery-swapping hubs.

The project focuses on three interconnected resources: electricity, battery lifespan and customer waiting time.

Instead of managing charging demand, battery storage and electricity procurement independently, the proposed framework brings these decisions together to improve network reliability and operating economics.

## The Core Concept

The framework follows a Triple-Scarcity Decision approach:

1. **Energy:** Procure and allocate electricity according to renewable availability, market prices and charging demand.
2. **Battery lifespan:** Account for State of Health (SoH), charging cycles and degradation costs when scheduling storage.
3. **Customer time:** Use service-level priorities and flexible demand incentives to reduce congestion at charging hubs.

## Project Components

The repository contains two independent Streamlit applications.

### 1. Energy Decision Dashboard

Location: `Perishable_Grid_Interactive_Project/`

The primary application demonstrates:

- Five-metro energy network simulation.
- Energy procurement and dispatch decisions.
- Battery State of Charge and health-aware allocation.
- Customer demand management.
- Operating scenario simulation.
- Strategy comparison and performance visualisation.

Run locally:

```bash
streamlit run Perishable_Grid_Interactive_Project/app.py
```

### 2. EV Demand Forecasting Dashboard

Location: `perishable_grid_demo/`

The second application provides a separate forecasting and data-analysis environment built around ACN-Data.

Its components include:

- EV charging-session data preprocessing.
- Hourly charging-demand forecasting.
- Forecasting baseline comparison.
- Model training and evaluation scripts.
- Forecast visualisation and result exports.

Run locally:

```bash
streamlit run perishable_grid_demo/dashboard.py
```

### Technical Approach

The broader proposed framework combines:

- LightGBM / XGBoost for charging-demand forecasting.
- Weather-informed renewable generation forecasting.
- Mixed-Integer Linear Programming for constrained energy allocation.
- Model Predictive Control for rolling-horizon scheduling.
- Battery degradation modelling using SoH, DoD, temperature and throughput.
- SLA-weighted customer pricing and demand shifting.

The forecasting pipeline, heuristic simulator and proposed optimisation framework are maintained as distinct components. A method is not described as implemented unless it is supported by the corresponding code and evaluation.

## Data Sources

The project uses or references:

- ACN-Data: Public EV charging-session data from Caltech, JPL and Office 1.
- Indian regional electricity demand and renewable generation data.
- Weather observations from ERA5, where applicable.
- Historical electricity-market information from IEX.

Public dataset links and source references are maintained in the relevant project documentation.

Raw datasets are not included in this repository by default. Refer to the dataset download instructions before running the forecasting pipeline.

## Simulation and Evaluation

The energy simulation compares four operating approaches:

1. Rule-based dispatch.
2. Price-aware dispatch.
3. Battery-health-aware dispatch.
4. Battery-health-aware dispatch with SLA-based demand shifting.

The initial simulation uses synthetic operating conditions and explicitly labelled assumptions. Its results are experimental outputs, not measured performance from an actual charging operator.

Gross contribution, battery wear-cost proxies and service reliability are reported separately. Gross contribution must not be interpreted as EBITDA.

## Repository Structure

```text
perishable-grid/
│
├── Perishable_Grid_Interactive_Project/
│   ├── app.py
│   ├── simulate.py
│   ├── scenario_engine.py
│   ├── visualizations.py
│   ├── assumptions.json
│   ├── requirements.txt
│   └── tests/
│
├── perishable_grid_demo/
│   ├── dashboard.py
│   ├── train_forecaster.py
│   ├── forecasting/
│   ├── data/
│   ├── models/
│   ├── results/
│   ├── requirements.txt
│   └── tests/
│
├── docs/
├── README.md
└── .gitignore
```

## Local Installation

Python 3.11 is the recommended development environment, subject to the package versions supported by each application.

Install dependencies for the relevant application:

```bash
pip install -r Perishable_Grid_Interactive_Project/requirements.txt
```

Run the primary dashboard:

```bash
streamlit run Perishable_Grid_Interactive_Project/app.py
```

For the forecasting application, install its dependencies:

```bash
pip install -r perishable_grid_demo/requirements.txt
```

Then execute:

```bash
streamlit run perishable_grid_demo/dashboard.py
```

If the two applications require conflicting package versions, use separate Python virtual environments.

## Limitations

- Historical company-level operational data has not been supplied.
- Synthetic scenarios are used for the initial energy simulation.
- ACN-Data represents US charging locations and is not a direct substitute for Indian metro charging behaviour.
- Real-world battery degradation requires manufacturer or experimentally calibrated battery data.
- Live IEX bidding and VPP participation are outside the current prototype unless separately implemented and validated.
- Financial and operational improvements require validation using actual operator data.

## Objective

The project aims to demonstrate how forecasting, physical energy trading, battery-health-aware dispatch and customer demand management can be combined into a single decision-support framework for EV charging infrastructure.

The long-term objective is to make better use of existing charging capacity and storage while maintaining reliable service and sustainable operating economics.