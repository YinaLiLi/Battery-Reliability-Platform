# Automated Battery Reliability & Predictive Analytics Platform

An end-to-end platform that retrains and evaluates RUL and Survival models as progressive telemetry arrives, serves Current models, and powers real-time fleet and battery monitoring. It maintains incremental feature state without rebuilding history; BatteryLife MATR simulates production telemetry, with 130M+ measurements as supporting scale.

## Highlights

- Automatically retrains and evaluates RUL and Survival models as new telemetry arrives.
- Serves Current models and predictions through PostgreSQL and a Streamlit dashboard.
- Monitors fleet risk and individual battery health in real time.
- Appends incremental feature state to the Shared Feature Outlet without rebuilding history.

## Dashboard

The working Streamlit dashboard covers fleet risk, battery detail, RUL model monitoring, and Survival model monitoring.

<!-- TODO: add a real dashboard screenshot from a verified run; do not use a synthetic image. -->

## Architecture

```mermaid
flowchart TD
  classDef layer fill:transparent,stroke:#6b7280,stroke-width:2px,stroke-dasharray:5 4;
  classDef context fill:none,stroke:none,color:inherit,width:300px;
  classDef spacer fill:none,stroke:none,width:100px;

  subgraph ROWA[" "]
    direction LR
    A["<div style='width:360px;text-align:center'>Battery telemetry / lifecycle data</div>"]:::layer
    A_gap[" "]:::spacer
    A_info["<div style='width:320px;text-align:left'><b>DATA SOURCE</b><br/><i>BatteryLife MATR</i></div>"]:::context
  end
  subgraph ROWB[" "]
    direction LR
    B["<div style='width:360px;text-align:center'>Progressive telemetry ingestion</div>"]:::layer
    B_gap[" "]:::spacer
    B_info["<div style='width:320px;text-align:left'><b>DATA & INGESTION</b><br/><i>Kafka</i></div>"]:::context
  end
  subgraph ROWC[" "]
    direction LR
    C["<div style='width:360px;text-align:center'>Event processing<br/>Prefix-complete finalized battery state</div>"]:::layer
    C_gap[" "]:::spacer
    C_info["<div style='width:320px;text-align:left'><b>STREAMING & STATE</b><br/><i>Kafka · Spark Structured Streaming</i></div>"]:::context
  end
  subgraph ROWD[" "]
    direction LR
    D["<div style='width:360px;text-align:center'>Persistent Shared Feature Outlet<br/>Cumulative feature state</div>"]:::layer
    D_gap[" "]:::spacer
    D_info["<div style='width:320px;text-align:left'><b>FEATURE & RELIABILITY STATE</b><br/><i>Spark · Parquet</i></div>"]:::context
  end
  subgraph ROWE[" "]
    direction LR
    E["<div style='width:360px;text-align:center'>Generation / model orchestration<br/>RUL + Survival modeling</div>"]:::layer
    E_gap[" "]:::spacer
    E_info["<div style='width:320px;text-align:left'><b>ORCHESTRATION & MODELING</b><br/><i>Airflow · Spark · XGBoost<br/>scikit-learn · scikit-survival</i></div>"]:::context
  end
  subgraph ROWF[" "]
    direction LR
    F["<div style='width:360px;text-align:center'>Current RUL / Survival models<br/>Current predictions<br/>Operational dashboard</div>"]:::layer
    F_gap[" "]:::spacer
    F_info["<div style='width:320px;text-align:left'><b>SERVING & MONITORING</b><br/><i>Airflow · PostgreSQL · Streamlit</i></div>"]:::context
  end

  A --> B --> C --> D --> E --> F
  style ROWA fill:none,stroke:none
  style ROWB fill:none,stroke:none
  style ROWC fill:none,stroke:none
  style ROWD fill:none,stroke:none
  style ROWE fill:none,stroke:none
  style ROWF fill:none,stroke:none
```

**[View the detailed architecture →](docs/architecture.md)**

## Model Performance

### RUL fixed-test results

| Generation | Selected family | Arrived cohort | MAE (cycles) | RMSE (cycles) | R² |
|---|---:|---:|---:|---:|---:|
| 1.0 | MLP | 26 | 98.04 | 150.27 | 0.6684 |
| 1.1 | XGBoost | 51 | 46.71 | 71.77 | 0.9244 |
| 1.2 | XGBoost | 76 | 40.91 | 63.85 | 0.9401 |
| 1.3 | XGBoost | 94 | 39.41 | 62.59 | 0.9425 |

### Survival fixed-test results

| Generation | Selected family | Arrived cohort | Integrated Brier ↓ | IPCW C-index ↑ |
|---|---:|---:|---:|---:|
| 1.0 | RSF | 26 | 0.02397 | 0.8372 |
| 1.1 | RSF | 51 | 0.02101 | 0.7923 |
| 1.2 | RSF | 76 | 0.01929 | 0.8199 |
| 1.3 | RSF | 94 | 0.01966 | 0.8349 |

MAE/RMSE and integrated Brier are lower-is-better; R² and IPCW C-index are higher-is-better. Validation selects model family/configuration; the fixed test set is held out from training and selection.

## How It Works

1. Deterministic MATR cycle and lifecycle data are replayed through Kafka.
2. Spark Structured Streaming finalizes prefix-complete cycles and appends rows to the persistent Shared Feature Outlet.
3. Airflow trains and evaluates RUL and Survival generation models from that shared outlet.
4. Selected Current models publish predictions to PostgreSQL, where the Dashboard reads them for monitoring.

## Tech Stack

Python · PySpark · Kafka · Spark Structured Streaming · Airflow · PostgreSQL · XGBoost · scikit-learn · scikit-survival · Streamlit · Docker

## Setup

```sh
cp .env.example .env
python scripts/preflight.py --profile full --tracked
```

The setup commands only prepare and validate the local environment; they do not run the full platform.

**Run the Full Platform → [docs/environment.md](docs/environment.md)**

## Data & Evaluation

BatteryLife MATR is the laboratory data source used to simulate progressive telemetry; this project is not a live production-telemetry system. The canonical cohort contains 169 batteries, including 101 training, 34 validation, and 34 test batteries. Validation and test batteries are excluded from training; validation selects models, while the fixed test benchmark is held out for final reporting.

## Scope / License

This is a reproducible laboratory-data demonstration of battery reliability and predictive analytics, not a production deployment. The MIT License covers this repository's code. BatteryLife/MATR archives and labels are externally sourced and remain subject to their original provider/source terms.
