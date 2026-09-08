# Battery Reliability & Predictive Analytics Platform

An end-to-end platform for turning progressive battery telemetry into time-consistent health state, Remaining Useful Life (RUL), and Survival predictions. It replays the laboratory BatteryLife MATR dataset—169 batteries, 140K cycles, and 130M+ measurements—through Kafka and Spark to demonstrate streaming ML, PostgreSQL serving, and operational monitoring.

## Highlights

- Processes 130M+ laboratory measurements across a 169-battery monitoring population.
- Runs deterministic progressive replay with Kafka and Spark Structured Streaming.
- Improves RUL MAE from ~98 cycles (Generation 1.0) to ~39 cycles (Generation 1.3); R² rises from ~0.67 to ~0.94.
- Trains RUL and Survival independently from the same persistent Shared Feature Outlet.
- Serves Current models and predictions through PostgreSQL and a Streamlit dashboard.
- Verified clean-fork workflow from raw MATR archives through the Dashboard.

## Dashboard

The working Streamlit dashboard covers fleet risk, battery detail, RUL model monitoring, and Survival model monitoring.

<!-- TODO: add a real dashboard screenshot from a verified run; do not use a synthetic image. -->

## Architecture

### Simplified flow

```mermaid
flowchart TB
  subgraph SRC["DATA SOURCE"]
    BAT["BatteryLife MATR Dataset"]
  end

  subgraph ING["DATA & INGESTION (Kafka)"]
    I1["Telemetry / Lifecycle Data"]
  end

  subgraph STR["STREAMING & STATE (Kafka · Spark Structured Streaming)"]
    ST1["Event processing"]
    ST2["Prefix-complete finalized battery state"]
    ST1 --> ST2
  end

  subgraph FS["FEATURE & RELIABILITY STATE (Spark · Parquet)"]
    F1["Persistent Shared Feature Outlet"]
    F2["Cumulative feature state"]
    F1 --> F2
  end

  subgraph OM["ORCHESTRATION & MODELING (Airflow · Spark · XGBoost · scikit-learn · scikit-survival)"]
    OM1["Generation / model orchestration"]
    OM2["parallel RUL and Survival modeling"]
    OM1 --> OM2
  end

  subgraph SM["SERVING & MONITORING (Airflow · PostgreSQL · Streamlit)"]
    S1["Current RUL / Survival models"]
    S2["Current predictions"]
    S3["Operational dashboard"]
    S1 --> S2
    S2 --> S3
  end

  BAT --> I1
  I1 --> ST1
  ST2 --> F1
  F2 --> OM1
  OM2 --> S1

classDef layerBox fill:none,stroke:#6b7280,stroke-width:2px,stroke-dasharray:4 4;
classDef sourceBox fill:#f8fafc,stroke:#374151,stroke-width:2px;
class SRC sourceBox
class ING layerBox
class STR layerBox
class FS layerBox
class OM layerBox
class SM layerBox
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

Install the public requirements, obtain the pinned MATR archives, and follow the complete normalization, streaming, training, serving, and Dashboard workflow in [docs/environment.md](docs/environment.md).

## Data & Evaluation

BatteryLife MATR is the laboratory data source used to simulate progressive telemetry; this project is not a live production-telemetry system. The canonical cohort contains 169 batteries, including 101 training, 34 validation, and 34 test batteries. Validation and test batteries are excluded from training; validation selects models, while the fixed test benchmark is held out for final reporting.

## Scope / License

This is a reproducible laboratory-data demonstration of battery reliability and predictive analytics, not a production deployment. The MIT License covers this repository's code. BatteryLife/MATR archives and labels are externally sourced and remain subject to their original provider/source terms.
