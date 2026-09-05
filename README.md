# Battery Reliability & Predictive Analytics Platform

An end-to-end battery reliability and predictive analytics platform that processes progressive telemetry into time-consistent battery health state, RUL and survival predictions, and operational monitoring. BatteryLife MATR laboratory data is replayed through Kafka to simulate progressive real-world battery telemetry and lifecycle arrival for reproducible development and evaluation; this repository does not receive live production telemetry.

## Results at a Glance

Evaluation uses the canonical BatteryLife MATR laboratory corpus (169 batteries, 140,001 cycles, and 130,573,638 measurements). The fixed benchmark is held out from training and generation-cutoff selection.

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

MAE/RMSE and integrated Brier are lower-is-better; R² and IPCW C-index are higher-is-better. Model-family selection uses validation data only.

## Architecture

```mermaid
flowchart TD
    subgraph DATA["DATA"]
        MATR["BatteryLife MATR laboratory data"] --> CANON["Canonical cycle and measurement data"]
        CANON --> REPLAY["Deterministic progressive replay"]
        REPLAY --> KAFKA["Kafka telemetry and lifecycle events"]
    end

    subgraph STATE["STREAMING & STATE"]
        KAFKA --> SPARK["Spark Structured Streaming"]
        SPARK --> FINAL["Prefix-complete finalized state"]
        FINAL --> OUTLET["Persistent append-only Shared Feature Outlet"]
    end

    subgraph MODEL["MODEL TRAINING"]
        AIRFLOW["Airflow orchestration"] --> SELECT["Cumulative Generation ID selection"]
        OUTLET --> SELECT
        SELECT --> RULTRAIN["RUL training and evaluation"]
        SELECT --> SURVTRAIN["Survival training and evaluation"]
        RULTRAIN --> RULCAND["RUL candidate models"]
        SURVTRAIN --> SURVCAND["Survival candidate models"]
        BENCH["Fixed validation and test benchmark"] -.-> RULTRAIN
        BENCH -.-> SURVTRAIN
    end

    subgraph SERVE["CURRENT SERVING"]
        OUTLET --> CURRENT["Newest finalized cumulative rows"]
        CURRENT --> RULMODEL["Current RUL model"]
        CURRENT --> SURVMODEL["Current Survival model"]
        RULMODEL --> PG["PostgreSQL"]
        SURVMODEL --> PG
        PG --> DASH["Streamlit monitoring dashboard"]
    end

    classDef component fill:transparent,stroke:#9ca3af,color:#374151;
    class MATR,CANON,REPLAY,KAFKA,SPARK,FINAL,OUTLET,AIRFLOW,SELECT,RULTRAIN,SURVTRAIN,RULCAND,SURVCAND,BENCH,CURRENT,RULMODEL,SURVMODEL,PG,DASH component;
    style DATA fill:transparent,stroke:#5b6573,stroke-width:3px,stroke-dasharray:8 4;
    style STATE fill:transparent,stroke:#5b6573,stroke-width:3px,stroke-dasharray:8 4;
    style MODEL fill:transparent,stroke:#5b6573,stroke-width:3px,stroke-dasharray:8 4;
    style SERVE fill:transparent,stroke:#5b6573,stroke-width:3px,stroke-dasharray:8 4;
```

The Shared Feature Outlet is the common upstream outlet for both model families and current inference. New finalized cycle rows are appended once. Generation *N* consumes all outlet rows with `generation_id <= N`; RUL and Survival then apply their own labels, event semantics, and model-specific transformations to the same selected rows.

## Progressive Telemetry Simulation

BatteryLife MATR is the current laboratory source used to exercise the platform. Its canonical cycle and measurement data are replayed progressively through Kafka, with lifecycle events and cycle-completion boundaries preserved in time order. Spark streaming deduplicates events and admits only prefix-complete cycles, preventing later telemetry from changing an earlier finalized state.

## Modeling and Evaluation

State-bound features use only the current cycle and prior finalized history. RUL predicts remaining cycles to the source-supported approximately 80% SOH endpoint. Survival models estimate event and censoring behavior over time using Cox proportional hazards and Random Survival Forest candidates.

The platform evaluates multiple candidate families against a fixed, immutable benchmark of 34 validation batteries and 34 test batteries. Validation selects a family/configuration; the fixed test set is evaluated once for the selected winner and is never used for training or streaming cutoff selection.

## Generation IDs and Continuous Training

Airflow validates the finalized state, selects the cumulative shared-outlet rows for a generation, and fans those same rows out to parallel RUL and Survival training. A later generation appends only newly finalized rows; it does not rebuild or copy the historical feature dataset. Existing candidate models remain immutable, while Current Model selections are managed independently.

## Dashboard and Operational Use

PostgreSQL is the serving boundary for current RUL and Survival predictions, state availability, and model health. The Streamlit dashboard provides battery-level detail, current health and RUL views, survival horizons, cohort context, and operational model status. Candidate and historical predictions remain available for comparison and audit.

## Quick Start

The supported environment is Python **>=3.10,<3.14**, Java **>=17**, Docker Compose with the required Compose Specification capabilities, and Linux containers. Use Docker Desktop on macOS; use Docker Desktop with WSL2 and a checkout in the WSL filesystem on Windows. See [the environment guide](docs/environment.md) for prerequisites, data acquisition, service URLs, and clean-clone commands.

```sh
python scripts/preflight.py --profile unit
python -m pytest --tier unit -q
cp .env.example .env
python scripts/preflight.py --profile compose
docker compose up -d
```

Obtain the BatteryLife MATR laboratory archive separately, then follow the data bootstrap commands in [docs/environment.md](docs/environment.md). The repository’s compatibility workflow has validated the supported Python, Spark, Survival, and container paths; desktop full-stack runs remain environment-specific acceptance gates.

## Scope and Limitations

This is a reproducible laboratory-data demonstration, not a live field-telemetry deployment. Kafka replay simulates progressive arrival deterministically; it does not represent production network behavior or field coverage. Results depend on the current laboratory cohort, feature contract, and benchmark definitions. The complete demonstration includes RUL, conditional Survival, parallel training, PostgreSQL serving, a Linux `amd64` Survival runtime, and Streamlit monitoring.
