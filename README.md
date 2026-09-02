# Battery Reliability Platform

An end-to-end battery reliability platform built on BatteryLife MATR historical aging data. It processes 130,573,638 laboratory measurement rows from 169 batteries across 140,001 cycles with PySpark, supports deterministic simulated replay and Spark Structured Streaming battery-health state, predicts remaining useful life (RUL) with XGBoost, and serves health and model monitoring through PostgreSQL and Streamlit.

The platform is designed for battery reliability monitoring and predictive maintenance. SOH is a measured/derived health metric, while RUL is the ML-predicted number of cycles remaining.

SOH is a deterministic measured health metric, not an ML target:

`soh = discharge_capacity_in_Ah / nominal_capacity_in_Ah / SOC_width`

The current discharge-capacity signal reconstructs this value directly in MATR. The only ML target is remaining useful life (RUL) in cycles.

## Architecture

`MATR.zip → canonical Parquet → PySpark causal degradation features → XGBoost RUL evaluation/predictions → PostgreSQL → Streamlit`

`arrival manifest + deterministic staggered replay → Kafka battery_measurements → Structured Streaming cycle health → PostgreSQL`

Airflow orchestrates the batch path:

`ingest_matr → normalize_matr → build_degradation_features → train_evaluate_models → publish_predictions → load_serving_tables`

Continuous arrival and retraining adds:

- deterministic, staggered battery start-time simulation from `arrival_manifest.parquet`
- deterministic `replay_complete` / `eol_observed` lifecycle events
- training eligibility based only on valid observed endpoints
- threshold-driven cumulative training snapshots `26 → 51 → 76 → 94` with fixed split cohorts
- candidate-only `matr_continuous_retraining` refresh when a threshold is crossed
- idempotent refresh behavior when the same snapshot is reached again

Kafka replays historical measurements deterministically; it is not real-time field telemetry.

## Data and artifacts

The verified official archive is `data/raw/batterylife/MATR.zip`. Canonical outputs under `data/processed/matr/` contain 169 batteries, 140,001 cycles, and 130,573,638 measurement rows.

- Parquet is canonical historical storage.
- PySpark builds causal degradation features over roughly 130M measurements.
- Kafka provides deterministic historical replay keyed by `battery_id`.
- Spark Structured Streaming materializes deduplicated battery/cycle health state.
- Measured discharge capacity produces SOH; XGBoost predicts RUL.
- Airflow orchestrates batch feature engineering and model-refresh workflows.
- PostgreSQL serves compact health, prediction, replay-window, and evaluation snapshots to the dashboard.
- Streamlit provides battery reliability and model monitoring views.

The frozen lineage-disjoint benchmark cohorts are under `data/processed/matr/` and use fixed validation/test IDs across all generations.

Genuine XGBoost generations and constrained validation test MAE are:

- `matr-rul-xgboost-1.0-*` → 26 training batteries, Test MAE 50.81
- `matr-rul-xgboost-1.1-*` → 51 training batteries, Test MAE 43.81
- `matr-rul-xgboost-1.2-*` → 76 training batteries, Test MAE 39.89
- `matr-rul-xgboost-1.3-*` → 94 training batteries, Test MAE 38.70

Legacy full-pool generations are retained in PostgreSQL as retired audit rows.

Operational serving always enforces non-negative RUL and irreversible EOL behavior (first predicted EOL is frozen) before serving.

## Local services

Create a local `.env` from `.env.example`, then start the retained Docker services with `docker compose up -d`. The database is `battery_reliability`; its password belongs only in the ignored `.env`.

The primary batch DAG (`matr_reliability_pipeline`) reuses canonical MATR outputs and does not manage Kafka or the long-running streaming job. The continuous retraining DAG (`matr_continuous_retraining`) is conditional and no longer auto-promotes: it creates candidate generations only when thresholds are crossed and requires a manual champion promotion action.

Task-level `airflow dags test` verification has been used for both DAGs; the standalone local scheduler has experienced memory/OOM pressure and is not production-scheduler verification.

## Focused checks

Run the focused suite with `pytest -q tests/test_matr_data.py tests/test_matr_stage2.py tests/test_battery_events.py tests/test_kafka_streaming.py tests/test_spark_streaming.py tests/test_postgres_loader.py tests/test_airflow_dag.py`. Spark tests require Java and the project Python environment. Validate Compose with `docker compose config`.
