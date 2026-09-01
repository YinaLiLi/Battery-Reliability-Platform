# Battery Reliability Platform

An end-to-end battery reliability platform built on real BatteryLife MATR aging data. It processes 130,573,638 laboratory measurement rows from 169 batteries across 140,001 cycles with PySpark, supports deterministic Kafka replay and Spark Structured Streaming battery-health state, predicts remaining useful life (RUL) with XGBoost, and serves health and model monitoring through PostgreSQL and Streamlit.

The platform is designed for battery reliability monitoring and predictive maintenance. SOH is a measured/derived health metric, while RUL is the ML-predicted number of cycles remaining.

SOH is a deterministic measured health metric, not an ML target:

`soh = discharge_capacity_in_Ah / nominal_capacity_in_Ah / SOC_width`

The current discharge-capacity signal reconstructs this value directly in MATR. The only ML target is remaining useful life (RUL) in cycles.

## Architecture

`MATR.zip → canonical Parquet → PySpark causal degradation features → XGBoost RUL evaluation/predictions → PostgreSQL → Streamlit`

`canonical measurement replay → Kafka battery_measurements → Structured Streaming cycle health → PostgreSQL`

Airflow orchestrates the batch path:

`ingest_matr → normalize_matr → build_degradation_features → train_evaluate_models → publish_predictions → load_serving_tables`

Kafka replays the laboratory measurement order deterministically; it is not real-time field telemetry.

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

The frozen lineage-disjoint benchmark and model reports are under `data/processed/matr/`. The finalized XGBoost model has frozen test MAE/RMSE/R² of approximately 38.98 / 58.20 / 0.9503, with lifecycle MAE of 64.12 / 44.96 / 13.21 cycles (early/mid/late). After applying the physical serving constraint `predicted_rul_cycles = max(0, raw_predicted_rul_cycles)`, constrained operational test MAE is approximately 38.94 cycles and R² remains approximately 0.9503.

## Local services

Create a local `.env` from `.env.example`, then start the retained Docker services with `docker compose up -d`. The database is `battery_reliability`; its password belongs only in the ignored `.env`.

The Airflow DAG is manual (`matr_reliability_pipeline`) and limits active runs to one. It does not manage Kafka or the long-running streaming job. Canonical MATR outputs may be reused by the DAG to keep local memory use bounded. Task-level `airflow dags test` verification passed twice; the standalone local scheduler has experienced memory/OOM pressure and is not production-scheduler verification.

## Focused checks

Run the focused suite with `pytest -q tests/test_matr_data.py tests/test_matr_stage2.py tests/test_battery_events.py tests/test_kafka_streaming.py tests/test_spark_streaming.py tests/test_postgres_loader.py tests/test_airflow_dag.py`. Spark tests require Java and the project Python environment. Validate Compose with `docker compose config`.
