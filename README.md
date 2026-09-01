# Battery Reliability Platform — BatteryLife MATR

This project processes BatteryLife's 169-cell MATR corpus into a battery-reliability platform: measured cycle health (SOH), RUL prediction, deterministic historical replay, streaming health state, and model-serving snapshots for predictive maintenance.

SOH is a deterministic measured health metric, not an ML target:

`soh = discharge_capacity_in_Ah / nominal_capacity_in_Ah / SOC_width`

The current discharge-capacity signal reconstructs this value directly in MATR. The only ML target is remaining useful life (RUL) in cycles.

## Architecture

`MATR.zip → canonical Parquet → PySpark causal degradation features → XGBoost RUL evaluation/predictions → PostgreSQL`

`canonical measurement replay → Kafka battery_measurements → Structured Streaming cycle health → PostgreSQL`

Airflow orchestrates the batch path:

`ingest_matr → normalize_matr → build_degradation_features → train_evaluate_models → publish_predictions → load_serving_tables`

The historical Kafka path is a deterministic simulation of source measurement order; MATR does not provide real wall-clock telemetry.

## Data and artifacts

The verified official archive is `data/raw/batterylife/MATR.zip`. Canonical outputs under `data/processed/matr/` contain 169 batteries, 140,001 cycle-summary rows, and 130,573,638 measurements.

- Parquet is canonical historical storage.
- PySpark builds causal degradation features over roughly 130M measurements.
- Kafka provides a replay and streaming-ingestion abstraction keyed by `battery_id`.
- Structured Streaming materializes deduplicated battery/cycle health state.
- ML predicts RUL; measured capacity produces SOH.
- Airflow performs periodic batch feature and model refreshes.
- PostgreSQL serves compact health, prediction, replay-window, and evaluation snapshots.

The frozen lineage-disjoint benchmark and model reports are under `data/processed/matr/`. The selected RUL XGBoost result is validation MAE/RMSE/R² of 56.96 cycles / 95.57 / 0.9477 and test 38.98 / 58.20 / 0.9503. Test MAE is 64.12 early-life, 44.96 mid-life, and 13.21 late-life cycles.

## Local services

Create a local `.env` from `.env.example`, then start the retained Docker services with `docker compose up -d`. The database is `battery_reliability`; its password belongs only in the ignored `.env`.

The Airflow DAG is manual (`matr_reliability_pipeline`) and limits active runs to one. It does not manage Kafka or the long-running streaming job. Canonical MATR outputs may be reused by the DAG to keep local memory use bounded. Task-level `airflow dags test` verification passed twice; the standalone local scheduler has experienced memory/OOM pressure and is not production-scheduler verification.

## Focused checks

Run the focused suite with `pytest -q tests/test_matr_data.py tests/test_matr_stage2.py tests/test_battery_events.py tests/test_kafka_streaming.py tests/test_spark_streaming.py tests/test_postgres_loader.py tests/test_airflow_dag.py`. Spark tests require Java and the project Python environment. Validate Compose with `docker compose config`.
