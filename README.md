# EV Fleet Reliability

## Project Objective

Build an end-to-end EV fleet telemetry pipeline to predict battery failure risk using simulated fleet data.

## Telemetry Schema

### Vehicle metadata
- vehicle_id
- battery_age_days
- battery_type
- region

### Time-series telemetry
- vehicle_id
- timestamp
- soc
- pack_voltage
- pack_current
- module_temp_min
- module_temp_max
- outside_temp
- odometer
- is_charging

## Data Sources

The synthetic fleet telemetry schema is informed by Tesla's publicly documented Fleet Telemetry fields. Public battery datasets, including NASA battery aging data, will be used as references for realistic battery behavior and degradation patterns.

## Local Kafka MVP

Start the local broker and create its three-partition telemetry topic:

```sh
docker compose up -d
```

After generating `data/processed/synthetic_fleet_telemetry.parquet`, install the Python dependencies and verify the first 1,000 events:

```sh
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/kafka_producer.py --limit 1000
.venv/bin/python src/kafka_consumer.py --max-messages 1000
```

The producer uses `vehicle_id` as the Kafka key, so all events for a vehicle go to the same partition and retain their order. The consumer prints a partition and offset for every event, then commits that offset only after printing succeeds. Use `--limit 0` or `--max-messages 0` to process the complete stream.

## PySpark batch foundations

Build the Spark 4.1.3 image, start a small standalone cluster, then submit the batch feature job:

```sh
docker compose build spark-master
docker compose up -d spark-master spark-worker-1 spark-worker-2
docker compose run --rm spark-submit
docker compose run --rm -e SPARK_TEST_MASTER=spark://spark-master:7077 spark-submit python3 -m pytest -q
```

The image uses Apache Spark/PySpark 4.1.3 with its bundled Java 17 runtime, so
the host `JAVA_HOME` is not used. It runs one master and two one-core workers;
the repository is mounted at `/opt/project` in all Spark containers. The job
writes labeled features to `data/processed/spark_batch_features/`.

Its `spark.sql.shuffle.partitions=3` setting keeps plans small enough to
inspect. It is independent of the Kafka topic's three partitions: shuffle
partitions are Spark query tasks, while Kafka partitions are broker log shards.
Open the master UI at `http://localhost:8080` and worker UIs at
`http://localhost:8081` and `http://localhost:8082` while a job is running.

Use the job output and `notebooks/03_pyspark_batch.ipynb` (from the mounted
submit-container environment) to observe:

- lazy transformations becoming work at `count`, `show`, and `write` actions;
- repartitioning and the shuffle created by the label join;
- cache reuse for the telemetry branch used by both features and vehicle metadata;
- the explicit broadcast join of the real `vehicle_id` / `battery_type` / `region` dimension;
- vehicle history windows and a deliberately skewed `is_charging` repartition.

This is a batch-only milestone. Kafka Structured Streaming comes after these
plans and data movements are familiar; its event-time windows are not a direct
replacement for the row-based `lag` window used here.

## Airflow batch orchestration

Airflow 3.3.1 provides a manually triggered local DAG for the existing batch
chain: NASA download, cycle parsing, fleet simulation, then Spark feature
generation. Start the Spark cluster separately, then start Airflow:

```sh
docker compose build spark-master airflow
docker compose up -d spark-master spark-worker-1 spark-worker-2 airflow
```

Open `http://localhost:8083`, use the one-time credentials printed by
`docker compose logs airflow`, and trigger `fleet_batch_pipeline` manually.
The DAG permits one active run because every task regenerates the existing
fixed paths under `data/processed/`; retries are therefore safe and do not
create competing artifacts.

The Airflow container runs on the same Compose network and submits directly to
`spark://spark-master:7077`; it does not control the Spark services or require
Docker socket access. Kafka, the Kafka producer/consumer, and Structured
Streaming remain independently run services and are intentionally outside this
DAG. Model comparison also remains manual so its frozen evaluation reports are
not casually regenerated.

## Kafka → PySpark Structured Streaming

Start Kafka and the existing Spark cluster, publish telemetry, then run the
bounded streaming job:

```sh
docker compose up -d kafka spark-master spark-worker-1 spark-worker-2
.venv/bin/python src/kafka_producer.py --limit 300
docker compose run --rm spark-stream-submit
docker compose run --rm spark-submit /opt/spark/bin/spark-sql --master spark://spark-master:7077 -e 'SELECT * FROM parquet.`data/processed/spark_streaming_windows` LIMIT 20'
```

The query reads `vehicle_telemetry` in 100-record micro-batches, parses the
JSON schema, treats simulator timestamps as UTC event time, and writes
six-hour per-vehicle Parquet windows. It uses a two-hour event-time watermark
and `event_id` deduplication before aggregation, so repeated producer sends do
not increase `event_count` while their event times are within the watermark.

The checkpoint at `data/processed/spark_streaming_checkpoint/` owns Kafka
offset recovery and sink commits. Re-run the same command to resume without
replaying committed offsets. To intentionally replay from earliest offsets,
remove both that checkpoint and `data/processed/spark_streaming_windows/`.

## Battery failure model comparison

Run the local model comparison after generating the PySpark feature dataset:

```sh
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/model_training.py
```

The script makes vehicle-disjoint train/validation/test cohorts over the shared
calendar horizon, stratified by region, battery type, and first-positive-risk
timing band. It compares logistic regression, random forest, and XGBoost, and
writes the frozen primary evaluation to
`data/processed/primary_model_comparison_metrics.json`. It separately writes a
time-only stress test for the primary-selected configuration to
`data/processed/temporal_stress_metrics.json`; that stress result cannot change
the selected model. On macOS with a nonstandard Homebrew prefix, install `libomp`
and launch it with:

```sh
brew install libomp
DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix libomp)/lib" .venv/bin/python src/model_training.py
```

### Verified ML milestone

The primary task is predicting `failure_within_30_operating_days` for unseen
vehicles. The authoritative benchmark uses exact 60/20/20 vehicle-disjoint
cohorts over the shared 2025-01-01 through 2025-03-31 horizon. Vehicles are
stratified by `region × battery_type × first_positive_timing_band` where
timing is by first positive-risk row (`no_eol`, `early_positive_window`,
`mid_positive_window`, `late_positive_window`, `after_primary_horizon`). This
split is now **frozen** for future evaluation and must not be re-allocated based
on model performance.

Train/validation/test prevalence are 8.09%, 7.49%, and 7.79%, respectively.
Logistic regression was selected by validation PR-AUC (0.134), ahead of all
candidates.

The selected primary result is ROC-AUC 0.790 and PR-AUC 0.234 against 7.79%
test prevalence. The default operational threshold is fixed at `0.50`: on
validation it gives 13.5% precision, 55.4% recall, and a 30.7% alert rate; on
the untouched test set it gives 17.3% precision, 73.9% recall, and a 33.3%
alert rate (confusion matrix `[[26987, 11509], [850, 2402]]`). The former
recall-first threshold (`0.268`) remains an optional diagnostic only (80.0%
validation recall, 10.2% precision, 58.7% alert rate) and is not the alert
policy. Random forest and XGBoost show severe train/validation overfitting
gaps, with near-perfect train PR-AUC but validation PR-AUC at or below 0.092.

Post-split distribution audit is mostly comparable for key features, with minor
imbalances at the vehicle level. The largest notable imbalance is battery age:
validation vehicles are younger on average than test vehicles. This was retained as
an accepted limitation because battery age has weak association with the target
and the remaining imbalance is consistent with small-sample stratum allocation
noise rather than a proven distributional flaw.

The temporal stress test is diagnostic only: ROC-AUC collapses to 0.497 under
the fixed chronological windows. At the fixed 0.50 threshold it produces
17.9% precision, 84.8% recall, and a 90.7% alert rate on stress-test data.
The synthetic seasonal/temporal feature shift
and changing label prevalence make this a robustness limitation, not a
model-selection result. Do not add Airflow, PostgreSQL, Tableau, or MLOps in
this milestone.
