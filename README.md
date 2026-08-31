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
