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
