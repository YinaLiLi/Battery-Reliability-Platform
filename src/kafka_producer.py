"""Replay canonical MATR measurements to Kafka."""

import argparse
import json
import os
from pathlib import Path

import pyarrow.dataset as ds
try:
    from .battery_events import event_from_measurement
    from .continuous_arrival import lifecycle_events_for_manifest, schedule_measurement
except ImportError:
    from battery_events import event_from_measurement
    from continuous_arrival import lifecycle_events_for_manifest, schedule_measurement

try:
    from confluent_kafka import Producer
except ImportError:  # Lets unit tests run before the optional runtime dependency is installed.
    Producer = None


TOPIC = "battery_measurements"
LIFECYCLE_TOPIC = "battery_lifecycle"
DEFAULT_INPUT = Path("data/processed/matr/cycle_measurements")
DEFAULT_MANIFEST = Path("data/processed/matr/arrival_manifest.parquet")


def telemetry_rows(path, limit, battery_ids=None, limit_per_battery=0):
    """Yield bounded measurement rows without loading the canonical corpus into memory."""
    yielded = 0
    per_battery = {}
    dataset = ds.dataset(path, format="parquet")
    filter_expression = ds.field("battery_id").isin(sorted(battery_ids)) if battery_ids else None
    for fragment in dataset.get_fragments(filter=filter_expression):
        for batch in fragment.to_batches(batch_size=1000, filter=filter_expression):
            for row in batch.to_pylist():
                if battery_ids and limit_per_battery and all(per_battery.get(key, 0) >= limit_per_battery for key in battery_ids):
                    return
                if battery_ids and row["battery_id"] not in battery_ids:
                    continue
                if limit_per_battery and per_battery.get(row["battery_id"], 0) >= limit_per_battery:
                    break
                if limit and yielded >= limit:
                    return
                yield row
                yielded += 1
                per_battery[row["battery_id"]] = per_battery.get(row["battery_id"], 0) + 1


def scheduled_replay_events(measurements, manifest, *, include_lifecycle=True):
    """Schedule telemetry and independent lifecycle facts in one stable replay order."""
    by_battery = {row["battery_id"]: row for row in manifest}
    events = []
    completed_cycles = {}
    for row in measurements:
        scheduled = schedule_measurement(row, by_battery[row["battery_id"]], replay_sequence=0)
        events.append((0, scheduled))
        key = (scheduled["battery_id"], scheduled["cycle_index"])
        completed = completed_cycles.setdefault(key, {"last": scheduled, "count": 0})
        completed["count"] += 1
        if scheduled["replay_event_time"] > completed["last"]["replay_event_time"]:
            completed["last"] = scheduled
    for completed in completed_cycles.values():
        scheduled = completed["last"]
        events.append((1, {
            "event_id": f"matr-lifecycle:{scheduled['battery_id']}:cycle_complete:{scheduled['cycle_index']}",
            "event_type": "cycle_complete",
            "dataset": scheduled["dataset"],
            "battery_id": scheduled["battery_id"],
            "cycle_index": scheduled["cycle_index"],
            "expected_telemetry_rows": completed["count"],
            "replay_event_time": scheduled["replay_event_time"],
            "schema_version": "1.0",
        }))
    if include_lifecycle:
        for manifest_row in manifest:
            for lifecycle in lifecycle_events_for_manifest(manifest_row):
                events.append((2 if lifecycle["event_type"] == "eol_observed" else 3, lifecycle))
    for sequence, (_, event) in enumerate(sorted(events, key=lambda item: (item[1]["replay_event_time"], item[0], item[1]["battery_id"], item[1].get("cycle_index", 0), item[1].get("sample_index", 0)))):
        yield {**event, "replay_sequence": sequence}


def produce_rows(rows, producer, limit=1000):
    """Queue telemetry for asynchronous delivery and raise only after flushing."""
    delivery_errors = []

    def delivered(error, _message):
        if error is not None:
            delivery_errors.append(str(error))

    sent = 0
    for row in rows:
        if limit and sent >= limit:
            break
        event = row if row.get("event_type") else event_from_measurement(row)
        producer.produce(
            topic=LIFECYCLE_TOPIC if event.get("event_type") else TOPIC,
            key=event["battery_id"].encode(),
            value=json.dumps(event, separators=(",", ":")),
            on_delivery=delivered,
        )
        producer.poll(0)
        sent += 1

    undelivered = producer.flush()
    if undelivered:
        delivery_errors.append(f"{undelivered} message(s) were not delivered")
    if delivery_errors:
        raise RuntimeError(f"Kafka delivery failed: {'; '.join(delivery_errors)}")
    return sent


def parse_args():
    parser = argparse.ArgumentParser(description="Replay MATR measurements to Kafka.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bootstrap-server", default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    parser.add_argument("--limit", type=int, default=0, help="Rows to send; 0 sends all rows.")
    parser.add_argument("--battery-id", action="append", dest="battery_ids", help="Restrict bounded verification replay to one or more batteries.")
    parser.add_argument("--limit-per-battery", type=int, default=0, help="Maximum events per selected battery; 0 disables the per-battery bound.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be zero or positive")
    if Producer is None:
        raise SystemExit("Install dependencies first: python -m pip install -r requirements.txt")

    producer = Producer(
        {
            "bootstrap.servers": args.bootstrap_server,
            "enable.idempotence": True,
            "acks": "all",
        }
    )
    measurements = telemetry_rows(args.input, args.limit, set(args.battery_ids or []), args.limit_per_battery)
    manifest = ds.dataset(args.manifest, format="parquet").to_table().to_pylist()
    if args.battery_ids:
        manifest = [row for row in manifest if row["battery_id"] in set(args.battery_ids)]
    sent = produce_rows(scheduled_replay_events(measurements, manifest, include_lifecycle=not args.limit and not args.limit_per_battery), producer, 0)
    print(f"Delivered {sent} scheduled event(s) to {TOPIC} and {LIFECYCLE_TOPIC}.")


if __name__ == "__main__":
    main()
