"""Replay canonical MATR measurements to Kafka."""

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds
try:
    from .battery_events import event_from_measurement
except ImportError:
    from battery_events import event_from_measurement

try:
    from confluent_kafka import Producer
except ImportError:  # Lets unit tests run before the optional runtime dependency is installed.
    Producer = None


TOPIC = "battery_measurements"
DEFAULT_INPUT = Path("data/processed/matr/cycle_measurements")


def telemetry_rows(path, limit, battery_ids=None, limit_per_battery=0):
    """Yield bounded measurement rows without loading the canonical corpus into memory."""
    yielded = 0
    per_battery = {}
    dataset = ds.dataset(path, format="parquet")
    for fragment in dataset.get_fragments():
        for batch in fragment.to_batches(batch_size=1000):
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
        event = event_from_measurement(row)
        producer.produce(
            topic=TOPIC,
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
    parser.add_argument("--bootstrap-server", default="localhost:9092")
    parser.add_argument("--limit", type=int, default=1000, help="Rows to send; 0 sends all rows.")
    parser.add_argument("--battery-id", action="append", dest="battery_ids", help="Restrict bounded verification replay to one or more batteries.")
    parser.add_argument("--limit-per-battery", type=int, default=0, help="Maximum events per selected battery; 0 disables the per-battery bound.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be zero or positive")
    if Producer is None:
        raise SystemExit("Install dependencies first: .venv/bin/pip install -r requirements.txt")

    producer = Producer(
        {
            "bootstrap.servers": args.bootstrap_server,
            "enable.idempotence": True,
            "acks": "all",
        }
    )
    sent = produce_rows(telemetry_rows(args.input, args.limit, set(args.battery_ids or []), args.limit_per_battery), producer, args.limit)
    print(f"Delivered {sent} event(s) to {TOPIC}.")


if __name__ == "__main__":
    main()
