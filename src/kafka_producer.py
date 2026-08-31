"""Publish existing synthetic fleet telemetry to Kafka."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

try:
    from confluent_kafka import Producer
except ImportError:  # Lets unit tests run before the optional runtime dependency is installed.
    Producer = None


TOPIC = "vehicle_telemetry"
DEFAULT_INPUT = Path("data/processed/synthetic_fleet_telemetry.parquet")


def telemetry_rows(path, limit):
    """Yield up to limit rows without loading the complete fleet into memory."""
    yielded = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
        for row in batch.to_pylist():
            if limit and yielded >= limit:
                return
            yield row
            yielded += 1


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
        producer.produce(
            topic=TOPIC,
            key=row["vehicle_id"].encode(),
            value=json.dumps(row, separators=(",", ":")),
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
    parser = argparse.ArgumentParser(description="Publish synthetic fleet telemetry to Kafka.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bootstrap-server", default="localhost:9092")
    parser.add_argument("--limit", type=int, default=1000, help="Rows to send; 0 sends all rows.")
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
    sent = produce_rows(telemetry_rows(args.input, args.limit), producer, args.limit)
    print(f"Delivered {sent} event(s) to {TOPIC}.")


if __name__ == "__main__":
    main()
