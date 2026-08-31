"""Print synthetic fleet telemetry from Kafka with explicit offset commits."""

import argparse
import json

try:
    from confluent_kafka import Consumer
except ImportError:  # Lets unit tests run before the optional runtime dependency is installed.
    Consumer = None


TOPIC = "vehicle_telemetry"


def consume_messages(consumer, max_messages=1000, output=print):
    """Print and commit at most max_messages; zero means consume indefinitely."""
    processed = 0
    while not max_messages or processed < max_messages:
        message = consumer.poll(1.0)
        if message is None:
            continue
        if message.error():
            raise RuntimeError(f"Kafka consume failed: {message.error()}")

        output(
            json.dumps(
                {
                    "partition": message.partition(),
                    "offset": message.offset(),
                    "value": json.loads(message.value()),
                },
                sort_keys=True,
            )
        )
        consumer.commit(message=message, asynchronous=False)
        processed += 1
    return processed


def parse_args():
    parser = argparse.ArgumentParser(description="Print synthetic fleet telemetry from Kafka.")
    parser.add_argument("--bootstrap-server", default="localhost:9092")
    parser.add_argument("--max-messages", type=int, default=1000, help="Messages to print; 0 runs indefinitely.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_messages < 0:
        raise SystemExit("--max-messages must be zero or positive")
    if Consumer is None:
        raise SystemExit("Install dependencies first: .venv/bin/pip install -r requirements.txt")

    consumer = Consumer(
        {
            "bootstrap.servers": args.bootstrap_server,
            "group.id": "telemetry-console",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([TOPIC])
    try:
        print(f"Consuming up to {args.max_messages or 'unlimited'} event(s) from {TOPIC}.")
        consume_messages(consumer, args.max_messages)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
