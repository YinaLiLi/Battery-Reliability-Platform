import json

import pyarrow as pa
import pyarrow.parquet as pq

import pytest

from src.kafka_consumer import consume_messages
from src.kafka_producer import LIFECYCLE_TOPIC, produce_rows, scheduled_replay_events, telemetry_rows


def event(battery_id="MATR-1", event_id="one"):
    return {"event_id": event_id, "dataset": "MATR", "battery_id": battery_id, "cycle_index": 1, "sample_index": 0, "replay_event_time": "2020-01-01T00:00:00+00:00"}


class FakeProducer:
    def __init__(self, callback_error=None):
        self.callback_error = callback_error
        self.produced = []
        self.poll_calls = 0
        self.flushed = False

    def produce(self, **kwargs):
        self.produced.append(kwargs)
        kwargs["on_delivery"](self.callback_error, None)

    def poll(self, timeout):
        self.poll_calls += 1

    def flush(self):
        self.flushed = True
        return 0


class FakeMessage:
    def __init__(self, value, partition=1, offset=4):
        self._value = value
        self._partition = partition
        self._offset = offset

    def error(self):
        return None

    def value(self):
        return self._value

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset


class FakeConsumer:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.committed = []

    def poll(self, timeout):
        return next(self.messages, None)

    def commit(self, message, asynchronous=False):
        self.committed.append((message, asynchronous))


def test_producer_limits_rows_and_flushes_after_asynchronous_delivery():
    producer = FakeProducer()

    sent = produce_rows(
        [event(), event("MATR-2", "two")],
        producer,
        limit=1,
    )

    assert sent == 1
    assert producer.flushed
    assert producer.poll_calls == 1
    assert producer.produced[0]["key"] == b"MATR-1"
    assert json.loads(producer.produced[0]["value"])["schema_version"] == "1.0"


def test_producer_raises_after_flush_when_delivery_fails():
    producer = FakeProducer(callback_error=RuntimeError("broker unavailable"))

    with pytest.raises(RuntimeError, match="delivery failed"):
        produce_rows([event()], producer, limit=0)

    assert producer.flushed


def test_scheduled_replay_interleaves_batteries_and_emits_separate_lifecycle_events():
    manifest = [
        {"battery_id": "late", "start_time": "2020-01-02T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 1, "eol_cycle": 1, "valid_eol_label": True},
        {"battery_id": "early", "start_time": "2020-01-01T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 1, "eol_cycle": 2, "valid_eol_label": False},
    ]
    measurements = [
        {**event("late", "late-1"), "source_time_in_s": 0.0},
        {**event("early", "early-1"), "source_time_in_s": 0.0},
    ]

    replay = list(scheduled_replay_events(measurements, manifest))

    assert [(row.get("event_type"), row["battery_id"], row["replay_sequence"]) for row in replay] == [
        (None, "early", 0), ("cycle_complete", "early", 1), ("replay_complete", "early", 2),
        (None, "late", 3), ("cycle_complete", "late", 4), ("eol_observed", "late", 5), ("replay_complete", "late", 6),
    ]
    producer = FakeProducer()
    assert produce_rows(replay, producer, limit=0) == 7
    assert [record["topic"] for record in producer.produced].count(LIFECYCLE_TOPIC) == 5


def test_bounded_telemetry_rows_uses_battery_filter_before_reading_rows(tmp_path):
    pq.write_table(pa.Table.from_pylist([{**event("wanted", "wanted-1"), "source_time_in_s": 0.0}]), tmp_path / "wanted.parquet")
    pq.write_table(pa.Table.from_pylist([{**event("other", "other-1"), "source_time_in_s": 0.0}]), tmp_path / "other.parquet")

    assert [row["battery_id"] for row in telemetry_rows(tmp_path, 0, {"wanted"}, 0)] == ["wanted"]


def test_consumer_prints_then_commits_each_successful_message():
    message = FakeMessage(json.dumps({"battery_id": "MATR-1"}).encode())
    consumer = FakeConsumer([message])
    printed = []

    processed = consume_messages(consumer, max_messages=1, output=printed.append)

    assert processed == 1
    assert consumer.committed == [(message, False)]
    assert json.loads(printed[0]) == {
        "offset": 4,
        "partition": 1,
        "value": {"battery_id": "MATR-1"},
    }
